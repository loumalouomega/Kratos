//    |  /           |
//    ' /   __| _` | __|  _ \   __|
//    . \  |   (   | |   (   |\__ `
//   _|\_\_|  \__,_|\__|\___/ ____/
//                   Multi-Physics
//
//  License:         BSD License
//                   Kratos default license: kratos/license.txt
//
//  Main authors:    Vicente Mataix Ferrandiz
//

// System includes
#include <algorithm>
#include <numeric>
#include <set>
#include <unordered_set>

// External includes

// Project includes
#include "utilities/geometry_utilities.h"
#include "utilities/parallel_utilities.h"
#include "utilities/reduction_utilities.h"

// Application includes
#include "mesh_facet_table.h"

namespace Kratos
{

template<std::size_t TDim>
MeshFacetTable<TDim>::MeshFacetTable(const ModelPart& rModelPart)
{
    KRATOS_TRY

    const auto& r_nodes = rModelPart.Nodes();
    const auto& r_elements = rModelPart.Elements();
    mNumberOfNodes = r_nodes.size();
    const std::size_t n_elements = r_elements.size();

    KRATOS_ERROR_IF(n_elements == 0) << "Model part '" << rModelPart.FullName() << "' has no elements." << std::endl;

    // Map the node Ids to their position in the nodes container
    std::unordered_map<IndexType, std::size_t> node_id_to_index;
    node_id_to_index.reserve(mNumberOfNodes);
    std::size_t node_index = 0;
    for (const auto& r_node : r_nodes) {
        node_id_to_index[r_node.Id()] = node_index++;
    }

    mElementIdToIndex.reserve(n_elements);
    std::size_t element_index = 0;
    for (const auto& r_element : r_elements) {
        mElementIdToIndex[r_element.Id()] = element_index++;
        mElementIds.push_back(r_element.Id());
    }

    // Compute the element geometrical data
    mElementData.resize(n_elements);
    IndexPartition<std::size_t>(n_elements).for_each([&](const std::size_t iElement) {
        const auto& r_geometry = (r_elements.begin() + iElement)->GetGeometry();
        KRATOS_ERROR_IF(r_geometry.PointsNumber() != NumNodes || r_geometry.LocalSpaceDimension() != TDim)
            << "Element " << (r_elements.begin() + iElement)->Id() << " is not a linear simplex of dimension " << TDim << "." << std::endl;

        auto& r_data = mElementData[iElement];
        array_1d<double, NumNodes> N;
        double volume;
        GeometryUtils::CalculateGeometryData(r_geometry, r_data.DN_DX, N, volume);
        r_data.Volume = std::abs(volume);
        KRATOS_ERROR_IF(r_data.Volume < std::numeric_limits<double>::epsilon() * std::pow(r_geometry.Length(), TDim))
            << "Element " << (r_elements.begin() + iElement)->Id() << " is degenerated." << std::endl;

        noalias(r_data.Centroid) = ZeroVector(3);
        double max_gradient_norm = 0.0;
        for (std::size_t i_node = 0; i_node < NumNodes; ++i_node) {
            r_data.NodeIndices[i_node] = node_id_to_index.at(r_geometry[i_node].Id());
            noalias(r_data.Centroid) += r_geometry[i_node].Coordinates() / static_cast<double>(NumNodes);

            // The shape function gradient of node i is orthogonal to face i and points towards node i
            // Its norm is the inverse of the height of node i, hence |F_i| = TDim * V * |grad(N_i)|
            auto& r_area_normal = r_data.FaceAreaNormals[i_node];
            noalias(r_area_normal) = ZeroVector(3);
            double gradient_norm_squared = 0.0;
            for (std::size_t d = 0; d < TDim; ++d) {
                r_area_normal[d] = -static_cast<double>(TDim) * r_data.Volume * r_data.DN_DX(i_node, d);
                gradient_norm_squared += r_data.DN_DX(i_node, d) * r_data.DN_DX(i_node, d);
            }
            max_gradient_norm = std::max(max_gradient_norm, std::sqrt(gradient_norm_squared));
            r_data.Neighbours[i_node] = Invalid;
            r_data.Facets[i_node] = Invalid;
            r_data.PartitionRanks[i_node] = Invalid;
        }
        r_data.Size = 1.0 / max_gradient_norm;
    });

    mMinimumElementSize = block_for_each<MinReduction<double>>(mElementData, [](const ElementData& rData) {
        return rData.Size;
    });
    mMinimumElementSize = rModelPart.GetCommunicator().GetDataCommunicator().MinAll(mMinimumElementSize);

    // Sort all the element faces by their (sorted) node Ids so that the faces shared by two elements become consecutive
    struct FaceEntry
    {
        std::array<IndexType, NumFaceNodes> SortedIds;
        std::size_t Element;
        std::size_t Face;
    };
    std::vector<FaceEntry> faces(n_elements * NumFaces);
    IndexPartition<std::size_t>(n_elements).for_each([&](const std::size_t iElement) {
        const auto& r_geometry = (r_elements.begin() + iElement)->GetGeometry();
        for (std::size_t i_face = 0; i_face < NumFaces; ++i_face) {
            auto& r_entry = faces[iElement * NumFaces + i_face];
            r_entry.Element = iElement;
            r_entry.Face = i_face;
            std::size_t aux = 0;
            for (std::size_t i_node = 0; i_node < NumNodes; ++i_node) {
                if (i_node != i_face) {
                    r_entry.SortedIds[aux++] = r_geometry[i_node].Id();
                }
            }
            std::sort(r_entry.SortedIds.begin(), r_entry.SortedIds.end());
        }
    });
    std::sort(faces.begin(), faces.end(), [](const FaceEntry& rA, const FaceEntry& rB) {
        if (rA.SortedIds != rB.SortedIds) {
            return rA.SortedIds < rB.SortedIds;
        }
        return rA.Element < rB.Element;
    });

    // Match the faces shared by two local elements (interior facets)
    struct FacetEntry
    {
        std::array<IndexType, NumFaceNodes> SortedIds;
        std::array<std::pair<std::size_t, std::size_t>, 2> Sides; // (element, face) of each side present in this rank
        std::size_t NumberOfSides = 0;
        int NeighbourRank = Invalid;                               // Rank across a partition facet
    };
    std::vector<FacetEntry> facets;
    std::vector<std::size_t> unmatched_faces;
    std::size_t i_entry = 0;
    while (i_entry < faces.size()) {
        const auto& r_entry = faces[i_entry];
        if (i_entry + 1 < faces.size() && faces[i_entry + 1].SortedIds == r_entry.SortedIds) {
            const auto& r_other = faces[i_entry + 1];
            KRATOS_ERROR_IF(i_entry + 2 < faces.size() && faces[i_entry + 2].SortedIds == r_entry.SortedIds)
                << "Non-manifold mesh: a face is shared by more than two elements." << std::endl;

            auto& r_owner_data = mElementData[r_entry.Element];
            auto& r_other_data = mElementData[r_other.Element];
            r_owner_data.Neighbours[r_entry.Face] = static_cast<int>(r_other.Element);
            r_other_data.Neighbours[r_other.Face] = static_cast<int>(r_entry.Element);

            // Use the owner area normal on both sides so that the facet fluxes cancel exactly
            noalias(r_other_data.FaceAreaNormals[r_other.Face]) = -r_owner_data.FaceAreaNormals[r_entry.Face];

            FacetEntry facet;
            facet.SortedIds = r_entry.SortedIds;
            facet.Sides[0] = {r_entry.Element, r_entry.Face};
            facet.Sides[1] = {r_other.Element, r_other.Face};
            facet.NumberOfSides = 2;
            facets.push_back(facet);
            i_entry += 2;
        } else {
            unmatched_faces.push_back(i_entry);
            ++i_entry;
        }
    }
    const std::size_t n_interior_facets = facets.size();

    // Match the locally unmatched faces with the ones of the neighbour ranks (partition facets)
    const auto& r_communicator = rModelPart.GetCommunicator();
    mpDataCommunicator = &r_communicator.GetDataCommunicator();
    mIsDistributed = r_communicator.IsDistributed();
    const int rank = mpDataCommunicator->Rank();
    std::vector<std::pair<int, std::vector<std::size_t>>> partition_facets_by_rank; // Neighbour rank and its partition facets (sorted by key)
    if (mIsDistributed) {
        std::vector<bool> is_partition_face(unmatched_faces.size(), false);
        const auto& r_neighbours = r_communicator.NeighbourIndices();
        for (std::size_t i_color = 0; i_color < r_neighbours.size(); ++i_color) {
            const int neighbour = r_neighbours[i_color];
            if (neighbour < 0) {
                continue;
            }

            // Candidate faces: all their nodes are shared with the neighbour
            std::unordered_set<IndexType> interface_ids;
            for (const auto& r_node : r_communicator.InterfaceMesh(i_color).Nodes()) {
                interface_ids.insert(r_node.Id());
            }
            std::vector<std::size_t> candidates;
            std::vector<unsigned long> send_keys;
            for (std::size_t i_unmatched = 0; i_unmatched < unmatched_faces.size(); ++i_unmatched) {
                const auto& r_ids = faces[unmatched_faces[i_unmatched]].SortedIds;
                if (!is_partition_face[i_unmatched] && std::all_of(r_ids.begin(), r_ids.end(), [&interface_ids](const IndexType Id) {return interface_ids.count(Id) > 0;})) {
                    candidates.push_back(i_unmatched);
                    send_keys.insert(send_keys.end(), r_ids.begin(), r_ids.end());
                }
            }
            const std::vector<unsigned long> recv_keys = mpDataCommunicator->SendRecv(send_keys, neighbour, neighbour);

            // Matched faces, sorted by key so that both ranks list them in the same order
            std::set<std::array<IndexType, NumFaceNodes>> neighbour_keys;
            for (std::size_t i = 0; i + NumFaceNodes <= recv_keys.size(); i += NumFaceNodes) {
                std::array<IndexType, NumFaceNodes> key;
                std::copy(recv_keys.begin() + i, recv_keys.begin() + i + NumFaceNodes, key.begin());
                neighbour_keys.insert(key);
            }
            std::vector<std::size_t> matched;
            for (const std::size_t i_unmatched : candidates) {
                if (neighbour_keys.count(faces[unmatched_faces[i_unmatched]].SortedIds) > 0) {
                    matched.push_back(i_unmatched);
                    is_partition_face[i_unmatched] = true;
                }
            }
            std::sort(matched.begin(), matched.end(), [&](const std::size_t A, const std::size_t B) {
                return faces[unmatched_faces[A]].SortedIds < faces[unmatched_faces[B]].SortedIds;
            });

            std::vector<std::size_t> facet_indices;
            for (const std::size_t i_unmatched : matched) {
                const auto& r_face = faces[unmatched_faces[i_unmatched]];
                FacetEntry facet;
                facet.SortedIds = r_face.SortedIds;
                facet.Sides[0] = {r_face.Element, r_face.Face};
                facet.NumberOfSides = 1;
                facet.NeighbourRank = neighbour;
                facet_indices.push_back(facets.size());
                facets.push_back(facet);
            }
            partition_facets_by_rank.push_back({neighbour, facet_indices});
        }
    }

    // Number the owned facets contiguously: local interior facets first, then the partition facets owned by this rank (lowest rank)
    mFacetGlobalIds.assign(facets.size(), std::numeric_limits<std::size_t>::max());
    std::size_t n_owned = 0;
    for (std::size_t i_facet = 0; i_facet < facets.size(); ++i_facet) {
        if (i_facet < n_interior_facets || rank < facets[i_facet].NeighbourRank) {
            mFacetGlobalIds[i_facet] = n_owned++;
        }
    }
    mNumberOfOwnedFacets = n_owned;
    mFirstOwnedFacet = mpDataCommunicator->ScanSum(static_cast<unsigned long>(n_owned)) - n_owned;
    mTotalNumberOfFacets = mpDataCommunicator->SumAll(static_cast<unsigned long>(n_owned));
    for (std::size_t i_facet = 0; i_facet < facets.size(); ++i_facet) {
        if (mFacetGlobalIds[i_facet] != std::numeric_limits<std::size_t>::max()) {
            mFacetGlobalIds[i_facet] += mFirstOwnedFacet;
        }
    }

    // The owner of each partition facet sends its global index and its area normal
    for (const auto& [neighbour, r_facet_indices] : partition_facets_by_rank) {
        std::vector<unsigned long> send_ids;
        std::vector<double> send_normals;
        if (rank < neighbour) {
            for (const std::size_t i_facet : r_facet_indices) {
                send_ids.push_back(mFacetGlobalIds[i_facet]);
                const auto& r_side = facets[i_facet].Sides[0];
                const auto& r_normal = mElementData[r_side.first].FaceAreaNormals[r_side.second];
                send_normals.insert(send_normals.end(), r_normal.begin(), r_normal.end());
            }
        }
        const std::vector<unsigned long> recv_ids = mpDataCommunicator->SendRecv(send_ids, neighbour, neighbour);
        const std::vector<double> recv_normals = mpDataCommunicator->SendRecv(send_normals, neighbour, neighbour);
        if (rank > neighbour) {
            KRATOS_ERROR_IF(recv_ids.size() != r_facet_indices.size() || recv_normals.size() != 3 * r_facet_indices.size())
                << "Inconsistent partition facets between ranks " << rank << " and " << neighbour << "." << std::endl;
            for (std::size_t i = 0; i < r_facet_indices.size(); ++i) {
                const std::size_t i_facet = r_facet_indices[i];
                mFacetGlobalIds[i_facet] = recv_ids[i];
                const auto& r_side = facets[i_facet].Sides[0];
                auto& r_normal = mElementData[r_side.first].FaceAreaNormals[r_side.second];
                for (std::size_t d = 0; d < 3; ++d) {
                    r_normal[d] = -recv_normals[3 * i + d];
                }
            }
        }
    }

    // Fill the facet indices and the facet DOFs (from the rank of each vertex Id within the facet)
    std::array<std::array<std::size_t, NumNodes>, NumFaces> invalid_dofs;
    for (auto& r_face_dofs : invalid_dofs) {
        r_face_dofs.fill(std::numeric_limits<std::size_t>::max());
    }
    mFacetDofs.assign(n_elements, invalid_dofs);
    for (std::size_t i_facet = 0; i_facet < facets.size(); ++i_facet) {
        const auto& r_facet = facets[i_facet];
        const std::size_t global_id = mFacetGlobalIds[i_facet];
        for (std::size_t i_side = 0; i_side < r_facet.NumberOfSides; ++i_side) {
            const auto [element, face] = r_facet.Sides[i_side];
            auto& r_data = mElementData[element];
            r_data.Facets[face] = static_cast<int>(i_facet);
            r_data.PartitionRanks[face] = r_facet.NeighbourRank;
            const auto& r_geometry = (r_elements.begin() + element)->GetGeometry();
            for (std::size_t i_node = 0; i_node < NumNodes; ++i_node) {
                if (i_node != face) {
                    const auto it = std::lower_bound(r_facet.SortedIds.begin(), r_facet.SortedIds.end(), r_geometry[i_node].Id());
                    const std::size_t vertex_rank = std::distance(r_facet.SortedIds.begin(), it);
                    mFacetDofs[element][face][i_node] = global_id * NumFaceNodes + vertex_rank;
                }
            }
            if (r_facet.NeighbourRank != Invalid) {
                mPartitionFacets[global_id] = {element, face};
            }
        }
    }

    KRATOS_CATCH("")
}

/***********************************************************************************/
/***********************************************************************************/

template<std::size_t TDim>
void MeshFacetTable<TDim>::ShapeFunctions(
    const std::size_t ElementIndex,
    const array_1d<double, 3>& rCoordinates,
    ShapeFunctionsType& rN
    ) const
{
    const auto& r_data = mElementData[ElementIndex];
    for (std::size_t i_node = 0; i_node < NumNodes; ++i_node) {
        double value = 1.0 / static_cast<double>(NumNodes);
        for (std::size_t d = 0; d < TDim; ++d) {
            value += r_data.DN_DX(i_node, d) * (rCoordinates[d] - r_data.Centroid[d]);
        }
        rN[i_node] = value;
    }
}

/***********************************************************************************/
/***********************************************************************************/

template<std::size_t TDim>
void MeshFacetTable<TDim>::ComputeElementFacetDofs(std::vector<std::vector<std::size_t>>& rElementDofs) const
{
    KRATOS_TRY

    rElementDofs.resize(mElementData.size());
    IndexPartition<std::size_t>(mElementData.size()).for_each([&](const std::size_t iElement) {
        auto& r_dofs = rElementDofs[iElement];
        r_dofs.clear();
        const auto& r_data = mElementData[iElement];
        for (std::size_t i_face = 0; i_face < NumFaces; ++i_face) {
            if (r_data.Facets[i_face] != Invalid) {
                for (std::size_t i_node = 0; i_node < NumNodes; ++i_node) {
                    if (i_node != i_face) {
                        r_dofs.push_back(mFacetDofs[iElement][i_face][i_node]);
                    }
                }
            }
        }
    });

    KRATOS_CATCH("")
}

/***********************************************************************************/
/***********************************************************************************/

template<std::size_t TDim>
bool MeshFacetTable<TDim>::FindPartitionFacet(
    const std::size_t GlobalFacet,
    std::size_t& rElementIndex,
    std::size_t& rFace
    ) const
{
    const auto it = mPartitionFacets.find(GlobalFacet);
    if (it == mPartitionFacets.end()) {
        return false;
    }
    rElementIndex = it->second.first;
    rFace = it->second.second;
    return true;
}

/***********************************************************************************/
/***********************************************************************************/

template<std::size_t TDim>
std::size_t MeshFacetTable<TDim>::GetElementIndex(const IndexType ElementId) const
{
    const auto it = mElementIdToIndex.find(ElementId);
    KRATOS_ERROR_IF(it == mElementIdToIndex.end()) << "Element " << ElementId << " is not in the facet table." << std::endl;
    return it->second;
}

/***********************************************************************************/
/***********************************************************************************/

template class MeshFacetTable<2>;
template class MeshFacetTable<3>;

} // namespace Kratos

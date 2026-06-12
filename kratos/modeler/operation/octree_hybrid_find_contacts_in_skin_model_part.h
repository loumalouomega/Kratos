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

#pragma once

// System includes

// External includes

// Project includes
#include "modeler/operation/octree_hybrid_mesher_operation.h"

namespace Kratos
{

///@name Kratos Classes
///@{

/**
 * @class OctreeHybridFindContactsInSkinModelPart
 * @ingroup KratosCore
 * @brief Post-processing operation that splits the conditions of a skin (boundary)
 *        ModelPart into per-neighbour-colour "contact" sub-ModelParts.
 * @details This operation is applied to a skin ModelPart whose conditions are the
 * boundary quads of a colour region (typically produced by
 * @ref GenerateHybridOctreeQuadrilateralConditionsWithFaceColor with `"color"` set to
 * `"cell_color"`).  For every condition it determines, on the **octree hex mesh**
 * (`OctreeHybridMesherData::mCells` / `mCellColor`), the cell on the other side of the
 * shared quad face.  If that neighbour cell's colour matches one of the
 * `"outside_color"` values listed in `"contact_model_parts"`, the condition (and its
 * four nodes) is additionally added to the corresponding `"contact_model_part_name"`.
 *
 * A single boundary condition may be added to more than one contact ModelPart if its
 * face is shared (in a non-manifold sense) by more than one neighbouring cell with
 * different colours that both appear in `"contact_model_parts"`.
 *
 * ### Face-to-cell adjacency
 * The operation builds, once, a map from a sorted 4-node-index face key to the list of
 * cell indices (over the **full**, unfiltered `mCells`) that own that face.  For a
 * boundary condition's face this is typically:
 * - **2 cells**: the `"cell_color"` cell on this side, and one neighbour cell whose
 *   colour is tested against every `"outside_color"`.
 * - **1 cell**: the face lies on the outer boundary of the whole octree domain — no
 *   contact is possible and the condition is skipped.
 *
 * ### Node-index recovery
 * Because the skin ModelPart's nodes were created via
 * @ref OctreeHybridMeshGeneratorModeler::GenerateOrRetrieveNode, this operation builds a
 * reverse map from Kratos `Node::Id()` to the mesh-node index in
 * `OctreeHybridMesherData::mNodes` using `OctreeHybridMesherData::mNodePtrs`. Conditions
 * whose nodes were never materialised (no entry in `mNodePtrs`) are skipped.
 *
 * ### Registry paths
 * Registered at:
 * - `OctreeHybridMesherOperation.KratosMultiphysics`
 * - `OctreeHybridMesherOperation.All`
 *
 * ### Parameters schema
 * | Key                    | Type   | Default                                        | Description |
 * |------------------------|--------|------------------------------------------------|-------------|
 * | `type`                 | string | `"OctreeHybridFindContactsInSkinModelPart"`     | Registry lookup key. |
 * | `model_part_name`      | string | `"Undefined"`                                   | Skin ModelPart whose conditions are classified. |
 * | `contact_model_parts`  | array  | `[]`                                            | List of `{ "outside_color", "contact_model_part_name" }` objects. |
 * | `cell_color`           | int    | `-1`                                            | Colour of the cells on "this" side of `model_part_name`'s conditions. |
 *
 * Each entry of `"contact_model_parts"` accepts:
 * | Key                       | Type   | Default       | Description |
 * |---------------------------|--------|---------------|-------------|
 * | `outside_color`           | int    | `-1`          | Colour of the neighbour cell that identifies a contact. |
 * | `contact_model_part_name` | string | `"Undefined"` | ModelPart (created if needed) receiving the matching conditions and nodes. |
 *
 * ### Example JSON
 * @code{.json}
 * {
 *     "type"                : "OctreeHybridFindContactsInSkinModelPart",
 *     "model_part_name"     : "destination_model_part.submodelpart_liquid",
 *     "cell_color"          : -1,
 *     "contact_model_parts" : [
 *         {
 *             "outside_color": 0.0,
 *             "contact_model_part_name": "destination_model_part.submodelpart_liquid.contact_liquid2air"
 *         },
 *         {
 *             "outside_color": -2,
 *             "contact_model_part_name": "destination_model_part.submodelpart_liquid.contact_liquid2solid.contact_liquid2mold_105"
 *         }
 *     ]
 * }
 * @endcode
 *
 * @see OctreeHybridMesherOperation
 * @see GenerateHybridOctreeQuadrilateralConditionsWithFaceColor
 * @see OctreeHybridMeshUtility::ExtractBoundaryFaces
 * @author Vicente Mataix Ferrandiz
 */
class KRATOS_API(KRATOS_CORE) OctreeHybridFindContactsInSkinModelPart : public OctreeHybridMesherOperation
{
public:
    ///@name Life Cycle
    ///@{

    /// Default constructor.
    OctreeHybridFindContactsInSkinModelPart() = default;

    /**
     * @brief Copy constructor.
     * @param rOther The instance to copy.  No data members to copy in this class.
     */
    OctreeHybridFindContactsInSkinModelPart(OctreeHybridFindContactsInSkinModelPart const& rOther) {}

    ///@}
    ///@name Operations
    ///@{

    /**
     * @brief Classifies the conditions of the configured skin ModelPart into the
     *        requested contact ModelParts based on octree cell-colour adjacency.
     * @details Algorithm:
     * 1. Validates each entry of `"contact_model_parts"` against its own default
     *    schema and creates (or retrieves) every `"contact_model_part_name"`.
     * 2. Builds a reverse map from `Node::Id()` to mesh-node index using
     *    `OctreeHybridMesherData::mNodePtrs`.
     * 3. Builds a map from sorted face-node-index tuples to the list of owning cell
     *    indices over the full `OctreeHybridMesherData::mCells`.
     * 4. For every condition in `"model_part_name"`, recovers its face's mesh-node
     *    indices, looks up the owning cells, and — for every owning cell whose colour
     *    differs from `"cell_color"` — checks whether that colour matches any
     *    `"outside_color"`.  On a match, the condition's id and its four node ids are
     *    queued for the corresponding contact ModelPart.
     * 5. Adds the queued conditions and nodes (looked up by id from the root
     *    ModelPart) to each contact ModelPart.
     *
     * @param rModeler            The @ref OctreeHybridMeshGeneratorModeler that holds the Model
     *                            and the extracted hex mesh data.
     * @param OperationParameters Validated JSON parameters.  Must contain
     *                            `"model_part_name"`, `"cell_color"` and
     *                            `"contact_model_parts"`.
     */
    void Execute(OctreeHybridMeshGeneratorModeler& rModeler, Parameters OperationParameters) const override;

    /**
     * @brief Returns the default parameter schema for this operation.
     * @details Schema:
     * @code{.json}
     * {
     *     "type"                : "OctreeHybridFindContactsInSkinModelPart",
     *     "model_part_name"     : "Undefined",
     *     "contact_model_parts" : [],
     *     "cell_color"          : -1
     * }
     * @endcode
     * @return A Parameters object with all accepted keys and their default values.
     */
    const Parameters GetDefaultParameters() const override;

    ///@}
private:
    ///@name Registry
    ///@{

    /// Registers this class as a prototype under the KratosMultiphysics sub-path.
    KRATOS_REGISTRY_ADD_PROTOTYPE("OctreeHybridMesherOperation.KratosMultiphysics", OctreeHybridMesherOperation, OctreeHybridFindContactsInSkinModelPart)
    /// Registers this class as a prototype under the All sub-path.
    KRATOS_REGISTRY_ADD_PROTOTYPE("OctreeHybridMesherOperation.All", OctreeHybridMesherOperation, OctreeHybridFindContactsInSkinModelPart)

    ///@}
};

///@}

} // namespace Kratos

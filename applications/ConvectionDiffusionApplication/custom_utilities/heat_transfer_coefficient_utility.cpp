// KRATOS ___ ___  _  ___   __   ___ ___ ___ ___
//       / __/ _ \| \| \ \ / /__|   \_ _| __| __|
//      | (_| (_) | .` |\ V /___| |) | || _|| _|
//       \___\___/|_|\_| \_/    |___/___|_| |_|  APPLICATION
//
//  License:         BSD License
//                   Kratos default license: kratos/license.txt
//
//  Main authors:    Vicente Mataix Ferrandiz
//

// System includes
#include <algorithm>
#include <cmath>
#include <limits>

// External includes

// Project includes
#include "includes/variables.h"
#include "utilities/intersection_utilities.h"
#include "utilities/math_utils.h"

// Application includes
#include "heat_transfer_coefficient_utility.h"

namespace Kratos
{

namespace
{

/// Local ordering of the twelve edges of a hexahedron, referred to the Hexahedra3D8 corners.
constexpr std::array<std::array<std::size_t, 2>, 12> HexahedronEdges = {{
    {{0, 1}}, {{1, 2}}, {{2, 3}}, {{3, 0}},   // bottom face
    {{4, 5}}, {{5, 6}}, {{6, 7}}, {{7, 4}},   // top face
    {{0, 4}}, {{1, 5}}, {{2, 6}}, {{3, 7}}    // vertical edges
}};

/**
 * @brief Reads a three component vector from the parameters.
 * @param ThisParameters The parameters holding the entry
 * @param rEntryName The name of the entry
 * @return The parsed vector
 */
array_1d<double, 3> ReadVector3(
    Parameters ThisParameters,
    const std::string& rEntryName)
{
    const Vector aux_vector = ThisParameters[rEntryName].GetVector();
    KRATOS_ERROR_IF_NOT(aux_vector.size() == 3)
        << "The '" << rEntryName << "' must have three components, but " << aux_vector.size()
        << " were given." << std::endl;

    array_1d<double, 3> result;
    for (std::size_t i = 0; i < 3; ++i) {
        result[i] = aux_vector[i];
    }

    return result;
}

} // unnamed namespace

/* Public functions *******************************************************/

HeatTransferCoefficientUtility::HeatTransferCoefficientUtility(
    Model& rModel,
    Parameters ThisParameters)
{
    KRATOS_TRY

    // Validate input settings with defaults
    ThisParameters.ValidateAndAssignDefaults(GetDefaultParameters());

    // Retrieve the domain definition
    mBoxMin = ReadVector3(ThisParameters, "box_min");
    mBoxMax = ReadVector3(ThisParameters, "box_max");
    for (std::size_t i = 0; i < 3; ++i) {
        KRATOS_ERROR_IF(mBoxMax[i] <= mBoxMin[i])
            << "The 'box_max' must be strictly greater than 'box_min' in all directions. "
            << "Component " << i << " is " << mBoxMax[i] << " against " << mBoxMin[i] << "." << std::endl;
    }

    // Retrieve the dividing plane definition
    mPlanePoint = ReadVector3(ThisParameters, "plane_point");
    mPlaneNormal = ReadVector3(ThisParameters, "plane_normal");
    const double normal_norm = norm_2(mPlaneNormal);
    KRATOS_ERROR_IF(normal_norm < std::numeric_limits<double>::epsilon())
        << "The 'plane_normal' cannot be a zero vector." << std::endl;
    mPlaneNormal /= normal_norm;

    // Set the octree refinement depth
    const int max_octree_level = ThisParameters["max_octree_level"].GetInt();
    KRATOS_ERROR_IF(max_octree_level < 0 || max_octree_level > 10)
        << "The 'max_octree_level' must lie in [0,10], but " << max_octree_level << " was given. "
        << "Note that level 10 already implies up to 8^10 cells." << std::endl;
    mMaxOctreeLevel = static_cast<IndexType>(max_octree_level);
    mLatticeSize = static_cast<std::int64_t>(1) << mMaxOctreeLevel;

    // Retrieve the material data
    Parameters material_1 = ThisParameters["material_1"];
    Parameters material_2 = ThisParameters["material_2"];
    material_1.ValidateAndAssignDefaults(GetDefaultParameters()["material_1"]);
    material_2.ValidateAndAssignDefaults(GetDefaultParameters()["material_2"]);

    mConductivityTable1 = ReadDataTable(material_1["conductivity_table"], "material_1.conductivity_table");
    mConductivityTable2 = ReadDataTable(material_2["conductivity_table"], "material_2.conductivity_table");
    mAirConductivityTable = ReadDataTable(ThisParameters["air_conductivity_table"], "air_conductivity_table");

    mEmissivity1 = material_1["emissivity"].GetDouble();
    mEmissivity2 = material_2["emissivity"].GetDouble();
    mThickness1 = material_1["thickness"].GetDouble();
    mThickness2 = material_2["thickness"].GetDouble();

    // Retrieve the interface model settings
    mIsAir = ThisParameters["is_air"].GetBool();
    mAmbientTemperature = ThisParameters["ambient_temperature"].GetDouble();
    mContactFraction = ThisParameters["f_contact"].GetDouble();
    mGapThickness = ThisParameters["gap_thickness"].GetDouble();

    mIncludeConduction = ThisParameters["include_conduction"].GetBool();
    mIncludeGapConduction = ThisParameters["include_gap_conduction"].GetBool();
    mIncludeRadiation = ThisParameters["include_radiation"].GetBool();

    // Validate the interface model settings
    KRATOS_ERROR_IF(mEmissivity1 <= 0.0 || mEmissivity1 > 1.0)
        << "The 'material_1.emissivity' must lie in (0,1], but " << mEmissivity1 << " was given." << std::endl;
    KRATOS_ERROR_IF(!mIsAir && (mEmissivity2 <= 0.0 || mEmissivity2 > 1.0))
        << "The 'material_2.emissivity' must lie in (0,1], but " << mEmissivity2 << " was given." << std::endl;
    KRATOS_ERROR_IF(mContactFraction < 0.0 || mContactFraction > 1.0)
        << "The 'f_contact' must lie in [0,1], but " << mContactFraction << " was given." << std::endl;
    KRATOS_ERROR_IF(mIncludeGapConduction && mGapThickness <= 0.0)
        << "The 'gap_thickness' must be strictly positive when 'include_gap_conduction' is true, "
        << "but " << mGapThickness << " was given. Set 'include_gap_conduction' to false to "
        << "disable the gas conduction term instead." << std::endl;
    KRATOS_ERROR_IF(mIncludeConduction && mContactFraction > 0.0 && (mThickness1 <= 0.0 || mThickness2 <= 0.0))
        << "The material thicknesses must be strictly positive when the solid conduction term is "
        << "active, but " << mThickness1 << " and " << mThickness2 << " were given." << std::endl;

    // The gray-body exchange factor is only meaningful when the emissivities are compatible
    if (!mIsAir) {
        const double aux_exchange = 1.0 / mEmissivity1 + 1.0 / mEmissivity2 - 1.0;
        KRATOS_ERROR_IF(aux_exchange <= 0.0)
            << "The gray-body exchange factor 1/(1/e1 + 1/e2 - 1) is not positive for the given "
            << "emissivities " << mEmissivity1 << " and " << mEmissivity2 << "." << std::endl;
    }

    // Retrieve the sampling settings
    const Vector sample_temperatures = ThisParameters["sample_temperatures"].GetVector();
    mSampleTemperatures.resize(sample_temperatures.size());
    for (std::size_t i = 0; i < sample_temperatures.size(); ++i) {
        mSampleTemperatures[i] = sample_temperatures[i];
    }
    mPartnerTemperatureMode = ThisParameters["partner_temperature_mode"].GetString();
    mDeltaTemperature = ThisParameters["delta_temperature"].GetDouble();
    mSinkTemperature = ThisParameters["sink_temperature"].GetDouble();
    mPlateauEpsilon = ThisParameters["plateau_epsilon"].GetDouble();

    KRATOS_ERROR_IF(mPartnerTemperatureMode != "delta" && mPartnerTemperatureMode != "fixed")
        << "The 'partner_temperature_mode' must be either 'delta' or 'fixed', but '"
        << mPartnerTemperatureMode << "' was given." << std::endl;
    KRATOS_ERROR_IF(mSampleTemperatures.size() == 0)
        << "The 'sample_temperatures' list cannot be empty." << std::endl;
    KRATOS_ERROR_IF(mPlateauEpsilon <= 0.0)
        << "The 'plateau_epsilon' must be strictly positive, but " << mPlateauEpsilon
        << " was given." << std::endl;

    // The samples must be strictly increasing so that the resulting table is well defined
    for (std::size_t i = 1; i < mSampleTemperatures.size(); ++i) {
        KRATOS_ERROR_IF(mSampleTemperatures[i] <= mSampleTemperatures[i - 1])
            << "The 'sample_temperatures' must be strictly increasing, but entry " << i << " ("
            << mSampleTemperatures[i] << ") does not exceed the previous one ("
            << mSampleTemperatures[i - 1] << ")." << std::endl;
        KRATOS_ERROR_IF(mPlateauEpsilon >= mSampleTemperatures[i] - mSampleTemperatures[i - 1])
            << "The 'plateau_epsilon' (" << mPlateauEpsilon << ") must be strictly smaller than "
            << "the gap between consecutive samples (" << mSampleTemperatures[i] - mSampleTemperatures[i - 1]
            << " between entries " << i - 1 << " and " << i << "). Otherwise the emitted table "
            << "would not have a strictly increasing input column." << std::endl;
    }

    // All the temperatures involved are absolute, as required by the radiative term
    for (std::size_t i = 0; i < mSampleTemperatures.size(); ++i) {
        const double temperature = mSampleTemperatures[i];
        KRATOS_ERROR_IF(temperature <= 0.0)
            << "The sample temperatures are absolute (K) and must be strictly positive, but entry "
            << i << " is " << temperature << "." << std::endl;
        const double partner_temperature = ComputePartnerTemperature(temperature);
        KRATOS_ERROR_IF(partner_temperature <= 0.0)
            << "The partner temperature of the sample " << temperature << " K is "
            << partner_temperature << " K, which is not a valid absolute temperature." << std::endl;
    }

    mOutputVariableName = ThisParameters["output_variable"].GetString();

    // Create the model part holding the generated mesh
    const std::string model_part_name = ThisParameters["model_part_name"].GetString();
    KRATOS_ERROR_IF(rModel.HasModelPart(model_part_name))
        << "The model already contains a model part named '" << model_part_name << "'. The "
        << "utility creates its own mesh and requires a free name." << std::endl;
    mpModelPart = &rModel.CreateModelPart(model_part_name);

    // The nodal area is stored historically, so it must be added before creating any node
    mpModelPart->AddNodalSolutionStepVariable(NODAL_AREA);
    mpModelPart->CreateSubModelPart("Material1");
    mpModelPart->CreateSubModelPart("Material2");
    mpModelPart->CreateSubModelPart("Interface");

    KRATOS_CATCH("")
}

void HeatTransferCoefficientUtility::GenerateMesh()
{
    KRATOS_TRY

    // Clear any previously generated data so that the call is idempotent
    mInterfaceAreas.clear();

    // The properties are irrelevant for the generated elements, which are never assembled
    if (!mpModelPart->HasProperties(1)) {
        mpModelPart->CreateNewProperties(1);
    }

    Plane3D plane(mPlaneNormal, Point(mPlanePoint[0], mPlanePoint[1], mPlanePoint[2]));

    std::map<LatticeKeyType, IndexType> node_ids;
    IndexType node_counter = 0;
    IndexType element_counter = 0;

    const LatticeKeyType root_min = {{0, 0, 0}};
    const LatticeKeyType root_max = {{mLatticeSize, mLatticeSize, mLatticeSize}};
    RefineCell(root_min, root_max, 0, plane, node_ids, node_counter, element_counter);

    KRATOS_INFO("HeatTransferCoefficientUtility")
        << "Generated " << mpModelPart->NumberOfElements() << " cells and "
        << mInterfaceAreas.size() << " interface nodes covering an area of "
        << GetInterfaceArea() << "." << std::endl;

    KRATOS_CATCH("")
}

double HeatTransferCoefficientUtility::ComputeHeatTransferCoefficient(
    const double Temperature1,
    const double Temperature2) const
{
    KRATOS_TRY

    KRATOS_ERROR_IF(Temperature1 <= 0.0 || Temperature2 <= 0.0)
        << "The temperatures are absolute (K) and must be strictly positive, but " << Temperature1
        << " and " << Temperature2 << " were given." << std::endl;

    // Solid conduction in series across both materials
    double h_cond = 0.0;
    if (mIncludeConduction) {
        const double k_1 = Interpolate(mConductivityTable1, Temperature1);
        const double k_2 = Interpolate(mConductivityTable2, Temperature2);
        KRATOS_ERROR_IF(k_1 <= 0.0 || k_2 <= 0.0)
            << "The interpolated conductivities must be strictly positive, but " << k_1 << " and "
            << k_2 << " were obtained at " << Temperature1 << " K and " << Temperature2
            << " K." << std::endl;
        h_cond = 1.0 / (mThickness1 / k_1 + mThickness2 / k_2);
    }

    // Gas conduction across the gap, evaluated at the mean film temperature
    double h_gap = 0.0;
    if (mIncludeGapConduction) {
        const double mean_temperature = 0.5 * (Temperature1 + Temperature2);
        const double k_air = Interpolate(mAirConductivityTable, mean_temperature);
        h_gap = k_air / mGapThickness;
    }

    // Linearised Stefan-Boltzmann radiation across the gap
    double h_rad = 0.0;
    if (mIncludeRadiation) {
        // For a free surface the exchange factor degenerates to the solid emissivity
        const double exchange_factor = mIsAir
            ? mEmissivity1
            : 1.0 / (1.0 / mEmissivity1 + 1.0 / mEmissivity2 - 1.0);
        h_rad = StefanBoltzmann * exchange_factor
            * (Temperature1 * Temperature1 + Temperature2 * Temperature2)
            * (Temperature1 + Temperature2);
    }

    return mContactFraction * h_cond + (1.0 - mContactFraction) * (h_gap + h_rad);

    KRATOS_CATCH("")
}

double HeatTransferCoefficientUtility::GetInterfaceArea() const
{
    KRATOS_TRY

    double total_area = 0.0;
    for (const double area : mInterfaceAreas) {
        total_area += area;
    }

    return total_area;

    KRATOS_CATCH("")
}

Parameters HeatTransferCoefficientUtility::ComputeTable()
{
    KRATOS_TRY

    // Build the interface on demand so that the utility can be driven with a single call
    if (mInterfaceAreas.empty()) {
        GenerateMesh();
    }

    const double total_area = GetInterfaceArea();
    KRATOS_ERROR_IF(total_area <= 0.0)
        << "The interface area is " << total_area << ", so the dividing plane does not cut the "
        << "given box. Check the 'plane_point' and 'plane_normal' against 'box_min' and "
        << "'box_max'." << std::endl;

    Parameters output = Parameters(R"({
        "input_variable"  : "TEMPERATURE",
        "output_variable" : "",
        "data"            : []
    })");
    output["output_variable"].SetString(mOutputVariableName);

    for (const double temperature : mSampleTemperatures) {
        const double partner_temperature = ComputePartnerTemperature(temperature);

        // Area-weighted average over the interface. The material data is currently uniform, so
        // every contribution is identical and this reduces to the pointwise value, but the
        // weighted form is the correct one and admits spatially varying properties later.
        double weighted_sum = 0.0;
        for (const double area : mInterfaceAreas) {
            weighted_sum += area * ComputeHeatTransferCoefficient(temperature, partner_temperature);
        }
        const double coefficient = weighted_sum / total_area;

        // Each sample is written twice so that the linear interpolation of Kratos' Table yields a
        // flat plateau at the sample and a ramp in between consecutive samples
        Vector plateau_start(2);
        plateau_start[0] = temperature - mPlateauEpsilon;
        plateau_start[1] = coefficient;
        output["data"].Append(plateau_start);

        Vector plateau_end(2);
        plateau_end[0] = temperature;
        plateau_end[1] = coefficient;
        output["data"].Append(plateau_end);
    }

    return output;

    KRATOS_CATCH("")
}

void HeatTransferCoefficientUtility::Execute()
{
    KRATOS_TRY

    GenerateMesh();
    ComputeTable();

    KRATOS_CATCH("")
}

Parameters HeatTransferCoefficientUtility::GetDefaultParameters()
{
    const Parameters default_parameters = Parameters(R"({
        "model_part_name"          : "HTCDomain",
        "box_min"                  : [0.0, 0.0, 0.0],
        "box_max"                  : [1.0, 1.0, 1.0],
        "plane_point"              : [0.0, 0.0, 0.5],
        "plane_normal"             : [0.0, 0.0, 1.0],
        "max_octree_level"         : 3,
        "material_1"               : {
            "conductivity_table" : [[0.0, 1.0]],
            "emissivity"         : 1.0,
            "thickness"          : 1.0e-3
        },
        "material_2"               : {
            "conductivity_table" : [[0.0, 1.0]],
            "emissivity"         : 1.0,
            "thickness"          : 1.0e-3
        },
        "is_air"                   : false,
        "ambient_temperature"      : 300.0,
        "f_contact"                : 0.0,
        "gap_thickness"            : 1.0e-5,
        "air_conductivity_table"   : [[0.0, 0.026]],
        "include_conduction"       : true,
        "include_gap_conduction"   : true,
        "include_radiation"        : true,
        "sample_temperatures"      : [],
        "partner_temperature_mode" : "delta",
        "delta_temperature"        : 100.0,
        "sink_temperature"         : 300.0,
        "plateau_epsilon"          : 0.01,
        "output_variable"          : "CONVECTION_COEFFICIENT"
    })");

    return default_parameters;
}

/* Private functions ******************************************************/

void HeatTransferCoefficientUtility::RefineCell(
    const LatticeKeyType& rMinKey,
    const LatticeKeyType& rMaxKey,
    const IndexType Level,
    Plane3D& rPlane,
    std::map<LatticeKeyType, IndexType>& rNodeIds,
    IndexType& rNodeCounter,
    IndexType& rElementCounter)
{
    KRATOS_TRY

    std::array<LatticeKeyType, 8> corner_keys;
    std::array<array_1d<double, 3>, 8> corners;
    ComputeCellCorners(rMinKey, rMaxKey, corner_keys, corners);

    // A corner is classified as positive if its signed distance is strictly positive, and as
    // negative otherwise. This half-open convention is what keeps the interface area exact when
    // the plane happens to coincide with a lattice plane: in that case the corners lying on the
    // plane count as negative, so only the cell on the positive side registers as cut and its
    // clipped polygon, which is the shared face, is accounted for exactly once.
    bool has_positive = false;
    bool has_negative = false;
    for (const auto& r_corner : corners) {
        const double distance = rPlane.CalculateSignedDistance(Point(r_corner[0], r_corner[1], r_corner[2]));
        if (distance > 0.0) {
            has_positive = true;
        } else {
            has_negative = true;
        }
    }
    const bool is_cut = has_positive && has_negative;

    if (is_cut && Level < mMaxOctreeLevel) {
        // Refine towards the dividing plane. The span of a cell is a power of two at the finest
        // resolution, so the midpoint is always an exact integer.
        LatticeKeyType mid_key;
        for (std::size_t i = 0; i < 3; ++i) {
            mid_key[i] = (rMinKey[i] + rMaxKey[i]) / 2;
        }

        for (std::size_t i_child = 0; i_child < 8; ++i_child) {
            LatticeKeyType child_min;
            LatticeKeyType child_max;
            for (std::size_t i = 0; i < 3; ++i) {
                const bool upper_half = (i_child >> i) & 1;
                child_min[i] = upper_half ? mid_key[i] : rMinKey[i];
                child_max[i] = upper_half ? rMaxKey[i] : mid_key[i];
            }
            RefineCell(child_min, child_max, Level + 1, rPlane, rNodeIds, rNodeCounter, rElementCounter);
        }
    } else {
        CreateLeafElement(rMinKey, rMaxKey, rPlane, rNodeIds, rNodeCounter, rElementCounter);
        if (is_cut) {
            CreateInterfaceNode(rMinKey, rMaxKey, rPlane, rNodeCounter);
        }
    }

    KRATOS_CATCH("")
}

void HeatTransferCoefficientUtility::CreateLeafElement(
    const LatticeKeyType& rMinKey,
    const LatticeKeyType& rMaxKey,
    Plane3D& rPlane,
    std::map<LatticeKeyType, IndexType>& rNodeIds,
    IndexType& rNodeCounter,
    IndexType& rElementCounter)
{
    KRATOS_TRY

    std::array<LatticeKeyType, 8> corner_keys;
    std::array<array_1d<double, 3>, 8> corners;
    ComputeCellCorners(rMinKey, rMaxKey, corner_keys, corners);

    // Retrieve, or create, the nodes of the cell. The lattice keys are exact integers, so
    // corners shared by neighbouring cells are matched without any geometric tolerance.
    std::vector<IndexType> element_node_ids(8);
    for (std::size_t i = 0; i < 8; ++i) {
        const auto it_node = rNodeIds.find(corner_keys[i]);
        if (it_node != rNodeIds.end()) {
            element_node_ids[i] = it_node->second;
        } else {
            const IndexType new_node_id = ++rNodeCounter;
            mpModelPart->CreateNewNode(new_node_id, corners[i][0], corners[i][1], corners[i][2]);
            rNodeIds[corner_keys[i]] = new_node_id;
            element_node_ids[i] = new_node_id;
        }
    }

    // The side of the cell is decided from the signed distance at its centroid
    array_1d<double, 3> centroid = ZeroVector(3);
    for (const auto& r_corner : corners) {
        noalias(centroid) += r_corner;
    }
    centroid /= 8.0;
    const double centroid_distance = rPlane.CalculateSignedDistance(Point(centroid[0], centroid[1], centroid[2]));

    const std::string sub_model_part_name = centroid_distance > 0.0 ? "Material2" : "Material1";
    auto& r_sub_model_part = mpModelPart->GetSubModelPart(sub_model_part_name);

    // The nodes must belong to the sub model part before the element is added to it
    r_sub_model_part.AddNodes(element_node_ids);
    r_sub_model_part.CreateNewElement(
        "Element3D8N", ++rElementCounter, element_node_ids, mpModelPart->pGetProperties(1));

    KRATOS_CATCH("")
}

void HeatTransferCoefficientUtility::CreateInterfaceNode(
    const LatticeKeyType& rMinKey,
    const LatticeKeyType& rMaxKey,
    Plane3D& rPlane,
    IndexType& rNodeCounter)
{
    KRATOS_TRY

    std::array<LatticeKeyType, 8> corner_keys;
    std::array<array_1d<double, 3>, 8> corners;
    ComputeCellCorners(rMinKey, rMaxKey, corner_keys, corners);

    std::array<double, 8> distances;
    for (std::size_t i = 0; i < 8; ++i) {
        distances[i] = rPlane.CalculateSignedDistance(Point(corners[i][0], corners[i][1], corners[i][2]));
    }

    // A characteristic length of the cell, used to make the merging tolerance dimensionless
    const array_1d<double, 3> cell_diagonal = corners[6] - corners[0];
    const double cell_size = norm_2(cell_diagonal);
    const double merge_tolerance = RelativeTolerance * cell_size;

    // Clip the plane against the cell by intersecting it with the twelve edges
    std::vector<array_1d<double, 3>> polygon;
    for (const auto& r_edge : HexahedronEdges) {
        const bool first_positive = distances[r_edge[0]] > 0.0;
        const bool second_positive = distances[r_edge[1]] > 0.0;
        if (first_positive == second_positive) {
            continue;
        }

        array_1d<double, 3> intersection_point;
        const int intersection_type = IntersectionUtilities::ComputePlaneLineIntersection(
            mPlanePoint, mPlaneNormal, corners[r_edge[0]], corners[r_edge[1]], intersection_point);

        // A return value of 2 means the edge lies in the plane, in which case its endpoints are
        // already contributed by the adjacent edges
        if (intersection_type != 1) {
            continue;
        }

        // Discard the duplicates that arise when the plane passes exactly through a corner
        bool is_duplicated = false;
        for (const auto& r_point : polygon) {
            if (norm_2(r_point - intersection_point) <= merge_tolerance) {
                is_duplicated = true;
                break;
            }
        }
        if (!is_duplicated) {
            polygon.push_back(intersection_point);
        }
    }

    if (polygon.size() < 3) {
        return;
    }

    // Order the vertices. The intersection of a plane with a convex box is always a convex
    // polygon, so sorting by the angle about the vertex average recovers the boundary exactly.
    array_1d<double, 3> polygon_average = ZeroVector(3);
    for (const auto& r_point : polygon) {
        noalias(polygon_average) += r_point;
    }
    polygon_average /= static_cast<double>(polygon.size());

    // Build an orthonormal basis of the plane, starting from the least aligned Cartesian axis
    array_1d<double, 3> first_axis = ZeroVector(3);
    const std::size_t min_component = std::distance(
        mPlaneNormal.begin(),
        std::min_element(mPlaneNormal.begin(), mPlaneNormal.end(),
            [](const double a, const double b) { return std::abs(a) < std::abs(b); }));
    first_axis[min_component] = 1.0;

    array_1d<double, 3> basis_1;
    MathUtils<double>::CrossProduct(basis_1, mPlaneNormal, first_axis);
    basis_1 /= norm_2(basis_1);

    array_1d<double, 3> basis_2;
    MathUtils<double>::CrossProduct(basis_2, mPlaneNormal, basis_1);

    std::sort(polygon.begin(), polygon.end(),
        [&](const array_1d<double, 3>& rPointA, const array_1d<double, 3>& rPointB) {
            const array_1d<double, 3> local_a = rPointA - polygon_average;
            const array_1d<double, 3> local_b = rPointB - polygon_average;
            return std::atan2(inner_prod(local_a, basis_2), inner_prod(local_a, basis_1))
                 < std::atan2(inner_prod(local_b, basis_2), inner_prod(local_b, basis_1));
        });

    // Area and area centroid by fan triangulation from the first vertex. Note that the area
    // centroid is not the vertex average, which is why it is accumulated explicitly here.
    double area = 0.0;
    array_1d<double, 3> centroid = ZeroVector(3);
    for (std::size_t i = 1; i + 1 < polygon.size(); ++i) {
        const array_1d<double, 3> side_1 = polygon[i] - polygon[0];
        const array_1d<double, 3> side_2 = polygon[i + 1] - polygon[0];

        array_1d<double, 3> cross_product;
        MathUtils<double>::CrossProduct(cross_product, side_1, side_2);
        const double triangle_area = 0.5 * norm_2(cross_product);

        area += triangle_area;
        noalias(centroid) += triangle_area * (polygon[0] + polygon[i] + polygon[i + 1]) / 3.0;
    }

    if (area <= merge_tolerance * merge_tolerance) {
        return;
    }
    centroid /= area;

    // Create the interface node carrying its tributary area
    auto& r_interface_model_part = mpModelPart->GetSubModelPart("Interface");
    auto p_node = r_interface_model_part.CreateNewNode(
        ++rNodeCounter, centroid[0], centroid[1], centroid[2]);
    p_node->FastGetSolutionStepValue(NODAL_AREA) = area;

    mInterfaceAreas.push_back(area);

    KRATOS_CATCH("")
}

array_1d<double, 3> HeatTransferCoefficientUtility::LatticeToCoordinates(const LatticeKeyType& rKey) const
{
    array_1d<double, 3> coordinates;
    for (std::size_t i = 0; i < 3; ++i) {
        // The bounds are reproduced exactly, which matters for the cells touching the box faces
        if (rKey[i] == 0) {
            coordinates[i] = mBoxMin[i];
        } else if (rKey[i] == mLatticeSize) {
            coordinates[i] = mBoxMax[i];
        } else {
            const double ratio = static_cast<double>(rKey[i]) / static_cast<double>(mLatticeSize);
            coordinates[i] = mBoxMin[i] + ratio * (mBoxMax[i] - mBoxMin[i]);
        }
    }

    return coordinates;
}

void HeatTransferCoefficientUtility::ComputeCellCorners(
    const LatticeKeyType& rMinKey,
    const LatticeKeyType& rMaxKey,
    std::array<LatticeKeyType, 8>& rCornerKeys,
    std::array<array_1d<double, 3>, 8>& rCorners) const
{
    // Hexahedra3D8 local ordering: the bottom face counter-clockwise, then the top one
    constexpr std::array<std::array<std::size_t, 3>, 8> corner_pattern = {{
        {{0, 0, 0}}, {{1, 0, 0}}, {{1, 1, 0}}, {{0, 1, 0}},
        {{0, 0, 1}}, {{1, 0, 1}}, {{1, 1, 1}}, {{0, 1, 1}}
    }};

    for (std::size_t i = 0; i < 8; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            rCornerKeys[i][j] = corner_pattern[i][j] == 0 ? rMinKey[j] : rMaxKey[j];
        }
        rCorners[i] = LatticeToCoordinates(rCornerKeys[i]);
    }
}

double HeatTransferCoefficientUtility::Interpolate(
    const DataTableType& rTable,
    const double Temperature)
{
    KRATOS_TRY

    KRATOS_ERROR_IF(rTable.empty()) << "Cannot interpolate an empty table." << std::endl;

    // A single entry describes a constant property
    if (rTable.size() == 1) {
        return rTable[0][1];
    }

    // Clamp outside the tabulated range
    if (Temperature <= rTable.front()[0]) {
        return rTable.front()[1];
    }
    if (Temperature >= rTable.back()[0]) {
        return rTable.back()[1];
    }

    for (std::size_t i = 1; i < rTable.size(); ++i) {
        if (Temperature <= rTable[i][0]) {
            const double x_0 = rTable[i - 1][0];
            const double x_1 = rTable[i][0];
            const double y_0 = rTable[i - 1][1];
            const double y_1 = rTable[i][1];
            return y_0 + (y_1 - y_0) * (Temperature - x_0) / (x_1 - x_0);
        }
    }

    return rTable.back()[1];

    KRATOS_CATCH("")
}

HeatTransferCoefficientUtility::DataTableType HeatTransferCoefficientUtility::ReadDataTable(
    Parameters TableParameters,
    const std::string& rTableName)
{
    KRATOS_TRY

    KRATOS_ERROR_IF_NOT(TableParameters.IsArray())
        << "The '" << rTableName << "' must be an array of [temperature, value] pairs." << std::endl;
    KRATOS_ERROR_IF(TableParameters.size() == 0)
        << "The '" << rTableName << "' cannot be empty." << std::endl;

    DataTableType table;
    table.reserve(TableParameters.size());
    for (std::size_t i = 0; i < TableParameters.size(); ++i) {
        Parameters row = TableParameters.GetArrayItem(i);
        KRATOS_ERROR_IF_NOT(row.IsArray() && row.size() == 2)
            << "The entry " << i << " of '" << rTableName << "' must be a [temperature, value] "
            << "pair." << std::endl;
        table.push_back({{row.GetArrayItem(0).GetDouble(), row.GetArrayItem(1).GetDouble()}});
    }

    // A well defined interpolation requires a strictly increasing input column
    for (std::size_t i = 1; i < table.size(); ++i) {
        KRATOS_ERROR_IF(table[i][0] <= table[i - 1][0])
            << "The temperatures of '" << rTableName << "' must be strictly increasing, but entry "
            << i << " (" << table[i][0] << ") does not exceed the previous one ("
            << table[i - 1][0] << ")." << std::endl;
    }

    return table;

    KRATOS_CATCH("")
}

double HeatTransferCoefficientUtility::ComputePartnerTemperature(const double Temperature) const
{
    // A free surface always exchanges with the ambient
    if (mIsAir) {
        return mAmbientTemperature;
    }

    return mPartnerTemperatureMode == "fixed" ? mSinkTemperature : Temperature - mDeltaTemperature;
}

} // namespace Kratos

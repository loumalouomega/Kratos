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

#pragma once

// System includes
#include <array>
#include <map>
#include <vector>

// External includes

// Project includes
#include "includes/define.h"
#include "containers/model.h"
#include "geometries/plane_3d.h"
#include "includes/kratos_parameters.h"

// Application includes
#include "custom_conditions/thermal_face.h"

namespace Kratos
{

///@name Kratos Globals
///@{

///@}
///@name Type Definitions
///@{

///@}
///@name  Enum's
///@{

///@}
///@name  Functions
///@{

///@}
///@name Kratos Classes
///@{

/**
 * @class HeatTransferCoefficientUtility
 * @ingroup ConvectionDiffusionApplication
 * @brief Estimates the effective heat transfer coefficient (HTC) of the interface between two
 * materials, or between one material and air, and exports it as a temperature-HTC table.
 * @details A cube domain is split by a plane into two parts. An octree is refined towards the
 * dividing plane up to @p max_octree_level, and every leaf becomes an @a Element3D8N hexahedron
 * in the sub model part matching its side. For every leaf cut by the plane, the plane is clipped
 * against the cell to obtain the exact intersection polygon; its area is the tributary interface
 * area and its area centroid becomes an interface node carrying @a NODAL_AREA.
 *
 * For each sample temperature @a T1, with partner temperature @a T2, the pointwise coefficient is
 * @code
 *   h(T1,T2) = f_contact * h_cond + (1 - f_contact) * (h_gap + h_rad)
 *   h_cond   = 1 / (L1/k1(T1) + L2/k2(T2))              solid conduction, series
 *   h_gap    = k_air(Tm) / gap_thickness                gas conduction across the gap
 *   h_rad    = sigma * F12 * (T1^2 + T2^2) * (T1 + T2)  linearised Stefan-Boltzmann
 *   F12      = 1 / (1/e1 + 1/e2 - 1)                    gray-body parallel surfaces
 * @endcode
 * The radiative term is the algebraic linearisation of @a Q = sigma*A*F12*(T1^4 - T2^4), using
 * @a T1^4-T2^4 = (T1-T2)(T1+T2)(T1^2+T2^2), so it has no singularity as @a T1 approaches @a T2.
 * For a free surface (@p is_air) the exchange factor degenerates to @a F12 = e1 and the partner
 * temperature is the ambient one.
 *
 * The interface value is then area-weighted, @a h(T) = sum(A_i*h_i) / sum(A_i).
 *
 * @note The generated octree mesh is intentionally non-conforming: leaves of different refinement
 * levels meet at hanging nodes. The mesh is a diagnostic and area-bookkeeping artefact only and
 * must never be assembled into a system.
 * @note With spatially uniform material data, which is what the current input describes, every
 * @a h_i is identical and the area weighting reduces to the pointwise value. The weighted form is
 * kept because it is the correct one and admits spatially varying properties later.
 * @note All temperatures are absolute (K); the radiative term requires it.
 * @author Vicente Mataix Ferrandiz
 */
class KRATOS_API(CONVECTION_DIFFUSION_APPLICATION) HeatTransferCoefficientUtility
{
public:
    ///@name Type Definitions
    ///@{

    /// The index type
    using IndexType = std::size_t;

    /// A temperature-value table given as a list of (x,y) pairs
    using DataTableType = std::vector<std::array<double, 2>>;

    /// Integer lattice key used to deduplicate the octree nodes
    using LatticeKeyType = std::array<std::int64_t, 3>;

    ///@}
    ///@name Pointer Definitions

    /// Pointer definition of HeatTransferCoefficientUtility
    KRATOS_CLASS_POINTER_DEFINITION(HeatTransferCoefficientUtility);

    ///@}
    ///@name Life Cycle
    ///@{

    /// Default constructor.
    HeatTransferCoefficientUtility() = delete;

    /**
     * @brief Constructor.
     * @param rModel The model containing the model part to be created
     * @param ThisParameters The configuration parameters
     */
    HeatTransferCoefficientUtility(
        Model& rModel,
        Parameters ThisParameters);

    /// Destructor.
    virtual ~HeatTransferCoefficientUtility() = default;

    ///@}
    ///@name Operations
    ///@{

    /**
     * @brief Builds the octree mesh and the interface nodes.
     * @details Fills the @a Material1 and @a Material2 sub model parts with @a Element3D8N
     * hexahedra and the @a Interface sub model part with one node per cut leaf, each carrying
     * its tributary area in @a NODAL_AREA.
     */
    void GenerateMesh();

    /**
     * @brief Computes the temperature-HTC table.
     * @details Requires GenerateMesh() to have been called; it is invoked automatically if the
     * interface has not been built yet.
     * @return The table in the schema understood by ReadMaterialsUtility, that is
     * @a {"input_variable", "output_variable", "data"}
     */
    Parameters ComputeTable();

    /**
     * @brief Computes the pointwise heat transfer coefficient for a temperature pair.
     * @param Temperature1 The temperature of the first material [K]
     * @param Temperature2 The temperature of the partner material [K]
     * @return The heat transfer coefficient [W/(m^2 K)]
     */
    double ComputeHeatTransferCoefficient(
        const double Temperature1,
        const double Temperature2) const;

    /**
     * @brief Returns the total interface area, that is the sum of NODAL_AREA over the interface.
     * @return The total interface area [m^2]
     */
    double GetInterfaceArea() const;

    /// Generates the mesh and computes the table.
    void Execute();

    ///@}
    ///@name Access
    ///@{

    /**
     * @brief Returns the default parameters of this utility.
     * @return The default parameters
     */
    static Parameters GetDefaultParameters();

    ///@}
    ///@name Input and output
    ///@{

    /// Turn back information as a string.
    std::string Info() const
    {
        return "HeatTransferCoefficientUtility";
    }

    /// Print information about this object.
    void PrintInfo(std::ostream& rOStream) const
    {
        rOStream << Info();
    }

    /// Print object's data.
    void PrintData(std::ostream& rOStream) const
    {
        rOStream << Info();
    }

    ///@}
private:
    ///@name Static Member Variables
    ///@{

    /// Stefan Boltzmann constant for radiation in SI units: [W / (m^2 K^4)].
    constexpr static double StefanBoltzmann = ThermalFace::StefanBoltzmann;

    /// Relative tolerance used to merge coincident intersection points.
    constexpr static double RelativeTolerance = 1.0e-12;

    ///@}
    ///@name Member Variables
    ///@{

    ModelPart* mpModelPart = nullptr;        /// The model part holding the generated mesh

    array_1d<double, 3> mBoxMin;             /// Minimum point of the cube domain
    array_1d<double, 3> mBoxMax;             /// Maximum point of the cube domain
    array_1d<double, 3> mPlanePoint;         /// A point belonging to the dividing plane
    array_1d<double, 3> mPlaneNormal;        /// Unit normal of the dividing plane
    IndexType mMaxOctreeLevel;               /// Maximum octree refinement level
    std::int64_t mLatticeSize;               /// Number of finest cells per direction, 2^mMaxOctreeLevel

    DataTableType mConductivityTable1;       /// Conductivity of material 1 as a function of T
    DataTableType mConductivityTable2;       /// Conductivity of material 2 as a function of T
    DataTableType mAirConductivityTable;     /// Conductivity of the gap gas as a function of T

    double mEmissivity1;                     /// Emissivity of material 1
    double mEmissivity2;                     /// Emissivity of material 2
    double mThickness1;                      /// Conduction length in material 1
    double mThickness2;                      /// Conduction length in material 2

    bool mIsAir;                             /// If true material 2 is a free surface exposed to air
    double mAmbientTemperature;              /// Ambient temperature used when mIsAir is true
    double mContactFraction;                 /// Fraction of the interface in solid contact
    double mGapThickness;                    /// Thickness of the gas gap

    bool mIncludeConduction;                 /// If the solid conduction term is accounted for
    bool mIncludeGapConduction;              /// If the gas conduction term is accounted for
    bool mIncludeRadiation;                  /// If the radiative term is accounted for

    std::vector<double> mSampleTemperatures; /// Sample temperatures of the resulting table
    std::string mPartnerTemperatureMode;     /// Either "delta" or "fixed"
    double mDeltaTemperature;                /// Temperature drop used in the "delta" mode
    double mSinkTemperature;                 /// Partner temperature used in the "fixed" mode
    double mPlateauEpsilon;                  /// Half width of the plateau around each sample

    std::string mOutputVariableName;         /// Name of the output variable of the table

    std::vector<double> mInterfaceAreas;     /// Tributary area of every interface node

    ///@}
    ///@name Private Operations
    ///@{

    /**
     * @brief Recursively refines a cell towards the dividing plane.
     * @details The cell is addressed through integer lattice coordinates at the finest
     * resolution, so that the geometry of a corner shared by several cells is bit-identical and
     * the node deduplication needs no floating point tolerance.
     * @param rMinKey Minimum lattice coordinates of the cell
     * @param rMaxKey Maximum lattice coordinates of the cell
     * @param Level Current refinement level
     * @param rPlane The dividing plane
     * @param rNodeIds Map from the integer lattice keys to the already created node ids
     * @param rNodeCounter Running node id counter
     * @param rElementCounter Running element id counter
     */
    void RefineCell(
        const LatticeKeyType& rMinKey,
        const LatticeKeyType& rMaxKey,
        const IndexType Level,
        Plane3D& rPlane,
        std::map<LatticeKeyType, IndexType>& rNodeIds,
        IndexType& rNodeCounter,
        IndexType& rElementCounter);

    /**
     * @brief Creates the Element3D8N corresponding to a leaf cell.
     * @param rMinKey Minimum lattice coordinates of the cell
     * @param rMaxKey Maximum lattice coordinates of the cell
     * @param rPlane The dividing plane
     * @param rNodeIds Map from the integer lattice keys to the already created node ids
     * @param rNodeCounter Running node id counter
     * @param rElementCounter Running element id counter
     */
    void CreateLeafElement(
        const LatticeKeyType& rMinKey,
        const LatticeKeyType& rMaxKey,
        Plane3D& rPlane,
        std::map<LatticeKeyType, IndexType>& rNodeIds,
        IndexType& rNodeCounter,
        IndexType& rElementCounter);

    /**
     * @brief Clips the dividing plane against a cell and creates the corresponding interface node.
     * @param rMinKey Minimum lattice coordinates of the cell
     * @param rMaxKey Maximum lattice coordinates of the cell
     * @param rPlane The dividing plane
     * @param rNodeCounter Running node id counter
     */
    void CreateInterfaceNode(
        const LatticeKeyType& rMinKey,
        const LatticeKeyType& rMaxKey,
        Plane3D& rPlane,
        IndexType& rNodeCounter);

    /**
     * @brief Transforms lattice coordinates into physical coordinates.
     * @param rKey The lattice coordinates
     * @return The corresponding physical point
     */
    array_1d<double, 3> LatticeToCoordinates(const LatticeKeyType& rKey) const;

    /**
     * @brief Computes the eight corners of an axis aligned cell.
     * @param rMinKey Minimum lattice coordinates of the cell
     * @param rMaxKey Maximum lattice coordinates of the cell
     * @param rCornerKeys The resulting corner lattice keys, in the Hexahedra3D8 local ordering
     * @param rCorners The resulting corners, in the Hexahedra3D8 local ordering
     */
    void ComputeCellCorners(
        const LatticeKeyType& rMinKey,
        const LatticeKeyType& rMaxKey,
        std::array<LatticeKeyType, 8>& rCornerKeys,
        std::array<array_1d<double, 3>, 8>& rCorners) const;

    /**
     * @brief Linearly interpolates a temperature dependent table, clamping outside its range.
     * @param rTable The table to be interpolated
     * @param Temperature The temperature at which the table is evaluated
     * @return The interpolated value
     */
    static double Interpolate(
        const DataTableType& rTable,
        const double Temperature);

    /**
     * @brief Reads a list of (x,y) pairs from the parameters.
     * @param TableParameters The parameters holding the list of pairs
     * @param rTableName The name of the entry, used in the error messages
     * @return The parsed table
     */
    static DataTableType ReadDataTable(
        Parameters TableParameters,
        const std::string& rTableName);

    /**
     * @brief Returns the partner temperature associated to a sample temperature.
     * @param Temperature The sample temperature
     * @return The partner temperature
     */
    double ComputePartnerTemperature(const double Temperature) const;

    ///@}
}; // Class HeatTransferCoefficientUtility

///@}
///@name Input and output
///@{

/// input stream function
inline std::istream& operator >> (
    std::istream& rIStream,
    HeatTransferCoefficientUtility& rThis);

/// output stream function
inline std::ostream& operator << (
    std::ostream& rOStream,
    const HeatTransferCoefficientUtility& rThis)
{
    rThis.PrintInfo(rOStream);
    rOStream << std::endl;
    rThis.PrintData(rOStream);

    return rOStream;
}

///@}

} // namespace Kratos

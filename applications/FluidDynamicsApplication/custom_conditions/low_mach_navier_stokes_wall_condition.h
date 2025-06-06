//    |  /           |
//    ' /   __| _` | __|  _ \   __|
//    . \  |   (   | |   (   |\__ \.
//   _|\_\_|  \__,_|\__|\___/ ____/
//                   Multi-Physics
//
//  License:         BSD License
//                   Kratos default license: kratos/license.txt
//
//  Main authors:    Ruben Zorrilla
//

#pragma once

// System includes
#include <string>
#include <iostream>


// External includes


// Project includes
#include "includes/define.h"
#include "includes/condition.h"
#include "includes/model_part.h"
#include "includes/serializer.h"
#include "includes/process_info.h"

// Application includes
#include "fluid_dynamics_application_variables.h"

namespace Kratos
{

///@addtogroup FluidDynamicsApplication
///@{

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
 * @brief Implements a wall condition for the Navier-Stokes (and Stokes) monolithic formulations
 * This condition is intended to be used in combination with Navier-Stokes (or Stokes) monolithic
 * formulations. It supports the Neumann BC contribution as well as the addition of a wall law
 * contribution through the TWallModel template argument. Such TWallModel must be a class implementing
 * the wall model RHS and LHS Gauss point contributions (as example see @NavierSlipWallLaw).
 * Current condition also has optional features that help numerical stability such as the outlet
 * inflow energy correction or the spurious tangential velocity correction for pure slip boundaries.
 * @tparam TDim Number of dimensions
 * @tparam TNumNodes Number of nodes
 * @tparam TWallModel Optional class implementing a LHS and RHS wall contribution
 */
template<unsigned int TDim, unsigned int TNumNodes, class... TWallModel>
class KRATOS_API(FLUID_DYNAMICS_APPLICATION) LowMachNavierStokesWallCondition : public Condition
{
public:
    ///@name Type Definitions
    ///@{

    /// Pointer definition of NavierStokesWallCondition
    KRATOS_CLASS_INTRUSIVE_POINTER_DEFINITION(LowMachNavierStokesWallCondition);

    // struct ConditionDataStruct
    // {
    //     double wGauss;                                  // Gauss point weight
    //     array_1d<double, 3> Normal;                     // Condition normal
    //     array_1d<double, TNumNodes> N;                  // Gauss point shape functions values
    //     Vector ViscousStress;                           // Viscous stresses that are retrieved from parent
    // };

    using SizeType = Condition::SizeType;

    typedef Node NodeType;

    typedef Properties PropertiesType;

    typedef Geometry<NodeType> GeometryType;

    typedef Geometry<NodeType>::PointsArrayType NodesArrayType;

    typedef Vector VectorType;

    typedef Matrix MatrixType;

    typedef std::size_t IndexType;

    typedef std::vector<std::size_t> EquationIdVectorType;

    typedef std::vector< Dof<double>::Pointer > DofsVectorType;

    static constexpr SizeType Dim = TDim;

    static constexpr SizeType NumNodes = TNumNodes;

    static constexpr SizeType VoigtSize = 3 * (TDim-1);

    static constexpr SizeType BlockSize = TDim + 2;

    static constexpr SizeType LocalSize = TNumNodes*BlockSize;

    ///@}
    ///@name Life Cycle
    ///@{

    LowMachNavierStokesWallCondition(IndexType NewId = 0):Condition(NewId)
    {
    }

    LowMachNavierStokesWallCondition(
        IndexType NewId,
        const NodesArrayType& ThisNodes)
        : Condition(NewId, ThisNodes)
    {
    }

    LowMachNavierStokesWallCondition(
        IndexType NewId,
        GeometryType::Pointer pGeometry)
        : Condition(NewId, pGeometry)
    {
    }

    LowMachNavierStokesWallCondition(
        IndexType NewId,
        GeometryType::Pointer pGeometry,
        PropertiesType::Pointer pProperties)
        : Condition(NewId, pGeometry, pProperties)
    {
    }

    LowMachNavierStokesWallCondition(LowMachNavierStokesWallCondition const& rOther)
        : Condition(rOther)
    {
    }

    ~LowMachNavierStokesWallCondition() override {}

    ///@}
    ///@name Operators
    ///@{

    LowMachNavierStokesWallCondition& operator=(LowMachNavierStokesWallCondition const& rOther)
    {
        Condition::operator=(rOther);
        return *this;
    }

    ///@}
    ///@name Operations
    ///@{

    Condition::Pointer Create(
        IndexType NewId,
        NodesArrayType const& ThisNodes,
        PropertiesType::Pointer pProperties) const override
    {
        return Kratos::make_intrusive<LowMachNavierStokesWallCondition>(NewId, GetGeometry().Create(ThisNodes), pProperties);
    }

    Condition::Pointer Create(
        IndexType NewId,
        GeometryType::Pointer pGeom,
        PropertiesType::Pointer pProperties) const override
    {
        return Kratos::make_intrusive< LowMachNavierStokesWallCondition >(NewId, pGeom, pProperties);
    }

    Condition::Pointer Clone(
        IndexType NewId,
        NodesArrayType const& rThisNodes) const override
    {
        Condition::Pointer pNewCondition = Create(NewId, GetGeometry().Create( rThisNodes ), pGetProperties() );

        pNewCondition->SetData(this->GetData());
        pNewCondition->SetFlags(this->GetFlags());

        return pNewCondition;
    }

    void CalculateLocalSystem(
        MatrixType& rLeftHandSideMatrix,
        VectorType& rRightHandSideVector,
        const ProcessInfo& rCurrentProcessInfo) override;

    void CalculateLeftHandSide(
        MatrixType& rLeftHandSideMatrix,
        const ProcessInfo& rCurrentProcessInfo) override;

    void CalculateRightHandSide(
        VectorType& rRightHandSideVector,
        const ProcessInfo& rCurrentProcessInfo) override;

    int Check(const ProcessInfo& rCurrentProcessInfo) const override;

    void EquationIdVector(
        EquationIdVectorType& rResult,
        const ProcessInfo& rCurrentProcessInfo) const override;

    void GetDofList(
        DofsVectorType& rConditionDofList,
        const ProcessInfo& rCurrentProcessInfo) const override;

    void Calculate(
        const Variable< array_1d<double,3> >& rVariable,
        array_1d<double,3>& Output,
        const ProcessInfo& rCurrentProcessInfo) override;

    ///@}
    ///@name Access
    ///@{


    ///@}
    ///@name Inquiry
    ///@{


    ///@}
    ///@name Input and output
    ///@{

    /// Turn back information as a string.
    std::string Info() const override
    {
        std::stringstream buffer;
        buffer << "LowMachNavierStokesWallCondition" << TDim << "D";
        return buffer.str();
    }

    /// Print information about this object.
    void PrintInfo(std::ostream& rOStream) const override
    {
        rOStream << "LowMachNavierStokesWallCondition";
    }

    /// Print object's data.
    void PrintData(std::ostream& rOStream) const override {}


    ///@}
    ///@name Friends
    ///@{


    ///@}
protected:
    ///@name Protected static Member Variables
    ///@{


    ///@}
    ///@name Protected member Variables
    ///@{


    ///@}
    ///@name Protected Operators
    ///@{


    ///@}
    ///@name Protected Operations
    ///@{

    /**
     * @brief Calculate the condition area normal
     * This method calculates the current condition area normal
     * @param rAreaNormal Reference to the current condition area normal
     */
    void CalculateNormal(array_1d<double,3>& rAreaNormal);

    // /**
    //  * @brief Calculates the Gauss point LHS contribution
    //  * This method calculates the current Gauss point LHS contribution and saves it
    //  * in the provided array. Note that the input data container is expected to
    //  * already contain the data at the Gauss point of interest.
    //  * @param rLHS Reference to the LHS output matrix
    //  * @param rData Condition data container
    //  * @param rProcessInfo Reference to the ProcessInfo container
    //  */
    // void ComputeGaussPointLHSContribution(
    //     BoundedMatrix<double, LocalSize, LocalSize>& rLHS,
    //     const ConditionDataStruct& rData,
    //     const ProcessInfo& rProcessInfo);

    // /**
    //  * @brief Calculates the Gauss point RHS contribution
    //  * This method calculates the current Gauss point RHS contribution and saves it
    //  * in the provided array. Note that the input data container is expected to
    //  * already contain the data at the Gauss point of interest.
    //  * @param rLHS Reference to the RHS output vector
    //  * @param rData Condition data container
    //  * @param rProcessInfo Reference to the ProcessInfo container
    //  */
    // void ComputeGaussPointRHSContribution(
    //     array_1d<double, LocalSize>& rRHS,
    //     const ConditionDataStruct& rData,
    //     const ProcessInfo& rProcessInfo);

    // /**
    //  * @brief Calculates the RHS Neumann BC contribution
    //  * This method calculates the Neumann BC pressure flux contribution
    //  * Note that the Neumann BC value is expected to be stored in the historical
    //  * database within the EXTERNAL_PRESSURE variable.
    //  * @param rRHS Reference to the RHS output vector
    //  * @param data Condition data container
    //  */
    // void ComputeRHSNeumannContribution(
    //     array_1d<double,LocalSize>& rRHS,
    //     const ConditionDataStruct& data);

    // /**
    //  * @brief Calculates and adds the RHS outlet inflow prevention contribution
    //  * This method calculates and adds an extra numerical contribution to the RHS in order
    //  * to prevent uncontrolled system energy growth coming from inflow in free-boundaries.
    //  * More information can be found in Dong et al. 2014 (https://doi.org/10.1016/j.jcp.2013.12.042).
    //  * @param rRHS Reference to RHS vector
    //  * @param rData Condition data container
    //  * @param rProcessInfo Reference to the ProcessInfo container
    //  */
    // void ComputeRHSOutletInflowContribution(
    //     array_1d<double, LocalSize>& rRHS,
    //     const ConditionDataStruct& rData,
    //     const ProcessInfo& rProcessInfo);

    ///@}
    ///@name Protected  Access
    ///@{


    ///@}
    ///@name Protected Inquiry
    ///@{


    ///@}
    ///@name Protected LifeCycle
    ///@{


    ///@}
private:
    ///@name Static Member Variables
    ///@{


    ///@}
    ///@name Member Variables
    ///@{


    ///@}
    ///@name Serialization
    ///@{

    friend class Serializer;

    void save(Serializer& rSerializer) const override
    {
        KRATOS_SERIALIZE_SAVE_BASE_CLASS(rSerializer, Condition );
    }

    void load(Serializer& rSerializer) override
    {
        KRATOS_SERIALIZE_LOAD_BASE_CLASS(rSerializer, Condition );
    }

    ///@}
    ///@name Private Operators
    ///@{


    ///@}
    ///@name Private Operations
    ///@{

    // /**
    //  * @brief Computes the left-hand side contribution for the slip tangential correction
    //  * This specific implementation of the slip condition avoids spurious velocities
    //  * at points were the normal directions of the adjacent boundary geometries do not
    //  * coincide (Reference Behr (2004): https://onlinelibrary.wiley.com/doi/abs/10.1002/fld.663)
    //  * @param rLeftHandSideMatrix reference to the LHS matrix
    //  * @param rDataStruct reference to a struct to hand over data
    //  */
    // void CalculateGaussPointSlipTangentialCorrectionLHSContribution(
    //     BoundedMatrix<double,LocalSize,LocalSize>& rLeftHandSideMatrix,
    //     const ConditionDataStruct& rDataStruct);

    // /**
    //  * @brief Computes the right-hand side contribution for the slip tangential correction
    //  * This specific implementation of the slip condition avoids spurious velocities
    //  * at points were the normal directions of the adjacent boundary geometries do not
    //  * coincide (Reference Behr (2004): https://onlinelibrary.wiley.com/doi/abs/10.1002/fld.663)
    //  * @param rRightHandSideVector reference to the RHS vector
    //  * @param rDataStruct reference to a struct to hand over data
    //  */
    // void CalculateGaussPointSlipTangentialCorrectionRHSContribution(
    //     array_1d<double,LocalSize>& rRightHandSideVector,
    //     const ConditionDataStruct& rDataStruct);

    // /**
    //  * @brief Project the viscous stress
    //  * Provided a viscous stress tensor (in Voigt notation) and a unit normal vector,
    //  * this method calculates and returns the projection of the shear stress onto the normal
    //  * @param rViscousStress The viscous stress in Voigt notation
    //  * @param rNormal The unit normal vector to project onto
    //  * @param rProjectedViscousStress The projected viscous stress
    //  */
    // void ProjectViscousStress(
    //     const Vector& rViscousStress,
    //     const array_1d<double,3> rNormal,
    //     array_1d<double,3>& rProjectedViscousStress);

    // /**
    //  * @brief Set the Tangential Projection Matrix
    //  * For the given unit normal, this method sets the corresponding tangential projection matrix
    //  * @param rUnitNormal Reference to the unit normal
    //  * @param rTangProjMat Reference to the output tangential projection matrix
    //  */
    // void SetTangentialProjectionMatrix(
    //     const array_1d<double,3>& rUnitNormal,
    //     BoundedMatrix<double,TDim,TDim>& rTangProjMat)
    // {
    //     noalias(rTangProjMat) = IdentityMatrix(TDim,TDim);
    //     for (std::size_t d1 = 0; d1 < TDim; ++d1) {
    //         for (std::size_t d2 = 0; d2 < TDim; ++d2) {
    //             rTangProjMat(d1,d2) -= rUnitNormal[d1]*rUnitNormal[d2];
    //         }
    //     }
    // }

    template<typename TWallModelType>
    int WallModelCheckCall(const ProcessInfo& rProcessInfo) const
    {
        return TWallModelType::Check(this, rProcessInfo);
    }

    template<typename TWallModelType>
    void AddWallModelRightHandSideCall(
        VectorType& rRHS,
        const ProcessInfo& rProcessInfo)
    {
        TWallModelType::AddWallModelRightHandSide(rRHS, this, rProcessInfo);
    }

    template<typename TWallModelType>
    void AddWallModelLeftHandSideCall(
        MatrixType& rLHS,
        const ProcessInfo& rProcessInfo)
    {
        TWallModelType::AddWallModelLeftHandSide(rLHS, this, rProcessInfo);
    }

    template<typename TWallModelType>
    void AddWallModelLocalSystemCall(
        MatrixType& rLHS,
        VectorType& rRHS,
        const ProcessInfo& rProcessInfo)
    {
        TWallModelType::AddWallModelLocalSystem(rLHS, rRHS, this, rProcessInfo);
    }

    ///@}
    ///@name Private  Access
    ///@{


    ///@}
    ///@name Private Inquiry
    ///@{


    ///@}
    ///@name Un accessible methods
    ///@{


    ///@}
}; // Class LowMachNavierStokesWallCondition

///@}
///@name Type Definitions
///@{


///@}
///@name Input and output
///@{

/// input stream function
template< unsigned int TDim, unsigned int TNumNodes, class TWallModel >
inline std::istream& operator >> (
    std::istream& rIStream,
    LowMachNavierStokesWallCondition<TDim,TNumNodes,TWallModel>& rThis)
{
    return rIStream;
}

/// output stream function
template< unsigned int TDim, unsigned int TNumNodes, class TWallModel >
inline std::ostream& operator << (
    std::ostream& rOStream,
    const LowMachNavierStokesWallCondition<TDim,TNumNodes,TWallModel>& rThis)
{
    rThis.PrintInfo(rOStream);
    rOStream << std::endl;
    rThis.PrintData(rOStream);

    return rOStream;
}

///@}

///@} addtogroup block

}  // namespace Kratos.

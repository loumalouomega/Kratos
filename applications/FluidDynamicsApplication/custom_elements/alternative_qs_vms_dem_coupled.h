//    |  /           |
//    ' /   __| _` | __|  _ \   __|
//    . \  |   (   | |   (   |\__ `
//   _|\_\_|  \__,_|\__|\___/ ____/
//                   Multi-Physics
//
//  License:         BSD License
//                   Kratos default license: kratos/license.txt
//
//  Main authors:    Joaquin Gonzalez-Usua
//

#ifndef KRATOS_ALTERNATIVE_QS_VMS_DEM_COUPLED_H
#define KRATOS_ALTERNATIVE_QS_VMS_DEM_COUPLED_H

// System includes
#include <string>
#include <iostream>

// External includes

// Project includes
#include "includes/checks.h"
#include "includes/element.h"
#include "includes/serializer.h"
#include "geometries/geometry.h"
#include "includes/cfd_variables.h"

// Application includes
#include "custom_elements/fluid_element.h"
#include "custom_elements/qs_vms.h"
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

template< class TElementData >
class AlternativeQSVMSDEMCoupled : public QSVMS<TElementData>
{
public:
    ///@name Type Definitions
    ///@{

    /// Pointer definition of AlternativeQSVMSDEMCoupled
    KRATOS_CLASS_INTRUSIVE_POINTER_DEFINITION(AlternativeQSVMSDEMCoupled);

    /// Node type (default is: Node)
    typedef Node NodeType;

    /// Geometry type (using with given NodeType)
    typedef Geometry<NodeType> GeometryType;

    /// Definition of nodes container type, redefined from GeometryType
    typedef Geometry<NodeType>::PointsArrayType NodesArrayType;

    /// Vector type for local contributions to the linear system
    typedef Vector VectorType;

    /// Matrix type for local contributions to the linear system
    typedef Matrix MatrixType;

    typedef std::size_t IndexType;

    typedef std::size_t SizeType;

    typedef std::vector<std::size_t> EquationIdVectorType;

    typedef std::vector< Dof<double>::Pointer > DofsVectorType;

    typedef PointerVectorSet<Dof<double>, IndexedObject> DofsArrayType;

    /// Type for shape function values container
    typedef Kratos::Vector ShapeFunctionsType;

    /// Type for a matrix containing the shape function gradients
    typedef Kratos::Matrix ShapeFunctionDerivativesType;

    /// Type for an array of shape function gradient matrices
    typedef GeometryType::ShapeFunctionsGradientsType ShapeFunctionDerivativesArrayType;

    typedef GeometryType::ShapeFunctionsSecondDerivativesType ShapeFunctionsSecondDerivativesType;

    constexpr static unsigned int Dim = QSVMS<TElementData>::Dim;
    constexpr static unsigned int NumNodes = QSVMS<TElementData>::NumNodes;
    constexpr static unsigned int BlockSize = QSVMS<TElementData>::BlockSize;
    constexpr static unsigned int LocalSize = QSVMS<TElementData>::LocalSize;
    constexpr static unsigned int StrainSize = QSVMS<TElementData>::StrainSize;

    ///@}
    ///@name Life Cycle
    ///@{

    //Constructors.

    /// Default constructor.
    /**
     * @param NewId Index number of the new element (optional)
     */
    AlternativeQSVMSDEMCoupled(IndexType NewId = 0);

    /// Constructor using an array of nodes.
    /**
     * @param NewId Index of the new element
     * @param ThisNodes An array containing the nodes of the new element
     */
    AlternativeQSVMSDEMCoupled(IndexType NewId, const NodesArrayType& ThisNodes);

    /// Constructor using a geometry object.
    /**
     * @param NewId Index of the new element
     * @param pGeometry Pointer to a geometry object
     */
    AlternativeQSVMSDEMCoupled(IndexType NewId, GeometryType::Pointer pGeometry);

    /// Constructor using geometry and properties.
    /**
     * @param NewId Index of the new element
     * @param pGeometry Pointer to a geometry object
     * @param pProperties Pointer to the element's properties
     */
    AlternativeQSVMSDEMCoupled(IndexType NewId, GeometryType::Pointer pGeometry, Properties::Pointer pProperties);

    /// Destructor.
    ~AlternativeQSVMSDEMCoupled();

    ///@}
    ///@name Operators
    ///@{


    ///@}
    ///@name Operations
    ///@{


    /// Create a new element of this type
    /**
     * Returns a pointer to a new AlternativeQSVMSDEMCoupled element, created using given input
     * @param NewId the ID of the new element
     * @param ThisNodes the nodes of the new element
     * @param pProperties the properties assigned to the new element
     * @return a Pointer to the new element
     */
    Element::Pointer Create(
        IndexType NewId,
        NodesArrayType const& ThisNodes,
        Properties::Pointer pProperties) const override;

    /// Create a new element of this type using given geometry
    /**
     * Returns a pointer to a new FluidElement element, created using given input
     * @param NewId the ID of the new element
     * @param pGeom a pointer to the geometry to be used to create the element
     * @param pProperties the properties assigned to the new element
     * @return a Pointer to the new element
     */
    Element::Pointer Create(
        IndexType NewId,
        GeometryType::Pointer pGeom,
        Properties::Pointer pProperties) const override;


    void Initialize(const ProcessInfo& rCurrentProcessInfo) override;

    GeometryData::IntegrationMethod GetIntegrationMethod() const override;

    ///@}
    ///@name Access
    ///@{

    ///@}
    ///@name Inquiry
    ///@{

    int Check(const ProcessInfo &rCurrentProcessInfo) const override;

    ///@}
    ///@name Input and output
    ///@{

    /// Turn back information as a string.
    std::string Info() const override;


    /// Print information about this object.
    void PrintInfo(std::ostream& rOStream) const override;


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


    // Velocity subscale history, stored at integration points
    DenseVector< array_1d<double,Dim> > mPredictedSubscaleVelocity;
    DenseVector< array_1d<double,Dim> > mPreviousVelocity;
    DenseVector <BoundedMatrix<double,Dim,Dim>> mViscousResistanceTensor;
    int mInterpolationOrder = 1;
    std::vector<double> mPorosity;
    std::vector<double> mPorosityRate;
    std::vector<Vector> mPorosityGradient;
    std::vector<Vector> mBodyForce;
    ///@}
    ///@name Protected Operators
    ///@{


    ///@}
    ///@name Protected Operations
    ///@{

    // Protected interface of FluidElement ////////////////////////////////////

    /// Determine the shape second derivative in the gauss point

    void AlgebraicMomentumResidual(
        const TElementData& rData,
        const array_1d<double,3> &rConvectionVelocity,
        array_1d<double,3>& rResidual) const override;

    void MomentumProjTerm(
        const TElementData& rData,
        const array_1d<double,3>& rConvectionVelocity,
        array_1d<double,3> &rMomentumRHS) const override;

    void AddMassStabilization(
        TElementData& rData,
        MatrixType &rMassMatrix);

    void AddReactionStabilization(
        TElementData& rData,
        BoundedMatrix<double,NumNodes*(Dim+1),NumNodes*(Dim+1)>& rLHS,
        VectorType& rLocalRHS);

    using QSVMS<TElementData>::CalculateTau;
    void CalculateTau(
        const TElementData& rData,
        const array_1d<double,3> &Velocity,
        BoundedMatrix<double,Dim,Dim> &TauOne,
        double &TauTwo) const;

    void CalculateProjections(
        const ProcessInfo &rCurrentProcessInfo) override;

    void UpdateIntegrationPointDataSecondDerivatives(
        TElementData& rData,
        unsigned int IntegrationPointIndex,
        double Weight,
        const typename TElementData::MatrixRowType& rN,
        const typename TElementData::ShapeDerivativesType& rDN_DX,
        const typename TElementData::ShapeFunctionsSecondDerivativesType& rDDN_DDX) const;

    void AddVelocitySystem(
        TElementData& rData,
        MatrixType &rLocalLHS,
        VectorType &rLocalRHS) override;

    void CalculateMassMatrix(MatrixType& rMassMatrix,
                            const ProcessInfo& rCurrentProcessInfo) override;

    void CalculateLocalVelocityContribution(MatrixType& rDampMatrix,
                                            VectorType& rRightHandSideVector,
                                            const ProcessInfo& rCurrentProcessInfo) override;

    void AddMassLHS(
        TElementData& rData,
        MatrixType& rMassMatrix) override;

    void CalculateResistanceTensor(
        const TElementData& rData);

    void MassProjTerm(
        const TElementData& rData,
        double &rMassRHS) const override;

    void SubscaleVelocity(
        const TElementData& rData,
        array_1d<double,3> &rVelocitySubscale) const override;

    void SubscalePressure(
        const TElementData& rData,
        double &rPressureSubscale) const override;

    void Calculate(
        const Variable<array_1d<double, 3>>& rVariable,
        array_1d<double, 3>& rOutput, const ProcessInfo& rCurrentProcessInfo) override;

    void Calculate(
        const Variable<Matrix>& rVariable,
        Matrix& rOutput, const ProcessInfo& rCurrentProcessInfo) override;

    void CalculateOnIntegrationPoints(
        const Variable<array_1d<double, 3>>& rVariable,
        std::vector<array_1d<double, 3>>& rOutput,
        const ProcessInfo& rCurrentProcessInfo) override;

    void CalculateOnIntegrationPoints(
        const Variable<double>& rVariable,
        std::vector<double>& rOutput,
        const ProcessInfo& rCurrentProcessInfo) override;

    void CalculateOnIntegrationPoints(
        Variable<Matrix> const& rVariable,
        std::vector<Matrix>& rOutput,
        ProcessInfo const& rCurrentProcessInfo) override;

    void InitializeNonLinearIteration(const ProcessInfo& rCurrentProcessInfo) override;

    void FinalizeNonLinearIteration(const ProcessInfo& rCurrentProcessInfo) override;

    void UpdateSubscaleVelocity(const TElementData& rData);

    array_1d<double,3> FullConvectiveVelocity(const TElementData& rData) const;

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

    void save(Serializer& rSerializer) const override;

    void load(Serializer& rSerializer) override;

    ///@}
    ///@name Private Operators
    ///@{


    ///@}
    ///@name Private Operations
    ///@{


    ///@}
    ///@name Private  Access
    ///@{


    ///@}
    ///@name Private Inquiry
    ///@{


    ///@}
    ///@name Un accessible methods
    ///@{

    /// Assignment operator.
    AlternativeQSVMSDEMCoupled& operator=(AlternativeQSVMSDEMCoupled const& rOther);

    /// Copy constructor.
    AlternativeQSVMSDEMCoupled(AlternativeQSVMSDEMCoupled const& rOther);
    ///@}


}; // Class AlternativeQSVMSDEMCoupled

///@}

///@name Type Definitions
///@{


///@}
///@name Input and output
///@{


/// input stream function
template< class TElementData >
inline std::istream& operator >>(std::istream& rIStream,
                                 AlternativeQSVMSDEMCoupled<TElementData>& rThis)
{
    return rIStream;
}

/// output stream function
template< class TElementData >
inline std::ostream& operator <<(std::ostream& rOStream,
                                 const AlternativeQSVMSDEMCoupled<TElementData>& rThis)
{
    rThis.PrintInfo(rOStream);
    rOStream << std::endl;
    rThis.PrintData(rOStream);

    return rOStream;
}
///@}

///@} // Fluid Dynamics Application group

} // namespace Kratos.

#endif // KRATOS_ALTERNATIVE_QS_VMS_DEM_COUPLED_H
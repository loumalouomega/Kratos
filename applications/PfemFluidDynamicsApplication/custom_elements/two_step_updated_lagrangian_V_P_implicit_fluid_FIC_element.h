//
//   Project Name:        KratosFluidDynamicsApplication $
//   Last modified by:    $Author:               AFranci $
//   Date:                $Date:           February 2016 $
//   Revision:            $Revision:                 0.0 $
//
//

#if !defined(KRATOS_TWO_STEP_UPDATED_LAGRANGIAN_V_P_IMPLICIT_FLUID_FIC_ELEMENT_H_INCLUDED)
#define KRATOS_TWO_STEP_UPDATED_LAGRANGIAN_V_P_IMPLICIT_FLUID_FIC_ELEMENT_H_INCLUDED

// System includes
#include <string>
#include <iostream>

// External includes

// Project includes
#include "containers/array_1d.h"
#include "includes/define.h"
/* #include "includes/element.h" */
#include "includes/serializer.h"
#include "geometries/geometry.h"
#include "utilities/math_utils.h"
#include "includes/constitutive_law.h"

#include "custom_elements/two_step_updated_lagrangian_V_P_implicit_fluid_element.h"

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

  /// A stabilized element for the incompressible Navier-Stokes equations.
  /**
   */
  template <unsigned int TDim>
  class TwoStepUpdatedLagrangianVPImplicitFluidFicElement : public TwoStepUpdatedLagrangianVPImplicitFluidElement<TDim>
  {

  public:
    ///@name Type Definitions
    ///@{

    /// Pointer definition of TwoStepUpdatedLagrangianVPImplicitFluidFicElement
    KRATOS_CLASS_INTRUSIVE_POINTER_DEFINITION(TwoStepUpdatedLagrangianVPImplicitFluidFicElement);

    ///base type:
    typedef TwoStepUpdatedLagrangianVPImplicitFluidElement<TDim> BaseType;

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

    typedef std::vector<Dof<double>::Pointer> DofsVectorType;

    typedef PointerVectorSet<Dof<double>, IndexedObject> DofsArrayType;

    /// Type for shape function values container
    typedef Kratos::Vector ShapeFunctionsType;

    /// Type for a matrix containing the shape function gradients
    typedef Kratos::Matrix ShapeFunctionDerivativesType;

    /// Type for an array of shape function gradient matrices
    typedef GeometryType::ShapeFunctionsGradientsType ShapeFunctionDerivativesArrayType;

    typedef typename BaseType::PropertiesType PropertiesType;

    typedef typename BaseType::PropertiesType::Pointer pPropertiesType;

    typedef typename BaseType::ElementalVariables ElementalVariables;

    typedef GlobalPointersVector<Node> NodeWeakPtrVectorType;

    /// Reference type definition for constitutive laws
    typedef ConstitutiveLaw ConstitutiveLawType;

    ///Pointer type for constitutive laws
    typedef ConstitutiveLawType::Pointer ConstitutiveLawPointerType;

    ///@}
    ///@name Life Cycle
    ///@{

    //Constructors.

    /// Default constructor.
    /**
       * @param NewId Index number of the new element (optional)
       */
    TwoStepUpdatedLagrangianVPImplicitFluidFicElement(IndexType NewId = 0) : BaseType(NewId)
    {
    }

    /// Constructor using an array of nodes.
    /**
       * @param NewId Index of the new element
       * @param ThisNodes An array containing the nodes of the new element
       */
    TwoStepUpdatedLagrangianVPImplicitFluidFicElement(IndexType NewId, const NodesArrayType &ThisNodes) : BaseType(NewId, ThisNodes)
    {
    }

    /// Constructor using a geometry object.
    /**
       * @param NewId Index of the new element
       * @param pGeometry Pointer to a geometry object
       */
    TwoStepUpdatedLagrangianVPImplicitFluidFicElement(IndexType NewId, GeometryType::Pointer pGeometry) : BaseType(NewId, pGeometry)
    {
    }

    /// Constructor using geometry and properties.
    /**
       * @param NewId Index of the new element
       * @param pGeometry Pointer to a geometry object
       * @param pProperties Pointer to the element's properties
       */
    TwoStepUpdatedLagrangianVPImplicitFluidFicElement(IndexType NewId, GeometryType::Pointer pGeometry, pPropertiesType pProperties) : BaseType(NewId, pGeometry, pProperties)
    {
    }

    /// copy constructor

    TwoStepUpdatedLagrangianVPImplicitFluidFicElement(TwoStepUpdatedLagrangianVPImplicitFluidFicElement const &rOther) : BaseType(rOther)
    {
    }

    /// Destructor.
    virtual ~TwoStepUpdatedLagrangianVPImplicitFluidFicElement()
    {
    }

    ///@}
    ///@name Operators
    ///@{

    ///@}
    ///@name Operations
    ///@{

    /// Create a new element of this type
    /**
       * Returns a pointer to a new TwoStepUpdatedLagrangianVPImplicitFluidFicElement element, created using given input
       * @param NewId: the ID of the new element
       * @param ThisNodes: the nodes of the new element
       * @param pProperties: the properties assigned to the new element
       * @return a Pointer to the new element
       */
    Element::Pointer Create(IndexType NewId, NodesArrayType const &ThisNodes,
                            pPropertiesType pProperties) const override
    {
      return Element::Pointer(new TwoStepUpdatedLagrangianVPImplicitFluidFicElement(NewId, BaseType::GetGeometry().Create(ThisNodes), pProperties));
    }

    Element::Pointer Clone(IndexType NewId, NodesArrayType const &ThisNodes) const override;

    ///@}
    ///@name Access
    ///@{

    ///@}
    ///@name Elemental Data
    ///@{

    /// Checks the input and that all required Kratos variables have been registered.
    /**
       * This function provides the place to perform checks on the completeness of the input.
       * It is designed to be called only once (or anyway, not often) typically at the beginning
       * of the calculations, so to verify that nothing is missing from the input
       * or that no common error is found.
       * @param rCurrentProcessInfo The ProcessInfo of the ModelPart that contains this element.
       * @return 0 if no errors were found.
       */
    int Check(const ProcessInfo &rCurrentProcessInfo) const override;

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
      buffer << "TwoStepUpdatedLagrangianVPImplicitFluidFicElement #" << BaseType::Id();
      return buffer.str();
    }

    /// Print information about this object.
    void PrintInfo(std::ostream &rOStream) const override
    {
      rOStream << "TwoStepUpdatedLagrangianVPImplicitFluidFicElement" << TDim << "D";
    }

    //        /// Print object's data.
    //        virtual void PrintData(std::ostream& rOStream) const;

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
       * A consistent mass matrix is used.
       * @param rMassMatrix The local matrix where the result will be added.
       * @param rN Elemental shape functions.
       * @param Weight Multiplication coefficient for the matrix, typically Density times integration point weight.
       */

    void ComputeBoundLHSMatrix(MatrixType &BoundLHSMatrix,
                               const ShapeFunctionsType &rN,
                               const double Weight) override;

    void ComputeBoundRHSVector(VectorType &BoundRHSVector,
                               const ShapeFunctionsType &rN,
                               const double TimeStep,
                               const double BoundRHSCoeffAcc,
                               const double BoundRHSCoeffDev) override;

    void ComputeBoundRHSVectorComplete(VectorType &BoundRHSVector,
                                       const double TimeStep,
                                       const double BoundRHSCoeffAcc,
                                       const double BoundRHSCoeffDev,
                                       const VectorType SpatialDefRate);

    void ComputeStabLaplacianMatrix(MatrixType &StabLaplacianMatrix,
                                    const ShapeFunctionDerivativesType &rShapeDeriv,
                                    const double Weight) override;

    void CalculateTauFIC(double &TauOne,
                         double ElemSize,
                         const double Density,
                         const double Viscosity,
                         const ProcessInfo &rCurrentProcessInfo) override;

    void AddStabilizationMatrixLHS(MatrixType &rLeftHandSideMatrix,
                                   Matrix &BulkAccMatrix,
                                   const ShapeFunctionsType &rN,
                                   const double Weight) override;

    void AddStabilizationNodalTermsLHS(MatrixType &rLeftHandSideMatrix,
                                       const double Tau,
                                       const double Weight,
                                       const ShapeFunctionDerivativesType &rDN_DX,
                                       const SizeType i) override;

    void AddStabilizationNodalTermsRHS(VectorType &rRightHandSideVector,
                                       const double Tau,
                                       const double Density,
                                       const double Weight,
                                       const ShapeFunctionDerivativesType &rDN_DX,
                                       const SizeType i) override;

    void CalculateLocalContinuityEqForPressure(MatrixType &rLeftHandSideMatrix,
                                               VectorType &rRightHandSideVector,
                                               const ProcessInfo &rCurrentProcessInfo) override;

    void GetPressureAccelerationValues(Vector &rValues,
                                       const int Step);

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

    void save(Serializer &rSerializer) const override
    {
      KRATOS_SERIALIZE_SAVE_BASE_CLASS(rSerializer, Element);
    }

    void load(Serializer &rSerializer) override
    {
      KRATOS_SERIALIZE_LOAD_BASE_CLASS(rSerializer, Element);
    }

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
    TwoStepUpdatedLagrangianVPImplicitFluidFicElement &operator=(TwoStepUpdatedLagrangianVPImplicitFluidFicElement const &rOther);

    /* /// Copy constructor. */
    /* TwoStepUpdatedLagrangianVPImplicitFluidFicElement(TwoStepUpdatedLagrangianVPImplicitFluidFicElement const& rOther); */

    ///@}

  }; // Class TwoStepUpdatedLagrangianVPImplicitFluidFicElement

  ///@}

  ///@name Type Definitions
  ///@{

  ///@}
  ///@name Input and output
  ///@{

  /// input stream function
  template <unsigned int TDim>
  inline std::istream &operator>>(std::istream &rIStream,
                                  TwoStepUpdatedLagrangianVPImplicitFluidFicElement<TDim> &rThis)
  {
    return rIStream;
  }

  /// output stream function
  template <unsigned int TDim>
  inline std::ostream &operator<<(std::ostream &rOStream,
                                  const TwoStepUpdatedLagrangianVPImplicitFluidFicElement<TDim> &rThis)
  {
    rThis.PrintInfo(rOStream);
    rOStream << std::endl;
    rThis.PrintData(rOStream);

    return rOStream;
  }

} // namespace Kratos.

#endif // KRATOS_TWO_STEP_UPDATED_LAGRANGIAN_V_P_FLUID_ELEMENT  defined

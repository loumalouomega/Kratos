// KRATOS  ___|  |                   |                   |
//       \___ \  __|  __| |   |  __| __| |   |  __| _` | |
//             | |   |    |   | (    |   |   | |   (   | |
//       _____/ \__|_|   \__,_|\___|\__|\__,_|_|  \__,_|_| MECHANICS
//
//  License:         BSD License
//                   license: StructuralMechanicsApplication/license.txt
//
//  Main authors:    Ruben Zorrilla
//                   Riccardo Rossi
//

#pragma once

// System includes

// External includes

// Project includes
#include "includes/define.h"
#include "includes/element.h"
#include "utilities/integration_utilities.h"

// Application includes
#include "structural_mechanics_application_variables.h"

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
 * @class SmallDisplacementMixedVolumetricStrainElement
 * @ingroup StructuralMechanicsApplication
 * @brief Small displacement with strain based mixed formulation element
 * @details This implements a small displacements element formulation with an extra volumetric strain nodal DOF
 * @author Ruben Zorrilla
 * @author Riccardo Rossi
 */
class KRATOS_API(STRUCTURAL_MECHANICS_APPLICATION) SmallDisplacementMixedVolumetricStrainElement
    : public Element
{

protected:

    /// The definition of the index type
    typedef std::size_t IndexType;

    /// The definition of the sizetype
    typedef std::size_t SizeType;

    /**
     * Internal variables used in the kinematic calculations
     */
    struct KinematicVariables
    {
        Vector N;
        Matrix B;
        double detF;
        Matrix F;
        double detJ0;
        Matrix J0;
        Matrix InvJ0;
        Matrix DN_DX;
        Vector Displacements;
        Vector VolumetricNodalStrains;
        Vector EquivalentStrain;

        /**
         * The default constructor
         * @param StrainSize The size of the strain vector in Voigt notation
         * @param Dimension The problem dimension: 2D or 3D
         * @param NumberOfNodes The number of nodes in the element
         */
        KinematicVariables(
            const SizeType StrainSize,
            const SizeType Dimension,
            const SizeType NumberOfNodes
            )
        {
            detF = 1.0;
            detJ0 = 1.0;
            N = ZeroVector(NumberOfNodes);
            B = ZeroMatrix(StrainSize, Dimension * NumberOfNodes);
            F = IdentityMatrix(Dimension);
            DN_DX = ZeroMatrix(NumberOfNodes, Dimension);
            J0 = ZeroMatrix(Dimension, Dimension);
            InvJ0 = ZeroMatrix(Dimension, Dimension);
            Displacements = ZeroVector(Dimension * NumberOfNodes);
            VolumetricNodalStrains = ZeroVector(NumberOfNodes);
            EquivalentStrain = ZeroVector(StrainSize);
        }
    };

    /**
     * Internal variables used in the kinematic calculations
     */
    struct ConstitutiveVariables
    {
        Vector StrainVector;
        Vector StressVector;
        Matrix D;

        /**
         * The default constructor
         * @param StrainSize The size of the strain vector in Voigt notation
         */
        ConstitutiveVariables(const SizeType StrainSize)
        {
            StrainVector = ZeroVector(StrainSize);
            StressVector = ZeroVector(StrainSize);
            D = ZeroMatrix(StrainSize, StrainSize);
        }
    };

    struct GaussPointAuxiliaryVariables
    {
        double Dt;
        double Rho;
        double Tau1;
        double Tau2;
        double Weight;
        double BulkModulus;

        Vector m; // Voigt identity
        Vector m_T; // Voigt identity x anisotropy tensor
        Vector invT_m; // Anisotropy tensor inverse x Voigt identity
        Vector BodyForce; // Body force value at current Gauss point
        Vector VolumetricStrainGradient; // Volumetric strain gradient at current Gauss point
        Vector DisplacementSubscaleAcceleration; // Displacement subscale acceleration term for dynamic subscale calculation

        GaussPointAuxiliaryVariables(
            const SmallDisplacementMixedVolumetricStrainElement* pElement,
            const SizeType Dimension,
            const SizeType StrainSize)
        {
            // Array allocation
            BodyForce.resize(Dimension);
            VolumetricStrainGradient.resize(Dimension);

            // Initialize the subscale acceleration to zero for the static case
            // Note that the zero value is only overwritten in the dynamic case
            DisplacementSubscaleAcceleration = ZeroVector(Dimension);

            // Define the identity tensor in Voigt notation
            m =  ZeroVector(StrainSize);
            for (IndexType d = 0; d < Dimension; ++d) {
                m[d] = 1.0;
            }

            // Calculate the anisotropy tensor products
            m_T = prod(m, pElement->mAnisotropyTensor);
            invT_m = prod(pElement->mInverseAnisotropyTensor, m);
        }
    };

public:

    ///@name Type Definitions
    ///@{

    ///Reference type definition for constitutive laws
    typedef ConstitutiveLaw ConstitutiveLawType;

    ///Pointer type for constitutive laws
    typedef ConstitutiveLawType::Pointer ConstitutiveLawPointerType;

    ///StressMeasure from constitutive laws
    typedef ConstitutiveLawType::StressMeasure StressMeasureType;

    ///Type definition for integration methods
    typedef GeometryData::IntegrationMethod IntegrationMethod;

    /// This is the definition of the node.
    typedef Node NodeType;

    /// The base element type
    typedef Element BaseType;

    // Counted pointer of SmallDisplacementMixedVolumetricStrainElement
    KRATOS_CLASS_INTRUSIVE_POINTER_DEFINITION( SmallDisplacementMixedVolumetricStrainElement );

    ///@}
    ///@name Life Cycle
    ///@{

    // Constructor void
    SmallDisplacementMixedVolumetricStrainElement()
    {};

    // Constructor using an array of nodes
    SmallDisplacementMixedVolumetricStrainElement(
        IndexType NewId,
        GeometryType::Pointer pGeometry
        ) : Element(
            NewId,
            pGeometry)
    {};

    // Constructor using an array of nodes with properties
    SmallDisplacementMixedVolumetricStrainElement(
        IndexType NewId,
        GeometryType::Pointer pGeometry,
        PropertiesType::Pointer pProperties
        ) : Element(
            NewId,
            pGeometry,
            pProperties)
    {};

    // Copy constructor
    SmallDisplacementMixedVolumetricStrainElement(SmallDisplacementMixedVolumetricStrainElement const& rOther)
        : BaseType(rOther),
          mThisIntegrationMethod(rOther.mThisIntegrationMethod),
          mConstitutiveLawVector(rOther.mConstitutiveLawVector)
    {};

    // Destructor
    ~SmallDisplacementMixedVolumetricStrainElement() override
    {};

    ///@}
    ///@name Operators
    ///@{

    ///@}
    ///@name Operations
    ///@{

    /**
     * @brief Called to initialize the element.
     * @warning Must be called before any calculation is done
     */
    void Initialize(const ProcessInfo &rCurrentProcessInfo) override;

    /**
     * @brief Called at the beginning of each solution step
     * @param rCurrentProcessInfo the current process info instance
     */
    void InitializeSolutionStep(const ProcessInfo& rCurrentProcessInfo) override;

    /**
     * @brief Called at the end of each solution step
     * @param rCurrentProcessInfo the current process info instance
     */
    void FinalizeSolutionStep(const ProcessInfo& rCurrentProcessInfo) override;

    /**
     * @brief It creates a new element pointer
     * @param NewId the ID of the new element
     * @param ThisNodes the nodes of the new element
     * @param pProperties the properties assigned to the new element
     * @return a Pointer to the new element
     */
    Element::Pointer Create(
        IndexType NewId,
        NodesArrayType const& ThisNodes,
        PropertiesType::Pointer pProperties) const override;

    /**
     * @brief It creates a new element pointer
     * @param NewId the ID of the new element
     * @param pGeom the geometry to be employed
     * @param pProperties the properties assigned to the new element
     * @return a Pointer to the new element
     */
    Element::Pointer Create(
        IndexType NewId,
        GeometryType::Pointer pGeom,
        PropertiesType::Pointer pProperties) const override;

    /**
     * @brief It creates a new element pointer and clones the previous element data
     * @param NewId the ID of the new element
     * @param ThisNodes the nodes of the new element
     * @param pProperties the properties assigned to the new element
     * @return a Pointer to the new element
     */
    Element::Pointer Clone(
        IndexType NewId,
        NodesArrayType const& rThisNodes) const override;

    /**
     * @brief Sets on rResult the ID's of the element degrees of freedom
     * @param rResult The vector containing the equation id
     * @param rCurrentProcessInfo The current process info instance
     */
    void EquationIdVector(
        EquationIdVectorType& rResult,
        const ProcessInfo& rCurrentProcessInfo) const override;

    /**
     * @brief Sets on rElementalDofList the degrees of freedom of the considered element geometry
     * @param rElementalDofList The vector containing the dof of the element
     * @param rCurrentProcessInfo The current process info instance
     */
    void GetDofList(
        DofsVectorType& rElementalDofList,
        const ProcessInfo& rCurrentProcessInfo) const override;

    /**
     * @brief Returns the used integration method
     * @return default integration method of the used Geometry
     */
    IntegrationMethod GetIntegrationMethod() const override
    {
        return mThisIntegrationMethod;
    }

    /**
     * @brief This function provides a more general interface to the element.
     * @details It is designed so that rLHSvariables and rRHSvariables are passed to the element thus telling what is the desired output
     * @param rLeftHandSideMatrix container with the output Left Hand Side matrix
     * @param rRightHandSideVector container for the desired RHS output
     * @param rCurrentProcessInfo the current process info instance
     */
    void CalculateLocalSystem(
        MatrixType& rLeftHandSideMatrix,
        VectorType& rRightHandSideVector,
        const ProcessInfo& rCurrentProcessInfo) override;

    /**
     * @brief This is called during the assembling process in order to calculate the elemental left hand side matrix only
     * @param rLeftHandSideMatrix the elemental left hand side matrix
     * @param rCurrentProcessInfo the current process info instance
     */
    void CalculateLeftHandSide(
        MatrixType& rLeftHandSideMatrix,
        const ProcessInfo& rCurrentProcessInfo) override;

    /**
      * @brief This is called during the assembling process in order to calculate the elemental right hand side vector only
      * @param rRightHandSideVector the elemental right hand side vector
      * @param rCurrentProcessInfo the current process info instance
      */
    void CalculateRightHandSide(
        VectorType& rRightHandSideVector,
        const ProcessInfo& rCurrentProcessInfo) override;

    /**
     * @brief This is called during the assembling process in order to calculate the elemental mass matrix
     * @param rMassMatrix The elemental mass matrix
     * @param rCurrentProcessInfo The current process info instance
     */
    void CalculateMassMatrix(
        MatrixType& rMassMatrix,
        const ProcessInfo& rCurrentProcessInfo) override;

    void GetSecondDerivativesVector(
        Vector& rValues,
        int Step = 0) const override;

    /**
     * @brief This function provides the place to perform checks on the completeness of the input.
     * @details It is designed to be called only once (or anyway, not often) typically at the beginning
     * of the calculations, so to verify that nothing is missing from the input
     * or that no common error is found.
     * @param rCurrentProcessInfo the current process info instance
     */
    int Check(const ProcessInfo& rCurrentProcessInfo) const override;

    /**
     * @brief Calculate a double Variable on the element Gauss points
     * @param rVariable The variable we want to get
     * @param rOutput The values obtained int the integration points
     * @param rCurrentProcessInfo the current process info instance
     */
    void CalculateOnIntegrationPoints(
        const Variable<double>& rVariable,
        std::vector<double>& rOutput,
        const ProcessInfo& rCurrentProcessInfo) override;

    /**
     * @brief Calculate a Vector Variable on the element Gauss points
     * @param rVariable The variable we want to get
     * @param rOutput The values obtained int the integration points
     * @param rCurrentProcessInfo the current process info instance
     */
    void CalculateOnIntegrationPoints(
        const Variable<Vector>& rVariable,
        std::vector<Vector>& rOutput,
        const ProcessInfo& rCurrentProcessInfo) override;

    /**
     * @brief Calculate an array Variable on the element Gauss points
     * @param rVariable The variable we want to get
     * @param rOutput The values obtained int the integration points
     * @param rCurrentProcessInfo the current process info instance
     */
    void CalculateOnIntegrationPoints(
        const Variable<array_1d<double,3>>& rVariable,
        std::vector<array_1d<double,3>>& rOutput,
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

    const Parameters GetSpecifications() const override;

    /// Turn back information as a string.
    std::string Info() const override
    {
        std::stringstream buffer;
        buffer << "Small Displacement Mixed Strain Element #" << Id() << "\nConstitutive law: " << mConstitutiveLawVector[0]->Info();
        return buffer.str();
    }

    /// Print information about this object.
    void PrintInfo(std::ostream& rOStream) const override
    {
        rOStream << "Small Displacement Mixed Strain Element #" << Id() << "\nConstitutive law: " << mConstitutiveLawVector[0]->Info();
    }

    /// Print object's data.
    void PrintData(std::ostream& rOStream) const override
    {
        pGetGeometry()->PrintData(rOStream);
    }

    ///@}
    ///@name Friends
    ///@{

protected:

    ///@name Protected static Member Variables
    ///@{

    ///@}
    ///@name Protected member Variables
    ///@{

    bool mIsDynamic; // Bool variable to indicate if the problem is dynamic

    std::vector<Vector> mDisplacementSubscale1; /// Gauss points displacement subscale at previous time step

    std::vector<Vector> mDisplacementSubscale2; /// Gauss points displacement subscale at two previous time step

    IntegrationMethod mThisIntegrationMethod; /// Currently selected integration methods

    std::vector<ConstitutiveLaw::Pointer> mConstitutiveLawVector; /// The vector containing the constitutive laws

    ///@}
    ///@name Protected Operators
    ///@{

    ///@}
    ///@name Protected Operations
    ///@{

    /**
     * @brief Sets the used integration method
     * @param ThisIntegrationMethod Integration method used
     */
    void SetIntegrationMethod(const IntegrationMethod& ThisIntegrationMethod)
    {
        mThisIntegrationMethod = ThisIntegrationMethod;
    }

    /**
     * @brief Sets the used constitutive laws
     * @param ThisConstitutiveLawVector Constitutive laws used
     */
    void SetConstitutiveLawVector(const std::vector<ConstitutiveLaw::Pointer>& ThisConstitutiveLawVector)
    {
        mConstitutiveLawVector = ThisConstitutiveLawVector;
    }

    /**
     * @brief It initializes the material
     */
    virtual void InitializeMaterial();

    /**
     * @brief This method returns if the element provides the strain
     */
    virtual bool UseElementProvidedStrain() const;

    /**
     * @brief Calculate the kinematics
     * @details This method calculates the kinematics of the element for a given integration point
     * @param rThisKinematicVariables Integration point kinematics container
     * @param PointNumber Integration point index
     * @param rIntegrationMethod Integration rule
     */
    void CalculateKinematicVariables(
        KinematicVariables &rThisKinematicVariables,
        const IndexType PointNumber,
        const GeometryType::IntegrationMethod &rIntegrationMethod) const;

    /**
     * @brief This functions updates the data structure passed to the CL
     * @param rThisKinematicVariables The kinematic variables to be calculated
     * @param rThisConstitutiveVariables The constitutive variables
     * @param rValues The CL parameters
     * @param PointNumber The integration point considered
     * @param IntegrationPoints The list of integration points
     */
    virtual void SetConstitutiveVariables(
        KinematicVariables& rThisKinematicVariables,
        ConstitutiveVariables& rThisConstitutiveVariables,
        ConstitutiveLaw::Parameters& rValues,
        const IndexType PointNumber,
        const GeometryType::IntegrationPointsArrayType& IntegrationPoints
        ) const;

    /**
     * @brief This functions updates the constitutive variables
     * @param rThisKinematicVariables The kinematic variables to be calculated
     * @param rThisConstitutiveVariables The constitutive variables
     * @param rValues The CL parameters
     * @param PointNumber The integration point considered
     * @param IntegrationPoints The list of integration points
     * @param ThisStressMeasure The stress measure considered
     */
    virtual void CalculateConstitutiveVariables(
        KinematicVariables& rThisKinematicVariables,
        ConstitutiveVariables& rThisConstitutiveVariables,
        ConstitutiveLaw::Parameters& rValues,
        const IndexType PointNumber,
        const GeometryType::IntegrationPointsArrayType& IntegrationPoints,
        const ConstitutiveLaw::StressMeasure ThisStressMeasure = ConstitutiveLaw::StressMeasure_PK2
        ) const;

    /**
     * @brief Calculates the Gauss point data
     * @param rThisGaussPointAuxiliaryVariables The Gauss point data container to be filled
     * @param rThisKinematicVariables The kinematic variables to be used to fill the data
     * @param rThisConstitutiveVariables The constitutive variables to be used to fill the data
     * @param rProcessInfo Reference to the current ProcessInfo container
     * @param PointNumber Gauss point index
     */
    void CalculateGaussPointAuxiliaryVariables(
        GaussPointAuxiliaryVariables &rThisGaussPointAuxiliaryVariables,
        const KinematicVariables &rThisKinematicVariables,
        const ConstitutiveVariables &rThisConstitutiveVariables,
        const ProcessInfo &rProcessInfo,
        const IndexType PointNumber) const;

    /**
     * @brief This function computes the body force
     * @param IntegrationPoints The array containing the integration points
     * @param PointNumber The id of the integration point considered
     * @return The vector of body forces
     */
    virtual Vector GetBodyForce(
        const GeometryType::IntegrationPointsArrayType& rIntegrationPoints,
        const IndexType PointNumber) const;

    /**
     * @brief Calculate the isotropic bulk modulus
     * Calculates the bulk modulus for the transformation to the closest isotropic tensor
     * @param rC Input constitutive matrx
     * @return double Isotropic bulk modulus
     */
    double CalculateBulkModulus(const Matrix &rC) const;

    /**
     * @brief Calculates the displacement subscale stabilization parameter
     * Calculates the first stabilization parameter corresponding to the displacement subscale
     * @param rVoigtIdAnysotropyMatrixProd Matrix containing the Voigt identity and the anisotropy tensor product
     * @param rThisKinematicVariables Reference to the filled kinematics container
     * @param rThisConstitutiveVariables Reference to the filled constitutive variables container
     * @param rCurrentProcessInfo Reference to the current ProcessInfo container
     * @return double The displacement subscale stabilization constant
     */
    double CalculateTau1(
        const Vector& rVoigtIdAnysotropyMatrixProd,
        const KinematicVariables& rThisKinematicVariables,
        const ConstitutiveVariables& rThisConstitutiveVariables,
        const ProcessInfo& rCurrentProcessInfo) const;

    /**
     * @brief Calculates the volumetric strain subscale stabilization parameter
     * Calculates the second stabilization parameter corresponding to the volumetric strain subscale
     * @param rThisConstitutiveVariables Reference to the filled constitutive variables container
     * @return double The volumetric strain subscale stabilization constant
     */
    double CalculateTau2(const ConstitutiveVariables& rThisConstitutiveVariables) const;

    /**
     * @brief Calculates the current Gauss point local system (LHS and RHS) contributions
     * @param rRightHandSideVector The elemental right hand side vector
     * @param rLeftHandSideMatrix The elemental left hand side matrix
     * @param rThisKinematicVariables Reference to the filled kinematics container
     * @param rThisConstitutiveVariables Reference to the filled constitutive variables container
     * @param rThisGaussPointAuxiliaryVariables Reference to the Gauss point data container
     */
    virtual void CalculateLocalSystemGaussPointContribution(
        VectorType& rRightHandSideVector,
        MatrixType& rLeftHandSideMatrix,
        const KinematicVariables& rThisKinematicVariables,
        const ConstitutiveVariables& rThisConstitutiveVariables,
        const GaussPointAuxiliaryVariables& rThisGaussPointAuxiliaryVariables) const;

    /**
     * @brief Calculates the current Gauss point RHS contribution
     * @param rRightHandSideVector The elemental right hand side vector
     * @param rThisKinematicVariables Reference to the filled kinematics container
     * @param rThisConstitutiveVariables Reference to the filled constitutive variables container
     * @param rThisGaussPointAuxiliaryVariables Reference to the Gauss point data container
     */
    virtual void CalculateRightHandSideGaussPointContribution(
        VectorType& rRightHandSideVector,
        const KinematicVariables& rThisKinematicVariables,
        const ConstitutiveVariables& rThisConstitutiveVariables,
        const GaussPointAuxiliaryVariables& rThisGaussPointAuxiliaryVariables) const;

    /**
     * @brief Calculates the current Gauss point LHS contribution
     * @param rLeftHandSideMatrix The elemental left hand side matrix
     * @param rThisKinematicVariables Reference to the filled kinematics container
     * @param rThisConstitutiveVariables Reference to the filled constitutive variables container
     * @param rThisGaussPointAuxiliaryVariables Reference to the Gauss point data container
     */
    virtual void CalculateLeftHandSideGaussPointContribution(
        MatrixType& rLeftHandSideMatrix,
        const KinematicVariables& rThisKinematicVariables,
        const ConstitutiveVariables& rThisConstitutiveVariables,
        const GaussPointAuxiliaryVariables& rThisGaussPointAuxiliaryVariables) const;

    /**
     * @brief Updates the subscale historical values for the dynamic case
     * @param rThisKinematicVariables Reference to the filled kinematics container
     * @param rThisConstitutiveVariables Reference to the filled constitutive variables container
     * @param rProcessInfo Reference to the current ProcessInfo container
     * @param PointIndex Gauss point index
     */
    virtual void UpdateGaussPointDisplacementSubscaleHistory(
        const KinematicVariables& rThisKinematicVariables,
        const ConstitutiveVariables& rThisConstitutiveVariables,
        const ProcessInfo& rProcessInfo,
        const IndexType PointIndex);

    ///@}
    ///@name Protected  Access
    ///@{

    /**
     * @brief Get the Anisotropy Tensor object
     * @return const Matrix& Reference to the anisotropy tensor
     */
    const Matrix& GetAnisotropyTensor() const
    {
        return mAnisotropyTensor;
    }

    ///@}
    ///@name Protected Inquiry
    ///@{

    /**
     * @brief Returns a bool variable indicating if the problem is dynamic
     * The variable is set in the Initialize function according to the presence of the ACCELERATION variable
     * @return true If the problem is dynamic
     * @return false If the problem is not dynamic (e.g., static or eigenvalue analysis)
     */
    bool IsDynamic() const
    {
        return mIsDynamic;
    }

    ///@}
    ///@name Protected LifeCycle
    ///@{

private:
    ///@name Static Member Variables
    ///@{

    ///@}
    ///@name Member Variables
    ///@{

    Matrix mAnisotropyTensor; // The anisotropy transformation tensor

    Matrix mInverseAnisotropyTensor; // The inverse of the anisotropy transformation tensor

    ///@}
    ///@name Private Operators
    ///@{

    ///@}
    ///@name Private Operations
    ///@{

    /**
     * @brief Calculation of the Deformation Matrix B
     * @param rB The deformation matrix
     * @param rDN_DX The derivatives of the shape functions
     * @param IntegrationPoints The array containing the integration points
     * @param PointNumber The integration point considered
     */
    void CalculateB(
        Matrix& rB,
        const Matrix& rDN_DX
        ) const;

    /**
     * @brief Calculate the equivalent strain
     * This function computes the equivalent strain vector.
     * The equivalent strain is defined as the deviatoric part of the displacement
     * symmetric gradient plus a volumetric strain coming from the interpolation
     * of the nodal volumetric strain.
     * @param rThisKinematicVariables Kinematic variables container
     */
    void CalculateEquivalentStrain(KinematicVariables& rThisKinematicVariables) const;

    /**
     * @brief Calculate the anisotropy tensor
     * This function calculates the anisotropy transformation tensor from the constitutive matrix
     */
    void CalculateAnisotropyTensor(const ProcessInfo &rCurrentProcessInfo);

    /**
     * @brief Calculate the inverse of the anisotropy tensor
     * This function calculates the inverse of the anisotropy transformation tensor
     * Note that we take advantage of the fact that the anisotropy tensor is diagonal
     */
    void CalculateInverseAnisotropyTensor();

    /**
     * @brief Calculate the isotropic shear modulus
     * Calculates the shear modulus for the transformation to the closest isotropic tensor
     * @param rC Input constitutive matrx
     * @return double Isotropic shear modulus
     */
    double CalculateShearModulus(const Matrix &rC) const;

    /**
     * @brief Calculation of the deformation gradient F
     * @param rF The deformation gradient
     * @param rStrainTensor The strain tensor in Voigt notation
     */
    void ComputeEquivalentF(
        Matrix& rF,
        const Vector& rStrainTensor
        ) const;

    /**
     * @brief This method gets a value directly in the CL
     * @details Avoids code repetition
     * @param rVariable The variable we want to get
     * @param rOutput The values obtained int the integration points
     * @tparam TType The type considered
     */
    template<class TType>
    void GetValueOnConstitutiveLaw(
        const Variable<TType>& rVariable,
        std::vector<TType>& rOutput)
    {
        const auto& r_geometry = GetGeometry();
        const SizeType n_gauss = r_geometry.IntegrationPointsNumber(GetIntegrationMethod());

        for (IndexType i_gauss = 0; i_gauss < n_gauss; ++i_gauss) {
            mConstitutiveLawVector[i_gauss]->GetValue(rVariable, rOutput[i_gauss]);
        }
    }

    /**
     * @brief This method computes directly in the CL
     * @details Avoids code repetition
     * @param rVariable The variable we want to get
     * @param rOutput The values obtained int the integration points
     * @param rCurrentProcessInfo the current process info instance
     * @tparam TType The type considered
     */
    template<class TType>
    void CalculateOnConstitutiveLaw(
        const Variable<TType>& rVariable,
        std::vector<TType>& rOutput,
        const ProcessInfo& rCurrentProcessInfo)
    {
        const auto& r_geometry = GetGeometry();
        const SizeType n_nodes = r_geometry.size();
        const SizeType dim = r_geometry.WorkingSpaceDimension();
        const SizeType strain_size = mConstitutiveLawVector[0]->GetStrainSize();
        const SizeType n_gauss = r_geometry.IntegrationPointsNumber(GetIntegrationMethod());
        const auto& r_integration_points = r_geometry.IntegrationPoints(GetIntegrationMethod());

        // Create the kinematics container and fill the nodal data
        KinematicVariables kinematic_variables(strain_size, dim, n_nodes);
        for (IndexType i_node = 0; i_node < n_nodes; ++i_node) {
            const auto& r_disp = r_geometry[i_node].FastGetSolutionStepValue(DISPLACEMENT);
            for (IndexType d = 0; d < dim; ++d) {
                kinematic_variables.Displacements(i_node * dim + d) = r_disp[d];
            }
            kinematic_variables.VolumetricNodalStrains[i_node] = r_geometry[i_node].FastGetSolutionStepValue(VOLUMETRIC_STRAIN);
        }

        // Create the constitutive variables and values containers
        ConstitutiveVariables constitutive_variables(strain_size);
        ConstitutiveLaw::Parameters cons_law_values(r_geometry, GetProperties(), rCurrentProcessInfo);
        auto& r_cons_law_options = cons_law_values.GetOptions();
        r_cons_law_options.Set(ConstitutiveLaw::COMPUTE_STRESS, true);
        r_cons_law_options.Set(ConstitutiveLaw::USE_ELEMENT_PROVIDED_STRAIN, true);
        r_cons_law_options.Set(ConstitutiveLaw::COMPUTE_CONSTITUTIVE_TENSOR, false);

        // Call the initialize material response
        for (IndexType i_gauss = 0; i_gauss < n_gauss; ++i_gauss) {
            // Recompute the kinematics
            CalculateKinematicVariables(kinematic_variables, i_gauss, GetIntegrationMethod());

            // Set the constitutive variables
            SetConstitutiveVariables(kinematic_variables, constitutive_variables, cons_law_values, i_gauss, r_integration_points);

            // Calculate the output value
            rOutput[i_gauss] = mConstitutiveLawVector[i_gauss]->CalculateValue(cons_law_values, rVariable, rOutput[i_gauss] );
        }
    }

    ///@}
    ///@name Private  Access
    ///@{

    ///@}
    ///@name Private Inquiry
    ///@{

    ///@}
    ///@name Serialization
    ///@{

    friend class Serializer;

    void save( Serializer& rSerializer ) const override;

    void load( Serializer& rSerializer ) override;

}; // class SmallDisplacementMixedVolumetricStrainElement.

///@}
///@name Type Definitions
///@{


///@}
///@name Input and output
///@{

} // namespace Kratos.

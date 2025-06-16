#pragma once

#include "includes/define.h"
#include "includes/element.h"
#include "includes/process_info.h"
#include "includes/geometry.h"
#include "includes/properties.h"
#include "includes/constitutive_law.h"
#include "structural_mechanics_application_variables.h" // For KinematicVariables, ConstitutiveVariables if not defined locally

namespace Kratos
{

// Forward declare structs if they are not included from elsewhere or defined here
struct KinematicVariables; // Assuming this will be defined properly, maybe move its definition here or include
struct ConstitutiveVariables; // Assuming this will be defined properly

class SolidElementUtilities
{
public:
    KRATOS_CLASS_POINTER_DEFINITION(SolidElementUtilities);

    typedef Element::IndexType IndexType;
    typedef Element::SizeType SizeType;
    typedef Element::MatrixType MatrixType;
    typedef Element::VectorType VectorType;
    typedef Element::EquationIdVectorType EquationIdVectorType;
    typedef Element::DofsVectorType DofsVectorType;
    typedef GeometryData::IntegrationMethod IntegrationMethod;
    typedef ConstitutiveLaw ConstitutiveLawType;


    // Structs moved from BaseSolidElement for broader use by utilities
    // These might need to be more generically defined or included if they are complex
    struct LocalKinematicVariables
    {
        Vector  N;
        Matrix  B;
        double  detF;
        Matrix  F;
        double  detJ0;
        Matrix  J0;
        Matrix  InvJ0;
        Matrix  DN_DX;
        Vector Displacements;

        LocalKinematicVariables(
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
        }
    };

    struct LocalConstitutiveVariables
    {
        ConstitutiveLaw::StrainVectorType StrainVector;
        ConstitutiveLaw::StressVectorType StressVector;
        ConstitutiveLaw::VoigtSizeMatrixType D;

        LocalConstitutiveVariables(const SizeType StrainSize)
        {
            if (StrainVector.size() != StrainSize) StrainVector.resize(StrainSize);
            if (StressVector.size() != StrainSize) StressVector.resize(StrainSize);
            if (D.size1() != StrainSize || D.size2() != StrainSize) D.resize(StrainSize, StrainSize);
            noalias(StrainVector) = ZeroVector(StrainSize);
            noalias(StressVector) = ZeroVector(StrainSize);
            noalias(D)            = ZeroMatrix(StrainSize, StrainSize);
        }
    };


    static double CalculateDerivativesOnReferenceConfiguration(
        Matrix& rJ0,
        Matrix& rInvJ0,
        Matrix& rDN_DX,
        const IndexType PointNumber,
        IntegrationMethod ThisIntegrationMethod,
        const Element::GeometryType& rGeom,
        bool UseGeometryIntegrationMethod // Added this parameter
        );

    static double GetIntegrationWeight(
        const Element::GeometryType::IntegrationPointsArrayType& rThisIntegrationPoints,
        const IndexType PointNumber,
        const double detJ,
        const SizeType Dimension,
        const Properties& rProperties // Added for THICKNESS
        );

    static void CalculateBStrainLinear(
        const Element::GeometryType& rGeom,
        const Matrix& rDN_DX,
        Matrix& rB
        );

    static void ComputeEquivalentFStrainLinear(
        const Element::GeometryType& rGeom,
        const Vector& rStrainTensor,
        Matrix& rF
        );

    static void SetSmallDisplacementConstitutiveVariables(
        const LocalKinematicVariables& rThisKinematicVariables, // Use Local version
        LocalConstitutiveVariables& rThisConstitutiveVariables, // Use Local version
        ConstitutiveLaw::Parameters& rValues,
        const Vector& rDisplacements // Pass displacements explicitly
        );

    static void CalculateMaterialStiffnessMatrix(
        MatrixType& rLeftHandSideMatrix,
        const Matrix& B,
        const Matrix& D,
        const double IntegrationWeight
        );

    static void CalculateAndAddInternalForces( // Renamed from CalculateAndAddResidualVector for clarity part 1
        VectorType& rRightHandSideVector,
        const Matrix& B,
        const Vector& rStressVector,
        const double IntegrationWeight
        );

    static void AddBodyForceContribution(
        const Vector& rN,
        const ProcessInfo& rCurrentProcessInfo,
        const array_1d<double, 3>& rBodyForce,
        VectorType& rRightHandSideVector,
        const double Weight,
        const SizeType Dimension,
        const SizeType NumberOfNodes
        );

    static void InitializeConstitutiveLaw(
        ConstitutiveLaw::Pointer& pConstitutiveLaw,
        const Properties& rProperties,
        const Element::GeometryType& rGeom,
        const Vector& rShapeFunctionValues
        );

    static void CalculateKinematicVariablesSmallDisplacement(
        LocalKinematicVariables& rThisKinematicVariables, // Use Local version
        const IndexType PointNumber,
        const IntegrationMethod& rIntegrationMethod,
        const Element::GeometryType& rGeom,
        const Vector& rDisplacements, // Added
        const SizeType StrainSize, // Added
        bool UseGeometryIntegrationMethod // Added
        );

    static double GetDensityForMassMatrixComputation(const Element& rElement);

    static int SolidElementCheck(
        const Element& rElement,
        const ProcessInfo& rCurrentProcessInfo,
        const std::vector<ConstitutiveLaw::Pointer>& rConstitutiveLawVector);

    static bool ComputeLumpedMassMatrix(
        const Properties& rProperties,
        const ProcessInfo& rCurrentProcessInfo);

    static bool HasRayleighDamping(
        const Properties& rProperties,
        const ProcessInfo& rCurrentProcessInfo);

    static void CalculateRayleighDampingMatrix(
        const Element& rElement,
        MatrixType& rDampingMatrix,
        const ProcessInfo& rCurrentProcessInfo,
        SizeType MatrixSize,
        const std::vector<ConstitutiveLaw::Pointer>& rConstitutiveLawVector, // Added
        IntegrationMethod ThisIntegrationMethod // Added
        );

    static array_1d<double, 3> GetBodyForce(
        const Element& rElement,
        const Element::GeometryType::IntegrationPointsArrayType& rIntegrationPoints,
        const IndexType PointNumber);

    static void CalculateLumpedMassVector(
        const Element& rElement,
        VectorType& rLumpedMassVector,
        const ProcessInfo& rCurrentProcessInfo
        );

    static bool IsElementRotated(
        const Element& rElement, // To access LOCAL_AXIS_X variables
        const ConstitutiveLaw::Pointer& pConstitutiveLaw // To get strain size
        );

    static void BuildRotationMatrix(
        BoundedMatrix<double, 3, 3>& rRotationMatrix,
        const array_1d<double, 3>& rLocalAxis1,
        const array_1d<double, 3>& rLocalAxis2, // Optional, for 3D
        const SizeType StrainSize);

    static void RotateVectorToLocalAxes(
        Vector& rVector,
        const BoundedMatrix<double, 3, 3>& rRotationMatrix,
        const SizeType StrainSize);

    static void RotateVectorToGlobalAxes(
        Vector& rVector,
        const BoundedMatrix<double, 3, 3>& rRotationMatrix,
        const SizeType StrainSize);

    static void RotateMatrixToLocalAxes(
        Matrix& rMatrix,
        const BoundedMatrix<double, 3, 3>& rRotationMatrix,
        const SizeType StrainSize);

    static void RotateMatrixToGlobalAxes(
        Matrix& rMatrix,
        const BoundedMatrix<double, 3, 3>& rRotationMatrix,
        const SizeType StrainSize);

    static void RotateFToLocalAxes(
        Matrix& rF,
        const BoundedMatrix<double, 3, 3>& rRotationMatrix);

    static void RotateFToGlobalAxes(
        Matrix& rF,
        const BoundedMatrix<double, 3, 3>& rRotationMatrix);

    // Forward declaration from base_solid_element.h
    static void CalculateConstitutiveVariables(
        LocalKinematicVariables& rThisKinematicVariables,
        LocalConstitutiveVariables& rThisConstitutiveVariables,
        ConstitutiveLaw::Parameters& rValues,
        const ConstitutiveLaw::Pointer& pConstitutiveLaw, // Pass CL directly
        const ConstitutiveLaw::StressMeasure ThisStressMeasure,
        const bool IsElementRotated, // To call rotation utilities
        const Element& rElementForRotation // To get local axes for rotation
        );

    static void CalculateKinematicVariablesTotalLagrangian(
        LocalKinematicVariables& rThisKinematicVariables,
        const IndexType PointNumber,
        const IntegrationMethod& rIntegrationMethod,
        const Element::GeometryType& rGeom,
        bool UseGeometryIntegrationMethod);

    static void CalculateBTotalLagrangian(
        Matrix& rB,
        const Matrix& rF, // Deformation Gradient
        const Matrix& rDN_DX, // Shape function derivatives in reference config
        const SizeType StrainSize, // To handle 2D/3D/Axisym
        const SizeType Dimension,
        const SizeType NumberOfNodes,
        const Vector& rN // Shape functions (for axisymmetric)
        );

    static void SetTotalLagrangianConstitutiveVariables( // Similar to SetSmallDisplacementConstitutiveVariables but might differ
        const LocalKinematicVariables& rThisKinematicVariables,
        LocalConstitutiveVariables& rThisConstitutiveVariables,
        ConstitutiveLaw::Parameters& rValues
        // No displacements needed here as strain is from F
        );

    // Add GetDefaultSolidSpecifications for use in elements
    static Parameters GetDefaultSolidSpecifications(SizeType Dimension);

    template<class TValueType>
    static void GetValueFromConstitutiveLaw(
        const std::vector<ConstitutiveLaw::Pointer>& rConstitutiveLawVector, // Pass CL vector directly
        const Variable<TValueType>& rVariable,
        std::vector<TValueType>& rOutput // Output directly
        )
    {
        KRATOS_TRY
        KRATOS_ERROR_IF(rConstitutiveLawVector.empty()) << "ConstitutiveLawVector is empty." << std::endl;
        if (rOutput.size() != rConstitutiveLawVector.size()) {
            rOutput.resize(rConstitutiveLawVector.size());
        }
        for (IndexType i = 0; i < rConstitutiveLawVector.size(); ++i) {
            // Ensure TValueType is default constructible if GetValue requires it for some types
            // For basic types like double, int, bool, array_1d, Vector, Matrix, this should be fine.
            rConstitutiveLawVector[i]->GetValue(rVariable, rOutput[i]);
        }
        KRATOS_CATCH("")
    }

    // Declaration for CalculateValueOnConstitutiveLaw (similar to GetValue but calls CalculateValue)
    template<class TValueType>
    static void CalculateValueOnConstitutiveLaw(
        const Element& rElement, // For CL params, geometry, properties
        const std::vector<ConstitutiveLaw::Pointer>& rConstitutiveLawVector,
        const Variable<TValueType>& rVariable,
        const ProcessInfo& rCurrentProcessInfo,
        const IntegrationMethod& rIntegrationMethod,
        bool UseElementProvidedStrain,
        ConstitutiveLaw::StressMeasure rStressMeasure,
        std::vector<TValueType>& rOutput // Output directly
        );


    static void CalculateKinematicVariablesAxisymSmallDisplacement(
        LocalKinematicVariables& rThisKinematicVariables, // Will use strain_size = 4
        const IndexType PointNumber,
        const IntegrationMethod& rIntegrationMethod,
        const Element::GeometryType& rGeom,
        const Vector& rDisplacements, // Current displacements
        bool UseGeometryIntegrationMethod);

    static void CalculateBAxisymSmallDisplacement(
        Matrix& rB, // Output B matrix (4x(N*2))
        const Matrix& rDN_DX, // Derivatives w.r.t. R, Z (N_nodes x 2)
        const Vector& rN,     // Shape functions (N_nodes)
        const double CurrentRadius, // Current radius at integration point
        const SizeType NumberOfNodes);

    static void CalculateKinematicVariablesAxisymTotalLagrangian(
        LocalKinematicVariables& rThisKinematicVariables, // Strain size 4. F will be 3x3 effectively for CL, or 2x2 + hoop.
        const IndexType PointNumber,
        const IntegrationMethod& rIntegrationMethod,
        const Element::GeometryType& rGeom,
        bool UseGeometryIntegrationMethod);

    static void CalculateBAxisymTotalLagrangian( // B for Green-Lagrange strain
        Matrix& rB, // Output B matrix (4x(N*2))
        const Matrix& rF, // Deformation Gradient (either 2x2 RZ-plane or effective 3x3)
        const Matrix& rDN_DX, // Derivatives w.r.t. initial R, Z (N_nodes x 2)
        const Vector& rN,     // Shape functions (N_nodes)
        const double InitialRadius, // Initial radius at integration point
        const SizeType NumberOfNodes);
}; // End of SolidElementUtilities class
}; // namespace Kratos

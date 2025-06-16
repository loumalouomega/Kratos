#pragma once
#include "includes/element.h"
#include "custom_utilities/solid_elements_utilities.h"
#include "includes/variables.h" // Required for ACCELERATION, VELOCITY, DISPLACEMENT etc.
#include "includes/constitutive_law.h" // Required for ConstitutiveLaw
#include "includes/process_info.h" // Required for ProcessInfo
#include "includes/properties.h" // Required for Properties
#include "geometries/geometry.h" // Required for Geometry
#include "structural_mechanics_application_variables.h" // For application specific variables if any


namespace Kratos {
class KRATOS_API(STRUCTURAL_MECHANICS_APPLICATION) UpdatedLagrangian : public Element {
public:
    KRATOS_CLASS_INTRUSIVE_POINTER_DEFINITION(UpdatedLagrangian);
    typedef Element BaseType;
    typedef BaseType::IndexType IndexType;
    typedef BaseType::SizeType SizeType;
    typedef BaseType::MatrixType MatrixType;
    typedef BaseType::VectorType VectorType;
    typedef BaseType::EquationIdVectorType EquationIdVectorType;
    typedef BaseType::DofsVectorType DofsVectorType;
    typedef GeometryData::IntegrationMethod IntegrationMethod;
    typedef ConstitutiveLaw ConstitutiveLawType;
    typedef ConstitutiveLawType::Pointer ConstitutiveLawPointerType;

    UpdatedLagrangian(IndexType NewId, GeometryType::Pointer pGeometry);
    UpdatedLagrangian(IndexType NewId, GeometryType::Pointer pGeometry, PropertiesType::Pointer pProperties);
    ~UpdatedLagrangian() override;
    Element::Pointer Create(IndexType NewId, NodesArrayType const& ThisNodes, PropertiesType::Pointer pProperties) const override;
    Element::Pointer Create(IndexType NewId, GeometryType::Pointer pGeom, PropertiesType::Pointer pProperties) const override;
    Element::Pointer Clone(IndexType NewId, NodesArrayType const& rThisNodes) const override;

    void Initialize(const ProcessInfo& rCurrentProcessInfo) override;
    void ResetConstitutiveLaw() override;
    void InitializeSolutionStep(const ProcessInfo& rCurrentProcessInfo) override;
    void FinalizeSolutionStep(const ProcessInfo& rCurrentProcessInfo) override;
    void InitializeNonLinearIteration(const ProcessInfo& rCurrentProcessInfo) override;
    void FinalizeNonLinearIteration(const ProcessInfo& rCurrentProcessInfo) override;

    void CalculateLocalSystem(MatrixType& rLeftHandSideMatrix, VectorType& rRightHandSideVector, const ProcessInfo& rCurrentProcessInfo) override;
    void CalculateLeftHandSide(MatrixType& rLeftHandSideMatrix, const ProcessInfo& rCurrentProcessInfo) override;
    void CalculateRightHandSide(VectorType& rRightHandSideVector, const ProcessInfo& rCurrentProcessInfo) override;
    void CalculateMassMatrix(MatrixType& rMassMatrix, const ProcessInfo& rCurrentProcessInfo) override;
    void CalculateDampingMatrix(MatrixType& rDampingMatrix, const ProcessInfo& rCurrentProcessInfo) override;

    void EquationIdVector(EquationIdVectorType& rResult, const ProcessInfo& rCurrentProcessInfo) const override;
    void GetDofList(DofsVectorType& rElementalDofList, const ProcessInfo& rCurrentProcessInfo) const override;

    void GetValuesVector(Vector& rValues, int Step = 0) const override;
    void GetFirstDerivativesVector(Vector& rValues, int Step = 0) const override;
    void GetSecondDerivativesVector(Vector& rValues, int Step = 0) const override;

    void AddExplicitContribution(const VectorType& rRHSVector, const Variable<VectorType>& rRHSVariable, const Variable<double>& rDestinationVariable, const ProcessInfo& rCurrentProcessInfo) override;
    void AddExplicitContribution(const VectorType& rRHSVector, const Variable<VectorType>& rRHSVariable, const Variable<array_1d<double,3>>& rDestinationVariable, const ProcessInfo& rCurrentProcessInfo) override;

    int Check(const ProcessInfo& rCurrentProcessInfo) const override;
    const Parameters GetSpecifications() const override;

    void CalculateOnIntegrationPoints(const Variable<double>& rVariable, std::vector<double>& rOutput, const ProcessInfo& rCurrentProcessInfo) override;
    void CalculateOnIntegrationPoints(const Variable<Vector>& rVariable, std::vector<Vector>& rOutput, const ProcessInfo& rCurrentProcessInfo) override;
    void CalculateOnIntegrationPoints(const Variable<Matrix>& rVariable, std::vector<Matrix>& rOutput, const ProcessInfo& rCurrentProcessInfo) override;
    void CalculateOnIntegrationPoints(const Variable<array_1d<double,3>>& rVariable, std::vector<array_1d<double,3>>& rOutput, const ProcessInfo& rCurrentProcessInfo) override;

    void SetValuesOnIntegrationPoints(const Variable<double>& rVariable, const std::vector<double>& rValues, const ProcessInfo& rCurrentProcessInfo) override;
    void SetValuesOnIntegrationPoints(const Variable<Vector>& rVariable, const std::vector<Vector>& rValues, const ProcessInfo& rCurrentProcessInfo) override;
    void SetValuesOnIntegrationPoints(const Variable<Matrix>& rVariable, const std::vector<Matrix>& rValues, const ProcessInfo& rCurrentProcessInfo) override;
    void SetValuesOnIntegrationPoints(const Variable<array_1d<double,3>>& rVariable, const std::vector<array_1d<double,3>>& rValues, const ProcessInfo& rCurrentProcessInfo) override;
    void SetValuesOnIntegrationPoints(const Variable<ConstitutiveLaw::Pointer>& rVariable, const std::vector<ConstitutiveLaw::Pointer>& rValues, const ProcessInfo& rCurrentProcessInfo) override;
    void SetValuesOnIntegrationPoints(const Variable<bool>& rVariable, const std::vector<bool>& rValues, const ProcessInfo& rCurrentProcessInfo) override;
    void SetValuesOnIntegrationPoints(const Variable<int>& rVariable, const std::vector<int>& rValues, const ProcessInfo& rCurrentProcessInfo) override;
    void SetValuesOnIntegrationPoints(const Variable<array_1d<double,6>>& rVariable, const std::vector<array_1d<double,6>>& rValues, const ProcessInfo& rCurrentProcessInfo) override;

    std::string Info() const override;
    void PrintInfo(std::ostream& rOStream) const override;
    void PrintData(std::ostream& rOStream) const override;


protected:
    IntegrationMethod mThisIntegrationMethod;
    std::vector<ConstitutiveLaw::Pointer> mConstitutiveLawVector;

    // Variables for UL specific data (e.g. historical F, if needed for rate-dependent CLs)
    // std::vector<Matrix> mF0_history; // Example: if CL needs F_n for F_n+1 = df * F_n

    ConstitutiveLaw::StressMeasure GetStressMeasure() const;
    bool UseElementProvidedStrain() const;

private:
    friend class Serializer;
    void save(Serializer& rSerializer) const override;
    void load(Serializer& rSerializer) override;
    UpdatedLagrangian() : Element() {}
};
} // namespace Kratos

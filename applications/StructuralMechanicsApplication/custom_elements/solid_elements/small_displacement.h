#pragma once

#include "includes/define.h"
#include "includes/element.h" // Inherit from Element
#include "includes/variables.h"
#include "includes/constitutive_law.h" // For mConstitutiveLawVector
#include "custom_utilities/solid_elements_utilities.h" // Include the new utilities

namespace Kratos
{

class KRATOS_API(STRUCTURAL_MECHANICS_APPLICATION) SmallDisplacement
    : public Element // Changed inheritance
{
public:
    KRATOS_CLASS_INTRUSIVE_POINTER_DEFINITION(SmallDisplacement);

    typedef Element BaseType; // BaseType is now Element
    typedef BaseType::IndexType IndexType;
    typedef BaseType::SizeType SizeType;
    typedef BaseType::MatrixType MatrixType;
    typedef BaseType::VectorType VectorType;
    typedef BaseType::EquationIdVectorType EquationIdVectorType;
    typedef BaseType::DofsVectorType DofsVectorType;
    typedef BaseType::IntegrationMethod IntegrationMethod; // Keep this typedef
    typedef ConstitutiveLaw ConstitutiveLawType;
    typedef ConstitutiveLawType::Pointer ConstitutiveLawPointerType;


    SmallDisplacement(IndexType NewId, GeometryType::Pointer pGeometry);
    SmallDisplacement(IndexType NewId, GeometryType::Pointer pGeometry, PropertiesType::Pointer pProperties);
    ~SmallDisplacement() override;

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

    // CalculateOnIntegrationPoints (abbreviated for brevity in this draft - will need full list)
    void CalculateOnIntegrationPoints(const Variable<double>& rVariable, std::vector<double>& rOutput, const ProcessInfo& rCurrentProcessInfo) override;
    void CalculateOnIntegrationPoints(const Variable<Vector>& rVariable, std::vector<Vector>& rOutput, const ProcessInfo& rCurrentProcessInfo) override;
    void CalculateOnIntegrationPoints(const Variable<Matrix>& rVariable, std::vector<Matrix>& rOutput, const ProcessInfo& rCurrentProcessInfo) override;
    void CalculateOnIntegrationPoints(const Variable<array_1d<double,3>>& rVariable, std::vector<array_1d<double,3>>& rOutput, const ProcessInfo& rCurrentProcessInfo) override;
    // ... other CalculateOnIntegrationPoints and SetValuesOnIntegrationPoints if needed
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
    const Parameters GetSpecifications() const override;

protected:
    // SmallDisplacement() : Element() {} // Protected default constructor for serialization if needed by KRATOS_SERIALIZE_LOAD_BASE_CLASS

    // Members copied from BaseSolidElement
    IntegrationMethod mThisIntegrationMethod;
    std::vector<ConstitutiveLaw::Pointer> mConstitutiveLawVector;

    // Methods that were virtual in BaseSolidElement, now need to be implemented or are specific
    virtual bool UseElementProvidedStrain() const;
    // This was virtual in BaseSolidElement, SmallDisplacement overrides it.
    // If it's always true for SmallDisplacement, it can be hardcoded or made non-virtual.

    // The following are no longer needed as protected virtuals; their logic is in CalculateAll or utilities
    // void CalculateAll(...) // This is now the public CalculateLocalSystem etc.
    // void CalculateKinematicVariables(...) // Logic moved
    // void SetConstitutiveVariables(...) // Logic moved
    // void CalculateB(...) // Logic moved
    // void ComputeEquivalentF(...) // Logic moved

    // Copied from BaseSolidElement for direct use, or to be replaced by utility calls
    ConstitutiveLaw::StressMeasure GetStressMeasure() const;


private:
    friend class Serializer;
    void save(Serializer& rSerializer) const override;
    void load(Serializer& rSerializer) override;

    SmallDisplacement() : Element() {} // Private default constructor for serialization

}; // Class SmallDisplacement

} // namespace Kratos

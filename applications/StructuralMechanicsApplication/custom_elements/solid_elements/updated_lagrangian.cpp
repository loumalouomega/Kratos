#include "updated_lagrangian.h"
#include "utilities/math_utils.h"
#include "custom_utilities/constitutive_law_utilities.h"
#include "structural_mechanics_application_variables.h"
#include "utilities/atomic_utilities.h"


namespace Kratos {
UpdatedLagrangian::UpdatedLagrangian(IndexType NewId, GeometryType::Pointer pGeometry) : Element(NewId, pGeometry) { mThisIntegrationMethod = GetGeometry().GetDefaultIntegrationMethod(); }
UpdatedLagrangian::UpdatedLagrangian(IndexType NewId, GeometryType::Pointer pGeometry, PropertiesType::Pointer pProperties) : Element(NewId, pGeometry, pProperties) { mThisIntegrationMethod = GetGeometry().GetDefaultIntegrationMethod(); }
UpdatedLagrangian::~UpdatedLagrangian() {}

Element::Pointer UpdatedLagrangian::Create(IndexType NewId, NodesArrayType const& ThisNodes, PropertiesType::Pointer pProperties) const {
    return Kratos::make_intrusive<UpdatedLagrangian>(NewId, GetGeometry().Create(ThisNodes), pProperties);
}
Element::Pointer UpdatedLagrangian::Create(IndexType NewId, GeometryType::Pointer pGeom, PropertiesType::Pointer pProperties) const {
    return Kratos::make_intrusive<UpdatedLagrangian>(NewId, pGeom, pProperties);
}

Element::Pointer UpdatedLagrangian::Clone(IndexType NewId, NodesArrayType const& rThisNodes) const {
    KRATOS_TRY
    auto p_new_elem = Kratos::make_intrusive<UpdatedLagrangian>(NewId, GetGeometry().Create(rThisNodes), pGetProperties());
    p_new_elem->SetData(this->GetData());
    p_new_elem->Set(Flags(*this));
    p_new_elem->mThisIntegrationMethod = this->mThisIntegrationMethod;
    p_new_elem->mConstitutiveLawVector.resize(this->mConstitutiveLawVector.size());
    for(size_t i=0; i<this->mConstitutiveLawVector.size(); ++i) {
        if(this->mConstitutiveLawVector[i] != nullptr)
            p_new_elem->mConstitutiveLawVector[i] = this->mConstitutiveLawVector[i]->Clone();
    }
    // Copy any UL-specific history data if needed
    return p_new_elem;
    KRATOS_CATCH("");
}
void UpdatedLagrangian::Initialize(const ProcessInfo& rCurrentProcessInfo) {
    KRATOS_TRY
    if (!rCurrentProcessInfo[IS_RESTARTED]) {
        if( GetProperties().Has(INTEGRATION_ORDER) ) {
            const SizeType integration_order = GetProperties()[INTEGRATION_ORDER];
            switch ( integration_order ) {
                case 1: mThisIntegrationMethod = GeometryData::IntegrationMethod::GI_GAUSS_1; break;
                case 2: mThisIntegrationMethod = GeometryData::IntegrationMethod::GI_GAUSS_2; break;
                case 3: mThisIntegrationMethod = GeometryData::IntegrationMethod::GI_GAUSS_3; break;
                case 4: mThisIntegrationMethod = GeometryData::IntegrationMethod::GI_GAUSS_4; break;
                case 5: mThisIntegrationMethod = GeometryData::IntegrationMethod::GI_GAUSS_5; break;
                default:
                    KRATOS_WARNING("UpdatedLagrangian") << "Integration order " << integration_order << " for element " << Id() << " not available, using default." << std::endl;
                    mThisIntegrationMethod = GetGeometry().GetDefaultIntegrationMethod();
            }
        } else {
            mThisIntegrationMethod = GetGeometry().GetDefaultIntegrationMethod();
        }
        const auto& integration_points = GetGeometry().IntegrationPoints(mThisIntegrationMethod);
        if ( mConstitutiveLawVector.size() != integration_points.size() ) {
            mConstitutiveLawVector.resize(integration_points.size());
        }
        const auto& N_values = GetGeometry().ShapeFunctionsValues(mThisIntegrationMethod);
        for ( IndexType i = 0; i < mConstitutiveLawVector.size(); ++i ) {
            SolidElementUtilities::InitializeConstitutiveLaw(mConstitutiveLawVector[i], GetProperties(), GetGeometry(), row(N_values, i));
        }
        // Initialize UL specific history data if needed (e.g. F0_history)
    }
    KRATOS_CATCH( "" )
}

ConstitutiveLaw::StressMeasure UpdatedLagrangian::GetStressMeasure() const { return ConstitutiveLaw::StressMeasure_Cauchy; }
bool UpdatedLagrangian::UseElementProvidedStrain() const { return true; } // Strain rate D is provided

void UpdatedLagrangian::CalculateLocalSystem(MatrixType& rLeftHandSideMatrix, VectorType& rRightHandSideVector, const ProcessInfo& rCurrentProcessInfo) {
    KRATOS_TRY
    auto& r_geom = GetGeometry();
    const SizeType num_nodes = r_geom.size();
    const SizeType dim = r_geom.WorkingSpaceDimension();
    KRATOS_ERROR_IF(mConstitutiveLawVector.empty() || !mConstitutiveLawVector[0]) << "CL not init! Elm ID: " << Id() << std::endl;
    const SizeType strain_size = mConstitutiveLawVector[0]->GetStrainSize();

    SolidElementUtilities::LocalKinematicVariables kin_vars(strain_size, dim, num_nodes);
    SolidElementUtilities::LocalConstitutiveVariables const_vars(strain_size);

    // Get current nodal velocities for strain rate calculation
    GetFirstDerivativesVector(kin_vars.Displacements); // Store velocities in kin_vars.Displacements

    const SizeType mat_size = num_nodes * dim;
    if (rLeftHandSideMatrix.size1() != mat_size) rLeftHandSideMatrix.resize(mat_size, mat_size, false);
    noalias(rLeftHandSideMatrix) = ZeroMatrix(mat_size, mat_size);
    if (rRightHandSideVector.size() != mat_size) rRightHandSideVector.resize(mat_size, false);
    noalias(rRightHandSideVector) = ZeroVector(mat_size);

    const auto& integration_points = r_geom.IntegrationPoints(mThisIntegrationMethod);
    ConstitutiveLaw::Parameters cl_params(r_geom, GetProperties(), rCurrentProcessInfo);
    Flags& cl_options = cl_params.GetOptions();
    cl_options.Set(ConstitutiveLaw::USE_ELEMENT_PROVIDED_STRAIN, UseElementProvidedStrain()); // True for UL (strain rate)
    cl_options.Set(ConstitutiveLaw::COMPUTE_STRESS, true);
    cl_options.Set(ConstitutiveLaw::COMPUTE_CONSTITUTIVE_TENSOR, true); // Tangent for LHS

    cl_params.SetStrainVector(const_vars.StrainVector); // CL reads strain rate (D) from here
    cl_params.SetStressVector(const_vars.StressVector); // CL writes Cauchy stress sigma_n+1 here
    cl_params.SetConstitutiveMatrix(const_vars.D);      // CL writes C_tangent here

    const Matrix& N_container = r_geom.ShapeFunctionsValues(mThisIntegrationMethod);

    for (IndexType i = 0; i < integration_points.size(); ++i) {
        SolidElementUtilities::CalculateKinematicVariablesUpdatedLagrangian(
            kin_vars, i, mThisIntegrationMethod, r_geom, rCurrentProcessInfo, true);
        // kin_vars.DN_DX is dN/dx (current config), kin_vars.B is for strain rate D, kin_vars.F is df (or L*dt)

        // Strain rate D = B_UL * v_nodes
        noalias(const_vars.StrainVector) = prod(kin_vars.B, kin_vars.Displacements); // Displacements here are velocities
        cl_params.SetStrainVector(const_vars.StrainVector); // Set strain rate for CL

        cl_params.SetShapeFunctionsValues(kin_vars.N);
        cl_params.SetDeformationGradientF(kin_vars.F); // Set incremental deformation gradient df (or L*dt)
        cl_params.SetDeterminantF(kin_vars.detF);      // Set det(df)

        // TODO: Handle F0 history for CLs that need F_total = df * F_n
        // cl_params.SetDeformationGradientF(mF0_history[i]); // Pass F_n if CL uses it
        // Then after CL call: mF0_history[i] = prod(kin_vars.F, mF0_history[i]); // Update F_n+1 = df * F_n

        bool is_rotated = SolidElementUtilities::IsElementRotated(*this, mConstitutiveLawVector[i]);
        // For UL, rotation is typically handled by objective stress rates within CL.
        // If CL expects pre-rotated inputs (strain rate D), then rotation utilities would be used here.
        // Assuming CL handles objectivity.

        mConstitutiveLawVector[i]->CalculateMaterialResponse(cl_params, GetStressMeasure());
        // const_vars.StressVector (Cauchy sigma_n+1) and const_vars.D (tangent C) are now populated

        double integration_weight = SolidElementUtilities::GetIntegrationWeight(integration_points, i, kin_vars.detJ0, dim, GetProperties()); // detJ0 here is detJ_current

        // LHS: Material part (B_UL^T * C_tangent * B_UL * w)
        SolidElementUtilities::CalculateMaterialStiffnessMatrix(rLeftHandSideMatrix, kin_vars.B, const_vars.D, integration_weight);

        // LHS: Geometric part (Kg_UL * w) using current Cauchy stress sigma_n+1
        Matrix Kg_gauss(mat_size, mat_size);
        // DN_DX for Kg is dN/dx (current config), Stress is Cauchy sigma_n+1
        StructuralMechanicsElementUtilities::CalculateKgMatrix(Kg_gauss, kin_vars.DN_DX, const_vars.StressVector, integration_weight, dim, num_nodes);
        rLeftHandSideMatrix += Kg_gauss;

        // RHS
        array_1d<double, 3> body_force = SolidElementUtilities::GetBodyForce(*this, integration_points, i);
        SolidElementUtilities::AddBodyForceContribution(kin_vars.N, rCurrentProcessInfo, body_force, rRightHandSideVector, integration_weight, dim, num_nodes);
        SolidElementUtilities::CalculateAndAddInternalForces(rRightHandSideVector, kin_vars.B, const_vars.StressVector, integration_weight);
    }
    KRATOS_CATCH("")
}

// Implement other stubs: LHS, RHS, Mass, Damping, EqID, Dofs, GetValues, Check, Specs, COIP, Info, save/load
// These will be very similar to TotalLagrangian/SmallDisplacement, adapted for UL where necessary.
void UpdatedLagrangian::ResetConstitutiveLaw() { /* Similar to TotalLagrangian */ }
void UpdatedLagrangian::InitializeSolutionStep( const ProcessInfo& rCurrentProcessInfo ) { /* Similar to TotalLagrangian, manage F0_history if needed */ }
void UpdatedLagrangian::FinalizeSolutionStep( const ProcessInfo& rCurrentProcessInfo ) { /* Similar to TotalLagrangian, manage F0_history if needed */ }
void UpdatedLagrangian::InitializeNonLinearIteration( const ProcessInfo& rCurrentProcessInfo ) { /* Usually empty */ }
void UpdatedLagrangian::FinalizeNonLinearIteration( const ProcessInfo& rCurrentProcessInfo ) { /* Usually empty */ }

void UpdatedLagrangian::CalculateLeftHandSide(MatrixType& rLeftHandSideMatrix, const ProcessInfo& rCurrentProcessInfo) { VectorType temp_rhs; CalculateLocalSystem(rLeftHandSideMatrix, temp_rhs, rCurrentProcessInfo); }
void UpdatedLagrangian::CalculateRightHandSide(VectorType& rRightHandSideVector, const ProcessInfo& rCurrentProcessInfo) { MatrixType temp_lhs; CalculateLocalSystem(temp_lhs, rRightHandSideVector, rCurrentProcessInfo); }
void UpdatedLagrangian::CalculateMassMatrix(MatrixType& rMassMatrix, const ProcessInfo& rCurrentProcessInfo) { /* Adapt from TotalLagrangian, use current volume for consistent mass */ }
void UpdatedLagrangian::CalculateDampingMatrix(MatrixType& rDampingMatrix, const ProcessInfo& rCurrentProcessInfo) { /* Adapt from TotalLagrangian */ }

void UpdatedLagrangian::EquationIdVector(EquationIdVectorType& rResult, const ProcessInfo& rCurrentProcessInfo) const { /* Same as TotalLagrangian/SmallDisplacement */ }
void UpdatedLagrangian::GetDofList(DofsVectorType& rElementalDofList, const ProcessInfo& rCurrentProcessInfo) const { /* Same as TotalLagrangian/SmallDisplacement */ }
void UpdatedLagrangian::GetValuesVector(Vector& rValues, int Step) const { /* Same as TotalLagrangian/SmallDisplacement */ }
void UpdatedLagrangian::GetFirstDerivativesVector(Vector& rValues, int Step) const { /* Same as TotalLagrangian/SmallDisplacement */ }
void UpdatedLagrangian::GetSecondDerivativesVector(Vector& rValues, int Step) const { /* Same as TotalLagrangian/SmallDisplacement */ }

void UpdatedLagrangian::AddExplicitContribution(const VectorType& rRHSVector, const Variable<VectorType>& rRHSVariable, const Variable<double>& rDestinationVariable, const ProcessInfo& rCurrentProcessInfo) { /* Same as TotalLagrangian/SmallDisplacement */ }
void UpdatedLagrangian::AddExplicitContribution(const VectorType& rRHSVector, const Variable<VectorType>& rRHSVariable, const Variable<array_1d<double,3>>& rDestinationVariable, const ProcessInfo& rCurrentProcessInfo) { /* Same as TotalLagrangian/SmallDisplacement */ }

int UpdatedLagrangian::Check(const ProcessInfo& rCurrentProcessInfo) const {
    KRATOS_TRY
    int check = Element::Check(rCurrentProcessInfo);
    check = SolidElementUtilities::SolidElementCheck(*this, rCurrentProcessInfo, mConstitutiveLawVector);
    // Add UL specific checks if any (e.g., det(df) > 0 if df is computed and used)
    return check;
    KRATOS_CATCH("")
}
const Parameters UpdatedLagrangian::GetSpecifications() const {
    Parameters specs = SolidElementUtilities::GetDefaultSolidSpecifications(GetGeometry().WorkingSpaceDimension());
    specs["framework"].SetString("updated_lagrangian"); // Or Eulerian if CL handles rates in spatial frame
    return specs;
}

void UpdatedLagrangian::CalculateOnIntegrationPoints(const Variable<double>& rVariable, std::vector<double>& rOutput, const ProcessInfo& rCurrentProcessInfo) { /* Adapt from TotalLagrangian, use UL kinematics */ }
void UpdatedLagrangian::CalculateOnIntegrationPoints(const Variable<Vector>& rVariable, std::vector<Vector>& rOutput, const ProcessInfo& rCurrentProcessInfo) { /* Adapt from TotalLagrangian, use UL kinematics */ }
void UpdatedLagrangian::CalculateOnIntegrationPoints(const Variable<Matrix>& rVariable, std::vector<Matrix>& rOutput, const ProcessInfo& rCurrentProcessInfo) { /* Adapt from TotalLagrangian, use UL kinematics */ }
void UpdatedLagrangian::CalculateOnIntegrationPoints(const Variable<array_1d<double,3>>& rVariable, std::vector<array_1d<double,3>>& rOutput, const ProcessInfo& rCurrentProcessInfo) { /* Adapt from TotalLagrangian */ }

void UpdatedLagrangian::SetValuesOnIntegrationPoints(const Variable<double>& rVariable, const std::vector<double>& rValues, const ProcessInfo& rCurrentProcessInfo) { /* Same as TotalLagrangian */ }
// ... other SetValuesOnIntegrationPoints ... (copy from TotalLagrangian and change class name in KRATOS_WARNING)

std::string UpdatedLagrangian::Info() const { return "Updated Lagrangian Element #" + std::to_string(Id()); }
void UpdatedLagrangian::PrintInfo(std::ostream& rOStream) const { rOStream << Info(); }
void UpdatedLagrangian::PrintData(std::ostream& rOStream) const { pGetGeometry()->PrintData(rOStream); }

void UpdatedLagrangian::save(Serializer& rSerializer) const {
    KRATOS_SERIALIZE_SAVE_BASE_CLASS(rSerializer, Element);
    int IntMethod = int(mThisIntegrationMethod);
    rSerializer.save("IntegrationMethod", IntMethod);
    rSerializer.save("ConstitutiveLawVector", mConstitutiveLawVector);
    // KRATOS_SERIALIZE_SAVE("F0_history", mF0_history); // If F0 history is used
}
void UpdatedLagrangian::load(Serializer& rSerializer) {
    KRATOS_SERIALIZE_LOAD_BASE_CLASS(rSerializer, Element);
    int IntMethod;
    rSerializer.load("IntegrationMethod", IntMethod);
    mThisIntegrationMethod = IntegrationMethod(IntMethod);
    rSerializer.load("ConstitutiveLawVector", mConstitutiveLawVector);
    // KRATOS_SERIALIZE_LOAD("F0_history", mF0_history); // If F0 history is used
}

} // namespace Kratos

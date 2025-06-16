#include "total_lagrangian.h"
#include "utilities/math_utils.h"
#include "custom_utilities/constitutive_law_utilities.h" // For VonMises etc.
#include "structural_mechanics_application_variables.h"
#include "utilities/atomic_utilities.h" // For AddExplicitContribution


namespace Kratos
{

TotalLagrangian::TotalLagrangian(IndexType NewId, GeometryType::Pointer pGeometry)
    : Element(NewId, pGeometry) {
    mThisIntegrationMethod = GetGeometry().GetDefaultIntegrationMethod();
}

TotalLagrangian::TotalLagrangian(IndexType NewId, GeometryType::Pointer pGeometry, PropertiesType::Pointer pProperties)
    : Element(NewId, pGeometry, pProperties) {
    mThisIntegrationMethod = GetGeometry().GetDefaultIntegrationMethod();
}

TotalLagrangian::~TotalLagrangian() {}

Element::Pointer TotalLagrangian::Create(IndexType NewId, NodesArrayType const& ThisNodes, PropertiesType::Pointer pProperties) const {
    return Kratos::make_intrusive<TotalLagrangian>(NewId, GetGeometry().Create(ThisNodes), pProperties);
}

Element::Pointer TotalLagrangian::Create(IndexType NewId, GeometryType::Pointer pGeom, PropertiesType::Pointer pProperties) const {
    return Kratos::make_intrusive<TotalLagrangian>(NewId, pGeom, pProperties);
}

Element::Pointer TotalLagrangian::Clone(IndexType NewId, NodesArrayType const& rThisNodes) const {
    KRATOS_TRY
    TotalLagrangian::Pointer p_new_elem = Kratos::make_intrusive<TotalLagrangian>(NewId, GetGeometry().Create(rThisNodes), pGetProperties());
    p_new_elem->SetData(this->GetData());
    p_new_elem->Set(Flags(*this));
    p_new_elem->mThisIntegrationMethod = this->mThisIntegrationMethod;
    p_new_elem->mConstitutiveLawVector.resize(this->mConstitutiveLawVector.size());
    for(size_t i=0; i<this->mConstitutiveLawVector.size(); ++i) {
        if(this->mConstitutiveLawVector[i] != nullptr)
            p_new_elem->mConstitutiveLawVector[i] = this->mConstitutiveLawVector[i]->Clone();
    }
    return p_new_elem;
    KRATOS_CATCH("");
}

void TotalLagrangian::InitializeSolutionStep(const ProcessInfo& rCurrentProcessInfo) {
    KRATOS_TRY
    // Adapted from SmallDisplacement, should be generic enough for TL as well
    // In BaseSolidElement, this involved calling InitializeMaterialResponse on CL
    bool requires_init = false;
    for (const auto& cl : mConstitutiveLawVector) {
        if (cl->RequiresInitializeMaterialResponse()) {
            requires_init = true;
            break;
        }
    }
    if (requires_init) {
        const auto& r_geom = GetGeometry();
        const auto& r_props = GetProperties();
        const SizeType num_nodes = r_geom.size();
        const SizeType dim = r_geom.WorkingSpaceDimension();
        KRATOS_ERROR_IF(mConstitutiveLawVector.empty() || !mConstitutiveLawVector[0]) << "CL not init! Elm ID: " << Id() << std::endl;
        const SizeType strain_size = mConstitutiveLawVector[0]->GetStrainSize();

        SolidElementUtilities::LocalKinematicVariables kin_vars(strain_size, dim, num_nodes);
        SolidElementUtilities::LocalConstitutiveVariables const_vars(strain_size);

        ConstitutiveLaw::Parameters cl_params(r_geom, r_props, rCurrentProcessInfo);
        Flags& cl_options = cl_params.GetOptions();
        cl_options.Set(ConstitutiveLaw::USE_ELEMENT_PROVIDED_STRAIN, UseElementProvidedStrain());
        cl_options.Set(ConstitutiveLaw::COMPUTE_STRESS, true);
        cl_options.Set(ConstitutiveLaw::COMPUTE_CONSTITUTIVE_TENSOR, false);

        // For TL, F is primary. Strain is computed by CL from F.
        cl_params.SetStrainVector(const_vars.StrainVector);
        cl_params.SetStressVector(const_vars.StressVector);
        cl_params.SetConstitutiveMatrix(const_vars.D);

        const auto& integration_points = r_geom.IntegrationPoints(mThisIntegrationMethod);
        for (IndexType i = 0; i < mConstitutiveLawVector.size(); ++i) {
            SolidElementUtilities::CalculateKinematicVariablesTotalLagrangian(
                kin_vars, i, mThisIntegrationMethod, r_geom, true);

            SolidElementUtilities::SetTotalLagrangianConstitutiveVariables(kin_vars, const_vars, cl_params);
            // CL will use F from cl_params to compute strain if USE_ELEMENT_PROVIDED_STRAIN is false.

            bool is_rotated = SolidElementUtilities::IsElementRotated(*this, mConstitutiveLawVector[i]);
            // Apply rotation to F in cl_params before calling CL if needed
            if (is_rotated) {
                BoundedMatrix<double,3,3> rotation_matrix;
                array_1d<double,3> local_axis_1 = GetValue(LOCAL_AXIS_1); // Assuming it's set
                array_1d<double,3> local_axis_2 = Has(LOCAL_AXIS_2) ? GetValue(LOCAL_AXIS_2) : ZeroVector(3);
                SolidElementUtilities::BuildRotationMatrix(rotation_matrix, local_axis_1, local_axis_2, strain_size);
                SolidElementUtilities::RotateFToLocalAxes(cl_params.GetDeformationGradientF(), rotation_matrix);
            }

            mConstitutiveLawVector[i]->InitializeMaterialResponse(cl_params, GetStressMeasure());
            // No need to rotate F back here as it's just initialization
        }
    }
    KRATOS_CATCH("")
}

void TotalLagrangian::FinalizeSolutionStep(const ProcessInfo& rCurrentProcessInfo) {
    KRATOS_TRY
    // Similar to InitializeSolutionStep
    bool requires_finalize = false;
    for (const auto& cl : mConstitutiveLawVector) {
        if (cl->RequiresFinalizeMaterialResponse()) {
            requires_finalize = true;
            break;
        }
    }
    if (requires_finalize) {
        const auto& r_geom = GetGeometry();
        const auto& r_props = GetProperties();
        const SizeType num_nodes = r_geom.size();
        const SizeType dim = r_geom.WorkingSpaceDimension();
        KRATOS_ERROR_IF(mConstitutiveLawVector.empty() || !mConstitutiveLawVector[0]) << "CL not init! Elm ID: " << Id() << std::endl;
        const SizeType strain_size = mConstitutiveLawVector[0]->GetStrainSize();

        SolidElementUtilities::LocalKinematicVariables kin_vars(strain_size, dim, num_nodes);
        SolidElementUtilities::LocalConstitutiveVariables const_vars(strain_size);

        ConstitutiveLaw::Parameters cl_params(r_geom, r_props, rCurrentProcessInfo);
        Flags& cl_options = cl_params.GetOptions();
        cl_options.Set(ConstitutiveLaw::USE_ELEMENT_PROVIDED_STRAIN, UseElementProvidedStrain());
        cl_options.Set(ConstitutiveLaw::COMPUTE_STRESS, true);
        cl_options.Set(ConstitutiveLaw::COMPUTE_CONSTITUTIVE_TENSOR, false);

        cl_params.SetStrainVector(const_vars.StrainVector);
        cl_params.SetStressVector(const_vars.StressVector);
        cl_params.SetConstitutiveMatrix(const_vars.D);

        const auto& integration_points = r_geom.IntegrationPoints(mThisIntegrationMethod);
        for (IndexType i = 0; i < mConstitutiveLawVector.size(); ++i) {
            SolidElementUtilities::CalculateKinematicVariablesTotalLagrangian(
                kin_vars, i, mThisIntegrationMethod, r_geom, true);
            SolidElementUtilities::SetTotalLagrangianConstitutiveVariables(kin_vars, const_vars, cl_params);

            bool is_rotated = SolidElementUtilities::IsElementRotated(*this, mConstitutiveLawVector[i]);
            if (is_rotated) {
                BoundedMatrix<double,3,3> rotation_matrix;
                array_1d<double,3> local_axis_1 = GetValue(LOCAL_AXIS_1);
                array_1d<double,3> local_axis_2 = Has(LOCAL_AXIS_2) ? GetValue(LOCAL_AXIS_2) : ZeroVector(3);
                SolidElementUtilities::BuildRotationMatrix(rotation_matrix, local_axis_1, local_axis_2, strain_size);
                SolidElementUtilities::RotateFToLocalAxes(cl_params.GetDeformationGradientF(), rotation_matrix);
            }
            mConstitutiveLawVector[i]->FinalizeMaterialResponse(cl_params, GetStressMeasure());
        }
    }
    KRATOS_CATCH("")
}

void TotalLagrangian::ResetConstitutiveLaw() {
    KRATOS_TRY
    if (GetProperties().Has(CONSTITUTIVE_LAW)) {
        const auto& r_geom = GetGeometry();
        const auto& r_prop = GetProperties();
        const auto& N_values = r_geom.ShapeFunctionsValues(mThisIntegrationMethod);
        for (IndexType i = 0; i < mConstitutiveLawVector.size(); ++i) {
            mConstitutiveLawVector[i]->ResetMaterial(r_prop, r_geom, row(N_values, i));
        }
    }
    KRATOS_CATCH("")
}

void TotalLagrangian::Initialize(const ProcessInfo& rCurrentProcessInfo) {
    KRATOS_TRY
    // Copied from SmallDisplacement::Initialize, should be generic enough
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
                    KRATOS_WARNING("TotalLagrangian") << "Integration order " << integration_order << " is not available, using default." << std::endl;
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
    }
    KRATOS_CATCH( "" )
}

ConstitutiveLaw::StressMeasure TotalLagrangian::GetStressMeasure() const {
    return ConstitutiveLaw::StressMeasure_PK2; // Typical for Total Lagrangian
}

bool TotalLagrangian::UseElementProvidedStrain() const {
    return false; // CL calculates strain from F for TL
}

void TotalLagrangian::CalculateLocalSystem(MatrixType& rLeftHandSideMatrix, VectorType& rRightHandSideVector, const ProcessInfo& rCurrentProcessInfo) {
    KRATOS_TRY
    auto& r_geom = GetGeometry();
    const SizeType num_nodes = r_geom.size();
    const SizeType dim = r_geom.WorkingSpaceDimension();
    KRATOS_ERROR_IF(mConstitutiveLawVector.empty() || !mConstitutiveLawVector[0]) << "CL not init! Elm ID: " << Id() << std::endl;
    const SizeType strain_size = mConstitutiveLawVector[0]->GetStrainSize();

    SolidElementUtilities::LocalKinematicVariables kin_vars(strain_size, dim, num_nodes);
    SolidElementUtilities::LocalConstitutiveVariables const_vars(strain_size);

    const SizeType mat_size = num_nodes * dim;
    if (rLeftHandSideMatrix.size1() != mat_size) rLeftHandSideMatrix.resize(mat_size, mat_size, false);
    noalias(rLeftHandSideMatrix) = ZeroMatrix(mat_size, mat_size);
    if (rRightHandSideVector.size() != mat_size) rRightHandSideVector.resize(mat_size, false);
    noalias(rRightHandSideVector) = ZeroVector(mat_size);

    const auto& integration_points = r_geom.IntegrationPoints(mThisIntegrationMethod);
    ConstitutiveLaw::Parameters cl_params(r_geom, GetProperties(), rCurrentProcessInfo);
    Flags& cl_options = cl_params.GetOptions();
    cl_options.Set(ConstitutiveLaw::USE_ELEMENT_PROVIDED_STRAIN, UseElementProvidedStrain());
    cl_options.Set(ConstitutiveLaw::COMPUTE_STRESS, true);
    cl_options.Set(ConstitutiveLaw::COMPUTE_CONSTITUTIVE_TENSOR, true);

    // For TL, CL computes Green-Lagrange strain from F. F is set in cl_params.
    // CL will write to cl_params.GetStrainVector() (which points to const_vars.StrainVector)
    cl_params.SetStrainVector(const_vars.StrainVector);
    cl_params.SetStressVector(const_vars.StressVector);
    cl_params.SetConstitutiveMatrix(const_vars.D);

    const auto& N_container = r_geom.ShapeFunctionsValues(mThisIntegrationMethod); // For body forces and axisymmetric B

    for (IndexType i = 0; i < integration_points.size(); ++i) {
        SolidElementUtilities::CalculateKinematicVariablesTotalLagrangian(kin_vars, i, mThisIntegrationMethod, r_geom, true);

        SolidElementUtilities::SetTotalLagrangianConstitutiveVariables(kin_vars, const_vars, cl_params);

        // CL calculates strain (e.g. Green-Lagrange) from F and updates const_vars.StrainVector via cl_params
        bool is_rotated = SolidElementUtilities::IsElementRotated(*this, mConstitutiveLawVector[i]);
        SolidElementUtilities::CalculateConstitutiveVariables(kin_vars, const_vars, cl_params, mConstitutiveLawVector[i], GetStressMeasure(), is_rotated, *this);
        // const_vars.StressVector and const_vars.D are now populated (in global axes if rotation occurred)

        SolidElementUtilities::CalculateBTotalLagrangian(kin_vars.B, kin_vars.F, kin_vars.DN_DX, strain_size, dim, num_nodes, row(N_container, i));

        double integration_weight = SolidElementUtilities::GetIntegrationWeight(integration_points, i, kin_vars.detJ0, dim, GetProperties());

        // LHS = B^T * D * B * w
        SolidElementUtilities::CalculateMaterialStiffnessMatrix(rLeftHandSideMatrix, kin_vars.B, const_vars.D, integration_weight);

        // Geometric Stiffness Matrix (Kg)
        Matrix Kg_gauss(mat_size, mat_size);
        StructuralMechanicsElementUtilities::CalculateKgMatrix(Kg_gauss, kin_vars.DN_DX, const_vars.StressVector, integration_weight, dim, num_nodes);
        rLeftHandSideMatrix += Kg_gauss;


        // RHS
        array_1d<double, 3> body_force = SolidElementUtilities::GetBodyForce(*this, integration_points, i);
        SolidElementUtilities::AddBodyForceContribution(row(N_container,i), rCurrentProcessInfo, body_force, rRightHandSideVector, integration_weight, dim, num_nodes);
        SolidElementUtilities::CalculateAndAddInternalForces(rRightHandSideVector, kin_vars.B, const_vars.StressVector, integration_weight);
    }
    KRATOS_CATCH("")
}

void TotalLagrangian::CalculateLeftHandSide(MatrixType& rLeftHandSideMatrix, const ProcessInfo& rCurrentProcessInfo) {
    VectorType temp_rhs; CalculateLocalSystem(rLeftHandSideMatrix, temp_rhs, rCurrentProcessInfo);
}
void TotalLagrangian::CalculateRightHandSide(VectorType& rRightHandSideVector, const ProcessInfo& rCurrentProcessInfo) {
    MatrixType temp_lhs; CalculateLocalSystem(temp_lhs, rRightHandSideVector, rCurrentProcessInfo);
}

void TotalLagrangian::CalculateOnIntegrationPoints(const Variable<double>& rVariable, std::vector<double>& rOutput, const ProcessInfo& rCurrentProcessInfo) {
    KRATOS_TRY
    const auto& r_geom = GetGeometry();
    const auto& r_props = GetProperties();
    const SizeType dimension = r_geom.WorkingSpaceDimension();
    const SizeType num_nodes = r_geom.size();
    KRATOS_ERROR_IF(mConstitutiveLawVector.empty() || !mConstitutiveLawVector[0]) << "CL not init! Elm ID: " << Id() << std::endl;
    const SizeType strain_size = mConstitutiveLawVector[0]->GetStrainSize();

    const auto& integration_points = r_geom.IntegrationPoints(mThisIntegrationMethod);
    if (rOutput.size() != integration_points.size()) rOutput.resize(integration_points.size());

    if (rVariable == VON_MISES_STRESS) { // PK2 VonMises for TL
        std::vector<Vector> pk2_stresses;
        this->CalculateOnIntegrationPoints(PK2_STRESS_VECTOR, pk2_stresses, rCurrentProcessInfo);
        for(size_t i=0; i < pk2_stresses.size(); ++i) {
             if (dimension == 2 && strain_size == 4) { // Axisymmetric
                Vector aux_stress_for_vm(6);
                noalias(aux_stress_for_vm) = ZeroVector(6);
                aux_stress_for_vm[0] = pk2_stresses[i][0]; aux_stress_for_vm[1] = pk2_stresses[i][1];
                aux_stress_for_vm[2] = pk2_stresses[i][3]; aux_stress_for_vm[3] = pk2_stresses[i][2];
                rOutput[i] = ConstitutiveLawUtilities<6>::CalculateVonMisesEquivalentStress(aux_stress_for_vm);
            } else {
                 rOutput[i] = ConstitutiveLawUtilities<6>::CalculateVonMisesEquivalentStress(pk2_stresses[i]);
            }
        }
    } else if (rVariable == INTEGRATION_WEIGHT) {
        SolidElementUtilities::LocalKinematicVariables kin_vars(strain_size, dimension, num_nodes);
        for (IndexType i = 0; i < integration_points.size(); ++i) {
            kin_vars.detJ0 = SolidElementUtilities::CalculateDerivativesOnReferenceConfiguration(
                kin_vars.J0, kin_vars.InvJ0, kin_vars.DN_DX, i, mThisIntegrationMethod, r_geom, true);
            rOutput[i] = SolidElementUtilities::GetIntegrationWeight(integration_points, i, kin_vars.detJ0, dimension, r_props);
        }
    } else if (mConstitutiveLawVector[0]->Has(rVariable)) {
        SolidElementUtilities::GetValueFromConstitutiveLaw(mConstitutiveLawVector, rVariable, rOutput);
    } else {
        // Use the generic CalculateValue utility, making sure it uses TL kinematics
        // For now, this specific CalculateValueOnConstitutiveLaw in utilities uses SD kinematics.
        // So, we either make it truly generic or implement logic here.
        // For safety, implementing directly:
        SolidElementUtilities::LocalKinematicVariables kin_vars(strain_size, dimension, num_nodes);
        SolidElementUtilities::LocalConstitutiveVariables const_vars(strain_size);
        ConstitutiveLaw::Parameters cl_params(r_geom, r_props, rCurrentProcessInfo);
        // Set appropriate flags for cl_params based on rVariable
        cl_params.GetOptions().Set(ConstitutiveLaw::USE_ELEMENT_PROVIDED_STRAIN, UseElementProvidedStrain());
        // ... (other flags might be needed)
        cl_params.SetStrainVector(const_vars.StrainVector);
        cl_params.SetStressVector(const_vars.StressVector);
        cl_params.SetConstitutiveMatrix(const_vars.D);

        for (IndexType i = 0; i < integration_points.size(); ++i) {
            SolidElementUtilities::CalculateKinematicVariablesTotalLagrangian(kin_vars, i, mThisIntegrationMethod, r_geom, true);
            SolidElementUtilities::SetTotalLagrangianConstitutiveVariables(kin_vars, const_vars, cl_params);
            bool is_rotated = SolidElementUtilities::IsElementRotated(*this, mConstitutiveLawVector[i]);
            // Rotation logic for F if needed before CalculateValue
            rOutput[i] = mConstitutiveLawVector[i]->CalculateValue(cl_params, rVariable, rOutput[i]);
            // Rotation logic for output if needed
        }
    }
    KRATOS_CATCH("")
}

void TotalLagrangian::CalculateOnIntegrationPoints(const Variable<Vector>& rVariable, std::vector<Vector>& rOutput, const ProcessInfo& rCurrentProcessInfo) {
    KRATOS_TRY
    const auto& r_geom = GetGeometry();
    const auto& r_props = GetProperties();
    const SizeType dimension = r_geom.WorkingSpaceDimension();
    const SizeType num_nodes = r_geom.size();
    KRATOS_ERROR_IF(mConstitutiveLawVector.empty() || !mConstitutiveLawVector[0]) << "CL not init! Elm ID: " << Id() << std::endl;
    const SizeType strain_size = mConstitutiveLawVector[0]->GetStrainSize();

    const auto& integration_points = r_geom.IntegrationPoints(mThisIntegrationMethod);
    if (rOutput.size() != integration_points.size()) rOutput.resize(integration_points.size());
    for(auto& v : rOutput) if(v.size()!=strain_size) v.resize(strain_size, false);


    if (rVariable == PK2_STRESS_VECTOR || rVariable == CAUCHY_STRESS_VECTOR) { // Cauchy needs push-forward from PK2
        SolidElementUtilities::LocalKinematicVariables kin_vars(strain_size, dimension, num_nodes);
        SolidElementUtilities::LocalConstitutiveVariables const_vars(strain_size);
        ConstitutiveLaw::Parameters cl_params(r_geom, r_props, rCurrentProcessInfo);
        cl_params.GetOptions().Set(ConstitutiveLaw::USE_ELEMENT_PROVIDED_STRAIN, UseElementProvidedStrain());
        cl_params.GetOptions().Set(ConstitutiveLaw::COMPUTE_STRESS, true);
        cl_params.GetOptions().Set(ConstitutiveLaw::COMPUTE_CONSTITUTIVE_TENSOR, false);
        cl_params.SetStrainVector(const_vars.StrainVector);
        cl_params.SetStressVector(const_vars.StressVector);
        cl_params.SetConstitutiveMatrix(const_vars.D);

        for (IndexType i = 0; i < integration_points.size(); ++i) {
            SolidElementUtilities::CalculateKinematicVariablesTotalLagrangian(kin_vars, i, mThisIntegrationMethod, r_geom, true);
            SolidElementUtilities::SetTotalLagrangianConstitutiveVariables(kin_vars, const_vars, cl_params);

            bool is_rotated = SolidElementUtilities::IsElementRotated(*this, mConstitutiveLawVector[i]);
            // PK2 is calculated first, then potentially pushed forward for Cauchy
            SolidElementUtilities::CalculateConstitutiveVariables(kin_vars, const_vars, cl_params, mConstitutiveLawVector[i], ConstitutiveLaw::StressMeasure_PK2, is_rotated, *this);

            if (rVariable == PK2_STRESS_VECTOR) {
                rOutput[i] = const_vars.StressVector;
            } else { // CAUCHY_STRESS_VECTOR
                Vector cauchy_stress_gauss_point(strain_size);
                mConstitutiveLawVector[i]->TransformStresses(cauchy_stress_gauss_point, kin_vars.F, kin_vars.detF, ConstitutiveLaw::StressMeasure_Cauchy, ConstitutiveLaw::StressMeasure_PK2, const_vars.StressVector);
                rOutput[i] = cauchy_stress_gauss_point;
            }
        }
    } else if (rVariable == GREEN_LAGRANGE_STRAIN_VECTOR) { // Almansi not typical for TL directly from CL
        SolidElementUtilities::LocalKinematicVariables kin_vars(strain_size, dimension, num_nodes);
        SolidElementUtilities::LocalConstitutiveVariables const_vars(strain_size);
        ConstitutiveLaw::Parameters cl_params(r_geom, r_props, rCurrentProcessInfo);
        cl_params.GetOptions().Set(ConstitutiveLaw::USE_ELEMENT_PROVIDED_STRAIN, UseElementProvidedStrain());
        cl_params.GetOptions().Set(ConstitutiveLaw::COMPUTE_STRESS, false); // Only need strain
        cl_params.GetOptions().Set(ConstitutiveLaw::COMPUTE_CONSTITUTIVE_TENSOR, false);
        cl_params.SetStrainVector(const_vars.StrainVector); // CL writes strain here

        for (IndexType i = 0; i < integration_points.size(); ++i) {
            SolidElementUtilities::CalculateKinematicVariablesTotalLagrangian(kin_vars, i, mThisIntegrationMethod, r_geom, true);
            SolidElementUtilities::SetTotalLagrangianConstitutiveVariables(kin_vars, const_vars, cl_params);
            // CL calculates strain from F and sets it in const_vars.StrainVector via cl_params
            mConstitutiveLawVector[i]->CalculateMaterialResponse(cl_params, GetStressMeasure());
            rOutput[i] = const_vars.StrainVector;
        }
    } else if (mConstitutiveLawVector[0]->Has(rVariable)) {
        SolidElementUtilities::GetValueFromConstitutiveLaw(mConstitutiveLawVector, rVariable, rOutput);
    } else {
        SolidElementUtilities::CalculateValueOnConstitutiveLaw(*this, mConstitutiveLawVector, rVariable, rCurrentProcessInfo, mThisIntegrationMethod, UseElementProvidedStrain(), GetStressMeasure(), rOutput);
    }
    KRATOS_CATCH("")
}

void TotalLagrangian::CalculateOnIntegrationPoints(const Variable<Matrix>& rVariable, std::vector<Matrix>& rOutput, const ProcessInfo& rCurrentProcessInfo) {
    KRATOS_TRY
    const auto& r_geom = GetGeometry();
    const auto& r_props = GetProperties();
    const SizeType dimension = r_geom.WorkingSpaceDimension();
    const SizeType num_nodes = r_geom.size();
    KRATOS_ERROR_IF(mConstitutiveLawVector.empty() || !mConstitutiveLawVector[0]) << "CL not init! Elm ID: " << Id() << std::endl;
    const SizeType strain_size = mConstitutiveLawVector[0]->GetStrainSize();

    const auto& integration_points = r_geom.IntegrationPoints(mThisIntegrationMethod);
    if (rOutput.size() != integration_points.size()) rOutput.resize(integration_points.size());

    if (rVariable == CONSTITUTIVE_MATRIX) { // Secant Constitutive Matrix C_abcd (Voigt D_IJ)
        SolidElementUtilities::LocalKinematicVariables kin_vars(strain_size, dimension, num_nodes);
        SolidElementUtilities::LocalConstitutiveVariables const_vars(strain_size);
        ConstitutiveLaw::Parameters cl_params(r_geom, r_props, rCurrentProcessInfo);
        cl_params.GetOptions().Set(ConstitutiveLaw::USE_ELEMENT_PROVIDED_STRAIN, UseElementProvidedStrain());
        cl_params.GetOptions().Set(ConstitutiveLaw::COMPUTE_STRESS, false);
        cl_params.GetOptions().Set(ConstitutiveLaw::COMPUTE_CONSTITUTIVE_TENSOR, true);
        cl_params.SetStrainVector(const_vars.StrainVector);
        cl_params.SetStressVector(const_vars.StressVector);
        cl_params.SetConstitutiveMatrix(const_vars.D); // CL writes D here

        for (IndexType i = 0; i < integration_points.size(); ++i) {
            SolidElementUtilities::CalculateKinematicVariablesTotalLagrangian(kin_vars, i, mThisIntegrationMethod, r_geom, true);
            SolidElementUtilities::SetTotalLagrangianConstitutiveVariables(kin_vars, const_vars, cl_params);

            bool is_rotated = SolidElementUtilities::IsElementRotated(*this, mConstitutiveLawVector[i]);
            SolidElementUtilities::CalculateConstitutiveVariables(kin_vars, const_vars, cl_params, mConstitutiveLawVector[i], GetStressMeasure(), is_rotated, *this);
            if(rOutput[i].size1()!=strain_size || rOutput[i].size2()!=strain_size) rOutput[i].resize(strain_size,strain_size,false);
            rOutput[i] = const_vars.D;
        }
    } else if (rVariable == DEFORMATION_GRADIENT) {
        SolidElementUtilities::LocalKinematicVariables kin_vars(strain_size, dimension, num_nodes);
        for (IndexType i = 0; i < integration_points.size(); ++i) {
            SolidElementUtilities::CalculateKinematicVariablesTotalLagrangian(kin_vars, i, mThisIntegrationMethod, r_geom, true);
            if(rOutput[i].size1() != dimension || rOutput[i].size2() != dimension) rOutput[i].resize(dimension,dimension,false);
            rOutput[i] = kin_vars.F;
        }
    } else if (rVariable == CAUCHY_STRESS_TENSOR || rVariable == PK2_STRESS_TENSOR) {
        std::vector<Vector> stress_vector_output;
        const Variable<Vector>& vector_variable = (rVariable == CAUCHY_STRESS_TENSOR) ? CAUCHY_STRESS_VECTOR : PK2_STRESS_VECTOR;
        this->CalculateOnIntegrationPoints(vector_variable, stress_vector_output, rCurrentProcessInfo);
        for (IndexType i = 0; i < integration_points.size(); ++i) {
            if(rOutput[i].size1()!=dimension || rOutput[i].size2()!=dimension) rOutput[i].resize(dimension,dimension,false);
            rOutput[i] = MathUtils<double>::StressVectorToTensor(stress_vector_output[i]);
        }
    } else if (rVariable == GREEN_LAGRANGE_STRAIN_TENSOR) { // Almansi not typical for TL
        std::vector<Vector> strain_vector_output;
        this->CalculateOnIntegrationPoints(GREEN_LAGRANGE_STRAIN_VECTOR, strain_vector_output, rCurrentProcessInfo);
        for (IndexType i = 0; i < integration_points.size(); ++i) {
             if(rOutput[i].size1()!=dimension || rOutput[i].size2()!=dimension) rOutput[i].resize(dimension,dimension,false);
            rOutput[i] = MathUtils<double>::StrainVectorToTensor(strain_vector_output[i]);
        }
    }
    else if (mConstitutiveLawVector[0]->Has(rVariable)) {
        SolidElementUtilities::GetValueFromConstitutiveLaw(mConstitutiveLawVector, rVariable, rOutput);
    } else {
        SolidElementUtilities::CalculateValueOnConstitutiveLaw(*this, mConstitutiveLawVector, rVariable, rCurrentProcessInfo, mThisIntegrationMethod, UseElementProvidedStrain(), GetStressMeasure(), rOutput);
    }
    KRATOS_CATCH("")
}

void TotalLagrangian::CalculateOnIntegrationPoints(const Variable<array_1d<double,3>>& rVariable, std::vector<array_1d<double,3>>& rOutput, const ProcessInfo& rCurrentProcessInfo) {
    if (rVariable == INTEGRATION_COORDINATES) { // This part is generic
        const auto& r_geom = GetGeometry();
        const GeometryType::IntegrationPointsArrayType& integration_points = r_geom.IntegrationPoints(mThisIntegrationMethod);
        if (rOutput.size() != integration_points.size()) rOutput.resize(integration_points.size());
        for (IndexType i = 0; i < integration_points.size(); ++i) {
            Point global_point;
            r_geom.GlobalCoordinates(global_point, integration_points[i]);
            noalias(rOutput[i]) = global_point.Coordinates();
        }
    } else if (mConstitutiveLawVector[0]->Has(rVariable)){
        SolidElementUtilities::GetValueFromConstitutiveLaw(mConstitutiveLawVector, rVariable, rOutput);
    } else {
         SolidElementUtilities::CalculateValueOnConstitutiveLaw(*this, mConstitutiveLawVector, rVariable, rCurrentProcessInfo, mThisIntegrationMethod, UseElementProvidedStrain(), GetStressMeasure(), rOutput);
    }
}

void TotalLagrangian::CalculateMassMatrix(MatrixType& rMassMatrix, const ProcessInfo& rCurrentProcessInfo) {
    KRATOS_TRY;
    // Copied from SmallDisplacement, should be generic
    const auto& r_prop = GetProperties();
    const auto& r_geom = GetGeometry();
    SizeType dimension = r_geom.WorkingSpaceDimension();
    SizeType number_of_nodes = r_geom.size();
    SizeType mat_size = dimension * number_of_nodes;

    if (rMassMatrix.size1() != mat_size || rMassMatrix.size2() != mat_size) rMassMatrix.resize( mat_size, mat_size, false );
    noalias(rMassMatrix) = ZeroMatrix( mat_size, mat_size );

    KRATOS_ERROR_IF_NOT(r_prop.Has(DENSITY)) << "DENSITY missing!" << std::endl;
    const bool compute_lumped = SolidElementUtilities::ComputeLumpedMassMatrix(r_prop, rCurrentProcessInfo);

    if (compute_lumped) {
        VectorType temp_vector(mat_size);
        SolidElementUtilities::CalculateLumpedMassVector(*this, temp_vector, rCurrentProcessInfo);
        for (IndexType i = 0; i < mat_size; ++i) rMassMatrix(i, i) = temp_vector[i];
    } else { // Consistent mass
        const double density = SolidElementUtilities::GetDensityForMassMatrixComputation(*this);
        const double thickness = (dimension == 2 && r_prop.Has(THICKNESS)) ? r_prop[THICKNESS] : 1.0; // TL might not use thickness this way

        SolidElementUtilities::LocalKinematicVariables kin_vars(mConstitutiveLawVector[0]->GetStrainSize(), dimension, number_of_nodes); // For detJ0

        const IntegrationMethod ig_method = GetGeometry().GetDefaultIntegrationMethod(); // Or mThisIntegrationMethod
        const GeometryType::IntegrationPointsArrayType& integration_points = GetGeometry().IntegrationPoints(ig_method);
        const Matrix& Ncontainer = GetGeometry().ShapeFunctionsValues(ig_method);

        for ( IndexType point_number = 0; point_number < integration_points.size(); ++point_number ) {
            kin_vars.detJ0 = SolidElementUtilities::CalculateDerivativesOnReferenceConfiguration(
                kin_vars.J0, kin_vars.InvJ0, kin_vars.DN_DX, point_number, ig_method, r_geom, true);
            const double integration_weight = SolidElementUtilities::GetIntegrationWeight(integration_points, point_number, kin_vars.detJ0, dimension, r_prop);

            const Vector& rN = row(Ncontainer,point_number);
            for ( IndexType i = 0; i < number_of_nodes; ++i ) {
                const SizeType index_i = i * dimension;
                for ( IndexType j = 0; j < number_of_nodes; ++j ) {
                    const SizeType index_j = j * dimension;
                    const double NiNj_weight = rN[i] * rN[j] * integration_weight * density; // Using initial volume for TL mass
                    for ( IndexType k = 0; k < dimension; ++k ) rMassMatrix( index_i + k, index_j + k ) += NiNj_weight;
                }
            }
        }
    }
    KRATOS_CATCH("");
}

void TotalLagrangian::CalculateDampingMatrix(MatrixType& rDampingMatrix, const ProcessInfo& rCurrentProcessInfo) {
    // Copied from SmallDisplacement
    const unsigned int mat_size = GetGeometry().PointsNumber() * GetGeometry().WorkingSpaceDimension();
    SolidElementUtilities::CalculateRayleighDampingMatrix(*this, rDampingMatrix, rCurrentProcessInfo, mat_size, mConstitutiveLawVector, mThisIntegrationMethod);
}

void TotalLagrangian::EquationIdVector(EquationIdVectorType& rResult, const ProcessInfo& rCurrentProcessInfo) const {
    // Copied from SmallDisplacement
    KRATOS_TRY;
    const auto& r_geom = GetGeometry();
    const SizeType number_of_nodes = r_geom.size();
    const SizeType dimension = r_geom.WorkingSpaceDimension();
    if (rResult.size() != dimension * number_of_nodes) rResult.resize(dimension * number_of_nodes,false);
    const SizeType pos = r_geom[0].GetDofPosition(DISPLACEMENT_X);

    if(dimension == 2) {
        for (IndexType i = 0; i < number_of_nodes; ++i) {
            const SizeType index = i * 2;
            rResult[index] = r_geom[i].GetDof(DISPLACEMENT_X,pos).EquationId();
            rResult[index + 1] = r_geom[i].GetDof(DISPLACEMENT_Y,pos+1).EquationId();
        }
    } else {
        for (IndexType i = 0; i < number_of_nodes; ++i) {
            const SizeType index = i * 3;
            rResult[index] = r_geom[i].GetDof(DISPLACEMENT_X,pos).EquationId();
            rResult[index + 1] = r_geom[i].GetDof(DISPLACEMENT_Y,pos+1).EquationId();
            rResult[index + 2] = r_geom[i].GetDof(DISPLACEMENT_Z,pos+2).EquationId();
        }
    }
    KRATOS_CATCH("")
}

void TotalLagrangian::GetDofList(DofsVectorType& rElementalDofList, const ProcessInfo& rCurrentProcessInfo) const {
    // Copied from SmallDisplacement
    KRATOS_TRY;
    const auto& r_geom = GetGeometry();
    const SizeType number_of_nodes = r_geom.size();
    const SizeType dimension = r_geom.WorkingSpaceDimension();
    rElementalDofList.resize(0);
    rElementalDofList.reserve(dimension * number_of_nodes);
    if(dimension == 2) {
        for (IndexType i = 0; i < number_of_nodes; ++i) {
            rElementalDofList.push_back(r_geom[i].pGetDof(DISPLACEMENT_X));
            rElementalDofList.push_back(r_geom[i].pGetDof(DISPLACEMENT_Y));
        }
    } else {
        for (IndexType i = 0; i < number_of_nodes; ++i) {
            rElementalDofList.push_back(r_geom[i].pGetDof(DISPLACEMENT_X));
            rElementalDofList.push_back(r_geom[i].pGetDof(DISPLACEMENT_Y));
            rElementalDofList.push_back(r_geom[i].pGetDof(DISPLACEMENT_Z));
        }
    }
    KRATOS_CATCH("")
}

void TotalLagrangian::GetValuesVector(Vector& rValues, int Step) const {
    // Copied from SmallDisplacement
    const auto& r_geom = GetGeometry();
    const SizeType number_of_nodes = r_geom.size();
    const SizeType dimension = r_geom.WorkingSpaceDimension();
    const SizeType mat_size = number_of_nodes * dimension;
    if (rValues.size() != mat_size) rValues.resize(mat_size, false);
    for (IndexType i = 0; i < number_of_nodes; ++i) {
        const array_1d<double, 3 >& displacement = r_geom[i].FastGetSolutionStepValue(DISPLACEMENT, Step);
        const SizeType index = i * dimension;
        for(unsigned int k = 0; k < dimension; ++k) {
            rValues[index + k] = displacement[k];
        }
    }
}

std::string TotalLagrangian::Info() const { return "Total Lagrangian Element #" + std::to_string(Id()); }
void TotalLagrangian::PrintInfo(std::ostream& rOStream) const { rOStream << Info(); }
void TotalLagrangian::PrintData(std::ostream& rOStream) const { pGetGeometry()->PrintData(rOStream); }

// SetValuesOnIntegrationPoints (copied from SmallDisplacement, should be mostly generic)
void TotalLagrangian::SetValuesOnIntegrationPoints(const Variable<double>& rVariable, const std::vector<double>& rValues, const ProcessInfo& rCurrentProcessInfo)
{ KRATOS_TRY if(!mConstitutiveLawVector.empty() && mConstitutiveLawVector[0]->Has(rVariable)) for(IndexType i=0; i<mConstitutiveLawVector.size(); ++i) mConstitutiveLawVector[i]->SetValue(rVariable, rValues[i], rCurrentProcessInfo); else KRATOS_WARNING("TL")<<"Set for double "<<rVariable.Name()<<" not in CL"<<std::endl; KRATOS_CATCH("") }
void TotalLagrangian::SetValuesOnIntegrationPoints(const Variable<Vector>& rVariable, const std::vector<Vector>& rValues, const ProcessInfo& rCurrentProcessInfo)
{ KRATOS_TRY if(!mConstitutiveLawVector.empty() && mConstitutiveLawVector[0]->Has(rVariable)) for(IndexType i=0; i<mConstitutiveLawVector.size(); ++i) mConstitutiveLawVector[i]->SetValue(rVariable, rValues[i], rCurrentProcessInfo); else KRATOS_WARNING("TL")<<"Set for Vector "<<rVariable.Name()<<" not in CL"<<std::endl; KRATOS_CATCH("") }
void TotalLagrangian::SetValuesOnIntegrationPoints(const Variable<Matrix>& rVariable, const std::vector<Matrix>& rValues, const ProcessInfo& rCurrentProcessInfo)
{ KRATOS_TRY if(!mConstitutiveLawVector.empty() && mConstitutiveLawVector[0]->Has(rVariable)) for(IndexType i=0; i<mConstitutiveLawVector.size(); ++i) mConstitutiveLawVector[i]->SetValue(rVariable, rValues[i], rCurrentProcessInfo); else KRATOS_WARNING("TL")<<"Set for Matrix "<<rVariable.Name()<<" not in CL"<<std::endl; KRATOS_CATCH("") }
void TotalLagrangian::SetValuesOnIntegrationPoints(const Variable<array_1d<double,3>>& rVariable, const std::vector<array_1d<double,3>>& rValues, const ProcessInfo& rCurrentProcessInfo)
{ KRATOS_TRY if(!mConstitutiveLawVector.empty() && mConstitutiveLawVector[0]->Has(rVariable)) for(IndexType i=0; i<mConstitutiveLawVector.size(); ++i) mConstitutiveLawVector[i]->SetValue(rVariable, rValues[i], rCurrentProcessInfo); else KRATOS_WARNING("TL")<<"Set for array_1d<double,3> "<<rVariable.Name()<<" not in CL"<<std::endl; KRATOS_CATCH("") }
void TotalLagrangian::SetValuesOnIntegrationPoints(const Variable<ConstitutiveLaw::Pointer>& rVariable, const std::vector<ConstitutiveLaw::Pointer>& rValues, const ProcessInfo& rCurrentProcessInfo)
{ if (rVariable == CONSTITUTIVE_LAW) { KRATOS_ERROR_IF(rValues.size() != mConstitutiveLawVector.size()) << "CL vector size mismatch." << std::endl; mConstitutiveLawVector = rValues; } else { KRATOS_WARNING("TL") << "Set for CL Ptr var " << rVariable.Name() << " not CONSTITUTIVE_LAW." << std::endl;}}
void TotalLagrangian::SetValuesOnIntegrationPoints(const Variable<bool>& rVariable, const std::vector<bool>& rValues, const ProcessInfo& rCurrentProcessInfo)
{ KRATOS_TRY if(!mConstitutiveLawVector.empty() && mConstitutiveLawVector[0]->Has(rVariable)) for(IndexType i=0; i<mConstitutiveLawVector.size(); ++i) mConstitutiveLawVector[i]->SetValue(rVariable, rValues[i], rCurrentProcessInfo); else KRATOS_WARNING("TL")<<"Set for bool "<<rVariable.Name()<<" not in CL"<<std::endl; KRATOS_CATCH("") }
void TotalLagrangian::SetValuesOnIntegrationPoints(const Variable<int>& rVariable, const std::vector<int>& rValues, const ProcessInfo& rCurrentProcessInfo)
{ KRATOS_TRY if(!mConstitutiveLawVector.empty() && mConstitutiveLawVector[0]->Has(rVariable)) for(IndexType i=0; i<mConstitutiveLawVector.size(); ++i) mConstitutiveLawVector[i]->SetValue(rVariable, rValues[i], rCurrentProcessInfo); else KRATOS_WARNING("TL")<<"Set for int "<<rVariable.Name()<<" not in CL"<<std::endl; KRATOS_CATCH("") }
void TotalLagrangian::SetValuesOnIntegrationPoints(const Variable<array_1d<double,6>>& rVariable, const std::vector<array_1d<double,6>>& rValues, const ProcessInfo& rCurrentProcessInfo)
{ KRATOS_TRY if(!mConstitutiveLawVector.empty() && mConstitutiveLawVector[0]->Has(rVariable)) for(IndexType i=0; i<mConstitutiveLawVector.size(); ++i) mConstitutiveLawVector[i]->SetValue(rVariable, rValues[i], rCurrentProcessInfo); else KRATOS_WARNING("TL")<<"Set for array_1d<double,6> "<<rVariable.Name()<<" not in CL"<<std::endl; KRATOS_CATCH("") }

void TotalLagrangian::GetFirstDerivativesVector(Vector& rValues, int Step) const {
    // Copied from SmallDisplacement
    const auto& r_geom = GetGeometry();
    const SizeType number_of_nodes = r_geom.size();
    const SizeType dimension = r_geom.WorkingSpaceDimension();
    const SizeType mat_size = number_of_nodes * dimension;
    if (rValues.size() != mat_size) rValues.resize(mat_size, false);
    for (IndexType i = 0; i < number_of_nodes; ++i) {
        const array_1d<double, 3 >& velocity = r_geom[i].FastGetSolutionStepValue(VELOCITY, Step);
        const SizeType index = i * dimension;
        for(unsigned int k = 0; k < dimension; ++k) rValues[index + k] = velocity[k];
    }
}
void TotalLagrangian::GetSecondDerivativesVector(Vector& rValues, int Step) const {
    // Copied from SmallDisplacement
    const auto& r_geom = GetGeometry();
    const SizeType number_of_nodes = r_geom.size();
    const SizeType dimension = r_geom.WorkingSpaceDimension();
    const SizeType mat_size = number_of_nodes * dimension;
    if (rValues.size() != mat_size) rValues.resize(mat_size, false);
    for (IndexType i = 0; i < number_of_nodes; ++i) {
        const array_1d<double, 3 >& acceleration = r_geom[i].FastGetSolutionStepValue(ACCELERATION, Step);
        const SizeType index = i * dimension;
        for(unsigned int k = 0; k < dimension; ++k) rValues[index + k] = acceleration[k];
    }
}

void TotalLagrangian::AddExplicitContribution(const VectorType& rRHSVector, const Variable<VectorType>& rRHSVariable, const Variable<double>& rDestinationVariable, const ProcessInfo& rCurrentProcessInfo) {
    // Copied from SmallDisplacement
    KRATOS_TRY;
    auto& r_geom = this->GetGeometry();
    const SizeType dimension = r_geom.WorkingSpaceDimension();
    const SizeType number_of_nodes = r_geom.size();
    const SizeType mat_size = number_of_nodes * dimension;

    if (rDestinationVariable == NODAL_MASS ) {
        VectorType element_mass_vector(mat_size);
        SolidElementUtilities::CalculateLumpedMassVector(*this, element_mass_vector, rCurrentProcessInfo);
        for (IndexType i = 0; i < number_of_nodes; ++i) {
            AtomicAdd(r_geom[i].GetValue(NODAL_MASS), element_mass_vector[i * dimension]);
        }
    }
    KRATOS_CATCH("")
}
void TotalLagrangian::AddExplicitContribution(const VectorType& rRHSVector, const Variable<VectorType>& rRHSVariable, const Variable<array_1d<double,3>>& rDestinationVariable, const ProcessInfo& rCurrentProcessInfo) {
    // Copied from SmallDisplacement
     KRATOS_TRY;
    auto& r_geom = this->GetGeometry();
    const auto& r_prop = this->GetProperties();
    const SizeType dimension = r_geom.WorkingSpaceDimension();
    const SizeType number_of_nodes = r_geom.size();
    const SizeType element_size = dimension * number_of_nodes;

    Vector damping_residual_contribution = ZeroVector(element_size);

    if (SolidElementUtilities::HasRayleighDamping(r_prop, rCurrentProcessInfo)) {
        Vector current_nodal_velocities = ZeroVector(element_size);
        this->GetFirstDerivativesVector(current_nodal_velocities);
        MatrixType damping_matrix(element_size, element_size);
        this->CalculateDampingMatrix(damping_matrix, rCurrentProcessInfo);
        noalias(damping_residual_contribution) = prod(damping_matrix, current_nodal_velocities);
    }

    if (rRHSVariable == RESIDUAL_VECTOR && rDestinationVariable == FORCE_RESIDUAL) {
        for (IndexType i = 0; i < number_of_nodes; ++i) {
            const IndexType index = dimension * i;
            array_1d<double, 3>& r_force_residual = r_geom[i].FastGetSolutionStepValue(FORCE_RESIDUAL);
            for (IndexType j = 0; j < dimension; ++j) {
                AtomicAdd(r_force_residual[j], (rRHSVector[index + j] - damping_residual_contribution[index + j]));
            }
        }
    }
    KRATOS_CATCH("")
}
const Parameters TotalLagrangian::GetSpecifications() const {
    return SolidElementUtilities::GetDefaultSolidSpecifications(GetGeometry().WorkingSpaceDimension());
}
int TotalLagrangian::Check(const ProcessInfo& rCurrentProcessInfo) const {
    KRATOS_TRY
    int check = Element::Check(rCurrentProcessInfo);
    check = SolidElementUtilities::SolidElementCheck(*this, rCurrentProcessInfo, mConstitutiveLawVector);
    // Add TL specific checks if any
    // Check that deformation gradient is not too small (detF > 0)
    const auto& r_geom = GetGeometry();
    const SizeType strain_size = mConstitutiveLawVector[0]->GetStrainSize();
    const SizeType dim = r_geom.WorkingSpaceDimension();
    const SizeType num_nodes = r_geom.size();
    SolidElementUtilities::LocalKinematicVariables kin_vars(strain_size, dim, num_nodes);
    const auto& integration_points = r_geom.IntegrationPoints(mThisIntegrationMethod);
    for (IndexType i = 0; i < integration_points.size(); ++i) {
         SolidElementUtilities::CalculateKinematicVariablesTotalLagrangian(kin_vars, i, mThisIntegrationMethod, r_geom, true);
         KRATOS_ERROR_IF(kin_vars.detF <= 0.0) << "Element #" << Id() << " on integration point " << i << " has non-positive determinant of F: " << kin_vars.detF << std::endl;
    }
    return check;
    KRATOS_CATCH("")
}

void TotalLagrangian::save(Serializer& rSerializer) const {
    KRATOS_SERIALIZE_SAVE_BASE_CLASS(rSerializer, Element);
    int IntMethod = int(mThisIntegrationMethod);
    rSerializer.save("IntegrationMethod", IntMethod);
    rSerializer.save("ConstitutiveLawVector", mConstitutiveLawVector);
}
void TotalLagrangian::load(Serializer& rSerializer) {
    KRATOS_SERIALIZE_LOAD_BASE_CLASS(rSerializer, Element);
    int IntMethod;
    rSerializer.load("IntegrationMethod", IntMethod);
    mThisIntegrationMethod = IntegrationMethod(IntMethod);
    rSerializer.load("ConstitutiveLawVector", mConstitutiveLawVector);
}


} // Namespace Kratos

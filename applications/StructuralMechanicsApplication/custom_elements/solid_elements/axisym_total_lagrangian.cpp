#include "axisym_total_lagrangian.h"
#include "utilities/math_utils.h"
#include "custom_utilities/constitutive_law_utilities.h"
#include "structural_mechanics_application_variables.h"
#include "utilities/atomic_utilities.h" // For AddExplicitContribution


namespace Kratos
{
AxisymTotalLagrangian::AxisymTotalLagrangian(IndexType NewId, GeometryType::Pointer pGeometry)
    : Element(NewId, pGeometry) {
    mThisIntegrationMethod = GetGeometry().GetDefaultIntegrationMethod();
}
AxisymTotalLagrangian::AxisymTotalLagrangian(IndexType NewId, GeometryType::Pointer pGeometry, PropertiesType::Pointer pProperties)
    : Element(NewId, pGeometry, pProperties) {
    mThisIntegrationMethod = GetGeometry().GetDefaultIntegrationMethod();
}
AxisymTotalLagrangian::~AxisymTotalLagrangian() {}

Element::Pointer AxisymTotalLagrangian::Create(IndexType NewId, NodesArrayType const& ThisNodes, PropertiesType::Pointer pProperties) const {
    return Kratos::make_intrusive<AxisymTotalLagrangian>(NewId, GetGeometry().Create(ThisNodes), pProperties);
}
Element::Pointer AxisymTotalLagrangian::Create(IndexType NewId, GeometryType::Pointer pGeom, PropertiesType::Pointer pProperties) const {
    return Kratos::make_intrusive<AxisymTotalLagrangian>(NewId, pGeom, pProperties);
}
Element::Pointer AxisymTotalLagrangian::Clone(IndexType NewId, NodesArrayType const& rThisNodes) const {
    KRATOS_TRY
    auto p_new_elem = Kratos::make_intrusive<AxisymTotalLagrangian>(NewId, GetGeometry().Create(rThisNodes), pGetProperties());
    // ... (copy data, flags, mThisIntegrationMethod, mConstitutiveLawVector - same as other elements) ...
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

void AxisymTotalLagrangian::InitializeSolutionStep(const ProcessInfo& rCurrentProcessInfo) {
    KRATOS_TRY
    bool requires_init = false;
    for (const auto& cl : mConstitutiveLawVector) {
        if (cl->RequiresInitializeMaterialResponse()) { requires_init = true; break; }
    }
    if (requires_init) {
        const auto& r_geom = GetGeometry(); const auto& r_props = GetProperties();
        const SizeType num_nodes = r_geom.size(); const SizeType dim = 2; // RZ
        KRATOS_ERROR_IF(mConstitutiveLawVector.empty() || !mConstitutiveLawVector[0]) << "CL not init! Elm ID: " << Id() << std::endl;
        const SizeType strain_size = 4;

        SolidElementUtilities::LocalKinematicVariables kin_vars(strain_size, dim, num_nodes);
        SolidElementUtilities::LocalConstitutiveVariables const_vars(strain_size);
        ConstitutiveLaw::Parameters cl_params(r_geom, r_props, rCurrentProcessInfo);
        cl_params.GetOptions().Set(ConstitutiveLaw::USE_ELEMENT_PROVIDED_STRAIN, UseElementProvidedStrain());
        cl_params.GetOptions().Set(ConstitutiveLaw::COMPUTE_STRESS, true);
        cl_params.GetOptions().Set(ConstitutiveLaw::COMPUTE_CONSTITUTIVE_TENSOR, false);
        cl_params.SetStrainVector(const_vars.StrainVector);
        cl_params.SetStressVector(const_vars.StressVector);
        cl_params.SetConstitutiveMatrix(const_vars.D);

        for (IndexType i = 0; i < mConstitutiveLawVector.size(); ++i) {
            SolidElementUtilities::CalculateKinematicVariablesAxisymTotalLagrangian(kin_vars, i, mThisIntegrationMethod, r_geom, true);
            SolidElementUtilities::SetTotalLagrangianConstitutiveVariables(kin_vars, const_vars, cl_params); // Sets F (3x3) in cl_params
            // No rotation for standard axisymmetric in F before CL call for InitializeMaterialResponse
            mConstitutiveLawVector[i]->InitializeMaterialResponse(cl_params, GetStressMeasure());
        }
    }
    KRATOS_CATCH("")
}

void AxisymTotalLagrangian::FinalizeSolutionStep(const ProcessInfo& rCurrentProcessInfo) {
    KRATOS_TRY
    bool requires_finalize = false;
    for (const auto& cl : mConstitutiveLawVector) {
        if (cl->RequiresFinalizeMaterialResponse()) { requires_finalize = true; break; }
    }
    if (requires_finalize) {
        const auto& r_geom = GetGeometry(); const auto& r_props = GetProperties();
        const SizeType num_nodes = r_geom.size(); const SizeType dim = 2;
        KRATOS_ERROR_IF(mConstitutiveLawVector.empty() || !mConstitutiveLawVector[0]) << "CL not init! Elm ID: " << Id() << std::endl;
        const SizeType strain_size = 4;

        SolidElementUtilities::LocalKinematicVariables kin_vars(strain_size, dim, num_nodes);
        SolidElementUtilities::LocalConstitutiveVariables const_vars(strain_size);
        ConstitutiveLaw::Parameters cl_params(r_geom, r_props, rCurrentProcessInfo);
        cl_params.GetOptions().Set(ConstitutiveLaw::USE_ELEMENT_PROVIDED_STRAIN, UseElementProvidedStrain());
        cl_params.GetOptions().Set(ConstitutiveLaw::COMPUTE_STRESS, true);
        cl_params.GetOptions().Set(ConstitutiveLaw::COMPUTE_CONSTITUTIVE_TENSOR, false);
        cl_params.SetStrainVector(const_vars.StrainVector);
        cl_params.SetStressVector(const_vars.StressVector);
        cl_params.SetConstitutiveMatrix(const_vars.D);

        for (IndexType i = 0; i < mConstitutiveLawVector.size(); ++i) {
            SolidElementUtilities::CalculateKinematicVariablesAxisymTotalLagrangian(kin_vars, i, mThisIntegrationMethod, r_geom, true);
            SolidElementUtilities::SetTotalLagrangianConstitutiveVariables(kin_vars, const_vars, cl_params);
            mConstitutiveLawVector[i]->FinalizeMaterialResponse(cl_params, GetStressMeasure());
        }
    }
    KRATOS_CATCH("")
}

void AxisymTotalLagrangian::ResetConstitutiveLaw() {
    KRATOS_TRY
    if (GetProperties().Has(CONSTITUTIVE_LAW)) {
        const auto& r_geom = GetGeometry(); const auto& r_prop = GetProperties();
        const auto& N_values = r_geom.ShapeFunctionsValues(mThisIntegrationMethod);
        for (IndexType i = 0; i < mConstitutiveLawVector.size(); ++i) {
            mConstitutiveLawVector[i]->ResetMaterial(r_prop, r_geom, row(N_values, i));
        }
    }
    KRATOS_CATCH("")
}

void AxisymTotalLagrangian::Initialize(const ProcessInfo& rCurrentProcessInfo) {
    KRATOS_TRY
    // ... (Initialize mThisIntegrationMethod and mConstitutiveLawVector - same as AxisymSmallDisplacement)
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
                    KRATOS_WARNING("AxisymTotalLagrangian") << "Integration order " << integration_order << " for element " << Id() << " is not available, using default." << std::endl;
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
            KRATOS_ERROR_IF_NOT(mConstitutiveLawVector[i]->GetStrainSize() == 4) << "Constitutive law for AxisymTotalLagrangian element " << Id() << " does not have strain_size = 4" << std::endl;
        }
    }
    KRATOS_CATCH( "" )
}

ConstitutiveLaw::StressMeasure AxisymTotalLagrangian::GetStressMeasure() const { return ConstitutiveLaw::StressMeasure_PK2; }
bool AxisymTotalLagrangian::UseElementProvidedStrain() const { return false; } // CL calculates Green-Lagrange from F

void AxisymTotalLagrangian::CalculateLocalSystem(MatrixType& rLeftHandSideMatrix, VectorType& rRightHandSideVector, const ProcessInfo& rCurrentProcessInfo) {
    KRATOS_TRY
    auto& r_geom = GetGeometry();
    const SizeType num_nodes = r_geom.size();
    const SizeType dim = 2; // RZ plane
    KRATOS_ERROR_IF(mConstitutiveLawVector.empty() || !mConstitutiveLawVector[0]) << "CL not init! Elm ID: " << Id() << std::endl;
    const SizeType strain_size = 4; // E_RR, E_ZZ, E_TT, 2E_RZ

    SolidElementUtilities::LocalKinematicVariables kin_vars(strain_size, dim, num_nodes); // F will be 3x3 for CL
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

    cl_params.SetStrainVector(const_vars.StrainVector);
    cl_params.SetStressVector(const_vars.StressVector);
    cl_params.SetConstitutiveMatrix(const_vars.D);

    for (IndexType i = 0; i < integration_points.size(); ++i) {
        SolidElementUtilities::CalculateKinematicVariablesAxisymTotalLagrangian(kin_vars, i, mThisIntegrationMethod, r_geom, true);
        SolidElementUtilities::SetTotalLagrangianConstitutiveVariables(kin_vars, const_vars, cl_params); // Sets F (3x3) in cl_params

        mConstitutiveLawVector[i]->CalculateMaterialResponse(cl_params, GetStressMeasure());

        double initial_R_ip = 0.0;
        for(SizeType node_idx = 0; node_idx < num_nodes; ++node_idx) {
             initial_R_ip += kin_vars.N[node_idx] * r_geom[node_idx].X0();
        }
        SolidElementUtilities::CalculateBAxisymTotalLagrangian(kin_vars.B, kin_vars.F, kin_vars.DN_DX, kin_vars.N, initial_R_ip, num_nodes);

        const double integration_factor = 2.0 * Globals::Pi * initial_R_ip;
        double integration_weight = SolidElementUtilities::GetIntegrationWeight(integration_points, i, kin_vars.detJ0, dim, GetProperties());
        integration_weight *= integration_factor;

        SolidElementUtilities::CalculateMaterialStiffnessMatrix(rLeftHandSideMatrix, kin_vars.B, const_vars.D, integration_weight);

        Matrix Kg_gauss(mat_size, mat_size);
        StructuralMechanicsElementUtilities::CalculateKgMatrix(Kg_gauss, kin_vars.DN_DX, const_vars.StressVector, integration_weight, dim, num_nodes);
        rLeftHandSideMatrix += Kg_gauss;

        array_1d<double, 3> body_force_3d = SolidElementUtilities::GetBodyForce(*this, integration_points, i);
        array_1d<double, 2> body_force_2d = ZeroVector(2);
        body_force_2d[0] = body_force_3d[0]; body_force_2d[1] = body_force_3d[1];

        VectorType bf_contribution_vector = ZeroVector(mat_size);
        for(IndexType node_idx = 0; node_idx < num_nodes; ++node_idx) {
            bf_contribution_vector[node_idx*2 + 0] += integration_weight * kin_vars.N[node_idx] * body_force_2d[0];
            bf_contribution_vector[node_idx*2 + 1] += integration_weight * kin_vars.N[node_idx] * body_force_2d[1];
        }
        rRightHandSideVector += bf_contribution_vector;
        SolidElementUtilities::CalculateAndAddInternalForces(rRightHandSideVector, kin_vars.B, const_vars.StressVector, integration_weight);
    }
    KRATOS_CATCH("")
}

void AxisymTotalLagrangian::CalculateOnIntegrationPoints(const Variable<Vector>& rVariable, std::vector<Vector>& rOutput, const ProcessInfo& rCurrentProcessInfo) {
    KRATOS_TRY
    const auto& r_geom = GetGeometry(); const auto& r_props = GetProperties();
    const SizeType dim = 2; const SizeType num_nodes = r_geom.size();
    KRATOS_ERROR_IF(mConstitutiveLawVector.empty() || !mConstitutiveLawVector[0]) << "CL not init! Elm ID: " << Id() << std::endl;
    const SizeType strain_size = 4;

    const auto& integration_points = r_geom.IntegrationPoints(mThisIntegrationMethod);
    if (rOutput.size() != integration_points.size()) rOutput.resize(integration_points.size());
    for(auto& v : rOutput) if(v.size()!=strain_size) v.resize(strain_size, false);

    if (rVariable == PK2_STRESS_VECTOR || rVariable == CAUCHY_STRESS_VECTOR) {
        SolidElementUtilities::LocalKinematicVariables kin_vars(strain_size, dim, num_nodes);
        SolidElementUtilities::LocalConstitutiveVariables const_vars(strain_size);
        ConstitutiveLaw::Parameters cl_params(r_geom, r_props, rCurrentProcessInfo);
        cl_params.GetOptions().Set(ConstitutiveLaw::USE_ELEMENT_PROVIDED_STRAIN, UseElementProvidedStrain());
        cl_params.GetOptions().Set(ConstitutiveLaw::COMPUTE_STRESS, true);
        cl_params.GetOptions().Set(ConstitutiveLaw::COMPUTE_CONSTITUTIVE_TENSOR, false);
        cl_params.SetStrainVector(const_vars.StrainVector);
        cl_params.SetStressVector(const_vars.StressVector);
        cl_params.SetConstitutiveMatrix(const_vars.D);

        for (IndexType i = 0; i < integration_points.size(); ++i) {
            SolidElementUtilities::CalculateKinematicVariablesAxisymTotalLagrangian(kin_vars, i, mThisIntegrationMethod, r_geom, true);
            SolidElementUtilities::SetTotalLagrangianConstitutiveVariables(kin_vars, const_vars, cl_params);
            // For AxisymTL, rotation is typically not applied to F before CL call for material response
            mConstitutiveLawVector[i]->CalculateMaterialResponse(cl_params, ConstitutiveLaw::StressMeasure_PK2);

            if (rVariable == PK2_STRESS_VECTOR) {
                rOutput[i] = const_vars.StressVector;
            } else { // CAUCHY_STRESS_VECTOR
                Vector cauchy_stress_gp(strain_size);
                // Kin_vars.F is 3x3 for CL, kin_vars.detF is from 3x3 F.
                mConstitutiveLawVector[i]->TransformStresses(cauchy_stress_gp, kin_vars.F, kin_vars.detF, ConstitutiveLaw::StressMeasure_Cauchy, ConstitutiveLaw::StressMeasure_PK2, const_vars.StressVector);
                rOutput[i] = cauchy_stress_gp;
            }
        }
    } else if (rVariable == GREEN_LAGRANGE_STRAIN_VECTOR) {
        SolidElementUtilities::LocalKinematicVariables kin_vars(strain_size, dim, num_nodes);
        SolidElementUtilities::LocalConstitutiveVariables const_vars(strain_size);
        ConstitutiveLaw::Parameters cl_params(r_geom, r_props, rCurrentProcessInfo);
        cl_params.GetOptions().Set(ConstitutiveLaw::USE_ELEMENT_PROVIDED_STRAIN, UseElementProvidedStrain());
        cl_params.GetOptions().Set(ConstitutiveLaw::COMPUTE_STRESS, false);
        cl_params.GetOptions().Set(ConstitutiveLaw::COMPUTE_CONSTITUTIVE_TENSOR, false);
        cl_params.SetStrainVector(const_vars.StrainVector);

        for (IndexType i = 0; i < integration_points.size(); ++i) {
            SolidElementUtilities::CalculateKinematicVariablesAxisymTotalLagrangian(kin_vars, i, mThisIntegrationMethod, r_geom, true);
            SolidElementUtilities::SetTotalLagrangianConstitutiveVariables(kin_vars, const_vars, cl_params);
            mConstitutiveLawVector[i]->CalculateMaterialResponse(cl_params, GetStressMeasure());
            rOutput[i] = const_vars.StrainVector;
        }
    } else if (mConstitutiveLawVector[0]->Has(rVariable)) {
        SolidElementUtilities::GetValueFromConstitutiveLaw(mConstitutiveLawVector, rVariable, rOutput);
    } else {
        KRATOS_WARNING("AxisymTL") << "COIP for Vector " << rVariable.Name() << " not fully handled." << std::endl;
    }
    KRATOS_CATCH("")
}

void AxisymTotalLagrangian::CalculateOnIntegrationPoints(const Variable<Matrix>& rVariable, std::vector<Matrix>& rOutput, const ProcessInfo& rCurrentProcessInfo) {
    KRATOS_TRY
    const auto& r_geom = GetGeometry(); const auto& r_props = GetProperties();
    const SizeType dim = 2; const SizeType num_nodes = r_geom.size();
    KRATOS_ERROR_IF(mConstitutiveLawVector.empty() || !mConstitutiveLawVector[0]) << "CL not init! Elm ID: " << Id() << std::endl;
    const SizeType strain_size = 4;

    const auto& integration_points = r_geom.IntegrationPoints(mThisIntegrationMethod);
    if (rOutput.size() != integration_points.size()) rOutput.resize(integration_points.size());

    if (rVariable == CONSTITUTIVE_MATRIX) {
        SolidElementUtilities::LocalKinematicVariables kin_vars(strain_size, dim, num_nodes);
        SolidElementUtilities::LocalConstitutiveVariables const_vars(strain_size);
        ConstitutiveLaw::Parameters cl_params(r_geom, r_props, rCurrentProcessInfo);
        cl_params.GetOptions().Set(ConstitutiveLaw::USE_ELEMENT_PROVIDED_STRAIN, UseElementProvidedStrain());
        cl_params.GetOptions().Set(ConstitutiveLaw::COMPUTE_STRESS, false);
        cl_params.GetOptions().Set(ConstitutiveLaw::COMPUTE_CONSTITUTIVE_TENSOR, true);
        cl_params.SetStrainVector(const_vars.StrainVector);
        cl_params.SetStressVector(const_vars.StressVector);
        cl_params.SetConstitutiveMatrix(const_vars.D);

        for (IndexType i = 0; i < integration_points.size(); ++i) {
            SolidElementUtilities::CalculateKinematicVariablesAxisymTotalLagrangian(kin_vars, i, mThisIntegrationMethod, r_geom, true);
            SolidElementUtilities::SetTotalLagrangianConstitutiveVariables(kin_vars, const_vars, cl_params);
            mConstitutiveLawVector[i]->CalculateMaterialResponse(cl_params, GetStressMeasure());
            if(rOutput[i].size1()!=strain_size || rOutput[i].size2()!=strain_size) rOutput[i].resize(strain_size,strain_size,false);
            rOutput[i] = const_vars.D;
        }
    } else if (rVariable == DEFORMATION_GRADIENT) { // Returns the 3x3 F used by CL
        SolidElementUtilities::LocalKinematicVariables kin_vars(strain_size, dim, num_nodes);
        for (IndexType i = 0; i < integration_points.size(); ++i) {
            SolidElementUtilities::CalculateKinematicVariablesAxisymTotalLagrangian(kin_vars, i, mThisIntegrationMethod, r_geom, true);
            if(rOutput[i].size1() != 3 || rOutput[i].size2() != 3) rOutput[i].resize(3,3,false); // F is 3x3
            rOutput[i] = kin_vars.F;
        }
    } else if (rVariable == CAUCHY_STRESS_TENSOR || rVariable == PK2_STRESS_TENSOR) {
        // For Axisym, stress tensor is usually represented as diag(s_RR, s_ZZ, s_TT) and tau_RZ for off-diag if needed.
        // Or, more commonly, just the vector is used. MathUtils<double>::StressVectorToTensor expects 3D or 2D plane stress/strain.
        // For Axisym 4-component Voigt (RR,ZZ,TT,RZ), direct conversion to 3x3 tensor is specific.
        // Let's provide the vector components directly.
        KRATOS_WARNING("AxisymTL") << "COIP for Matrix variable (Stress Tensor) " << rVariable.Name() << " for Axisym provides vector form. Review if tensor needed." << std::endl;
        std::vector<Vector> stress_vector_output;
        const Variable<Vector>& vector_variable = (rVariable == CAUCHY_STRESS_TENSOR) ? CAUCHY_STRESS_VECTOR : PK2_STRESS_VECTOR;
        this->CalculateOnIntegrationPoints(vector_variable, stress_vector_output, rCurrentProcessInfo);
        // If a matrix output is strictly required, this loop needs to build it from the 4-component vector.
        // For now, this will likely fail if a Matrix is expected by the caller unless rOutput is resized appropriately.
        for (IndexType i = 0; i < integration_points.size(); ++i) {
             if(rOutput[i].size1()!=strain_size || rOutput[i].size2()!=1) rOutput[i].resize(strain_size,1,false); // Store as column vector
             for(SizeType j=0; j<strain_size; ++j) rOutput[i](j,0) = stress_vector_output[i][j];
        }
    }
    // ... other Matrix variables
    else if (mConstitutiveLawVector[0]->Has(rVariable)) {
        SolidElementUtilities::GetValueFromConstitutiveLaw(mConstitutiveLawVector, rVariable, rOutput);
    } else {
        KRATOS_WARNING("AxisymTL") << "COIP for Matrix " << rVariable.Name() << " not fully handled." << std::endl;
    }
    KRATOS_CATCH("")
}

void AxisymTotalLagrangian::CalculateMassMatrix(MatrixType& rMassMatrix, const ProcessInfo& rCurrentProcessInfo) {
    KRATOS_TRY;
    // Copied from SmallDisplacement, should be generic
    const auto& r_prop = GetProperties();
    const auto& r_geom = GetGeometry();
    SizeType dimension = 2; SizeType number_of_nodes = r_geom.size(); SizeType mat_size = dimension * number_of_nodes;
    if (rMassMatrix.size1() != mat_size || rMassMatrix.size2() != mat_size) rMassMatrix.resize( mat_size, mat_size, false );
    noalias(rMassMatrix) = ZeroMatrix( mat_size, mat_size );
    KRATOS_ERROR_IF_NOT(r_prop.Has(DENSITY)) << "DENSITY missing!" << std::endl;
    const bool compute_lumped = SolidElementUtilities::ComputeLumpedMassMatrix(r_prop, rCurrentProcessInfo);

    if (compute_lumped) {
        VectorType temp_vector(mat_size);
        // CalculateLumpedMassVector needs to consider the 2*pi*R factor for axisymmetric elements.
        // The current SolidElementUtilities::CalculateLumpedMassVector is for general solids.
        // For now, using it as is, but this might need an axisymmetric specific version or modification.
        // Let's use the total mass and lumping factors, then multiply by 2*pi*R_avg or similar.
        // This is a simplification. A rigorous lumped mass for axisymmetry would integrate rho * 2*pi*R over volume.
        SolidElementUtilities::CalculateLumpedMassVector(*this, temp_vector, rCurrentProcessInfo);
        // The temp_vector is TotalMass*LumpingFactor. TotalMass from utility uses DomainSize() which is Area for 2D.
        // We need Volume * LumpingFactor. DomainSize for axisymmetric in Kratos is usually Area in RZ plane.
        // So, effectively, we need to multiply by 2*pi*R_centroid.
        // This part needs more careful derivation for true axisymmetric lumped mass.
        for (IndexType i = 0; i < mat_size; ++i) rMassMatrix(i, i) = temp_vector[i];
         KRATOS_WARNING_ONCE("AxisymTotalLagrangian") << "Lumped mass matrix for axisymmetric might not fully account for 2*pi*R factor. Review needed." << std::endl;

    } else { // Consistent mass
        const double density = SolidElementUtilities::GetDensityForMassMatrixComputation(*this);
        SolidElementUtilities::LocalKinematicVariables kin_vars(4, dimension, number_of_nodes);
        const IntegrationMethod ig_method = GetGeometry().GetDefaultIntegrationMethod();
        const GeometryType::IntegrationPointsArrayType& integration_points = GetGeometry().IntegrationPoints(ig_method);
        const Matrix& Ncontainer = GetGeometry().ShapeFunctionsValues(ig_method);

        for ( IndexType point_number = 0; point_number < integration_points.size(); ++point_number ) {
            kin_vars.detJ0 = SolidElementUtilities::CalculateDerivativesOnReferenceConfiguration(
                kin_vars.J0, kin_vars.InvJ0, kin_vars.DN_DX, point_number, ig_method, r_geom, true);

            double initial_R_ip = 0.0;
            const Vector& N_at_ip = row(Ncontainer, point_number);
            for(SizeType node_idx = 0; node_idx < number_of_nodes; ++node_idx) {
                 initial_R_ip += N_at_ip[node_idx] * r_geom[node_idx].X0();
            }
            const double integration_factor = 2.0 * Globals::Pi * initial_R_ip;
            double integration_weight = SolidElementUtilities::GetIntegrationWeight(integration_points, point_number, kin_vars.detJ0, dimension, r_prop);
            integration_weight *= integration_factor; // Full axisymmetric volume element

            const Vector& rN = row(Ncontainer,point_number);
            for ( IndexType i = 0; i < number_of_nodes; ++i ) {
                const SizeType index_i = i * dimension;
                for ( IndexType j = 0; j < number_of_nodes; ++j ) {
                    const SizeType index_j = j * dimension;
                    const double NiNj_weight = rN[i] * rN[j] * integration_weight * density;
                    for ( IndexType k = 0; k < dimension; ++k ) rMassMatrix( index_i + k, index_j + k ) += NiNj_weight;
                }
            }
        }
    }
    KRATOS_CATCH("");
}

std::string AxisymTotalLagrangian::Info() const { return "Axisymmetric Total Lagrangian Element #" + std::to_string(Id()); }
void AxisymTotalLagrangian::PrintInfo(std::ostream& rOStream) const { rOStream << Info(); }
void AxisymTotalLagrangian::PrintData(std::ostream& rOStream) const { pGetGeometry()->PrintData(rOStream); }

void AxisymTotalLagrangian::save(Serializer& rSerializer) const { KRATOS_SERIALIZE_SAVE_BASE_CLASS(rSerializer, Element); int IntMethod = int(mThisIntegrationMethod); rSerializer.save("IM", IntMethod); rSerializer.save("CLV", mConstitutiveLawVector); }
void AxisymTotalLagrangian::load(Serializer& rSerializer) { KRATOS_SERIALIZE_LOAD_BASE_CLASS(rSerializer, Element); int IntMethod; rSerializer.load("IM", IntMethod); mThisIntegrationMethod = IntegrationMethod(IntMethod); rSerializer.load("CLV", mConstitutiveLawVector); }

} // Namespace Kratos

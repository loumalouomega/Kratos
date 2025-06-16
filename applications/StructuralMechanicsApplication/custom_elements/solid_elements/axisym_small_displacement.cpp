#include "axisym_small_displacement.h"
#include "utilities/math_utils.h"
#include "custom_utilities/constitutive_law_utilities.h"
#include "structural_mechanics_application_variables.h"
#include "utilities/atomic_utilities.h" // For AddExplicitContribution


namespace Kratos
{

AxisymSmallDisplacement::AxisymSmallDisplacement(IndexType NewId, GeometryType::Pointer pGeometry)
    : Element(NewId, pGeometry) {
    mThisIntegrationMethod = GetGeometry().GetDefaultIntegrationMethod();
}

AxisymSmallDisplacement::AxisymSmallDisplacement(IndexType NewId, GeometryType::Pointer pGeometry, PropertiesType::Pointer pProperties)
    : Element(NewId, pGeometry, pProperties) {
    mThisIntegrationMethod = GetGeometry().GetDefaultIntegrationMethod();
}

AxisymSmallDisplacement::~AxisymSmallDisplacement() {}

Element::Pointer AxisymSmallDisplacement::Create(IndexType NewId, NodesArrayType const& ThisNodes, PropertiesType::Pointer pProperties) const {
    return Kratos::make_intrusive<AxisymSmallDisplacement>(NewId, GetGeometry().Create(ThisNodes), pProperties);
}

Element::Pointer AxisymSmallDisplacement::Create(IndexType NewId, GeometryType::Pointer pGeom, PropertiesType::Pointer pProperties) const {
    return Kratos::make_intrusive<AxisymSmallDisplacement>(NewId, pGeom, pProperties);
}

Element::Pointer AxisymSmallDisplacement::Clone(IndexType NewId, NodesArrayType const& rThisNodes) const {
    KRATOS_TRY
    auto p_new_elem = Kratos::make_intrusive<AxisymSmallDisplacement>(NewId, GetGeometry().Create(rThisNodes), pGetProperties());
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

void AxisymSmallDisplacement::Initialize(const ProcessInfo& rCurrentProcessInfo) {
    KRATOS_TRY
    // Copied from SmallDisplacement::Initialize
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
                    KRATOS_WARNING("AxisymSmallDisplacement") << "Integration order " << integration_order << " for element " << Id() << " is not available, using default." << std::endl;
                    mThisIntegrationMethod = GetGeometry().GetDefaultIntegrationMethod(); break;
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
            KRATOS_ERROR_IF_NOT(mConstitutiveLawVector[i]->GetStrainSize() == 4) << "Constitutive law for AxisymSmallDisplacement element " << Id() << " does not have strain_size = 4" << std::endl;
        }
    }
    KRATOS_CATCH( "" )
}

bool AxisymSmallDisplacement::UseElementProvidedStrain() const { return true; } // Strain = B*u
ConstitutiveLaw::StressMeasure AxisymSmallDisplacement::GetStressMeasure() const { return ConstitutiveLaw::StressMeasure_PK2; } // Or Cauchy for small disp


void AxisymSmallDisplacement::CalculateLocalSystem(MatrixType& rLeftHandSideMatrix, VectorType& rRightHandSideVector, const ProcessInfo& rCurrentProcessInfo) {
    KRATOS_TRY
    auto& r_geom = GetGeometry();
    const SizeType num_nodes = r_geom.size();
    const SizeType dim = 2; // R, Z for axisymmetric
    KRATOS_ERROR_IF(mConstitutiveLawVector.empty() || !mConstitutiveLawVector[0]) << "CL not init! Elm ID: " << Id() << std::endl;
    const SizeType strain_size = 4; // e_RR, e_ZZ, e_TT, gamma_RZ

    // Kinematic variables for axisymmetry (B is 4 x N*2)
    SolidElementUtilities::LocalKinematicVariables kin_vars(strain_size, dim, num_nodes);
    SolidElementUtilities::LocalConstitutiveVariables const_vars(strain_size);
    GetValuesVector(kin_vars.Displacements); // uR, uZ for each node

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

    cl_params.SetStrainVector(const_vars.StrainVector); // CL reads from this
    cl_params.SetStressVector(const_vars.StressVector); // CL writes here
    cl_params.SetConstitutiveMatrix(const_vars.D);      // CL writes here

    for (IndexType i = 0; i < integration_points.size(); ++i) {
        SolidElementUtilities::CalculateKinematicVariablesAxisymSmallDisplacement(
            kin_vars, i, mThisIntegrationMethod, r_geom, kin_vars.Displacements, true);

        noalias(const_vars.StrainVector) = prod(kin_vars.B, kin_vars.Displacements);
        cl_params.SetStrainVector(const_vars.StrainVector);

        cl_params.SetShapeFunctionsValues(kin_vars.N);
        cl_params.SetDeterminantF(kin_vars.detF);
        cl_params.SetDeformationGradientF(kin_vars.F);

        mConstitutiveLawVector[i]->CalculateMaterialResponse(cl_params, GetStressMeasure());

        Point global_coords_point; // Used for current radius calculation
        r_geom.GlobalCoordinates(global_coords_point, integration_points[i]);
        const double current_radius = global_coords_point.X();
        const double integration_factor = 2.0 * Globals::Pi * current_radius;

        double integration_weight = SolidElementUtilities::GetIntegrationWeight(integration_points, i, kin_vars.detJ0, dim, GetProperties());
        integration_weight *= integration_factor;

        SolidElementUtilities::CalculateMaterialStiffnessMatrix(rLeftHandSideMatrix, kin_vars.B, const_vars.D, integration_weight);

        array_1d<double, 3> body_force_3d = SolidElementUtilities::GetBodyForce(*this, integration_points, i);
        // For axisymmetric, body force in Z (axial) and R (radial) directions.
        // Assuming body_force_3d[0] is radial (X) and body_force_3d[1] is axial (Y/Z).
        Vector bf_contribution_vector = ZeroVector(mat_size);
        for(IndexType node_idx = 0; node_idx < num_nodes; ++node_idx) {
            bf_contribution_vector[node_idx*2 + 0] += integration_weight * kin_vars.N[node_idx] * body_force_3d[0]; // Radial
            bf_contribution_vector[node_idx*2 + 1] += integration_weight * kin_vars.N[node_idx] * body_force_3d[1]; // Axial
        }
        rRightHandSideVector += bf_contribution_vector;
        SolidElementUtilities::CalculateAndAddInternalForces(rRightHandSideVector, kin_vars.B, const_vars.StressVector, integration_weight);
    }
    KRATOS_CATCH("")
}

void AxisymSmallDisplacement::ResetConstitutiveLaw() {
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

void AxisymSmallDisplacement::InitializeSolutionStep(const ProcessInfo& rCurrentProcessInfo) { /* Similar to SmallDisplacement/TotalLagrangian */ }
void AxisymSmallDisplacement::FinalizeSolutionStep(const ProcessInfo& rCurrentProcessInfo) { /* Similar to SmallDisplacement/TotalLagrangian */ }
void AxisymSmallDisplacement::InitializeNonLinearIteration(const ProcessInfo& rCurrentProcessInfo) { /* Usually empty or CL call */ }
void AxisymSmallDisplacement::FinalizeNonLinearIteration(const ProcessInfo& rCurrentProcessInfo) { /* Usually empty or CL call */ }

void AxisymSmallDisplacement::CalculateLeftHandSide(MatrixType& rLeftHandSideMatrix, const ProcessInfo& rCurrentProcessInfo) {
    VectorType temp_rhs; CalculateLocalSystem(rLeftHandSideMatrix, temp_rhs, rCurrentProcessInfo);
}
void AxisymSmallDisplacement::CalculateRightHandSide(VectorType& rRightHandSideVector, const ProcessInfo& rCurrentProcessInfo) {
    MatrixType temp_lhs; CalculateLocalSystem(temp_lhs, rRightHandSideVector, rCurrentProcessInfo);
}
void AxisymSmallDisplacement::CalculateMassMatrix(MatrixType& rMassMatrix, const ProcessInfo& rCurrentProcessInfo) {
    // Adapt from SmallDisplacement, ensure 2*pi*R factor for integration weight
    KRATOS_TRY;
    const auto& r_prop = GetProperties();
    const auto& r_geom = GetGeometry();
    SizeType dimension = 2; // Axisymmetric
    SizeType number_of_nodes = r_geom.size();
    SizeType mat_size = dimension * number_of_nodes;

    if (rMassMatrix.size1() != mat_size || rMassMatrix.size2() != mat_size) rMassMatrix.resize( mat_size, mat_size, false );
    noalias(rMassMatrix) = ZeroMatrix( mat_size, mat_size );

    KRATOS_ERROR_IF_NOT(r_prop.Has(DENSITY)) << "DENSITY missing!" << std::endl;
    const bool compute_lumped = SolidElementUtilities::ComputeLumpedMassMatrix(r_prop, rCurrentProcessInfo);

    if (compute_lumped) {
        VectorType temp_vector(mat_size);
        // CalculateLumpedMassVector needs to be aware of axisymmetric integration factor
        // For now, assume base utility handles it or needs an axisymmetric specific version.
        // A simplified approach for lumped mass:
        const double density = SolidElementUtilities::GetDensityForMassMatrixComputation(*this);
        double total_volume = 0.0;
        const auto& integration_points = r_geom.IntegrationPoints(mThisIntegrationMethod);
         SolidElementUtilities::LocalKinematicVariables kin_vars(4, dimension, number_of_nodes);
        for(IndexType i=0; i < integration_points.size(); ++i) {
            kin_vars.detJ0 = SolidElementUtilities::CalculateDerivativesOnReferenceConfiguration(
                kin_vars.J0, kin_vars.InvJ0, kin_vars.DN_DX, i, mThisIntegrationMethod, r_geom, true);
            Point global_coords_point;
            r_geom.GlobalCoordinates(global_coords_point, integration_points[i]);
            const double current_radius = global_coords_point.X();
            total_volume += integration_points[i].Weight() * kin_vars.detJ0 * 2.0 * Globals::Pi * current_radius;
        }
        const double total_mass = total_volume * density;
        Vector lumping_factors = r_geom.LumpingFactors();
        for(IndexType k=0; k<number_of_nodes; ++k) {
            double temp_mass = lumping_factors[k] * total_mass;
            rMassMatrix(k*dimension, k*dimension) = temp_mass;
            rMassMatrix(k*dimension+1, k*dimension+1) = temp_mass;
        }

    } else { // Consistent mass
        const double density = SolidElementUtilities::GetDensityForMassMatrixComputation(*this);
        SolidElementUtilities::LocalKinematicVariables kin_vars(4, dimension, number_of_nodes);
        const IntegrationMethod ig_method = GetGeometry().GetDefaultIntegrationMethod();
        const GeometryType::IntegrationPointsArrayType& integration_points = GetGeometry().IntegrationPoints(ig_method);
        const Matrix& Ncontainer = GetGeometry().ShapeFunctionsValues(ig_method);

        for ( IndexType point_number = 0; point_number < integration_points.size(); ++point_number ) {
            kin_vars.detJ0 = SolidElementUtilities::CalculateDerivativesOnReferenceConfiguration(
                kin_vars.J0, kin_vars.InvJ0, kin_vars.DN_DX, point_number, ig_method, r_geom, true);

            Point global_coords_point;
            r_geom.GlobalCoordinates(global_coords_point, integration_points[point_number]);
            const double current_radius = global_coords_point.X();
            const double integration_factor = 2.0 * Globals::Pi * current_radius;
            const double integration_weight = integration_points[point_number].Weight() * kin_vars.detJ0 * integration_factor;

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
void AxisymSmallDisplacement::CalculateDampingMatrix(MatrixType& rDampingMatrix, const ProcessInfo& rCurrentProcessInfo) {
    const unsigned int mat_size = GetGeometry().PointsNumber() * 2; // dim = 2
    SolidElementUtilities::CalculateRayleighDampingMatrix(*this, rDampingMatrix, rCurrentProcessInfo, mat_size, mConstitutiveLawVector, mThisIntegrationMethod);
}

void AxisymSmallDisplacement::EquationIdVector(EquationIdVectorType& rResult, const ProcessInfo& rCurrentProcessInfo) const {
    const auto& r_geom = GetGeometry();
    const SizeType num_nodes = r_geom.size();
    const SizeType dim = 2; // Axisymmetric
    if (rResult.size() != dim * num_nodes) rResult.resize(dim * num_nodes, false);
    const SizeType pos = r_geom[0].GetDofPosition(DISPLACEMENT_X);
    for (IndexType i = 0; i < num_nodes; ++i) {
        const SizeType index = i * dim;
        rResult[index] = r_geom[i].GetDof(DISPLACEMENT_X, pos).EquationId();      // R-disp
        rResult[index + 1] = r_geom[i].GetDof(DISPLACEMENT_Y, pos + 1).EquationId(); // Z-disp (assuming Y is Z in 2D model)
    }
}
void AxisymSmallDisplacement::GetDofList(DofsVectorType& rElementalDofList, const ProcessInfo& rCurrentProcessInfo) const {
    const auto& r_geom = GetGeometry();
    const SizeType num_nodes = r_geom.size();
    const SizeType dim = 2; // Axisymmetric
    rElementalDofList.resize(0);
    rElementalDofList.reserve(dim * num_nodes);
    for (IndexType i = 0; i < num_nodes; ++i) {
        rElementalDofList.push_back(r_geom[i].pGetDof(DISPLACEMENT_X)); // R-disp
        rElementalDofList.push_back(r_geom[i].pGetDof(DISPLACEMENT_Y)); // Z-disp
    }
}
void AxisymSmallDisplacement::GetValuesVector(Vector& rValues, int Step) const {
    const auto& r_geom = GetGeometry();
    const SizeType num_nodes = r_geom.size();
    const SizeType dim = 2; // Axisymmetric
    const SizeType mat_size = num_nodes * dim;
    if (rValues.size() != mat_size) rValues.resize(mat_size, false);
    for (IndexType i = 0; i < num_nodes; ++i) {
        const array_1d<double, 3>& displacement = r_geom[i].FastGetSolutionStepValue(DISPLACEMENT, Step);
        const SizeType index = i * dim;
        rValues[index] = displacement[0];     // R-disp
        rValues[index + 1] = displacement[1]; // Z-disp (assuming Y is Z)
    }
}
void AxisymSmallDisplacement::GetFirstDerivativesVector(Vector& rValues, int Step) const {
    const auto& r_geom = GetGeometry();
    const SizeType num_nodes = r_geom.size();
    const SizeType dim = 2;
    const SizeType mat_size = num_nodes * dim;
    if (rValues.size() != mat_size) rValues.resize(mat_size, false);
    for (IndexType i = 0; i < num_nodes; ++i) {
        const array_1d<double, 3 >& velocity = r_geom[i].FastGetSolutionStepValue(VELOCITY, Step);
        const SizeType index = i * dim;
        rValues[index] = velocity[0];
        rValues[index + 1] = velocity[1];
    }
}
void AxisymSmallDisplacement::GetSecondDerivativesVector(Vector& rValues, int Step) const {
     const auto& r_geom = GetGeometry();
    const SizeType num_nodes = r_geom.size();
    const SizeType dim = 2;
    const SizeType mat_size = num_nodes * dim;
    if (rValues.size() != mat_size) rValues.resize(mat_size, false);
    for (IndexType i = 0; i < num_nodes; ++i) {
        const array_1d<double, 3 >& acceleration = r_geom[i].FastGetSolutionStepValue(ACCELERATION, Step);
        const SizeType index = i * dim;
        rValues[index] = acceleration[0];
        rValues[index + 1] = acceleration[1];
    }
}
void AxisymSmallDisplacement::AddExplicitContribution(const VectorType& rRHSVector, const Variable<VectorType>& rRHSVariable, const Variable<double>& rDestinationVariable, const ProcessInfo& rCurrentProcessInfo) {
    // Copied from SmallDisplacement
    KRATOS_TRY;
    auto& r_geom = this->GetGeometry();
    const SizeType dimension = 2; // Axisymmetric
    const SizeType number_of_nodes = r_geom.size();
     if (rDestinationVariable == NODAL_MASS ) { // Axisymmetric mass is scalar per node
        VectorType element_mass_vector(number_of_nodes * dimension); // Temp vector for full matrix
        MatrixType mass_matrix;
        this->CalculateMassMatrix(mass_matrix, rCurrentProcessInfo); // Use consistent or lumped as per props
        for (IndexType i = 0; i < number_of_nodes; ++i) {
            double nodal_mass = 0.0;
            for(SizeType d=0; d<dimension; ++d) nodal_mass += mass_matrix(i*dimension+d, i*dimension+d); // Sum diagonal entries for node i
            AtomicAdd(r_geom[i].GetValue(NODAL_MASS), nodal_mass/dimension); // Average or just take one component if lumped. This is a simplification.
                                                                              // Proper lumped mass for scalar NODAL_MASS would be total_mass * lumping_factor[i]
        }
    }
    KRATOS_CATCH("")
}
void AxisymSmallDisplacement::AddExplicitContribution(const VectorType& rRHSVector, const Variable<VectorType>& rRHSVariable, const Variable<array_1d<double,3>>& rDestinationVariable, const ProcessInfo& rCurrentProcessInfo) {
    // Copied from SmallDisplacement, ensure dim=2 for axisymmetric force residual
    KRATOS_TRY;
    auto& r_geom = this->GetGeometry();
    const auto& r_prop = this->GetProperties();
    const SizeType dimension = 2; // Axisymmetric
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
            // FORCE_RESIDUAL is array_1d<double,3>, but for axisym we only use 2 components
            AtomicAdd(r_force_residual[0], (rRHSVector[index + 0] - damping_residual_contribution[index + 0])); // R-component
            AtomicAdd(r_force_residual[1], (rRHSVector[index + 1] - damping_residual_contribution[index + 1])); // Z-component
        }
    }
    KRATOS_CATCH("")
}


int AxisymSmallDisplacement::Check(const ProcessInfo& rCurrentProcessInfo) const {
    KRATOS_TRY
    int check = Element::Check(rCurrentProcessInfo);
    check = SolidElementUtilities::SolidElementCheck(*this, rCurrentProcessInfo, mConstitutiveLawVector);
    KRATOS_ERROR_IF(GetGeometry().WorkingSpaceDimension() != 2) << "Axisymmetric element " << Id() << " expects 2D geometry (RZ plane)." << std::endl;
    for(const auto& cl : mConstitutiveLawVector) {
        KRATOS_ERROR_IF_NOT(cl) << "Constitutive law is null for element " << Id() << std::endl;
        KRATOS_ERROR_IF(cl->GetStrainSize() != 4) << "Constitutive law for Axisymmetric element " << Id() << " expects strain size 4. Provided: " << cl->GetStrainSize() << std::endl;
    }
    return check;
    KRATOS_CATCH("")
}
const Parameters AxisymSmallDisplacement::GetSpecifications() const {
    Parameters specs = SolidElementUtilities::GetDefaultSolidSpecifications(2); // 2D for nodal DOFs
    specs["compatible_constitutive_laws"]["strain_size"].SetIntArray({4}); // Specifically for axisymmetric CLs
    specs["documentation"].SetString("Axisymmetric small displacement element using R-Z plane for nodal DOFs. Strain order: e_RR, e_ZZ, e_TT (hoop), gamma_RZ. Stress order: s_RR, s_ZZ, s_TT (hoop), tau_RZ.");
    return specs;
}
void AxisymSmallDisplacement::CalculateOnIntegrationPoints(const Variable<double>& rVariable, std::vector<double>& rOutput, const ProcessInfo& rCurrentProcessInfo) {
    // Adapt from SmallDisp, use Axisym Kinematics. Strain size is 4.
    // Von Mises for axisymmetric needs specific handling for stress vector [sRR, sZZ, sTT, tRZ]
    // The utility CalculateValueOnConstitutiveLaw needs to be made aware of axisymmetric kinematics if used.
    // For now, direct implementation or specific utility call is safer.
    if (mConstitutiveLawVector[0]->Has(rVariable)){
        SolidElementUtilities::GetValueFromConstitutiveLaw(mConstitutiveLawVector, rVariable, rOutput);
    } else {
         KRATOS_WARNING("AxisymSmallDisplacement") << "COIP for double " << rVariable.Name() << " not fully implemented or CL does not have it." << std::endl;
        // SolidElementUtilities::CalculateValueOnConstitutiveLaw(*this, mConstitutiveLawVector, rVariable, rCurrentProcessInfo, mThisIntegrationMethod, UseElementProvidedStrain(), GetStressMeasure(), rOutput);
    }
}
void AxisymSmallDisplacement::CalculateOnIntegrationPoints(const Variable<Vector>& rVariable, std::vector<Vector>& rOutput, const ProcessInfo& rCurrentProcessInfo) {
    // Adapt for PK2_STRESS_VECTOR (size 4), GREEN_LAGRANGE_STRAIN_VECTOR (size 4)
    if (mConstitutiveLawVector[0]->Has(rVariable)){
        SolidElementUtilities::GetValueFromConstitutiveLaw(mConstitutiveLawVector, rVariable, rOutput);
    } else {
        KRATOS_WARNING("AxisymSmallDisplacement") << "COIP for Vector " << rVariable.Name() << " not fully implemented or CL does not have it." << std::endl;
    }
}
void AxisymSmallDisplacement::CalculateOnIntegrationPoints(const Variable<Matrix>& rVariable, std::vector<Matrix>& rOutput, const ProcessInfo& rCurrentProcessInfo) {
    if (mConstitutiveLawVector[0]->Has(rVariable)){
        SolidElementUtilities::GetValueFromConstitutiveLaw(mConstitutiveLawVector, rVariable, rOutput);
    } else {
        KRATOS_WARNING("AxisymSmallDisplacement") << "COIP for Matrix " << rVariable.Name() << " not fully implemented or CL does not have it." << std::endl;
    }
}
void AxisymSmallDisplacement::CalculateOnIntegrationPoints(const Variable<array_1d<double,3>>& rVariable, std::vector<array_1d<double,3>>& rOutput, const ProcessInfo& rCurrentProcessInfo) {
    if (rVariable == INTEGRATION_COORDINATES) {
        const auto& r_geom = GetGeometry();
        const GeometryType::IntegrationPointsArrayType& integration_points = r_geom.IntegrationPoints(mThisIntegrationMethod);
        if (rOutput.size() != integration_points.size()) rOutput.resize(integration_points.size());
        for (IndexType i = 0; i < integration_points.size(); ++i) {
            Point global_point;
            r_geom.GlobalCoordinates(global_point, integration_points[i]);
            noalias(rOutput[i]) = global_point.Coordinates(); // Global R,Z,0 (if input point is 2D)
        }
    } else if (mConstitutiveLawVector[0]->Has(rVariable)){
        SolidElementUtilities::GetValueFromConstitutiveLaw(mConstitutiveLawVector, rVariable, rOutput);
    } else {
        KRATOS_WARNING("AxisymSmallDisplacement") << "COIP for array_1d<double,3> " << rVariable.Name() << " not fully implemented or CL does not have it." << std::endl;
    }
}

// SetValuesOnIntegrationPoints (copied from SmallDisplacement, should be mostly generic)
void AxisymSmallDisplacement::SetValuesOnIntegrationPoints(const Variable<double>& rVariable, const std::vector<double>& rValues, const ProcessInfo& rCurrentProcessInfo)
{ KRATOS_TRY if(!mConstitutiveLawVector.empty() && mConstitutiveLawVector[0]->Has(rVariable)) for(IndexType i=0; i<mConstitutiveLawVector.size(); ++i) mConstitutiveLawVector[i]->SetValue(rVariable, rValues[i], rCurrentProcessInfo); else KRATOS_WARNING("ASD")<<"Set for double "<<rVariable.Name()<<" not in CL"<<std::endl; KRATOS_CATCH("") }
void AxisymSmallDisplacement::SetValuesOnIntegrationPoints(const Variable<Vector>& rVariable, const std::vector<Vector>& rValues, const ProcessInfo& rCurrentProcessInfo)
{ KRATOS_TRY if(!mConstitutiveLawVector.empty() && mConstitutiveLawVector[0]->Has(rVariable)) for(IndexType i=0; i<mConstitutiveLawVector.size(); ++i) mConstitutiveLawVector[i]->SetValue(rVariable, rValues[i], rCurrentProcessInfo); else KRATOS_WARNING("ASD")<<"Set for Vector "<<rVariable.Name()<<" not in CL"<<std::endl; KRATOS_CATCH("") }
void AxisymSmallDisplacement::SetValuesOnIntegrationPoints(const Variable<Matrix>& rVariable, const std::vector<Matrix>& rValues, const ProcessInfo& rCurrentProcessInfo)
{ KRATOS_TRY if(!mConstitutiveLawVector.empty() && mConstitutiveLawVector[0]->Has(rVariable)) for(IndexType i=0; i<mConstitutiveLawVector.size(); ++i) mConstitutiveLawVector[i]->SetValue(rVariable, rValues[i], rCurrentProcessInfo); else KRATOS_WARNING("ASD")<<"Set for Matrix "<<rVariable.Name()<<" not in CL"<<std::endl; KRATOS_CATCH("") }
void AxisymSmallDisplacement::SetValuesOnIntegrationPoints(const Variable<array_1d<double,3>>& rVariable, const std::vector<array_1d<double,3>>& rValues, const ProcessInfo& rCurrentProcessInfo)
{ KRATOS_TRY if(!mConstitutiveLawVector.empty() && mConstitutiveLawVector[0]->Has(rVariable)) for(IndexType i=0; i<mConstitutiveLawVector.size(); ++i) mConstitutiveLawVector[i]->SetValue(rVariable, rValues[i], rCurrentProcessInfo); else KRATOS_WARNING("ASD")<<"Set for array_1d<double,3> "<<rVariable.Name()<<" not in CL"<<std::endl; KRATOS_CATCH("") }
void AxisymSmallDisplacement::SetValuesOnIntegrationPoints(const Variable<ConstitutiveLaw::Pointer>& rVariable, const std::vector<ConstitutiveLaw::Pointer>& rValues, const ProcessInfo& rCurrentProcessInfo)
{ if (rVariable == CONSTITUTIVE_LAW) { KRATOS_ERROR_IF(rValues.size() != mConstitutiveLawVector.size()) << "CL vector size mismatch." << std::endl; mConstitutiveLawVector = rValues; } else { KRATOS_WARNING("ASD") << "Set for CL Ptr var " << rVariable.Name() << " not CONSTITUTIVE_LAW." << std::endl;}}
void AxisymSmallDisplacement::SetValuesOnIntegrationPoints(const Variable<bool>& rVariable, const std::vector<bool>& rValues, const ProcessInfo& rCurrentProcessInfo)
{ KRATOS_TRY if(!mConstitutiveLawVector.empty() && mConstitutiveLawVector[0]->Has(rVariable)) for(IndexType i=0; i<mConstitutiveLawVector.size(); ++i) mConstitutiveLawVector[i]->SetValue(rVariable, rValues[i], rCurrentProcessInfo); else KRATOS_WARNING("ASD")<<"Set for bool "<<rVariable.Name()<<" not in CL"<<std::endl; KRATOS_CATCH("") }
void AxisymSmallDisplacement::SetValuesOnIntegrationPoints(const Variable<int>& rVariable, const std::vector<int>& rValues, const ProcessInfo& rCurrentProcessInfo)
{ KRATOS_TRY if(!mConstitutiveLawVector.empty() && mConstitutiveLawVector[0]->Has(rVariable)) for(IndexType i=0; i<mConstitutiveLawVector.size(); ++i) mConstitutiveLawVector[i]->SetValue(rVariable, rValues[i], rCurrentProcessInfo); else KRATOS_WARNING("ASD")<<"Set for int "<<rVariable.Name()<<" not in CL"<<std::endl; KRATOS_CATCH("") }
void AxisymSmallDisplacement::SetValuesOnIntegrationPoints(const Variable<array_1d<double,6>>& rVariable, const std::vector<array_1d<double,6>>& rValues, const ProcessInfo& rCurrentProcessInfo)
{ KRATOS_TRY if(!mConstitutiveLawVector.empty() && mConstitutiveLawVector[0]->Has(rVariable)) for(IndexType i=0; i<mConstitutiveLawVector.size(); ++i) mConstitutiveLawVector[i]->SetValue(rVariable, rValues[i], rCurrentProcessInfo); else KRATOS_WARNING("ASD")<<"Set for array_1d<double,6> "<<rVariable.Name()<<" not in CL"<<std::endl; KRATOS_CATCH("") }


std::string AxisymSmallDisplacement::Info() const { return "Axisymmetric Small Displacement Element #" + std::to_string(Id()); }
void AxisymSmallDisplacement::PrintInfo(std::ostream& rOStream) const { rOStream << Info(); }
void AxisymSmallDisplacement::PrintData(std::ostream& rOStream) const { pGetGeometry()->PrintData(rOStream); }


void AxisymSmallDisplacement::save(Serializer& rSerializer) const {
    KRATOS_SERIALIZE_SAVE_BASE_CLASS(rSerializer, Element);
    int IntMethod = int(mThisIntegrationMethod);
    rSerializer.save("IntegrationMethod", IntMethod);
    rSerializer.save("ConstitutiveLawVector", mConstitutiveLawVector);
}
void AxisymSmallDisplacement::load(Serializer& rSerializer) {
    KRATOS_SERIALIZE_LOAD_BASE_CLASS(rSerializer, Element);
    int IntMethod;
    rSerializer.load("IntegrationMethod", IntMethod);
    mThisIntegrationMethod = IntegrationMethod(IntMethod);
    rSerializer.load("ConstitutiveLawVector", mConstitutiveLawVector);
}

} // Namespace Kratos

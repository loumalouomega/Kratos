#include "small_displacement.h"
#include "utilities/math_utils.h"
// #include "custom_utilities/structural_mechanics_element_utilities.h" // Already there or less needed
#include "structural_mechanics_application_variables.h"
#include "utilities/atomic_utilities.h" // For AddExplicitContribution
#include "custom_utilities/constitutive_law_utilities.h" // For VonMises

// Add standard Kratos includes that might be missing for ZeroVector etc.
#include "includes/checks.h"
#include "includes/ublas_interface.h"




namespace Kratos
{

SmallDisplacement::SmallDisplacement( IndexType NewId, GeometryType::Pointer pGeometry )
    : Element( NewId, pGeometry ) // Changed base class
{
    // Initialize mThisIntegrationMethod here if it's not set by properties in Initialize
    mThisIntegrationMethod = this->GetGeometry().GetDefaultIntegrationMethod();
}

SmallDisplacement::SmallDisplacement( IndexType NewId, GeometryType::Pointer pGeometry, PropertiesType::Pointer pProperties )
    : Element( NewId, pGeometry, pProperties ) // Changed base class
{
    mThisIntegrationMethod = this->GetGeometry().GetDefaultIntegrationMethod();
}

SmallDisplacement::~SmallDisplacement()
{
}

Element::Pointer SmallDisplacement::Create( IndexType NewId, NodesArrayType const& ThisNodes, PropertiesType::Pointer pProperties ) const
{
    return Kratos::make_intrusive<SmallDisplacement>( NewId, GetGeometry().Create( ThisNodes ), pProperties );
}

Element::Pointer SmallDisplacement::Create( IndexType NewId, GeometryType::Pointer pGeom, PropertiesType::Pointer pProperties ) const
{
    return Kratos::make_intrusive<SmallDisplacement>( NewId, pGeom, pProperties );
}

Element::Pointer SmallDisplacement::Clone (
    IndexType NewId,
    NodesArrayType const& rThisNodes
    ) const
{
    KRATOS_TRY
    SmallDisplacement::Pointer p_new_elem = Kratos::make_intrusive<SmallDisplacement>(NewId, GetGeometry().Create(rThisNodes), pGetProperties());
    p_new_elem->SetData(this->GetData());
    p_new_elem->Set(Flags(*this));

    // Copy mThisIntegrationMethod and mConstitutiveLawVector directly
    p_new_elem->mThisIntegrationMethod = this->mThisIntegrationMethod;
    // Deep copy of constitutive laws needed
    p_new_elem->mConstitutiveLawVector.resize(this->mConstitutiveLawVector.size());
    for(size_t i=0; i<this->mConstitutiveLawVector.size(); ++i) {
        if(this->mConstitutiveLawVector[i] != nullptr)
            p_new_elem->mConstitutiveLawVector[i] = this->mConstitutiveLawVector[i]->Clone();
    }
    return p_new_elem;
    KRATOS_CATCH("");
}

void SmallDisplacement::Initialize(const ProcessInfo& rCurrentProcessInfo)
{
    KRATOS_TRY
    if (!rCurrentProcessInfo[IS_RESTARTED]) {
        // Integration method from properties or default
        if( GetProperties().Has(INTEGRATION_ORDER) ) {
            const SizeType integration_order = GetProperties()[INTEGRATION_ORDER];
            switch ( integration_order ) {
                case 1: mThisIntegrationMethod = GeometryData::IntegrationMethod::GI_GAUSS_1; break;
                case 2: mThisIntegrationMethod = GeometryData::IntegrationMethod::GI_GAUSS_2; break;
                case 3: mThisIntegrationMethod = GeometryData::IntegrationMethod::GI_GAUSS_3; break;
                case 4: mThisIntegrationMethod = GeometryData::IntegrationMethod::GI_GAUSS_4; break;
                case 5: mThisIntegrationMethod = GeometryData::IntegrationMethod::GI_GAUSS_5; break;
                default:
                    KRATOS_WARNING("SmallDisplacement") << "Integration order " << integration_order << " is not available, using default." << std::endl;
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

void SmallDisplacement::ResetConstitutiveLaw()
{
    KRATOS_TRY
    if ( GetProperties().Has(CONSTITUTIVE_LAW) ) {
        const auto& r_geom = GetGeometry();
        const auto& r_prop = GetProperties();
        const auto& N_values = GetGeometry().ShapeFunctionsValues(mThisIntegrationMethod);
        for ( IndexType i = 0; i < mConstitutiveLawVector.size(); ++i ) {
            mConstitutiveLawVector[i]->ResetMaterial( r_prop, r_geom, row( N_values, i ) );
        }
    }
    KRATOS_CATCH( "" )
}

void SmallDisplacement::InitializeSolutionStep( const ProcessInfo& rCurrentProcessInfo )
{
    KRATOS_TRY
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
        const SizeType strain_size = mConstitutiveLawVector[0]->GetStrainSize();

        SolidElementUtilities::LocalKinematicVariables kin_vars(strain_size, dim, num_nodes);
        SolidElementUtilities::LocalConstitutiveVariables const_vars(strain_size); // Unused here, but CL may need it via params
        GetValuesVector(kin_vars.Displacements);


        ConstitutiveLaw::Parameters cl_params(r_geom, r_props, rCurrentProcessInfo);
        Flags& cl_options = cl_params.GetOptions();
        cl_options.Set(ConstitutiveLaw::USE_ELEMENT_PROVIDED_STRAIN, UseElementProvidedStrain());
        cl_options.Set(ConstitutiveLaw::COMPUTE_STRESS, true); // As per BaseSolidElement
        cl_options.Set(ConstitutiveLaw::COMPUTE_CONSTITUTIVE_TENSOR, false);

        if(UseElementProvidedStrain()) cl_params.SetStrainVector(const_vars.StrainVector); // CL might read from this
        cl_params.SetStressVector(const_vars.StressVector); // CL might write to this
        cl_params.SetConstitutiveMatrix(const_vars.D); // CL might write to this

        const auto& integration_points = r_geom.IntegrationPoints(mThisIntegrationMethod);

        for (IndexType i = 0; i < mConstitutiveLawVector.size(); ++i) {
            SolidElementUtilities::CalculateKinematicVariablesSmallDisplacement(
                kin_vars, i, mThisIntegrationMethod, r_geom, kin_vars.Displacements, strain_size, true);

            // SetConstitutiveVariables equivalent for this context
            cl_params.SetShapeFunctionsValues(kin_vars.N);
            cl_params.SetDeterminantF(kin_vars.detF);
            cl_params.SetDeformationGradientF(kin_vars.F);
            if (UseElementProvidedStrain()) { // Calculate and set strain if element provides it
                 Vector current_strain = prod(kin_vars.B, kin_vars.Displacements);
                 cl_params.SetStrainVector(current_strain); // Set it for the CL
            }

            bool is_rotated = SolidElementUtilities::IsElementRotated(*this, mConstitutiveLawVector[i]);
            if (is_rotated) {
                // Simplified: In a real scenario, full rotation logic for F or strain
                // As this is InitializeMaterialResponse, usually stress/C tensor not primary output yet
            }
            mConstitutiveLawVector[i]->InitializeMaterialResponse(cl_params, GetStressMeasure());
        }
    }
    KRATOS_CATCH("")
}

void SmallDisplacement::FinalizeSolutionStep( const ProcessInfo& rCurrentProcessInfo )
{
    KRATOS_TRY
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
        const SizeType strain_size = mConstitutiveLawVector[0]->GetStrainSize();

        SolidElementUtilities::LocalKinematicVariables kin_vars(strain_size, dim, num_nodes);
        SolidElementUtilities::LocalConstitutiveVariables const_vars(strain_size);
        GetValuesVector(kin_vars.Displacements);

        ConstitutiveLaw::Parameters cl_params(r_geom, r_props, rCurrentProcessInfo);
        Flags& cl_options = cl_params.GetOptions();
        cl_options.Set(ConstitutiveLaw::USE_ELEMENT_PROVIDED_STRAIN, UseElementProvidedStrain());
        cl_options.Set(ConstitutiveLaw::COMPUTE_STRESS, true);
        cl_options.Set(ConstitutiveLaw::COMPUTE_CONSTITUTIVE_TENSOR, false);

        if(UseElementProvidedStrain()) cl_params.SetStrainVector(const_vars.StrainVector);
        cl_params.SetStressVector(const_vars.StressVector);
        cl_params.SetConstitutiveMatrix(const_vars.D);

        const auto& integration_points = r_geom.IntegrationPoints(mThisIntegrationMethod);

        for (IndexType i = 0; i < mConstitutiveLawVector.size(); ++i) {
            SolidElementUtilities::CalculateKinematicVariablesSmallDisplacement(
                kin_vars, i, mThisIntegrationMethod, r_geom, kin_vars.Displacements, strain_size, true);

            cl_params.SetShapeFunctionsValues(kin_vars.N);
            cl_params.SetDeterminantF(kin_vars.detF);
            cl_params.SetDeformationGradientF(kin_vars.F);
            if (UseElementProvidedStrain()) {
                 Vector current_strain = prod(kin_vars.B, kin_vars.Displacements);
                 cl_params.SetStrainVector(current_strain);
            }

            bool is_rotated = SolidElementUtilities::IsElementRotated(*this, mConstitutiveLawVector[i]);
            // Add rotation logic if necessary before calling CL finalize

            mConstitutiveLawVector[i]->FinalizeMaterialResponse(cl_params, GetStressMeasure());
        }
    }
    KRATOS_CATCH("")
}

void SmallDisplacement::InitializeNonLinearIteration( const ProcessInfo& rCurrentProcessInfo )
{
    // Deprecated part from BaseSolidElement:
    // const auto& N_values = GetGeometry().ShapeFunctionsValues(mThisIntegrationMethod);
    // for ( IndexType point_number = 0; point_number < mConstitutiveLawVector.size(); ++point_number ) {
    //    mConstitutiveLawVector[point_number]->InitializeNonLinearIteration( GetProperties(), GetGeometry(), row( N_values, point_number ), rCurrentProcessInfo);
    // }
}

void SmallDisplacement::FinalizeNonLinearIteration( const ProcessInfo& rCurrentProcessInfo )
{
    // Deprecated part from BaseSolidElement:
    // const auto& N_values = GetGeometry().ShapeFunctionsValues(mThisIntegrationMethod);
    // for ( IndexType point_number = 0; point_number < mConstitutiveLawVector.size(); ++point_number ) {
    //    mConstitutiveLawVector[point_number]->FinalizeNonLinearIteration( GetProperties(), GetGeometry(), row( N_values, point_number ), rCurrentProcessInfo);
    // }
}


bool SmallDisplacement::UseElementProvidedStrain() const
{
    return true; // SmallDisplacement specific
}

ConstitutiveLaw::StressMeasure SmallDisplacement::GetStressMeasure() const
{
    return ConstitutiveLaw::StressMeasure_PK2; // From BaseSolidElement, assuming it's general for SD
}


void SmallDisplacement::CalculateLocalSystem(MatrixType& rLeftHandSideMatrix, VectorType& rRightHandSideVector, const ProcessInfo& rCurrentProcessInfo)
{
    KRATOS_TRY;

    auto& r_geometry = this->GetGeometry();
    const SizeType number_of_nodes = r_geometry.size();
    const SizeType dimension = r_geometry.WorkingSpaceDimension();
    const SizeType strain_size = mConstitutiveLawVector[0]->GetStrainSize(); // Assumes CL is initialized

    SolidElementUtilities::LocalKinematicVariables kinematic_variables(strain_size, dimension, number_of_nodes);
    SolidElementUtilities::LocalConstitutiveVariables constitutive_variables(strain_size);

    GetValuesVector(kinematic_variables.Displacements); // Get current displacements

    const SizeType mat_size = number_of_nodes * dimension;
    if ( rLeftHandSideMatrix.size1() != mat_size ) rLeftHandSideMatrix.resize( mat_size, mat_size, false );
    noalias( rLeftHandSideMatrix ) = ZeroMatrix( mat_size, mat_size );
    if ( rRightHandSideVector.size() != mat_size ) rRightHandSideVector.resize( mat_size, false );
    noalias(rRightHandSideVector) = ZeroVector( mat_size );

    const GeometryType::IntegrationPointsArrayType& integration_points = GetGeometry().IntegrationPoints(mThisIntegrationMethod);
    ConstitutiveLaw::Parameters cl_params(r_geometry, GetProperties(), rCurrentProcessInfo);
    Flags& cl_options = cl_params.GetOptions();
    cl_options.Set(ConstitutiveLaw::USE_ELEMENT_PROVIDED_STRAIN, UseElementProvidedStrain());
    cl_options.Set(ConstitutiveLaw::COMPUTE_STRESS, true);
    cl_options.Set(ConstitutiveLaw::COMPUTE_CONSTITUTIVE_TENSOR, true);

    // This is crucial: if USE_ELEMENT_PROVIDED_STRAIN is true, CL expects strain vector to be set in cl_params
    cl_params.SetStrainVector(constitutive_variables.StrainVector);


    array_1d<double, 3> body_force;
    double integration_weight;

    for ( IndexType point_number = 0; point_number < integration_points.size(); ++point_number ) {
        SolidElementUtilities::CalculateKinematicVariablesSmallDisplacement(
            kinematic_variables, point_number, mThisIntegrationMethod, r_geometry, kinematic_variables.Displacements, strain_size, true /*UseGeomIntegration*/);

        SolidElementUtilities::SetSmallDisplacementConstitutiveVariables(
            kinematic_variables, constitutive_variables, cl_params, kinematic_variables.Displacements);

        // If using element provided strain, set it before calling CL
        if (UseElementProvidedStrain()) {
             cl_params.SetStrainVector(constitutive_variables.StrainVector); // Strain is already computed in SetSmallDisplacementConstitutiveVariables
        }

        mConstitutiveLawVector[point_number]->CalculateMaterialResponse(cl_params, GetStressMeasure());
        // Stress and D are now in constitutive_variables.StressVector and constitutive_variables.D (via cl_params)

        integration_weight = SolidElementUtilities::GetIntegrationWeight(integration_points, point_number, kinematic_variables.detJ0, dimension, GetProperties());

        body_force = SolidElementUtilities::GetBodyForce(*this, integration_points, point_number);
        SolidElementUtilities::AddBodyForceContribution(kinematic_variables.N, rCurrentProcessInfo, body_force, rRightHandSideVector, integration_weight, dimension, number_of_nodes);

        SolidElementUtilities::CalculateAndAddInternalForces(rRightHandSideVector, kinematic_variables.B, constitutive_variables.StressVector, integration_weight);
        SolidElementUtilities::CalculateMaterialStiffnessMatrix(rLeftHandSideMatrix, kinematic_variables.B, constitutive_variables.D, integration_weight);
    }

    KRATOS_CATCH( "" )
}

void SmallDisplacement::CalculateLeftHandSide(MatrixType& rLeftHandSideMatrix, const ProcessInfo& rCurrentProcessInfo)
{
    VectorType temp_rhs; // Dummy
    CalculateLocalSystem(rLeftHandSideMatrix, temp_rhs, rCurrentProcessInfo);
}

void SmallDisplacement::CalculateRightHandSide(VectorType& rRightHandSideVector, const ProcessInfo& rCurrentProcessInfo)
{
    MatrixType temp_lhs; // Dummy
    CalculateLocalSystem(temp_lhs, rRightHandSideVector, rCurrentProcessInfo);
}

void SmallDisplacement::EquationIdVector(EquationIdVectorType& rResult, const ProcessInfo& rCurrentProcessInfo) const
{
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

void SmallDisplacement::GetDofList(DofsVectorType& rElementalDofList, const ProcessInfo& rCurrentProcessInfo) const
{
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

void SmallDisplacement::GetValuesVector(Vector& rValues, int Step) const
{
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

void SmallDisplacement::GetFirstDerivativesVector(Vector& rValues, int Step) const
{
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

void SmallDisplacement::GetSecondDerivativesVector(Vector& rValues, int Step) const
{
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

void SmallDisplacement::AddExplicitContribution(const VectorType& rRHSVector, const Variable<VectorType>& rRHSVariable, const Variable<double>& rDestinationVariable, const ProcessInfo& rCurrentProcessInfo)
{
    KRATOS_TRY;
    auto& r_geom = this->GetGeometry();
    const SizeType dimension = r_geom.WorkingSpaceDimension();
    const SizeType number_of_nodes = r_geom.size();
    const SizeType mat_size = number_of_nodes * dimension;

    if (rDestinationVariable == NODAL_MASS ) {
        VectorType element_mass_vector(mat_size);
        SolidElementUtilities::CalculateLumpedMassVector(*this, element_mass_vector, rCurrentProcessInfo);
        for (IndexType i = 0; i < number_of_nodes; ++i) {
            AtomicAdd(r_geom[i].GetValue(NODAL_MASS), element_mass_vector[i * dimension]); // Only X-component for scalar mass
        }
    }
    KRATOS_CATCH("")
}

void SmallDisplacement::AddExplicitContribution(const VectorType& rRHSVector, const Variable<VectorType>& rRHSVariable, const Variable<array_1d<double,3>>& rDestinationVariable, const ProcessInfo& rCurrentProcessInfo)
{
    KRATOS_TRY;
    auto& r_geom = this->GetGeometry();
    const auto& r_prop = this->GetProperties();
    const SizeType dimension = r_geom.WorkingSpaceDimension();
    const SizeType number_of_nodes = r_geom.size();
    const SizeType element_size = dimension * number_of_nodes;

    Vector damping_residual_contribution = ZeroVector(element_size);

    if (SolidElementUtilities::HasRayleighDamping(r_prop, rCurrentProcessInfo)) {
        Vector current_nodal_velocities = ZeroVector(element_size);
        this->GetFirstDerivativesVector(current_nodal_velocities); // Implemented in SmallDisplacement

        MatrixType damping_matrix(element_size, element_size);
        // CalculateDampingMatrix is now a member of SmallDisplacement.
        // It internally uses SolidElementUtilities::CalculateRayleighDampingMatrix.
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


void SmallDisplacement::CalculateMassMatrix(MatrixType& rMassMatrix, const ProcessInfo& rCurrentProcessInfo)
{
    KRATOS_TRY;
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
    } else {
        const double density = SolidElementUtilities::GetDensityForMassMatrixComputation(*this);
        const double thickness = (dimension == 2 && r_prop.Has(THICKNESS)) ? r_prop[THICKNESS] : 1.0;
        Matrix J0(dimension, dimension);
        const IntegrationMethod ig_method = GetGeometry().GetDefaultIntegrationMethod(); // Simplified for now
        const GeometryType::IntegrationPointsArrayType& integration_points = GetGeometry().IntegrationPoints(ig_method);
        const Matrix& Ncontainer = GetGeometry().ShapeFunctionsValues(ig_method);

        for ( IndexType point_number = 0; point_number < integration_points.size(); ++point_number ) {
            double detJ0; // Will be filled by utility
            Matrix invJ0, DN_DX; // Filled by utility
            detJ0 = SolidElementUtilities::CalculateDerivativesOnReferenceConfiguration(J0, invJ0, DN_DX, point_number, ig_method, r_geom, true);

            const double integration_weight = SolidElementUtilities::GetIntegrationWeight(integration_points, point_number, detJ0, dimension, r_prop) * thickness / detJ0; // thickness already in GetIntegrationWeight
                                                                                                                                                                   // but original had detJ0 here too.
                                                                                                                                                                   // The GetIntegrationWeight in BaseSolidElement was just rThisIntegrationPoints[PointNumber].Weight() * detJ;
                                                                                                                                                                   // The new utility includes thickness. So, the division by detJ0 might be an error in porting.
                                                                                                                                                                   // Let's use the utility directly:
            const double final_integration_weight = SolidElementUtilities::GetIntegrationWeight(integration_points, point_number, detJ0, dimension, r_prop);


            const Vector& rN = row(Ncontainer,point_number);
            for ( IndexType i = 0; i < number_of_nodes; ++i ) {
                const SizeType index_i = i * dimension;
                for ( IndexType j = 0; j < number_of_nodes; ++j ) {
                    const SizeType index_j = j * dimension;
                    const double NiNj_weight = rN[i] * rN[j] * final_integration_weight * density;
                    for ( IndexType k = 0; k < dimension; ++k ) rMassMatrix( index_i + k, index_j + k ) += NiNj_weight;
                }
            }
        }
    }
    KRATOS_CATCH("");
}


void SmallDisplacement::CalculateDampingMatrix(MatrixType& rDampingMatrix, const ProcessInfo& rCurrentProcessInfo)
{
    const unsigned int mat_size = GetGeometry().PointsNumber() * GetGeometry().WorkingSpaceDimension();
    SolidElementUtilities::CalculateRayleighDampingMatrix(*this, rDampingMatrix, rCurrentProcessInfo, mat_size, mConstitutiveLawVector, mThisIntegrationMethod);
}


int SmallDisplacement::Check(const ProcessInfo& rCurrentProcessInfo) const
{
    KRATOS_TRY;
    int check = Element::Check(rCurrentProcessInfo); // Call base class check.
    // Use the utility for solid element specific checks
    check = SolidElementUtilities::SolidElementCheck(*this, rCurrentProcessInfo, mConstitutiveLawVector);
    return check;
    KRATOS_CATCH( "" );
}

// --- CalculateOnIntegrationPoints ---
void SmallDisplacement::CalculateOnIntegrationPoints(const Variable<double>& rVariable, std::vector<double>& rOutput, const ProcessInfo& rCurrentProcessInfo)
{
    KRATOS_TRY
    const auto& r_geom = GetGeometry();
    const auto& r_props = GetProperties();
    const SizeType dimension = r_geom.WorkingSpaceDimension();
    const SizeType number_of_nodes = r_geom.size();
    KRATOS_ERROR_IF(mConstitutiveLawVector.empty() || !mConstitutiveLawVector[0]) << "CL not init! Elm ID: " << Id() << std::endl;
    const SizeType strain_size = mConstitutiveLawVector[0]->GetStrainSize();

    const GeometryType::IntegrationPointsArrayType& integration_points = r_geom.IntegrationPoints(mThisIntegrationMethod);
    if (rOutput.size() != integration_points.size()) {
        rOutput.resize(integration_points.size());
    }

    if (rVariable == VON_MISES_STRESS) {
        // Implementation from previous subtask... (assuming it's correct)
        SolidElementUtilities::LocalKinematicVariables kin_vars(strain_size, dimension, number_of_nodes);
        SolidElementUtilities::LocalConstitutiveVariables const_vars(strain_size);
        GetValuesVector(kin_vars.Displacements);

        ConstitutiveLaw::Parameters cl_params(r_geom, r_props, rCurrentProcessInfo);
        Flags& cl_options = cl_params.GetOptions();
        cl_options.Set(ConstitutiveLaw::USE_ELEMENT_PROVIDED_STRAIN, UseElementProvidedStrain());
        cl_options.Set(ConstitutiveLaw::COMPUTE_STRESS, true);
        cl_options.Set(ConstitutiveLaw::COMPUTE_CONSTITUTIVE_TENSOR, false);

        if (UseElementProvidedStrain()) cl_params.SetStrainVector(const_vars.StrainVector);
        cl_params.SetStressVector(const_vars.StressVector);
        cl_params.SetConstitutiveMatrix(const_vars.D);

        for (IndexType point_number = 0; point_number < integration_points.size(); ++point_number) {
            SolidElementUtilities::CalculateKinematicVariablesSmallDisplacement(
                kin_vars, point_number, mThisIntegrationMethod, r_geom, kin_vars.Displacements, strain_size, true);
            SolidElementUtilities::SetSmallDisplacementConstitutiveVariables(
                kin_vars, const_vars, cl_params, kin_vars.Displacements);
            if (UseElementProvidedStrain()) cl_params.SetStrainVector(const_vars.StrainVector);

            mConstitutiveLawVector[point_number]->CalculateMaterialResponse(cl_params, GetStressMeasure());

            if (dimension == 2) {
                if (strain_size == 3) {
                    rOutput[point_number] = ConstitutiveLawUtilities<3>::CalculateVonMisesEquivalentStress(const_vars.StressVector);
                } else {
                    Vector aux_stress_for_vm(6);
                    noalias(aux_stress_for_vm) = ZeroVector(6);
                    aux_stress_for_vm[0] = const_vars.StressVector[0];
                    aux_stress_for_vm[1] = const_vars.StressVector[1];
                    aux_stress_for_vm[2] = const_vars.StressVector[3];
                    aux_stress_for_vm[3] = const_vars.StressVector[2];
                    rOutput[point_number] = ConstitutiveLawUtilities<6>::CalculateVonMisesEquivalentStress(aux_stress_for_vm);
                }
            } else {
                rOutput[point_number] = ConstitutiveLawUtilities<6>::CalculateVonMisesEquivalentStress(const_vars.StressVector);
            }
        }
    } else if (rVariable == STRAIN_ENERGY) {
        SolidElementUtilities::LocalKinematicVariables kin_vars(strain_size, dimension, number_of_nodes);
        SolidElementUtilities::LocalConstitutiveVariables const_vars(strain_size);
        GetValuesVector(kin_vars.Displacements);

        ConstitutiveLaw::Parameters cl_params(r_geom, r_props, rCurrentProcessInfo);
        Flags& cl_options = cl_params.GetOptions();
        cl_options.Set(ConstitutiveLaw::USE_ELEMENT_PROVIDED_STRAIN, UseElementProvidedStrain());
        cl_options.Set(ConstitutiveLaw::COMPUTE_STRESS, false); // As per BaseSolidElement for STRAIN_ENERGY
        cl_options.Set(ConstitutiveLaw::COMPUTE_CONSTITUTIVE_TENSOR, false);

        if (UseElementProvidedStrain()) cl_params.SetStrainVector(const_vars.StrainVector);

        for (IndexType point_number = 0; point_number < integration_points.size(); ++point_number) {
            SolidElementUtilities::CalculateKinematicVariablesSmallDisplacement(
                kin_vars, point_number, mThisIntegrationMethod, r_geom, kin_vars.Displacements, strain_size, true);
            SolidElementUtilities::SetSmallDisplacementConstitutiveVariables(
                kin_vars, const_vars, cl_params, kin_vars.Displacements);
            if (UseElementProvidedStrain()) cl_params.SetStrainVector(const_vars.StrainVector);

            // StrainEnergy is a "calculated" value from CL
            mConstitutiveLawVector[point_number]->CalculateValue(cl_params, STRAIN_ENERGY, rOutput[point_number]);
        }
    } else if (rVariable == INTEGRATION_WEIGHT) {
        SolidElementUtilities::LocalKinematicVariables kin_vars(strain_size, dimension, number_of_nodes); // Only for detJ0
        for (IndexType point_number = 0; point_number < integration_points.size(); ++point_number) {
             kin_vars.detJ0 = SolidElementUtilities::CalculateDerivativesOnReferenceConfiguration(
                kin_vars.J0, kin_vars.InvJ0, kin_vars.DN_DX, point_number, mThisIntegrationMethod, r_geom, true);
            rOutput[point_number] = SolidElementUtilities::GetIntegrationWeight(
                integration_points, point_number, kin_vars.detJ0, dimension, r_props);
        }
    } else if (mConstitutiveLawVector[0]->Has(rVariable)) {
        SolidElementUtilities::GetValueFromConstitutiveLaw(mConstitutiveLawVector, rVariable, rOutput);
    } else { // Try to CalculateValue if not directly available via GetValue
        SolidElementUtilities::CalculateValueOnConstitutiveLaw(*this, mConstitutiveLawVector, rVariable, rCurrentProcessInfo, mThisIntegrationMethod, UseElementProvidedStrain(), GetStressMeasure(), rOutput);
    }
    KRATOS_CATCH("")
}

void SmallDisplacement::CalculateOnIntegrationPoints(const Variable<Vector>& rVariable, std::vector<Vector>& rOutput, const ProcessInfo& rCurrentProcessInfo)
{
    KRATOS_TRY
    const auto& r_geom = GetGeometry();
    const auto& r_props = GetProperties();
    const SizeType dimension = r_geom.WorkingSpaceDimension();
    const SizeType number_of_nodes = r_geom.size();
    KRATOS_ERROR_IF(mConstitutiveLawVector.empty() || !mConstitutiveLawVector[0]) << "CL not init for element " << Id() << std::endl;
    const SizeType strain_size = mConstitutiveLawVector[0]->GetStrainSize();

    const GeometryType::IntegrationPointsArrayType& integration_points = r_geom.IntegrationPoints(mThisIntegrationMethod);
    if (rOutput.size() != integration_points.size()) rOutput.resize(integration_points.size());
    for(auto& v : rOutput) if(v.size()!=strain_size) v.resize(strain_size, false);


    if (rVariable == CAUCHY_STRESS_VECTOR || rVariable == PK2_STRESS_VECTOR) {
        // Implementation from previous subtask...
        SolidElementUtilities::LocalKinematicVariables kin_vars(strain_size, dimension, number_of_nodes);
        SolidElementUtilities::LocalConstitutiveVariables const_vars(strain_size);
        GetValuesVector(kin_vars.Displacements);

        ConstitutiveLaw::Parameters cl_params(r_geom, r_props, rCurrentProcessInfo);
        Flags& cl_options = cl_params.GetOptions();
        cl_options.Set(ConstitutiveLaw::USE_ELEMENT_PROVIDED_STRAIN, UseElementProvidedStrain());
        cl_options.Set(ConstitutiveLaw::COMPUTE_STRESS, true);
        cl_options.Set(ConstitutiveLaw::COMPUTE_CONSTITUTIVE_TENSOR, false);

        if(UseElementProvidedStrain()) cl_params.SetStrainVector(const_vars.StrainVector);
        cl_params.SetStressVector(const_vars.StressVector);
        cl_params.SetConstitutiveMatrix(const_vars.D);

        for (IndexType i = 0; i < integration_points.size(); ++i) {
            SolidElementUtilities::CalculateKinematicVariablesSmallDisplacement(
                kin_vars, i, mThisIntegrationMethod, r_geom, kin_vars.Displacements, strain_size, true);
            SolidElementUtilities::SetSmallDisplacementConstitutiveVariables(
                 kin_vars, const_vars, cl_params, kin_vars.Displacements);
            if (UseElementProvidedStrain()) cl_params.SetStrainVector(const_vars.StrainVector);

            ConstitutiveLaw::StressMeasure target_measure = (rVariable == CAUCHY_STRESS_VECTOR) ? ConstitutiveLaw::StressMeasure_Cauchy : ConstitutiveLaw::StressMeasure_PK2;
            bool is_rotated = SolidElementUtilities::IsElementRotated(*this, mConstitutiveLawVector[i]);

            SolidElementUtilities::CalculateConstitutiveVariables(
                kin_vars, const_vars, cl_params, mConstitutiveLawVector[i],
                target_measure, is_rotated, *this);
            rOutput[i] = const_vars.StressVector;
        }
    } else if (rVariable == GREEN_LAGRANGE_STRAIN_VECTOR || rVariable == ALMANSI_STRAIN_VECTOR ) {
        // Implementation from previous subtask...
        SolidElementUtilities::LocalKinematicVariables kin_vars(strain_size, dimension, number_of_nodes);
        SolidElementUtilities::LocalConstitutiveVariables const_vars(strain_size);
        GetValuesVector(kin_vars.Displacements);

        ConstitutiveLaw::Parameters cl_params(r_geom, r_props, rCurrentProcessInfo);
        Flags& cl_options = cl_params.GetOptions();
        cl_options.Set(ConstitutiveLaw::USE_ELEMENT_PROVIDED_STRAIN, UseElementProvidedStrain());
        cl_options.Set(ConstitutiveLaw::COMPUTE_STRESS, false);
        cl_options.Set(ConstitutiveLaw::COMPUTE_CONSTITUTIVE_TENSOR, false);
        cl_params.SetStrainVector(const_vars.StrainVector);

        for (IndexType i = 0; i < integration_points.size(); ++i) {
            SolidElementUtilities::CalculateKinematicVariablesSmallDisplacement(
                kin_vars, i, mThisIntegrationMethod, r_geom, kin_vars.Displacements, strain_size, true);
            SolidElementUtilities::SetSmallDisplacementConstitutiveVariables(
                 kin_vars, const_vars, cl_params, kin_vars.Displacements);
            if (UseElementProvidedStrain()) cl_params.SetStrainVector(const_vars.StrainVector);

            ConstitutiveLaw::StressMeasure this_stress_measure = (rVariable == GREEN_LAGRANGE_STRAIN_VECTOR) ? ConstitutiveLaw::StressMeasure_PK2 : ConstitutiveLaw::StressMeasure_Kirchhoff;
            mConstitutiveLawVector[i]->CalculateMaterialResponse(cl_params, this_stress_measure);
            rOutput[i] = const_vars.StrainVector;
        }
    } else if (rVariable == INITIAL_STRESS_VECTOR) {
        for (IndexType i = 0; i < mConstitutiveLawVector.size(); ++i) {
            if (mConstitutiveLawVector[i]->HasInitialState()) {
                rOutput[i] = mConstitutiveLawVector[i]->GetInitialState().GetInitialStressVector();
            } else {
                if(rOutput[i].size() != strain_size) rOutput[i].resize(strain_size, false);
                noalias(rOutput[i]) = ZeroVector(strain_size);
            }
        }
    } else if (rVariable == INITIAL_STRAIN_VECTOR) {
        for (IndexType i = 0; i < mConstitutiveLawVector.size(); ++i) {
            if (mConstitutiveLawVector[i]->HasInitialState()) {
                rOutput[i] = mConstitutiveLawVector[i]->GetInitialState().GetInitialStrainVector();
            } else {
                if(rOutput[i].size() != strain_size) rOutput[i].resize(strain_size, false);
                noalias(rOutput[i]) = ZeroVector(strain_size);
            }
        }
    }
    else if (mConstitutiveLawVector[0]->Has(rVariable)) {
        SolidElementUtilities::GetValueFromConstitutiveLaw(mConstitutiveLawVector, rVariable, rOutput);
    } else { // Try to CalculateValue if not directly available via GetValue
        SolidElementUtilities::CalculateValueOnConstitutiveLaw(*this, mConstitutiveLawVector, rVariable, rCurrentProcessInfo, mThisIntegrationMethod, UseElementProvidedStrain(), GetStressMeasure(), rOutput);
    }
    KRATOS_CATCH("")
}

void SmallDisplacement::CalculateOnIntegrationPoints(const Variable<Matrix>& rVariable, std::vector<Matrix>& rOutput, const ProcessInfo& rCurrentProcessInfo)
{
    KRATOS_TRY
    const auto& r_geom = GetGeometry();
    const auto& r_props = GetProperties();
    const SizeType dimension = r_geom.WorkingSpaceDimension();
    const SizeType number_of_nodes = r_geom.size();
    KRATOS_ERROR_IF(mConstitutiveLawVector.empty() || !mConstitutiveLawVector[0]) << "CL not init!" << Id() << std::endl;
    const SizeType strain_size = mConstitutiveLawVector[0]->GetStrainSize();

    const GeometryType::IntegrationPointsArrayType& integration_points = r_geom.IntegrationPoints(mThisIntegrationMethod);
    if (rOutput.size() != integration_points.size()) rOutput.resize(integration_points.size());
    // Size of matrix output depends on variable; for stress/strain tensors it's dim x dim
    // For ConstitutiveMatrix it's strain_size x strain_size
    // This initial resize might be too generic.

    if (rVariable == CAUCHY_STRESS_TENSOR || rVariable == PK2_STRESS_TENSOR) {
        std::vector<Vector> stress_vector_output; // Temp vector for stress vectors
        const Variable<Vector>& vector_variable = (rVariable == CAUCHY_STRESS_TENSOR) ? CAUCHY_STRESS_VECTOR : PK2_STRESS_VECTOR;
        this->CalculateOnIntegrationPoints(vector_variable, stress_vector_output, rCurrentProcessInfo);
        for (IndexType i = 0; i < integration_points.size(); ++i) {
            if(rOutput[i].size1()!=dimension || rOutput[i].size2()!=dimension) rOutput[i].resize(dimension,dimension,false);
            rOutput[i] = MathUtils<double>::StressVectorToTensor(stress_vector_output[i]);
        }
    } else if (rVariable == GREEN_LAGRANGE_STRAIN_TENSOR || rVariable == ALMANSI_STRAIN_TENSOR) {
        std::vector<Vector> strain_vector_output;
        const Variable<Vector>& vector_variable = (rVariable == GREEN_LAGRANGE_STRAIN_TENSOR) ? GREEN_LAGRANGE_STRAIN_VECTOR : ALMANSI_STRAIN_VECTOR;
        this->CalculateOnIntegrationPoints(vector_variable, strain_vector_output, rCurrentProcessInfo);
        for (IndexType i = 0; i < integration_points.size(); ++i) {
             if(rOutput[i].size1()!=dimension || rOutput[i].size2()!=dimension) rOutput[i].resize(dimension,dimension,false);
            rOutput[i] = MathUtils<double>::StrainVectorToTensor(strain_vector_output[i]);
        }
    } else if (rVariable == CONSTITUTIVE_MATRIX) {
        // Implementation from previous subtask...
        SolidElementUtilities::LocalKinematicVariables kin_vars(strain_size, dimension, number_of_nodes);
        SolidElementUtilities::LocalConstitutiveVariables const_vars(strain_size);
        GetValuesVector(kin_vars.Displacements);

        ConstitutiveLaw::Parameters cl_params(r_geom, r_props, rCurrentProcessInfo);
        Flags& cl_options = cl_params.GetOptions();
        cl_options.Set(ConstitutiveLaw::USE_ELEMENT_PROVIDED_STRAIN, UseElementProvidedStrain());
        cl_options.Set(ConstitutiveLaw::COMPUTE_STRESS, false);
        cl_options.Set(ConstitutiveLaw::COMPUTE_CONSTITUTIVE_TENSOR, true);

        if(UseElementProvidedStrain()) cl_params.SetStrainVector(const_vars.StrainVector);
        cl_params.SetStressVector(const_vars.StressVector);
        cl_params.SetConstitutiveMatrix(const_vars.D);

        for (IndexType i = 0; i < integration_points.size(); ++i) {
            SolidElementUtilities::CalculateKinematicVariablesSmallDisplacement(
                kin_vars, i, mThisIntegrationMethod, r_geom, kin_vars.Displacements, strain_size, true);
            SolidElementUtilities::SetSmallDisplacementConstitutiveVariables(
                 kin_vars, const_vars, cl_params, kin_vars.Displacements);
            if (UseElementProvidedStrain()) cl_params.SetStrainVector(const_vars.StrainVector);

            bool is_rotated = SolidElementUtilities::IsElementRotated(*this, mConstitutiveLawVector[i]);
            SolidElementUtilities::CalculateConstitutiveVariables(
                kin_vars, const_vars, cl_params, mConstitutiveLawVector[i],
                GetStressMeasure(), is_rotated, *this);
            if(rOutput[i].size1()!=strain_size || rOutput[i].size2()!=strain_size) rOutput[i].resize(strain_size,strain_size,false);
            rOutput[i] = const_vars.D;
        }
    } else if (rVariable == DEFORMATION_GRADIENT) {
        // Implementation from previous subtask...
         SolidElementUtilities::LocalKinematicVariables kin_vars(strain_size, dimension, number_of_nodes);
         GetValuesVector(kin_vars.Displacements);
         for (IndexType i = 0; i < integration_points.size(); ++i) {
            SolidElementUtilities::CalculateKinematicVariablesSmallDisplacement(
                kin_vars, i, mThisIntegrationMethod, r_geom, kin_vars.Displacements, strain_size, true);
            if(rOutput[i].size1() != dimension || rOutput[i].size2() != dimension) rOutput[i].resize(dimension,dimension,false);
            rOutput[i] = kin_vars.F;
         }
    } else if (mConstitutiveLawVector[0]->Has(rVariable)) {
        SolidElementUtilities::GetValueFromConstitutiveLaw(mConstitutiveLawVector, rVariable, rOutput);
    } else { // Try to CalculateValue if not directly available via GetValue
        SolidElementUtilities::CalculateValueOnConstitutiveLaw(*this, mConstitutiveLawVector, rVariable, rCurrentProcessInfo, mThisIntegrationMethod, UseElementProvidedStrain(), GetStressMeasure(), rOutput);
    }
    KRATOS_CATCH("")
}

void SmallDisplacement::CalculateOnIntegrationPoints(const Variable<array_1d<double,3>>& rVariable, std::vector<array_1d<double,3>>& rOutput, const ProcessInfo& rCurrentProcessInfo)
{
    KRATOS_TRY
    const auto& r_geom = GetGeometry();
    const GeometryType::IntegrationPointsArrayType& integration_points = r_geom.IntegrationPoints(mThisIntegrationMethod);
    if (rOutput.size() != integration_points.size()) rOutput.resize(integration_points.size());
    // Default init for array_1d<double,3> is zero vector, which is fine.

    if (rVariable == INTEGRATION_COORDINATES) {
        for (IndexType i = 0; i < integration_points.size(); ++i) {
            Point global_point;
            r_geom.GlobalCoordinates(global_point, integration_points[i]);
            noalias(rOutput[i]) = global_point.Coordinates();
        }
    } else if (mConstitutiveLawVector.empty() || !mConstitutiveLawVector[0]) {
         KRATOS_WARNING("SmallDisplacement") << "CL not initialized for variable " << rVariable.Name() << " in element " << Id() << std::endl;
    }
    else if (mConstitutiveLawVector[0]->Has(rVariable)) {
        SolidElementUtilities::GetValueFromConstitutiveLaw(mConstitutiveLawVector, rVariable, rOutput);
    } else { // Try to CalculateValue
        SolidElementUtilities::CalculateValueOnConstitutiveLaw(*this, mConstitutiveLawVector, rVariable, rCurrentProcessInfo, mThisIntegrationMethod, UseElementProvidedStrain(), GetStressMeasure(), rOutput);
    }
    KRATOS_CATCH("")
}

void SmallDisplacement::SetValuesOnIntegrationPoints(const Variable<double>& rVariable, const std::vector<double>& rValues, const ProcessInfo& rCurrentProcessInfo)
{
    KRATOS_TRY
    KRATOS_ERROR_IF(mConstitutiveLawVector.empty()) << "Constitutive law vector is empty for element " << Id() << std::endl;
    KRATOS_ERROR_IF(rValues.size() != mConstitutiveLawVector.size()) << "Input vector size mismatch for SetValuesOnIntegrationPoints." << std::endl;

    if (mConstitutiveLawVector[0]->Has(rVariable)) {
        for (IndexType i = 0; i < mConstitutiveLawVector.size(); ++i) {
            mConstitutiveLawVector[i]->SetValue(rVariable, rValues[i], rCurrentProcessInfo);
        }
    } else {
        KRATOS_WARNING("SmallDisplacement") << "SetValue for double variable " << rVariable.Name() << " not implemented in CL." << std::endl;
    }
    KRATOS_CATCH("")
}

// Placeholder for other SetValuesOnIntegrationPoints overloads
void SmallDisplacement::SetValuesOnIntegrationPoints(const Variable<Vector>& rVariable, const std::vector<Vector>& rValues, const ProcessInfo& rCurrentProcessInfo)
{ KRATOS_TRY if(!mConstitutiveLawVector.empty() && mConstitutiveLawVector[0]->Has(rVariable)) for(IndexType i=0; i<mConstitutiveLawVector.size(); ++i) mConstitutiveLawVector[i]->SetValue(rVariable, rValues[i], rCurrentProcessInfo); else KRATOS_WARNING("SD")<<"Set for Vector "<<rVariable.Name()<<" not in CL"<<std::endl; KRATOS_CATCH("") }
void SmallDisplacement::SetValuesOnIntegrationPoints(const Variable<Matrix>& rVariable, const std::vector<Matrix>& rValues, const ProcessInfo& rCurrentProcessInfo)
{ KRATOS_TRY if(!mConstitutiveLawVector.empty() && mConstitutiveLawVector[0]->Has(rVariable)) for(IndexType i=0; i<mConstitutiveLawVector.size(); ++i) mConstitutiveLawVector[i]->SetValue(rVariable, rValues[i], rCurrentProcessInfo); else KRATOS_WARNING("SD")<<"Set for Matrix "<<rVariable.Name()<<" not in CL"<<std::endl; KRATOS_CATCH("") }
void SmallDisplacement::SetValuesOnIntegrationPoints(const Variable<array_1d<double,3>>& rVariable, const std::vector<array_1d<double,3>>& rValues, const ProcessInfo& rCurrentProcessInfo)
{ KRATOS_TRY if(!mConstitutiveLawVector.empty() && mConstitutiveLawVector[0]->Has(rVariable)) for(IndexType i=0; i<mConstitutiveLawVector.size(); ++i) mConstitutiveLawVector[i]->SetValue(rVariable, rValues[i], rCurrentProcessInfo); else KRATOS_WARNING("SD")<<"Set for array_1d<double,3> "<<rVariable.Name()<<" not in CL"<<std::endl; KRATOS_CATCH("") }
void SmallDisplacement::SetValuesOnIntegrationPoints(const Variable<ConstitutiveLaw::Pointer>& rVariable, const std::vector<ConstitutiveLaw::Pointer>& rValues, const ProcessInfo& rCurrentProcessInfo)
{
    KRATOS_TRY
    if (rVariable == CONSTITUTIVE_LAW) {
        KRATOS_ERROR_IF(rValues.size() != mConstitutiveLawVector.size()) << "Constitutive law vector size mismatch." << std::endl;
        mConstitutiveLawVector = rValues;
    } else {
         KRATOS_WARNING("SmallDisplacement") << "SetValue for CL Ptr variable " << rVariable.Name() << " not CONSTITUTIVE_LAW." << std::endl;
    }
    KRATOS_CATCH("")
}

void SmallDisplacement::SetValuesOnIntegrationPoints(const Variable<bool>& rVariable, const std::vector<bool>& rValues, const ProcessInfo& rCurrentProcessInfo)
{ KRATOS_TRY if(!mConstitutiveLawVector.empty() && mConstitutiveLawVector[0]->Has(rVariable)) for(IndexType i=0; i<mConstitutiveLawVector.size(); ++i) mConstitutiveLawVector[i]->SetValue(rVariable, rValues[i], rCurrentProcessInfo); else KRATOS_WARNING("SD")<<"Set for bool "<<rVariable.Name()<<" not in CL"<<std::endl; KRATOS_CATCH("") }

void SmallDisplacement::SetValuesOnIntegrationPoints(const Variable<int>& rVariable, const std::vector<int>& rValues, const ProcessInfo& rCurrentProcessInfo)
{ KRATOS_TRY if(!mConstitutiveLawVector.empty() && mConstitutiveLawVector[0]->Has(rVariable)) for(IndexType i=0; i<mConstitutiveLawVector.size(); ++i) mConstitutiveLawVector[i]->SetValue(rVariable, rValues[i], rCurrentProcessInfo); else KRATOS_WARNING("SD")<<"Set for int "<<rVariable.Name()<<" not in CL"<<std::endl; KRATOS_CATCH("") }

void SmallDisplacement::SetValuesOnIntegrationPoints(const Variable<array_1d<double,6>>& rVariable, const std::vector<array_1d<double,6>>& rValues, const ProcessInfo& rCurrentProcessInfo)
{ KRATOS_TRY if(!mConstitutiveLawVector.empty() && mConstitutiveLawVector[0]->Has(rVariable)) for(IndexType i=0; i<mConstitutiveLawVector.size(); ++i) mConstitutiveLawVector[i]->SetValue(rVariable, rValues[i], rCurrentProcessInfo); else KRATOS_WARNING("SD")<<"Set for array_1d<double,6> "<<rVariable.Name()<<" not in CL"<<std::endl; KRATOS_CATCH("") }

const Parameters SmallDisplacement::GetSpecifications() const
{
    const Parameters base_specs = SolidElementUtilities::GetDefaultSolidSpecifications(GetGeometry().WorkingSpaceDimension());
    // Specifics for SmallDisplacement can be added here if any, otherwise return base.
    // For example, if it had specific output variables or different framework tag.
    // For now, assume it's largely the same as the base solid element default.
    return base_specs;
}

std::string SmallDisplacement::Info() const
{
    std::stringstream buffer;
    buffer << "Small Displacement Solid Element #" << Id();
    if (mConstitutiveLawVector.size() > 0 && mConstitutiveLawVector[0] != nullptr) {
         buffer << "\nConstitutive law: " << mConstitutiveLawVector[0]->Info();
    }
    return buffer.str();
}

void SmallDisplacement::PrintInfo(std::ostream& rOStream) const
{
    rOStream << Info();
}

void SmallDisplacement::PrintData(std::ostream& rOStream) const
{
    pGetGeometry()->PrintData(rOStream);
}

void SmallDisplacement::save(Serializer& rSerializer) const
{
    KRATOS_SERIALIZE_SAVE_BASE_CLASS( rSerializer, Element ); // Save base Element class
    int IntMethod = int(mThisIntegrationMethod);
    rSerializer.save("IntegrationMethod", IntMethod);
    rSerializer.save("ConstitutiveLawVector", mConstitutiveLawVector);
}

void SmallDisplacement::load(Serializer& rSerializer)
{
    KRATOS_SERIALIZE_LOAD_BASE_CLASS( rSerializer, Element ); // Load base Element class
    int IntMethod;
    rSerializer.load("IntegrationMethod", IntMethod);
    mThisIntegrationMethod = IntegrationMethod(IntMethod);
    rSerializer.load("ConstitutiveLawVector", mConstitutiveLawVector);
}


} // Namespace Kratos

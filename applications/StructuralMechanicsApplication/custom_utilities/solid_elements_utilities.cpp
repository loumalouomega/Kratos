#include "solid_elements_utilities.h"
#include "utilities/math_utils.h"
#include "utilities/geometry_utilities.h"
#include "custom_utilities/structural_mechanics_element_utilities.h"
#include "includes/checks.h" // For KRATOS_ERROR_IF, etc.
#include "includes/ublas_interface.h" // For ZeroVector, ZeroMatrix etc.




namespace Kratos
{

double SolidElementUtilities::CalculateDerivativesOnReferenceConfiguration(
    Matrix& rJ0,
    Matrix& rInvJ0,
    Matrix& rDN_DX,
    const IndexType PointNumber,
    IntegrationMethod ThisIntegrationMethod,
    const Element::GeometryType& rGeom,
    bool UseGeometryIntegrationMethod)
{
    double detJ0;
    if (UseGeometryIntegrationMethod) {
        GeometryUtils::JacobianOnInitialConfiguration(
            rGeom,
            rGeom.IntegrationPoints(ThisIntegrationMethod)[PointNumber], rJ0);
        MathUtils<double>::InvertMatrix(rJ0, rInvJ0, detJ0);
        const Matrix& rDN_De =
            rGeom.ShapeFunctionsLocalGradients(ThisIntegrationMethod)[PointNumber];
        GeometryUtils::ShapeFunctionsGradients(rDN_De, rInvJ0, rDN_DX);
    } else {
        const auto& integration_points =  rGeom.IntegrationPoints(ThisIntegrationMethod); // Assuming default if not GI_GAUSS_X
        GeometryUtils::JacobianOnInitialConfiguration(rGeom, integration_points[PointNumber],rJ0);
        MathUtils<double>::InvertMatrix(rJ0, rInvJ0, detJ0);
        Matrix DN_De;
        rGeom.ShapeFunctionsLocalGradients(DN_De, integration_points[PointNumber]);
        GeometryUtils::ShapeFunctionsGradients(DN_De, rInvJ0, rDN_DX);
    }
    return detJ0;
}

double SolidElementUtilities::GetIntegrationWeight(
    const Element::GeometryType::IntegrationPointsArrayType& rThisIntegrationPoints,
    const IndexType PointNumber,
    const double detJ,
    const SizeType Dimension,
    const Properties& rProperties)
{
    double weight = rThisIntegrationPoints[PointNumber].Weight() * detJ;
    if (Dimension == 2 && rProperties.Has(THICKNESS)) {
        weight *= rProperties[THICKNESS];
    }
    return weight;
}

void SolidElementUtilities::CalculateBStrainLinear(
    const Element::GeometryType& rGeom,
    const Matrix& rDN_DX,
    Matrix& rB)
{
    StructuralMechanicsElementUtilities::CalculateBMatrix(rDN_DX, rB, rGeom.WorkingSpaceDimension(), rGeom.size());
}

void SolidElementUtilities::ComputeEquivalentFStrainLinear(
    const Element::GeometryType& rGeom,
    const Vector& rStrainTensor,
    Matrix& rF)
{
    StructuralMechanicsElementUtilities::ComputeEquivalentF(rGeom.WorkingSpaceDimension(), rGeom.size(), rStrainTensor, rF);
}

void SolidElementUtilities::SetSmallDisplacementConstitutiveVariables(
    const LocalKinematicVariables& rThisKinematicVariables,
    LocalConstitutiveVariables& rThisConstitutiveVariables,
    ConstitutiveLaw::Parameters& rValues,
    const Vector& rDisplacements)
{
    noalias(rThisConstitutiveVariables.StrainVector) = prod(rThisKinematicVariables.B, rDisplacements);

    rValues.SetShapeFunctionsValues(rThisKinematicVariables.N);
    rValues.SetDeterminantF(rThisKinematicVariables.detF);
    rValues.SetDeformationGradientF(rThisKinematicVariables.F);
    rValues.SetConstitutiveMatrix(rThisConstitutiveVariables.D);
    rValues.SetStressVector(rThisConstitutiveVariables.StressVector);
    // StrainVector is set on rValues by the CL itself if USE_ELEMENT_PROVIDED_STRAIN is true.
    // If USE_ELEMENT_PROVIDED_STRAIN is true, rValues.SetStrainVector(rThisConstitutiveVariables.StrainVector) should be done before CL call.
}


void SolidElementUtilities::CalculateMaterialStiffnessMatrix(
    MatrixType& rLeftHandSideMatrix,
    const Matrix& B,
    const Matrix& D,
    const double IntegrationWeight)
{
    noalias( rLeftHandSideMatrix ) += IntegrationWeight * prod( trans( B ), Matrix(prod(D, B)));
}

void SolidElementUtilities::CalculateAndAddInternalForces(
    VectorType& rRightHandSideVector,
    const Matrix& B,
    const Vector& rStressVector,
    const double IntegrationWeight)
{
    noalias( rRightHandSideVector ) -= IntegrationWeight * prod( trans( B ), rStressVector );
}

void SolidElementUtilities::AddBodyForceContribution(
    const Vector& rN,
    const ProcessInfo& rCurrentProcessInfo, // Unused for now, but kept for signature consistency
    const array_1d<double, 3>& rBodyForce,
    VectorType& rRightHandSideVector,
    const double Weight,
    const SizeType Dimension,
    const SizeType NumberOfNodes)
{
    for ( IndexType i = 0; i < NumberOfNodes; ++i ) {
        const SizeType index = Dimension * i;
        for ( IndexType j = 0; j < Dimension; ++j )
            rRightHandSideVector[index + j] += Weight * rN[i] * rBodyForce[j];
    }
}

void SolidElementUtilities::InitializeConstitutiveLaw(
    ConstitutiveLaw::Pointer& pConstitutiveLaw,
    const Properties& rProperties,
    const Element::GeometryType& rGeom,
    const Vector& rShapeFunctionValues)
{
    if ( rProperties.Has(CONSTITUTIVE_LAW) ) {
        pConstitutiveLaw = rProperties[CONSTITUTIVE_LAW]->Clone();
        pConstitutiveLaw->InitializeMaterial( rProperties, rGeom, rShapeFunctionValues);
    } else {
        KRATOS_ERROR << "A constitutive law needs to be specified for the element." << std::endl;
    }
}

void SolidElementUtilities::CalculateKinematicVariablesSmallDisplacement(
    LocalKinematicVariables& rThisKinematicVariables,
    const IndexType PointNumber,
    const IntegrationMethod& rIntegrationMethod,
    const Element::GeometryType& rGeom,
    const Vector& rDisplacements, // Added
    const SizeType StrainSize, // Added
    bool UseGeometryIntegrationMethod) // Added
{
    const Element::GeometryType::IntegrationPointsArrayType& r_integration_points = rGeom.IntegrationPoints(rIntegrationMethod);
    // Shape functions
    rThisKinematicVariables.N = rGeom.ShapeFunctionsValues(rThisKinematicVariables.N, r_integration_points[PointNumber].Coordinates());

    rThisKinematicVariables.detJ0 = SolidElementUtilities::CalculateDerivativesOnReferenceConfiguration(
        rThisKinematicVariables.J0, rThisKinematicVariables.InvJ0, rThisKinematicVariables.DN_DX, PointNumber, rIntegrationMethod, rGeom, UseGeometryIntegrationMethod);

    KRATOS_ERROR_IF(rThisKinematicVariables.detJ0 < 0.0) << " INVERTED. DETJ0: " << rThisKinematicVariables.detJ0 << std::endl;

    SolidElementUtilities::CalculateBStrainLinear( rGeom, rThisKinematicVariables.DN_DX, rThisKinematicVariables.B );

    // Compute equivalent F (Small Displacement specific part)
    Vector strain_vector(StrainSize); // Use StrainSize passed
    noalias(strain_vector) = prod(rThisKinematicVariables.B, rDisplacements);
    SolidElementUtilities::ComputeEquivalentFStrainLinear(rGeom, strain_vector, rThisKinematicVariables.F);
    rThisKinematicVariables.detF = MathUtils<double>::Det(rThisKinematicVariables.F);
}

double SolidElementUtilities::GetDensityForMassMatrixComputation(const Element& rElement)
{
    return StructuralMechanicsElementUtilities::GetDensityForMassMatrixComputation(rElement);
}

int SolidElementUtilities::SolidElementCheck(
    const Element& rElement,
    const ProcessInfo& rCurrentProcessInfo,
    const std::vector<ConstitutiveLaw::Pointer>& rConstitutiveLawVector)
{
    return StructuralMechanicsElementUtilities::SolidElementCheck(rElement, rCurrentProcessInfo, rConstitutiveLawVector);
}

bool SolidElementUtilities::ComputeLumpedMassMatrix(
    const Properties& rProperties,
    const ProcessInfo& rCurrentProcessInfo)
{
    return StructuralMechanicsElementUtilities::ComputeLumpedMassMatrix(rProperties, rCurrentProcessInfo);
}

bool SolidElementUtilities::HasRayleighDamping(
    const Properties& rProperties,
    const ProcessInfo& rCurrentProcessInfo)
{
    return StructuralMechanicsElementUtilities::HasRayleighDamping(rProperties, rCurrentProcessInfo);
}

void SolidElementUtilities::CalculateRayleighDampingMatrix(
    const Element& rElement,
    MatrixType& rDampingMatrix,
    const ProcessInfo& rCurrentProcessInfo,
    SizeType MatrixSize,
    const std::vector<ConstitutiveLaw::Pointer>& rConstitutiveLawVector, // Added
    IntegrationMethod ThisIntegrationMethod // Added
    )
{
    // This function in S-M-E-U depends on public methods of Element that might not be available
    // in the same way after refactoring (e.g. CalculateMassMatrix, CalculateLeftHandSide).
    // For now, let's assume it can be adapted or parts of it can be used.
    // A more complete refactoring of this utility might be needed.
    StructuralMechanicsElementUtilities::CalculateRayleighDampingMatrix(
        rElement, rDampingMatrix, rCurrentProcessInfo, MatrixSize);

    // The original S-M-E-U::CalculateRayleighDampingMatrix calls rElement.CalculateMassMatrix and rElement.CalculateLeftHandSide.
    // If SmallDisplacement (as an Element, not BaseSolidElement) implements these correctly using its own members
    // and other SolidElementUtilities, this *might* just work.
    // This is a placeholder, deeper changes might be needed if this utility itself relies on BaseSolidElement specifics.
}

array_1d<double, 3> SolidElementUtilities::GetBodyForce(
    const Element& rElement,
    const Element::GeometryType::IntegrationPointsArrayType& rIntegrationPoints,
    const IndexType PointNumber)
{
    return StructuralMechanicsElementUtilities::GetBodyForce(rElement, rIntegrationPoints, PointNumber);
}

void SolidElementUtilities::CalculateLumpedMassVector(
    const Element& rElement,
    VectorType& rLumpedMassVector,
    const ProcessInfo& rCurrentProcessInfo
    )
{
    // This function in S-M-E-U might need rElement to provide GetGeometry(), GetProperties(), etc.
    // which an Element does.
    // Original BaseSolidElement had its own version, this points to S-M-E-U's.
    // We need to ensure that the new SmallDisplacement element, when passed as 'rElement',
    // provides all necessary interfaces for this utility to work.
    // The BaseSolidElement::CalculateLumpedMassVector was:
    const auto& r_geom = rElement.GetGeometry();
    const auto& r_prop = rElement.GetProperties();
    const SizeType dimension = r_geom.WorkingSpaceDimension();
    const SizeType number_of_nodes = r_geom.size();
    const SizeType mat_size = dimension * number_of_nodes;

    if (rLumpedMassVector.size() != mat_size)
        rLumpedMassVector.resize( mat_size, false );

    const double density = GetDensityForMassMatrixComputation(rElement);
    const double thickness = (dimension == 2 && r_prop.Has(THICKNESS)) ? r_prop[THICKNESS] : 1.0;

    const double total_mass = r_geom.DomainSize() * density * thickness;

    Vector lumping_factors;
    lumping_factors = r_geom.LumpingFactors( lumping_factors );

    for ( IndexType i = 0; i < number_of_nodes; ++i ) {
        const double temp = lumping_factors[i] * total_mass;
        for ( IndexType j = 0; j < dimension; ++j ) {
            IndexType index = i * dimension + j;
            rLumpedMassVector[index] = temp;
        }
    }
}


bool SolidElementUtilities::IsElementRotated(
    const Element& rElement,
    const ConstitutiveLaw::Pointer& pConstitutiveLaw)
{
    if (!pConstitutiveLaw) return false;
    const SizeType strain_size = pConstitutiveLaw->GetStrainSize();
    if (strain_size == 6) { // 3D
        return (rElement.Has(LOCAL_AXIS_1) && rElement.Has(LOCAL_AXIS_2));
    } else if (strain_size == 3) { // 2D
        return (rElement.Has(LOCAL_AXIS_1));
    }
    return false;
}

void SolidElementUtilities::BuildRotationMatrix(
    BoundedMatrix<double, 3, 3>& rRotationMatrix,
    const array_1d<double, 3>& rLocalAxis1,
    const array_1d<double, 3>& rLocalAxis2, // Pass this in, even if zero for 2D
    const SizeType StrainSize)
{
    array_1d<double, 3> local_axis_1_norm = rLocalAxis1 / norm_2(rLocalAxis1);
    array_1d<double, 3> local_axis_2_computed;
    array_1d<double, 3> local_axis_3_computed;

    if (StrainSize == 6) { // 3D
        local_axis_2_computed = rLocalAxis2 - MathUtils<double>::Dot(rLocalAxis2, local_axis_1_norm) * local_axis_1_norm;
        local_axis_2_computed /= norm_2(local_axis_2_computed);
        local_axis_3_computed = MathUtils<double>::CrossProduct(local_axis_1_norm, local_axis_2_computed);
    } else if (StrainSize == 3) { // 2D (assuming XY plane)
        local_axis_2_computed[0] = -local_axis_1_norm[1];
        local_axis_2_computed[1] = local_axis_1_norm[0];
        local_axis_2_computed[2] = 0.0;
        local_axis_3_computed[0] = 0.0;
        local_axis_3_computed[1] = 0.0;
        local_axis_3_computed[2] = 1.0; // Normal to XY plane
    } else {
        KRATOS_ERROR << "BuildRotationMatrix: Unsupported strain size: " << StrainSize << std::endl;
    }
    StructuralMechanicsElementUtilities::BuildRotationMatrix(rRotationMatrix, local_axis_1_norm, local_axis_2_computed, local_axis_3_computed);
}


void SolidElementUtilities::RotateVectorToLocalAxes( Vector& rVector, const BoundedMatrix<double, 3, 3>& rRotationMatrix, const SizeType StrainSize)
{
    if (StrainSize == 6) { // 3D
        BoundedMatrix<double, 6, 6> voigt_rotation_matrix;
        ConstitutiveLawUtilities<6>::CalculateRotationOperatorVoigt(rRotationMatrix, voigt_rotation_matrix);
        rVector = prod(voigt_rotation_matrix, rVector);
    } else if (StrainSize == 3) { // 2D
        BoundedMatrix<double, 3, 3> voigt_rotation_matrix_2d; // Corrected to BoundedMatrix
        ConstitutiveLawUtilities<3>::CalculateRotationOperatorVoigt(rRotationMatrix, voigt_rotation_matrix_2d);
        rVector = prod(voigt_rotation_matrix_2d, rVector);
    }
}

void SolidElementUtilities::RotateVectorToGlobalAxes( Vector& rVector, const BoundedMatrix<double, 3, 3>& rRotationMatrix, const SizeType StrainSize)
{
    if (StrainSize == 6) { // 3D
        BoundedMatrix<double, 6, 6> voigt_rotation_matrix;
        ConstitutiveLawUtilities<6>::CalculateRotationOperatorVoigt(rRotationMatrix, voigt_rotation_matrix);
        rVector = prod(trans(voigt_rotation_matrix), rVector);
    } else if (StrainSize == 3) { // 2D
        BoundedMatrix<double, 3, 3> voigt_rotation_matrix_2d; // Corrected to BoundedMatrix
        ConstitutiveLawUtilities<3>::CalculateRotationOperatorVoigt(rRotationMatrix, voigt_rotation_matrix_2d);
        rVector = prod(trans(voigt_rotation_matrix_2d), rVector);
    }
}

void SolidElementUtilities::RotateMatrixToLocalAxes( Matrix& rMatrix, const BoundedMatrix<double, 3, 3>& rRotationMatrix, const SizeType StrainSize)
{
    Matrix aux_matrix = rMatrix; // Temporary copy
    if (StrainSize == 6) { // 3D
        BoundedMatrix<double, 6, 6> voigt_rotation_matrix;
        ConstitutiveLawUtilities<6>::CalculateRotationOperatorVoigt(rRotationMatrix, voigt_rotation_matrix);
        noalias(rMatrix) = prod(voigt_rotation_matrix, aux_matrix);
        noalias(aux_matrix) = prod(rMatrix, trans(voigt_rotation_matrix)); // rMatrix * trans(V)
        noalias(rMatrix) = aux_matrix;
    } else if (StrainSize == 3) { // 2D
        BoundedMatrix<double, 3, 3> voigt_rotation_matrix_2d;
        ConstitutiveLawUtilities<3>::CalculateRotationOperatorVoigt(rRotationMatrix, voigt_rotation_matrix_2d);
        noalias(rMatrix) = prod(voigt_rotation_matrix_2d, aux_matrix);
        noalias(aux_matrix) = prod(rMatrix, trans(voigt_rotation_matrix_2d));
        noalias(rMatrix) = aux_matrix;
    }
}

void SolidElementUtilities::RotateMatrixToGlobalAxes( Matrix& rMatrix, const BoundedMatrix<double, 3, 3>& rRotationMatrix, const SizeType StrainSize)
{
    Matrix aux_matrix = rMatrix; // Temporary copy
    if (StrainSize == 6) { // 3D
        BoundedMatrix<double, 6, 6> voigt_rotation_matrix;
        ConstitutiveLawUtilities<6>::CalculateRotationOperatorVoigt(rRotationMatrix, voigt_rotation_matrix);
        noalias(rMatrix) = prod(trans(voigt_rotation_matrix), aux_matrix);
        noalias(aux_matrix) = prod(rMatrix, voigt_rotation_matrix); // trans(V) * rMatrix * V
        noalias(rMatrix) = aux_matrix;
    } else if (StrainSize == 3) { // 2D
        BoundedMatrix<double, 3, 3> voigt_rotation_matrix_2d;
        ConstitutiveLawUtilities<3>::CalculateRotationOperatorVoigt(rRotationMatrix, voigt_rotation_matrix_2d);
        noalias(rMatrix) = prod(trans(voigt_rotation_matrix_2d), aux_matrix);
        noalias(aux_matrix) = prod(rMatrix, voigt_rotation_matrix_2d);
        noalias(rMatrix) = aux_matrix;
    }
}

void SolidElementUtilities::RotateFToLocalAxes( Matrix& rF, const BoundedMatrix<double, 3, 3>& rRotationMatrix)
{
    Matrix temp_F = rF;
    BoundedMatrix<double,3,3> inv_rotation_matrix;
    double det_rot;
    MathUtils<double>::InvertMatrix3(rRotationMatrix, inv_rotation_matrix, det_rot);
    noalias(rF) = prod(rRotationMatrix, temp_F);
    noalias(temp_F) = prod(rF, inv_rotation_matrix);
    noalias(rF) = temp_F;
}

void SolidElementUtilities::RotateFToGlobalAxes( Matrix& rF, const BoundedMatrix<double, 3, 3>& rRotationMatrix)
{
    Matrix temp_F = rF;
    BoundedMatrix<double,3,3> inv_rotation_matrix;
    double det_rot;
    MathUtils<double>::InvertMatrix3(rRotationMatrix, inv_rotation_matrix, det_rot);
    noalias(rF) = prod(inv_rotation_matrix, temp_F);
    noalias(temp_F) = prod(rF, rRotationMatrix);
    noalias(rF) = temp_F;
}

// Full specialization for template GetValueFromConstitutiveLaw might be tricky in one go.
// Let's assume it's defined elsewhere or will be added later.
// For now, a placeholder for the CalculateConstitutiveVariables.
void SolidElementUtilities::CalculateConstitutiveVariables(
    LocalKinematicVariables& rThisKinematicVariables,
    LocalConstitutiveVariables& rThisConstitutiveVariables,
    ConstitutiveLaw::Parameters& rValues,
    const ConstitutiveLaw::Pointer& pConstitutiveLaw,
    const ConstitutiveLaw::StressMeasure ThisStressMeasure,
    const bool IsElementRotatedFlag,
    const Element& rElementForRotation)
{
    // Set CL parameters from Kinematic and Constitutive Variables
    rValues.SetShapeFunctionsValues(rThisKinematicVariables.N);
    rValues.SetDeterminantF(rThisKinematicVariables.detF);
    rValues.SetDeformationGradientF(rThisKinematicVariables.F);
    // StrainVector might be set by element or CL
    // rValues.SetStrainVector(rThisConstitutiveVariables.StrainVector);
    rValues.SetStressVector(rThisConstitutiveVariables.StressVector);
    rValues.SetConstitutiveMatrix(rThisConstitutiveVariables.D);

    if (IsElementRotatedFlag) {
        BoundedMatrix<double, 3, 3> rotation_matrix;
        array_1d<double, 3> local_axis_1 = ZeroVector(3);
        array_1d<double, 3> local_axis_2 = ZeroVector(3); // Will be computed if needed
        if (rElementForRotation.Has(LOCAL_AXIS_1)) local_axis_1 = rElementForRotation.GetValue(LOCAL_AXIS_1);
        if (rElementForRotation.Has(LOCAL_AXIS_2)) local_axis_2 = rElementForRotation.GetValue(LOCAL_AXIS_2);

        BuildRotationMatrix(rotation_matrix, local_axis_1, local_axis_2, pConstitutiveLaw->GetStrainSize());

        if (rValues.GetOptions().Is(ConstitutiveLaw::USE_ELEMENT_PROVIDED_STRAIN)) {
            RotateVectorToLocalAxes(rValues.GetStrainVector(), rotation_matrix, pConstitutiveLaw->GetStrainSize());
        } else { // Rotate F
            RotateFToLocalAxes(rValues.GetDeformationGradientF(), rotation_matrix); // Modifies F in rValues
        }
    }

    pConstitutiveLaw->CalculateMaterialResponse(rValues, ThisStressMeasure);

    if (IsElementRotatedFlag) {
        BoundedMatrix<double, 3, 3> rotation_matrix; // Rebuild or pass from above
        array_1d<double, 3> local_axis_1 = ZeroVector(3);
        array_1d<double, 3> local_axis_2 = ZeroVector(3);
        if (rElementForRotation.Has(LOCAL_AXIS_1)) local_axis_1 = rElementForRotation.GetValue(LOCAL_AXIS_1);
        if (rElementForRotation.Has(LOCAL_AXIS_2)) local_axis_2 = rElementForRotation.GetValue(LOCAL_AXIS_2);
        BuildRotationMatrix(rotation_matrix, local_axis_1, local_axis_2, pConstitutiveLaw->GetStrainSize());

        if (rValues.GetOptions().Is(ConstitutiveLaw::USE_ELEMENT_PROVIDED_STRAIN)) {
             RotateVectorToGlobalAxes(rValues.GetStrainVector(), rotation_matrix, pConstitutiveLaw->GetStrainSize());
        } else { // Rotate F back
            RotateFToGlobalAxes(rValues.GetDeformationGradientF(), rotation_matrix);
        }
        if (rValues.GetOptions().Is(ConstitutiveLaw::COMPUTE_STRESS)) {
            RotateVectorToGlobalAxes(rValues.GetStressVector(), rotation_matrix, pConstitutiveLaw->GetStrainSize());
        }
        if (rValues.GetOptions().Is(ConstitutiveLaw::COMPUTE_CONSTITUTIVE_TENSOR)) {
            RotateMatrixToGlobalAxes(rValues.GetConstitutiveMatrix(), rotation_matrix, pConstitutiveLaw->GetStrainSize());
        }
    }
}

void SolidElementUtilities::CalculateKinematicVariablesTotalLagrangian(
    LocalKinematicVariables& rThisKinematicVariables,
    const IndexType PointNumber,
    const IntegrationMethod& rIntegrationMethod,
    const Element::GeometryType& rGeom,
    bool UseGeometryIntegrationMethod)
{
    rThisKinematicVariables.detJ0 = SolidElementUtilities::CalculateDerivativesOnReferenceConfiguration(
        rThisKinematicVariables.J0, rThisKinematicVariables.InvJ0, rThisKinematicVariables.DN_DX, PointNumber, rIntegrationMethod, rGeom, UseGeometryIntegrationMethod);
    KRATOS_ERROR_IF(rThisKinematicVariables.detJ0 < 0.0) << "DETJ0 is negative: " << rThisKinematicVariables.detJ0 << std::endl;

    // Compute deformation gradient F
    // F_ik = dx_i/dX_k = sum_L (x_L_i * dNL/dX_k)
    Matrix current_disp(rGeom.size(), rGeom.WorkingSpaceDimension());
    for(SizeType node_i = 0; node_i < rGeom.size(); ++node_i) {
        const auto& disp = rGeom[node_i].FastGetSolutionStepValue(DISPLACEMENT);
        for(SizeType comp = 0; comp < rGeom.WorkingSpaceDimension(); ++comp) {
            current_disp(node_i, comp) = disp[comp];
        }
    }

    rThisKinematicVariables.F = IdentityMatrix(rGeom.WorkingSpaceDimension()); // F = I + Grad_disp_ref
    for(SizeType i = 0; i < rGeom.WorkingSpaceDimension(); ++i) {
        for(SizeType j = 0; j < rGeom.WorkingSpaceDimension(); ++j) {
            for(SizeType node_k = 0; node_k < rGeom.size(); ++node_k) {
                 rThisKinematicVariables.F(i,j) += current_disp(node_k, i) * rThisKinematicVariables.DN_DX(node_k,j);
            }
        }
    }
    rThisKinematicVariables.detF = MathUtils<double>::Det(rThisKinematicVariables.F);
    KRATOS_ERROR_IF(rThisKinematicVariables.detF < 0.0) << "DETF is negative: " << rThisKinematicVariables.detF << std::endl;

    // Compute Green-Lagrange strain tensor E = 0.5 * (F^T * F - I)
    // This is typically done by the CL when USE_ELEMENT_PROVIDED_STRAIN is false.
    // The B matrix will be for Green-Lagrange.
}

void SolidElementUtilities::CalculateBTotalLagrangian(
    Matrix& rB,
    const Matrix& rF,
    const Matrix& rDN_DX,
    const SizeType StrainSize,
    const SizeType Dimension,
    const SizeType NumberOfNodes,
    const Vector& rN) // rN for axisymmetric
{
    KRATOS_TRY
    if (rB.size1() != StrainSize || rB.size2() != Dimension * NumberOfNodes) {
        rB.resize(StrainSize, Dimension * NumberOfNodes, false);
    }
    noalias(rB) = ZeroMatrix(StrainSize, Dimension * NumberOfNodes);

    if (Dimension == 2 && StrainSize == 3) { // 2D plane strain or plane stress (StrainSize = 3 for e_xx, e_yy, gamma_xy)
        for (IndexType i = 0; i < NumberOfNodes; ++i) {
            rB(0, i * 2 + 0) = rF(0, 0) * rDN_DX(i, 0) + rF(1, 0) * rDN_DX(i, 1);
            rB(1, i * 2 + 1) = rF(0, 1) * rDN_DX(i, 0) + rF(1, 1) * rDN_DX(i, 1);
            rB(2, i * 2 + 0) = rF(0, 1) * rDN_DX(i, 0) + rF(1, 1) * rDN_DX(i, 1);
            rB(2, i * 2 + 1) = rF(0, 0) * rDN_DX(i, 0) + rF(1, 0) * rDN_DX(i, 1);
        }
    } else if (Dimension == 3 && StrainSize == 6) { // 3D
        for (IndexType i = 0; i < NumberOfNodes; i++) {
            rB(0, i * 3 + 0) = rF(0, 0) * rDN_DX(i, 0) + rF(1, 0) * rDN_DX(i, 1) + rF(2, 0) * rDN_DX(i, 2);
            rB(1, i * 3 + 1) = rF(0, 1) * rDN_DX(i, 0) + rF(1, 1) * rDN_DX(i, 1) + rF(2, 1) * rDN_DX(i, 2);
            rB(2, i * 3 + 2) = rF(0, 2) * rDN_DX(i, 0) + rF(1, 2) * rDN_DX(i, 1) + rF(2, 2) * rDN_DX(i, 2);
            rB(3, i * 3 + 0) = rF(0, 1) * rDN_DX(i, 0) + rF(1, 1) * rDN_DX(i, 1) + rF(2, 1) * rDN_DX(i, 2);
            rB(3, i * 3 + 1) = rF(0, 0) * rDN_DX(i, 0) + rF(1, 0) * rDN_DX(i, 1) + rF(2, 0) * rDN_DX(i, 2);
            rB(4, i * 3 + 1) = rF(0, 2) * rDN_DX(i, 0) + rF(1, 2) * rDN_DX(i, 1) + rF(2, 2) * rDN_DX(i, 2);
            rB(4, i * 3 + 2) = rF(0, 1) * rDN_DX(i, 0) + rF(1, 1) * rDN_DX(i, 1) + rF(2, 1) * rDN_DX(i, 2);
            rB(5, i * 3 + 0) = rF(0, 2) * rDN_DX(i, 0) + rF(1, 2) * rDN_DX(i, 1) + rF(2, 2) * rDN_DX(i, 2);
            rB(5, i * 3 + 2) = rF(0, 0) * rDN_DX(i, 0) + rF(1, 0) * rDN_DX(i, 1) + rF(2, 0) * rDN_DX(i, 2);
        }
    } else if (Dimension == 2 && StrainSize == 4){ // Axisymmetric (e_RR, e_ZZ, e_TT, gamma_RZ)
         // This was adapted from TotalLagrangian::CalculateB in original BaseSolidElement version
        // double current_radius = 0.0;
        // For TL, B depends on F which is d(current_config)/d(ref_config).
        // The hoop strain component derivative involves current radius 'r' and initial radius 'R0'.
        // The original TotalLagrangianAxisymmetricElement had a specific B matrix calculation.
        // This utility should replicate it. For now, this is a simplified placeholder.
        // It is assumed F passed here is the 2x2 RZ-plane part of the deformation gradient.
        // The F_TT = r/R0 component is needed for the hoop strain term in B.

        // Calculate current radius 'r' using current displacements and initial radius 'R0'
        // R0 is needed at the integration point. rN are shape functions at IP.
        // r = sum(N_i * (X0_R_i + uR_i))
        // R0 = sum(N_i * X0_R_i)
        // This part needs to be consistent with how F_TT = r/R0 is defined for kinematic variables.
        // The CalculateBAxisymTotalLagrangian in the script uses 'InitialRadius' which seems to be R0.
        // Let's assume rN are shape functions and rF contains F_TT implicitly or explicitly.

        KRATOS_WARNING_ONCE("SolidElementUtilities") << "Axisymmetric B for TotalLagrangian in utilities is simplified and needs review for consistency with F_TT and strain definitions." << std::endl;
        for (IndexType i = 0; i < NumberOfNodes; ++i) {
            // Derivatives of Green-Lagrange strain components w.r.t nodal displacements
            // E_RR = 0.5 * (F_RR^2 + F_ZR^2 - 1)
            // E_ZZ = 0.5 * (F_RZ^2 + F_ZZ^2 - 1)
            // E_TT = 0.5 * (F_TT^2 - 1)  (where F_TT = r/R0)
            // 2E_RZ = F_RR*F_RZ + F_ZR*F_ZZ

            // Simplified B terms (these are linear in F, which is not fully correct for dE/du)
            rB(0, i * 2 + 0) = rF(0,0) * rDN_DX(i,0); // d(E_RR)/d(uR_i) type term
            rB(1, i * 2 + 1) = rF(1,1) * rDN_DX(i,1); // d(E_ZZ)/d(uZ_i) type term
            if (InitialRadius > 1e-9) {
                 // This F(2,2) should correspond to F_TT = r/R0. The derivative is more complex.
                 // d(E_TT)/d(uR_i) = F_TT * d(F_TT)/d(uR_i) = (r/R0) * (N_i/R0)
                 rB(2, i * 2 + 0) = (rF(2,2)) * rN[i] / InitialRadius; // This needs to be F_TT * N_i / R0
            } else {
                 rB(2, i * 2 + 0) = 0.0;
            }
            rB(3, i * 2 + 0) = rF(0,0) * rDN_DX(i,1) + rF(1,0) * rDN_DX(i,0); // d(2E_RZ)/d(uR_i) (F_RR*dN/dZ + F_ZR*dN/dR)
            rB(3, i * 2 + 1) = rF(0,1) * rDN_DX(i,1) + rF(1,1) * rDN_DX(i,0); // d(2E_RZ)/d(uZ_i) (F_RZ*dN/dZ + F_ZZ*dN/dR)
        }
    } else {
         KRATOS_ERROR << "CalculateBTotalLagrangian: Unsupported Dimension/StrainSize combination: " << Dimension << "/" << StrainSize << std::endl;
    }
    KRATOS_CATCH("")
}

void SolidElementUtilities::SetTotalLagrangianConstitutiveVariables(
    const LocalKinematicVariables& rThisKinematicVariables,
    LocalConstitutiveVariables& rThisConstitutiveVariables, // CL will fill StrainVector from F if not element provided
    ConstitutiveLaw::Parameters& rValues)
{
    // For TL, F is the primary kinematic quantity for the CL if strain is not element-provided.
    rValues.SetShapeFunctionsValues(rThisKinematicVariables.N);
    rValues.SetDeterminantF(rThisKinematicVariables.detF);
    rValues.SetDeformationGradientF(rThisKinematicVariables.F);

    // CL will calculate Green-Lagrange strain from F and set it in rValues.GetStrainVector() if USE_ELEMENT_PROVIDED_STRAIN is false.
    // rValues.SetStrainVector(rThisConstitutiveVariables.StrainVector); // CL reads from or writes to this via rValues
    rValues.SetStressVector(rThisConstitutiveVariables.StressVector);   // CL writes here
    rValues.SetConstitutiveMatrix(rThisConstitutiveVariables.D);        // CL writes here
}

Parameters SolidElementUtilities::GetDefaultSolidSpecifications(SizeType Dimension)
{
    Parameters specifications = Parameters(R"({
        "time_integration"           : ["static","implicit","explicit"],
        "framework"                  : "lagrangian",
        "symmetric_lhs"              : true,
        "positive_definite_lhs"      : true,
        "output"                     : {
            "gauss_point"            : ["INTEGRATION_WEIGHT","VON_MISES_STRESS","PK2_STRESS_VECTOR","GREEN_LAGRANGE_STRAIN_VECTOR","CONSTITUTIVE_MATRIX","DEFORMATION_GRADIENT","CAUCHY_STRESS_VECTOR"],
            "nodal_historical"       : ["DISPLACEMENT","VELOCITY","ACCELERATION"],
            "nodal_non_historical"   : ["NODAL_MASS", "FORCE_RESIDUAL"],
            "entity"                 : []
        },
        "required_variables"         : ["DISPLACEMENT"],
        "required_dofs"              : [],
        "flags_used"                 : [],
        "compatible_geometries"      : ["Triangle2D3", "Triangle2D6", "Quadrilateral2D4", "Quadrilateral2D8", "Quadrilateral2D9","Tetrahedra3D4", "Prism3D6", "Prism3D15", "Hexahedra3D8", "Hexahedra3D20", "Hexahedra3D27", "Tetrahedra3D10"],
        "element_integrates_in_time" : true,
        "compatible_constitutive_laws": {
            "type"        : ["PlaneStrain","ThreeDimensional", "Axisymmetric"],
            "dimension"   : ["2D","3D"],
            "strain_size" : [3, 4, 6]
        },
        "required_polynomial_degree_of_geometry" : -1,
        "documentation"   : "Default specifications for a solid displacement-based element."
    })");

    std::vector<std::string> dofs;
    if (Dimension == 2) {
        dofs = {"DISPLACEMENT_X","DISPLACEMENT_Y"};
    } else { // 3D
        dofs = {"DISPLACEMENT_X","DISPLACEMENT_Y","DISPLACEMENT_Z"};
    }
    specifications["required_dofs"].SetStringArray(dofs);
    return specifications;
}

template<class TValueType>
void SolidElementUtilities::CalculateValueOnConstitutiveLaw(
    const Element& rElement,
    const std::vector<ConstitutiveLaw::Pointer>& rConstitutiveLawVector,
    const Variable<TValueType>& rVariable,
    const ProcessInfo& rCurrentProcessInfo,
    const IntegrationMethod& rIntegrationMethod,
    bool UseElementProvidedStrainBool, // Renamed to avoid conflict
    ConstitutiveLaw::StressMeasure rStressMeasureEnum, // Renamed
    std::vector<TValueType>& rOutput)
{
    KRATOS_TRY
    KRATOS_ERROR_IF(rConstitutiveLawVector.empty()) << "ConstitutiveLawVector is empty for element " << rElement.Id() << std::endl;

    const auto& r_geom = rElement.GetGeometry();
    const auto& r_props = rElement.GetProperties();
    const SizeType num_nodes = r_geom.size();
    const SizeType dim = r_geom.WorkingSpaceDimension();
    const SizeType strain_size = rConstitutiveLawVector[0]->GetStrainSize();

    if (rOutput.size() != rConstitutiveLawVector.size()) {
        rOutput.resize(rConstitutiveLawVector.size());
    }

    LocalKinematicVariables kin_vars(strain_size, dim, num_nodes);
    LocalConstitutiveVariables const_vars(strain_size); // CL might use this via params
    // Get current displacements if needed by kinematic calculations for this variable
    // This depends on what variable is being calculated. For some, kinematics might not depend on current displacement.
    // However, CalculateKinematicVariablesSmallDisplacement does take displacements.
    rElement.GetValuesVector(kin_vars.Displacements);


    ConstitutiveLaw::Parameters cl_params(r_geom, r_props, rCurrentProcessInfo);
    Flags& cl_options = cl_params.GetOptions();
    cl_options.Set(ConstitutiveLaw::USE_ELEMENT_PROVIDED_STRAIN, UseElementProvidedStrainBool);
    // Set other flags based on the variable being calculated (e.g., COMPUTE_STRESS, COMPUTE_CONSTITUTIVE_TENSOR)
    // This is a generic function, so we assume flags are set appropriately by the caller or are general.
    // For a generic CalculateValue, we might not need to set COMPUTE_STRESS/TENSOR unless the specific variable implies it.
    // Let's assume for now they are not strictly needed for all CalculateValue calls or are handled by CL.
    cl_options.Set(ConstitutiveLaw::COMPUTE_STRESS, false);
    cl_options.Set(ConstitutiveLaw::COMPUTE_CONSTITUTIVE_TENSOR, false);

    if(UseElementProvidedStrainBool) cl_params.SetStrainVector(const_vars.StrainVector);
    cl_params.SetStressVector(const_vars.StressVector);
    cl_params.SetConstitutiveMatrix(const_vars.D);


    const auto& integration_points = r_geom.IntegrationPoints(rIntegrationMethod);

    for (IndexType i = 0; i < rConstitutiveLawVector.size(); ++i) {
        // Kinematics might be needed depending on the variable
        // THIS IS A HUGE ASSUMPTION: that CalculateValueOnConstitutiveLaw is only called by SD elements for now.
        // TODO: Pass a function pointer or similar for the kinematic calculation if this utility is to be truly general.
        CalculateKinematicVariablesSmallDisplacement(
            kin_vars, i, rIntegrationMethod, r_geom, kin_vars.Displacements, strain_size, true /*UseGeomIntegration*/);

        // Set CL params from kinematic variables
        cl_params.SetShapeFunctionsValues(kin_vars.N);
        cl_params.SetDeterminantF(kin_vars.detF);
        cl_params.SetDeformationGradientF(kin_vars.F);
        if (UseElementProvidedStrainBool) {
             Vector current_strain = prod(kin_vars.B, kin_vars.Displacements); // Recalculate strain for this IP
             cl_params.SetStrainVector(current_strain);
        }

        bool is_rotated = IsElementRotated(rElement, rConstitutiveLawVector[i]);
        if (is_rotated) {
            // Apply rotation to F or strain in cl_params before calling CL
            // This is simplified; actual rotation logic from CalculateConstitutiveVariables would be needed
        }

        rOutput[i] = rConstitutiveLawVector[i]->CalculateValue(cl_params, rVariable, rOutput[i]);

        if (is_rotated) {
            // Rotate output back if necessary (e.g. if rOutput[i] is a stress/strain vector/tensor)
        }
    }
    KRATOS_CATCH("")
}

// Explicit instantiations for common types used with CalculateValueOnConstitutiveLaw
// This is necessary if the definition is in a .cpp file.
template KRATOS_API(STRUCTURAL_MECHANICS_APPLICATION) void SolidElementUtilities::CalculateValueOnConstitutiveLaw<double>(const Element&, const std::vector<ConstitutiveLaw::Pointer>&, const Variable<double>&, const ProcessInfo&, const IntegrationMethod&, bool, ConstitutiveLaw::StressMeasure, std::vector<double>&);
template KRATOS_API(STRUCTURAL_MECHANICS_APPLICATION) void SolidElementUtilities::CalculateValueOnConstitutiveLaw<Vector>(const Element&, const std::vector<ConstitutiveLaw::Pointer>&, const Variable<Vector>&, const ProcessInfo&, const IntegrationMethod&, bool, ConstitutiveLaw::StressMeasure, std::vector<Vector>&);
template KRATOS_API(STRUCTURAL_MECHANICS_APPLICATION) void SolidElementUtilities::CalculateValueOnConstitutiveLaw<Matrix>(const Element&, const std::vector<ConstitutiveLaw::Pointer>&, const Variable<Matrix>&, const ProcessInfo&, const IntegrationMethod&, bool, ConstitutiveLaw::StressMeasure, std::vector<Matrix>&);
template KRATOS_API(STRUCTURAL_MECHANICS_APPLICATION) void SolidElementUtilities::CalculateValueOnConstitutiveLaw<array_1d<double,3>>(const Element&, const std::vector<ConstitutiveLaw::Pointer>&, const Variable<array_1d<double,3>>&, const ProcessInfo&, const IntegrationMethod&, bool, ConstitutiveLaw::StressMeasure, std::vector<array_1d<double,3>>&);


void SolidElementUtilities::CalculateKinematicVariablesAxisymSmallDisplacement(
    LocalKinematicVariables& rThisKinematicVariables,
    const IndexType PointNumber,
    const IntegrationMethod& rIntegrationMethod,
    const Element::GeometryType& rGeom,
    const Vector& rDisplacements,
    bool UseGeometryIntegrationMethod)
{
    // For axisymmetric, dimension is 2 (R, Z), but strain_size is 4 (e_RR, e_ZZ, e_RZ, e_TT)
    // rThisKinematicVariables.B should be sized for 4x(N*2)
    // rThisKinematicVariables.DN_DX is N_nodes x 2 (d/dR, d/dZ)

    const Element::GeometryType::IntegrationPointsArrayType& r_integration_points = rGeom.IntegrationPoints(rIntegrationMethod);
    rThisKinematicVariables.N = rGeom.ShapeFunctionsValues(rThisKinematicVariables.N, r_integration_points[PointNumber].Coordinates());

    rThisKinematicVariables.detJ0 = SolidElementUtilities::CalculateDerivativesOnReferenceConfiguration(
        rThisKinematicVariables.J0, rThisKinematicVariables.InvJ0, rThisKinematicVariables.DN_DX, PointNumber, rIntegrationMethod, rGeom, UseGeometryIntegrationMethod);
    KRATOS_ERROR_IF(rThisKinematicVariables.detJ0 < 0.0) << "DETJ0 is negative: " << rThisKinematicVariables.detJ0 << std::endl;

    // Calculate current radius. For small displacements, current radius ~ initial radius.
    // Global coordinates of the integration point
    Point global_coords_point;
    rGeom.GlobalCoordinates(global_coords_point, r_integration_points[PointNumber]);
    const double current_radius = global_coords_point.X(); // Assuming R is the X-coordinate in global system for axisymmetric
    KRATOS_ERROR_IF(current_radius < std::numeric_limits<double>::epsilon()) << "Radius is close to zero in axisymmetric element at point " << global_coords_point << std::endl;


    CalculateBAxisymSmallDisplacement(rThisKinematicVariables.B, rThisKinematicVariables.DN_DX, rThisKinematicVariables.N, current_radius, rGeom.size());

    // Compute equivalent F (Small Displacement specific part, adapted for axisymmetry if needed)
    // For small displacement, F is often I + grad(u). For axisymmetry, this concept for F might be less direct
    // if strain is directly computed. Usually, for small strain axisymmetry, strain_vector = B * u.
    // F is mainly for CLs that expect it.
    // For now, let's compute a simplified F, assuming small rotations.
    Matrix grad_u = ZeroMatrix(2,2); // d(uR)/dR, d(uR)/dZ; d(uZ)/dR, d(uZ)/dZ
    const SizeType strain_size_ax = 4; // Explicitly use 4 for safety here.
    Vector strain_vector(strain_size_ax);
    noalias(strain_vector) = prod(rThisKinematicVariables.B, rDisplacements); // e_RR, e_ZZ, e_RZ, e_TT

    grad_u(0,0) = strain_vector[0]; // duR/dR approx e_RR
    grad_u(1,1) = strain_vector[1]; // duZ/dZ approx e_ZZ
    grad_u(0,1) = 0.5 * strain_vector[2]; // duR/dZ approx 0.5 * gamma_RZ
    grad_u(1,0) = 0.5 * strain_vector[2]; // duZ/dR approx 0.5 * gamma_RZ

    rThisKinematicVariables.F.resize(2,2,false); // Axisymmetric F is 2x2 in RZ plane
    rThisKinematicVariables.F = IdentityMatrix(2) + grad_u;
    rThisKinematicVariables.detF = MathUtils<double>::Det(rThisKinematicVariables.F); // This detF is for the RZ plane
}

void SolidElementUtilities::CalculateBAxisymSmallDisplacement(
    Matrix& rB,
    const Matrix& rDN_DX, // N_nodes x 2 (d/dR, d/dZ)
    const Vector& rN,     // N_nodes
    const double CurrentRadius,
    const SizeType NumberOfNodes)
{
    // Strain order: e_RR, e_ZZ, e_TT (hoop), gamma_RZ (engineering shear)
    // Displacement order: uR1, uZ1, uR2, uZ2, ...
    const SizeType strain_size = 4;
    if (rB.size1() != strain_size || rB.size2() != NumberOfNodes * 2) {
        rB.resize(strain_size, NumberOfNodes * 2, false);
    }
    noalias(rB) = ZeroMatrix(strain_size, NumberOfNodes * 2);

    for (IndexType i = 0; i < NumberOfNodes; ++i) {
        const SizeType index_R = i * 2;     // Index for uR_i
        const SizeType index_Z = i * 2 + 1; // Index for uZ_i

        // e_RR = d(uR)/dR
        rB(0, index_R) = rDN_DX(i, 0);

        // e_ZZ = d(uZ)/dZ
        rB(1, index_Z) = rDN_DX(i, 1);

        // e_TT = uR/R
        if (CurrentRadius > std::numeric_limits<double>::epsilon()){ // Avoid division by zero
             rB(2, index_R) = rN[i] / CurrentRadius;
        } else { // Should not happen if radius check is done before
             rB(2, index_R) = 0.0; // Or some other appropriate handling
        }


        // gamma_RZ = d(uR)/dZ + d(uZ)/dR  (engineering shear)
        rB(3, index_R) = rDN_DX(i, 1);
        rB(3, index_Z) = rDN_DX(i, 0);
    }
}

void SolidElementUtilities::CalculateKinematicVariablesAxisymTotalLagrangian(
    LocalKinematicVariables& rThisKinematicVariables,
    const IndexType PointNumber,
    const IntegrationMethod& rIntegrationMethod,
    const Element::GeometryType& rGeom,
    bool UseGeometryIntegrationMethod)
{
    // F for axisymmetric TL:
    // F_RR = duR/dXR + 1, F_RZ = duR/dXZ
    // F_ZR = duZ/dXR    , F_ZZ = duZ/dXZ + 1
    // F_TT = uR/XR + 1 (or r/R)
    // The CL usually expects a 3x3 F. We form it here.
    // DN_DX is dN/dXR, dN/dXZ

    rThisKinematicVariables.detJ0 = SolidElementUtilities::CalculateDerivativesOnReferenceConfiguration(
        rThisKinematicVariables.J0, rThisKinematicVariables.InvJ0, rThisKinematicVariables.DN_DX, PointNumber, rIntegrationMethod, rGeom, UseGeometryIntegrationMethod);
    KRATOS_ERROR_IF(rThisKinematicVariables.detJ0 < 0.0) << "DETJ0 is negative: " << rThisKinematicVariables.detJ0 << std::endl;

    Matrix current_disp_mat(rGeom.size(), 2); // uR, uZ for each node
    for(SizeType node_i = 0; node_i < rGeom.size(); ++node_i) {
        const auto& disp = rGeom[node_i].FastGetSolutionStepValue(DISPLACEMENT);
        current_disp_mat(node_i, 0) = disp[0]; // uR
        current_disp_mat(node_i, 1) = disp[1]; // uZ
    }

    // Initial coordinates for radius
    const auto& r_integration_points = rGeom.IntegrationPoints(rIntegrationMethod);
    const Matrix& N_container = rGeom.ShapeFunctionsValues(rIntegrationMethod); // N at this IP
    const Vector& N_at_ip = row(N_container, PointNumber);

    double initial_R = 0.0;
    for(SizeType node_i = 0; node_i < rGeom.size(); ++node_i) {
        initial_R += N_at_ip[node_i] * rGeom[node_i].X0(); // X0 is initial R coordinate
    }
    KRATOS_ERROR_IF(initial_R < 1e-9) << "Initial radius is close to zero." << std::endl;

    double current_r = initial_R; // r = R + uR
    for(SizeType node_i = 0; node_i < rGeom.size(); ++node_i) {
        current_r += N_at_ip[node_i] * current_disp_mat(node_i, 0);
    }
    KRATOS_ERROR_IF(current_r < 1e-9) << "Current radius is close to zero." << std::endl;


    if (rThisKinematicVariables.F.size1() != 3 || rThisKinematicVariables.F.size2() != 3) {
        rThisKinematicVariables.F.resize(3,3,false);
    }
    Matrix F2D = ZeroMatrix(2,2); // For RZ plane
    // F_alpha_beta = delta_alpha_beta + Sum_L ( u_L_alpha * DN_L_beta )
    F2D(0,0) = 1.0; F2D(1,1) = 1.0;
    for(SizeType i=0; i<2; ++i) { // alpha
        for(SizeType j=0; j<2; ++j) { // beta
            for(SizeType L=0; L<rGeom.size(); ++L) {
                F2D(i,j) += current_disp_mat(L,i) * rThisKinematicVariables.DN_DX(L,j);
            }
        }
    }

    // Full 3D F for CL (F_RR, F_RZ, 0; F_ZR, F_ZZ, 0; 0, 0, F_TT)
    rThisKinematicVariables.F(0,0) = F2D(0,0); rThisKinematicVariables.F(0,1) = F2D(0,1); rThisKinematicVariables.F(0,2) = 0.0;
    rThisKinematicVariables.F(1,0) = F2D(1,0); rThisKinematicVariables.F(1,1) = F2D(1,1); rThisKinematicVariables.F(1,2) = 0.0;
    rThisKinematicVariables.F(2,0) = 0.0;    rThisKinematicVariables.F(2,1) = 0.0;    rThisKinematicVariables.F(2,2) = current_r / initial_R;

    rThisKinematicVariables.detF = MathUtils<double>::Det(rThisKinematicVariables.F);
    KRATOS_ERROR_IF(rThisKinematicVariables.detF < 0.0) << "DETF is negative: " << rThisKinematicVariables.detF << std::endl;

    // N (shape functions) are already computed in rThisKinematicVariables.N by SetTotalLagrangianConstitutiveVariables if called from there
    // For direct use, ensure N is available if B matrix calculation needs it.
    rThisKinematicVariables.N = N_at_ip;
}

void SolidElementUtilities::CalculateBAxisymTotalLagrangian(
    Matrix& rB,
    const Matrix& rF, // Full 3x3 F
    const Matrix& rDN_DX, // N_nodes x 2 (d/dXR, d/dXZ)
    const Vector& rN,     // N_nodes
    const double InitialRadius,
    const SizeType NumberOfNodes)
{
    // B for Green-Lagrange in Axisymmetric TL. Strain order: E_RR, E_ZZ, E_TT, 2*E_RZ (engineering)
    // This is a complex derivation. For simplicity, we refer to how Kratos's TotalLagrangianAxisymmetricElement does it.
    // It often involves terms like F_ij * dN_k/dX_L.
    // The original TotalLagrangianAxisymmetricElement directly implemented CalculateB.
    // This utility should replicate that logic. Given the complexity, this will be a simplified placeholder.
    // A full B matrix for TL axisymmetric Green-Lagrange strain is non-trivial.
    // Example structure (conceptual, actual terms are more complex):
    const SizeType strain_size = 4;
    if (rB.size1() != strain_size || rB.size2() != NumberOfNodes * 2) {
        rB.resize(strain_size, NumberOfNodes * 2, false);
    }
    noalias(rB) = ZeroMatrix(strain_size, NumberOfNodes * 2);

    for (IndexType i = 0; i < NumberOfNodes; ++i) {
        // E_RR component of B (related to uR_i)
        rB(0, i * 2 + 0) = rF(0,0) * rDN_DX(i,0); // Simplified d(E_RR)/d(uR_i)
        // E_ZZ component of B (related to uZ_i)
        rB(1, i * 2 + 1) = rF(1,1) * rDN_DX(i,1); // Simplified d(E_ZZ)/d(uZ_i)
        // E_TT component of B (related to uR_i)
        if (InitialRadius > 1e-9)
            rB(2, i * 2 + 0) = rF(2,2) * rN[i] / InitialRadius; // Simplified d(E_TT)/d(uR_i)
        // 2*E_RZ component of B (related to uR_i and uZ_i)
        rB(3, i * 2 + 0) = rF(0,0) * rDN_DX(i,1) + rF(1,0) * rDN_DX(i,0); // Simplified d(2E_RZ)/d(uR_i) (F_RR*dN/dZ + F_ZR*dN/dR)
        rB(3, i * 2 + 1) = rF(0,1) * rDN_DX(i,1) + rF(1,1) * rDN_DX(i,0); // Simplified d(2E_RZ)/d(uZ_i) (F_RZ*dN/dZ + F_ZZ*dN/dR)
    }
    // KRATOS_WARNING_ONCE("SolidElementUtilities") << "CalculateBAxisymTotalLagrangian is a simplified placeholder. Review derivation." << std::endl;
}

void SolidElementUtilities::CalculateKinematicVariablesUpdatedLagrangian(
    LocalKinematicVariables& rThisKinematicVariables, // F is deformation rate, B is for strain rate
    const IndexType PointNumber,
    const IntegrationMethod& rIntegrationMethod,
    const Element::GeometryType& rGeom,
    const ProcessInfo& rCurrentProcessInfo, // For time step
    bool UseGeometryIntegrationMethod)
{
    // For Updated Lagrangian, kinematics are usually in terms of rates.
    // DN_DX is computed on the current configuration.
    // F is often the incremental deformation gradient F_n+1 = F_n * df (df is incremental from current config)
    // Or, for rate formulations, we work with velocity gradient L.
    // Here, let's assume we need DN_DX in current config.
    // The 'F' in LocalKinematicVariables might represent 'df' or 'L*dt' depending on context.

    double detJ_current; // Jacobian determinant in current configuration
    Matrix J_current(rGeom.WorkingSpaceDimension(), rGeom.WorkingSpaceDimension());
    Matrix InvJ_current(rGeom.WorkingSpaceDimension(), rGeom.WorkingSpaceDimension());
    Matrix DN_De_current; // Shape function derivatives w.r.t local coords

    // Derivatives in CURRENT configuration
    if (UseGeometryIntegrationMethod) {
        GeometryUtils::JacobianOnCurrentConfiguration(
            rGeom, rGeom.IntegrationPoints(rIntegrationMethod)[PointNumber], J_current);
        MathUtils<double>::InvertMatrix(J_current, InvJ_current, detJ_current);
        const Matrix& rDN_De = rGeom.ShapeFunctionsLocalGradients(rIntegrationMethod)[PointNumber];
        GeometryUtils::ShapeFunctionsGradients(rDN_De, InvJ_current, rThisKinematicVariables.DN_DX); // DN_DX is now dN/dx (current)
    } else {
        const auto& integration_points = rGeom.IntegrationPoints(rIntegrationMethod);
        GeometryUtils::JacobianOnCurrentConfiguration(rGeom, integration_points[PointNumber], J_current);
        MathUtils<double>::InvertMatrix(J_current, InvJ_current, detJ_current);
        rGeom.ShapeFunctionsLocalGradients(DN_De_current, integration_points[PointNumber]);
        GeometryUtils::ShapeFunctionsGradients(DN_De_current, InvJ_current, rThisKinematicVariables.DN_DX);
    }
    KRATOS_ERROR_IF(detJ_current < 0.0) << "DETJ_CURRENT is negative: " << detJ_current << std::endl;
    rThisKinematicVariables.detJ0 = detJ_current; // Store current detJ in detJ0 for consistent use in GetIntegrationWeight

    // Velocity gradient L = sum( v_i * Grad_current(N_i)^T )
    // F in LocalKinematicVariables will represent L * dt (approx incremental deformation gradient df)
    // Displacements in LocalKinematicVariables will represent nodal velocities for UL rate formulation.
    Vector nodal_velocities;
    rGeom.GetFirstDerivativesVector(nodal_velocities, 0); // Get current velocities

    Matrix L = ZeroMatrix(rGeom.WorkingSpaceDimension(), rGeom.WorkingSpaceDimension());
    for(SizeType node_i = 0; node_i < rGeom.size(); ++node_i) {
        for(SizeType dim_i = 0; dim_i < rGeom.WorkingSpaceDimension(); ++dim_i) {
            for(SizeType dim_j = 0; dim_j < rGeom.WorkingSpaceDimension(); ++dim_j) {
                L(dim_i, dim_j) += nodal_velocities[node_i * rGeom.WorkingSpaceDimension() + dim_i] * rThisKinematicVariables.DN_DX(node_i, dim_j);
            }
        }
    }

    const double dt = rCurrentProcessInfo[DELTA_TIME];
    rThisKinematicVariables.F = IdentityMatrix(rGeom.WorkingSpaceDimension()) + L * dt; // df approx I + L*dt
    rThisKinematicVariables.detF = MathUtils<double>::Det(rThisKinematicVariables.F);

    // B matrix for UL relates nodal velocities to strain rate D = 0.5*(L + L^T)
    // This is the same form as small displacement B, but uses DN_DX in current config
    CalculateBUpdatedLagrangian(rThisKinematicVariables.B, rThisKinematicVariables.DN_DX, rGeom.WorkingSpaceDimension(), rGeom.size());

    // Shape functions N
    const Element::GeometryType::IntegrationPointsArrayType& r_integration_points = rGeom.IntegrationPoints(rIntegrationMethod);
    rThisKinematicVariables.N = rGeom.ShapeFunctionsValues(rThisKinematicVariables.N, r_integration_points[PointNumber].Coordinates());
}

void SolidElementUtilities::CalculateBUpdatedLagrangian(
    Matrix& rB,
    const Matrix& rDN_DX_current,
    const SizeType Dimension,
    const SizeType NumberOfNodes)
{
    // B for strain rate D (symmetric part of velocity gradient L)
    // D_ij = 0.5 * (dui/dxj + duj/dxi)
    // This is structurally the same as small displacement B matrix, but DN_DX is w.r.t current configuration.
    StructuralMechanicsElementUtilities::CalculateBMatrix(rDN_DX_current, rB, Dimension, NumberOfNodes);
}

} // namespace Kratos

// KRATOS  ___|  |                   |                   |
//       \___ \  __|  __| |   |  __| __| |   |  __| _` | |
//             | |   |    |   | (    |   |   | |   (   | |
//       _____/ \__|_|   \__,_|\___|\__|\__,_|_|  \__,_|_| MECHANICS
//
//  License:         BSD License
//                   license: StructuralMechanicsApplication/license.txt
//
//  Main authors:    Philipp Bucher
//                   Vicente Mataix Ferrandiz
//                   Riccardo Rossi
//                   Ruben Zorrilla
//                   Alejandro Cornejo
//

#pragma once

// System includes

// External includes

// Project includes
#include "includes/element.h"

namespace Kratos {
/**
 * @namespace ElementUtilities
 * @ingroup StructuralMechanicsApplication
 * @brief This class includes several utilities necessaries for the computation of the different elements
 * @author Philipp Bucher
 * @author Vicente Mataix Ferrandiz
 * @author Riccardo Rossi
 * @author Ruben Zorrilla
 */
namespace StructuralMechanicsElementUtilities {

/// The size type definition
typedef std::size_t SizeType;

/// The index type definition
typedef std::size_t IndexType;

/// Node type definition
typedef Node NodeType;

/// Geometry definitions
typedef Geometry<NodeType> GeometryType;

/**
 * @brief This method performs commons checks on the solid elements
 * @param rElement Reference to the element
 * @param rCurrentProcessInfo The current process info instance
 * @param rConstitutiveLaws The vector containing CL
 */
int KRATOS_API(STRUCTURAL_MECHANICS_APPLICATION) SolidElementCheck(
    const Element& rElement,
    const ProcessInfo& rCurrentProcessInfo,
    const std::vector<ConstitutiveLaw::Pointer>& rConstitutiveLaws
    );

/**
 * @brief This method computes the deformation gradient F (for small deformation solid elements)
 * @param rElement Reference to the element
 * @param rStrainTensor The strain tensor
 * @param rF The deformation gradient F
 * @tparam TVectorType The vector type
 * @tparam TMatrixType The matrix type
 */
template<class TVectorType, class TMatrixType>
void ComputeEquivalentF(
    const Element& rElement,
    const TVectorType& rStrainTensor,
    TMatrixType& rF
    )
{
    const auto& r_geometry = rElement.GetGeometry();
    const SizeType dimension = r_geometry.WorkingSpaceDimension();

    if(dimension == 2) {
        rF(0,0) = 1.0+rStrainTensor(0);
        rF(0,1) = 0.5*rStrainTensor(2);
        rF(1,0) = 0.5*rStrainTensor(2);
        rF(1,1) = 1.0+rStrainTensor(1);
    } else {
        rF(0,0) = 1.0+rStrainTensor(0);
        rF(0,1) = 0.5*rStrainTensor(3);
        rF(0,2) = 0.5*rStrainTensor(5);
        rF(1,0) = 0.5*rStrainTensor(3);
        rF(1,1) = 1.0+rStrainTensor(1);
        rF(1,2) = 0.5*rStrainTensor(4);
        rF(2,0) = 0.5*rStrainTensor(5);
        rF(2,1) = 0.5*rStrainTensor(4);
        rF(2,2) = 1.0+rStrainTensor(2);
    }
}

/**
 * @brief This method computes the deformation tensor B (for small deformation solid elements)
 * @param rElement Reference to the element
 * @param rDN_DX The shape function derivatives
 * @param rB The deformation tensor B
 * @tparam TMatrixType1 The first matrix type
 * @tparam TMatrixType2 The second matrix type
 */
template<class TMatrixType1, class TMatrixType2>
void CalculateB(
    const GeometricalObject& rElement,
    const TMatrixType1& rDN_DX,
    TMatrixType2& rB
    )
{
    const auto& r_geometry = rElement.GetGeometry();
    const SizeType number_of_nodes = r_geometry.PointsNumber();
    const SizeType dimension = rDN_DX.size2();

    rB.clear();

    if(dimension == 2) {
        for ( IndexType i = 0; i < number_of_nodes; ++i ) {
            const IndexType initial_index = i*2;
            rB(0, initial_index    ) = rDN_DX(i, 0);
            rB(1, initial_index + 1) = rDN_DX(i, 1);
            rB(2, initial_index    ) = rDN_DX(i, 1);
            rB(2, initial_index + 1) = rDN_DX(i, 0);
        }
    } else if(dimension == 3) {
        for ( IndexType i = 0; i < number_of_nodes; ++i ) {
            const IndexType initial_index = i*3;
            rB(0, initial_index    ) = rDN_DX(i, 0);
            rB(1, initial_index + 1) = rDN_DX(i, 1);
            rB(2, initial_index + 2) = rDN_DX(i, 2);
            rB(3, initial_index    ) = rDN_DX(i, 1);
            rB(3, initial_index + 1) = rDN_DX(i, 0);
            rB(4, initial_index + 1) = rDN_DX(i, 2);
            rB(4, initial_index + 2) = rDN_DX(i, 1);
            rB(5, initial_index    ) = rDN_DX(i, 2);
            rB(5, initial_index + 2) = rDN_DX(i, 0);
        }
    }
}

/**
 * @brief This method returns the computed the computed body force
 * @param rElement Reference to the element
 * @param rIntegrationPoints The integrations points
 * @param PointNumber The integration point number
 */
array_1d<double, 3> KRATOS_API(STRUCTURAL_MECHANICS_APPLICATION) GetBodyForce(
    const Element& rElement,
    const GeometryType::IntegrationPointsArrayType& rIntegrationPoints,
    const IndexType PointNumber
    );

/**
 * @brief Method to specify if the lumped or the consistent mass-matrix should be computed
 * @param rProperties The Properties where it is specified
 * @param rCurrentProcessInfo The ProcessInfo where it is specified
 * @return whether to compute the lumped mass-matrix
 */
bool KRATOS_API(STRUCTURAL_MECHANICS_APPLICATION) ComputeLumpedMassMatrix(
    const Properties& rProperties,
    const ProcessInfo& rCurrentProcessInfo);

/**
 * @brief Method to specify if rayligh-damping is specified
 * @param rProperties The Properties where it is specified
 * @param rCurrentProcessInfo The ProcessInfo where it is specified
 * @return whether rayleigh-damping was specified
 */
bool KRATOS_API(STRUCTURAL_MECHANICS_APPLICATION) HasRayleighDamping(
    const Properties& rProperties,
    const ProcessInfo& rCurrentProcessInfo);

/**
 * @brief Method to get the rayleigh-alpha parameter
 * @param rProperties The Properties where it is specified
 * @param rCurrentProcessInfo The ProcessInfo where it is specified
 * @return rayleigh-alpha
 */
double KRATOS_API(STRUCTURAL_MECHANICS_APPLICATION) GetRayleighAlpha(
    const Properties& rProperties,
    const ProcessInfo& rCurrentProcessInfo);

/**
 * @brief Method to get the rayleigh-beta parameter
 * @param rProperties The Properties where it is specified
 * @param rCurrentProcessInfo The ProcessInfo where it is specified
 * @return rayleigh-beta
 */
double KRATOS_API(STRUCTURAL_MECHANICS_APPLICATION) GetRayleighBeta(
    const Properties& rProperties,
    const ProcessInfo& rCurrentProcessInfo);

/**
 * @brief Method to returns the density to be consider for the mass-matrix computation
 * @param rElement The Element for which the mass-matrix should be computed
 * @return The density after apply the mass factor to the element
 */
double KRATOS_API(STRUCTURAL_MECHANICS_APPLICATION) GetDensityForMassMatrixComputation(const Element& rElement);

/**
 * @brief Method to calculate the rayleigh damping-matrix
 * @param rElement The Element for which the damping-matrix should be computed
 * @param rDampingMatrix The damping-matrix of the element
 * @param rCurrentProcessInfo The ProcessInfo where it is specified
 * @param MatrixSize The size of the damping-matrix
 */
void KRATOS_API(STRUCTURAL_MECHANICS_APPLICATION) CalculateRayleighDampingMatrix(
    Element& rElement,
    Element::MatrixType& rDampingMatrix,
    const ProcessInfo& rCurrentProcessInfo,
    const std::size_t MatrixSize);

/**
 * @brief This function calculates the reference length for 2D2N elements
 * @param rElement The Element for which the reference length should be computed
 * @return reference length
 */
double KRATOS_API(STRUCTURAL_MECHANICS_APPLICATION) CalculateReferenceLength2D2N(const Element& rElement);

/**
 * @brief This function calculates the current length for 2D2N elements
 * @param rElement The Element for which the current length should be computed
 * @return current length
 */
double KRATOS_API(STRUCTURAL_MECHANICS_APPLICATION) CalculateCurrentLength2D2N(const Element& rElement);

/**
 * @brief This function calculates the reference length for 3D2N elements
 * @param rElement The Element for which the reference length should be computed
 * @return reference length
 */
double KRATOS_API(STRUCTURAL_MECHANICS_APPLICATION) CalculateReferenceLength3D2N(const Element& rElement);

/**
 * @brief This function calculates the current length for 3D2N elements
 * @param rElement The Element for which the current length should be computed
 * @return current length
 */
double KRATOS_API(STRUCTURAL_MECHANICS_APPLICATION) CalculateCurrentLength3D2N(const Element& rElement);

/**
 * @brief This function checks the norm of the vectors
 * @param rvi The vector of which norm has to be different from null
 */
void KRATOS_API(STRUCTURAL_MECHANICS_APPLICATION) InitialCheckLocalAxes(
    const array_1d<double, 3>& rv1,
    const array_1d<double, 3>& rv2,
    const array_1d<double, 3>& rv3,
    const double Tolerance = 1.0e4*std::numeric_limits<double>::epsilon());

/**
 * @brief This function fills a rotation matrix from a set of vectors
 * @param rRotationMatrix The rotation matrix from global to local axes
 */
void KRATOS_API(STRUCTURAL_MECHANICS_APPLICATION) BuildRotationMatrix(
    BoundedMatrix<double, 3, 3>& rRotationMatrix,
    const array_1d<double, 3>& rv1,
    const array_1d<double, 3>& rv2,
    const array_1d<double, 3>& rv3);

/**
 * @brief This function fills a rotation matrix from an angle of the beam
 * @param rRotationMatrix The rotation matrix from local to global axes
 * It assumes 3 dofs per node: u,v,theta
 */
void BuildRotationMatrixForBeam(
    BoundedMatrix<double, 3, 3>& rRotationMatrix,
    const double AlphaAngle);

/**
 * @brief This function fills an element size rotation matrix a local rotation matrix
 * @param rRotationMatrix The rotation matrix from local to global axes
 * It assumes 3 dofs per node: u,v,theta
 */
void BuildElementSizeRotationMatrixFor2D2NBeam(
    const BoundedMatrix<double, 3, 3>& rRotationMatrix,
    BoundedMatrix<double, 6, 6>& rElementSizeRotationMatrix);

/**
 * @brief This function fills an element size rotation matrix a local rotation matrix
 * @param rRotationMatrix The rotation matrix from local to global axes
 * It assumes 3 dofs per node: u,v,theta
 */
void BuildElementSizeRotationMatrixFor2D3NBeam(
    const BoundedMatrix<double, 3, 3>& rRotationMatrix,
    BoundedMatrix<double, 9, 9>& rElementSizeRotationMatrix);

/**
 * @param rRotationMatrix The rotation matrix from local to global axes
 * It assumes 2 dofs per node: u,v
 */
void BuildRotationMatrixForTruss(
    BoundedMatrix<double, 2, 2>& rRotationMatrix,
    const double AlphaAngle);

/**
 * @brief This function fills an element size rotation matrix a local rotation matrix
 * @param rRotationMatrix The rotation matrix from local to global axes
 * It assumes 2 dofs per node: u,v
 */
void BuildElementSizeRotationMatrixFor2D2NTruss(
    const BoundedMatrix<double, 2, 2>& rRotationMatrix,
    BoundedMatrix<double, 4, 4>& rElementSizeRotationMatrix);

/**
 * @brief This function fills an element size rotation matrix a local rotation matrix
 * @param rRotationMatrix The rotation matrix from local to global axes
 * It assumes 2 dofs per node: u,v
 */
void BuildElementSizeRotationMatrixFor2D3NTruss(
    const BoundedMatrix<double, 2, 2>& rRotationMatrix,
    BoundedMatrix<double, 6, 6>& rElementSizeRotationMatrix);

/**
 * @brief This function fills an element size rotation matrix a local rotation matrix
 * @param rRotationMatrix The rotation matrix from local to global axes
 * It assumes 2 dofs per node: u,v
 */
void BuildElementSizeRotationMatrixFor3D2NTruss(
    const BoundedMatrix<double, 3, 3>& rRotationMatrix,
    BoundedMatrix<double, 6, 6>& rElementSizeRotationMatrix);

/**
 * @brief This function fills an element size rotation matrix a local rotation matrix
 * @param rRotationMatrix The rotation matrix from local to global axes
 * It assumes 2 dofs per node: u,v
 */
void BuildElementSizeRotationMatrixFor3D3NTruss(
    const BoundedMatrix<double, 3, 3>& rRotationMatrix,
    BoundedMatrix<double, 9, 9>& rElementSizeRotationMatrix);

/**
 * @brief This function computes the inclination angle of a 2 noded beam
 * @param rGeometry The geometry of the beam
 * It assumes 3 dofs per node: u,v,theta
 */
double GetReferenceRotationAngle2D2NBeam(const GeometryType &rGeometry);

/**
 * @brief This function computes the inclination angle of a 3 noded beam
 * @param rGeometry The geometry of the beam
 * It assumes 3 dofs per node: u,v,theta
 */
double GetReferenceRotationAngle2D3NBeam(const GeometryType &rGeometry);

/**
 * @brief This function computes the shear psi factor
 * @param rValues The constitutive law parameters
 */
KRATOS_API(STRUCTURAL_MECHANICS_APPLICATION) double CalculatePhi(const Properties& rProperties, const double L, const SizeType Plane = 0);

void InitializeConstitutiveLawValuesForStressCalculation(ConstitutiveLaw::Parameters& rValues,
    Vector& rStrainVector, Vector& rStressVector, Matrix& rConstitutiveMatrix);

void InitializeConstitutiveLawValuesForStressCalculation(ConstitutiveLaw::Parameters& rValues,
    Vector& rStrainVector, Vector& rStressVector);

/**
 * @brief This function builds a Frenet-Serret rotation matrix from local to global in 3D
 * This is valid for STRAIGHT lines
 * @param rGeometry: The geometry of the line element
 */
BoundedMatrix<double, 3, 3> GetFrenetSerretMatrix3D(const GeometryType& rGeometry);

} // namespace StructuralMechanicsElementUtilities.
}  // namespace Kratos.

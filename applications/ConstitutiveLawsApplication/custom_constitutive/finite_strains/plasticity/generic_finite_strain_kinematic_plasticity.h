// KRATOS ___                _   _ _         _   _             __                       _
//       / __\___  _ __  ___| |_(_) |_ _   _| |_(_)_   _____  / /  __ ___      _____   /_\  _ __  _ __
//      / /  / _ \| '_ \/ __| __| | __| | | | __| \ \ / / _ \/ /  / _` \ \ /\ / / __| //_\\| '_ \| '_  |
//     / /__| (_) | | | \__ \ |_| | |_| |_| | |_| |\ V /  __/ /__| (_| |\ V  V /\__ \/  _  \ |_) | |_) |
//     \____/\___/|_| |_|___/\__|_|\__|\__,_|\__|_| \_/ \___\____/\__,_| \_/\_/ |___/\_/ \_/ .__/| .__/
//                                                                                         |_|   |_|
//
//  License:         BSD License
//                   license: structural_mechanics_application/license.txt
//
//  Main authors:    Vicente Mataix Ferrandiz
//                   Alejandro Cornejo
//
//

#pragma once

// System includes

// External includes

// Project includes
#include "custom_constitutive/small_strains/plasticity/generic_small_strain_kinematic_plasticity.h"

namespace Kratos
{
///@name Kratos Globals
///@{

///@}
///@name Type Definitions
///@{

    // The size type definition
    typedef std::size_t SizeType;

///@}
///@name  Enum's
///@{

///@}
///@name  Functions
///@{

///@}
///@name Kratos Classes
///@{

/**
 * @class GenericFiniteStrainKinematicPlasticity
 * @ingroup StructuralMechanicsApplication
 * @brief This class is the base class which define all the constitutive laws for plasticity in strain framework
 * @details This class considers a constitutive law integrator as an intermediate utility to compute the plasticity
 * This implementation is based on the Eulerian logarithmic strain measure (COMPUTATIONAL METHODS FOR PLASTICITY THEORY AND APPLICATIONS. EA de Souza Neto,D Perić, DRJ Owen pag. 596).
 * @tparam TConstLawIntegratorType The constitutive law integrator considered
 * @author Alejandro Cornejo & Vicente Mataix Ferrandiz
 */
template<class TConstLawIntegratorType>
class KRATOS_API(CONSTITUTIVE_LAWS_APPLICATION) GenericFiniteStrainKinematicPlasticity
    : public GenericSmallStrainKinematicPlasticity<TConstLawIntegratorType>
{
public:
    ///@name Type Definitions
    ///@{

    /// The define the working dimension size, already defined in the integrator
    static constexpr SizeType Dimension = TConstLawIntegratorType::Dimension;

    /// The define the Voigt size, already defined in the  integrator
    static constexpr SizeType VoigtSize = TConstLawIntegratorType::VoigtSize;

    /// Definition of the base class
    typedef GenericSmallStrainKinematicPlasticity<TConstLawIntegratorType> BaseType;

    /// The definition of the Voigt array type
    typedef array_1d<double, VoigtSize> BoundedArrayType;

    /// The definition of the bounded matrix type
    typedef BoundedMatrix<double, Dimension, Dimension> BoundedMatrixType;

    /// Counted pointer of GenericFiniteStrainKinematicPlasticity
    KRATOS_CLASS_POINTER_DEFINITION(GenericFiniteStrainKinematicPlasticity);

    /// The node definition
    typedef Node NodeType;

    /// The geometry definition
    typedef Geometry<NodeType> GeometryType;

    ///@}
    ///@name Life Cycle
    ///@{

    /**
    * Default constructor.
    */
    GenericFiniteStrainKinematicPlasticity()
    {
    }

    /**
    * Clone.
    */
    ConstitutiveLaw::Pointer Clone() const override
    {
        return Kratos::make_shared<GenericFiniteStrainKinematicPlasticity<TConstLawIntegratorType>>(*this);
    }

    /**
    * Copy constructor.
    */
    GenericFiniteStrainKinematicPlasticity(const GenericFiniteStrainKinematicPlasticity &rOther)
        : BaseType(rOther)
    {
    }

    /**
    * Destructor.
    */
    ~GenericFiniteStrainKinematicPlasticity() override
    {
    }

    ///@}
    ///@name Operators
    ///@{

    ///@}
    ///@name Operations
    ///@{

    /**
     * @brief Computes the material response in terms of 1st Piola-Kirchhoff stresses and constitutive tensor
     * @see Parameters
     */
    void CalculateMaterialResponsePK1(ConstitutiveLaw::Parameters &rValues) override;

    /**
     * @brief Computes the material response in terms of 2nd Piola-Kirchhoff stresses and constitutive tensor
     * @see Parameters
     */
    void CalculateMaterialResponsePK2(ConstitutiveLaw::Parameters &rValues) override;

    /**
     * @brief Computes the material response in terms of Kirchhoff stresses and constitutive tensor
     * @see Parameters
     */
    void CalculateMaterialResponseKirchhoff(ConstitutiveLaw::Parameters &rValues) override;

    /**
     * @brief Computes the material response in terms of Cauchy stresses and constitutive tensor
     * @see Parameters
     */
    void CalculateMaterialResponseCauchy(ConstitutiveLaw::Parameters &rValues) override;

    /**
     * @brief Finalize the material response in terms of 1st Piola-Kirchhoff stresses
     * @see Parameters
     */
    void FinalizeMaterialResponsePK1(ConstitutiveLaw::Parameters &rValues) override;

    /**
     * @brief Finalize the material response in terms of 2nd Piola-Kirchhoff stresses
     * @see Parameters
     */
    void FinalizeMaterialResponsePK2(ConstitutiveLaw::Parameters &rValues) override;

    /**
     * @brief Finalize the material response in terms of Kirchhoff stresses
     * @see Parameters
     */
    void FinalizeMaterialResponseKirchhoff(ConstitutiveLaw::Parameters &rValues) override;
    /**
     * Finalize the material response in terms of Cauchy stresses
     * @see Parameters
     */
    void FinalizeMaterialResponseCauchy(ConstitutiveLaw::Parameters &rValues) override;

    /**
     * @brief If the CL requires to initialize the material response, called by the element in InitializeSolutionStep.
     */
    bool RequiresFinalizeMaterialResponse() override
    {
        return true;
    }

    /**
     * @brief If the CL requires to initialize the material response, called by the element in InitializeSolutionStep.
     */
    bool RequiresInitializeMaterialResponse() override
    {
        return false;
    }

    /**
     * @brief Returns the value of a specified variable (double)
     * @param rParameterValues the needed parameters for the CL calculation
     * @param rThisVariable the variable to be returned
     * @param rValue a reference to the returned value
     * @param rValue output: the value of the specified variable
     */
    double& CalculateValue(
        ConstitutiveLaw::Parameters& rParameterValues,
        const Variable<double>& rThisVariable,
        double& rValue) override;

    /**
     * @brief Returns the value of a specified variable (vector)
     * @param rParameterValues the needed parameters for the CL calculation
     * @param rThisVariable the variable to be returned
     * @param rValue a reference to the returned value
     * @param rValue output: the value of the specified variable
     */
    Vector& CalculateValue(
        ConstitutiveLaw::Parameters& rParameterValues,
        const Variable<Vector>& rThisVariable,
        Vector& rValue
        ) override;

    /**
     * @brief Returns the value of a specified variable (matrix)
     * @param rParameterValues the needed parameters for the CL calculation
     * @param rThisVariable the variable to be returned
     * @param rValue a reference to the returned value
     * @param rValue output: the value of the specified variable
     */
    Matrix& CalculateValue(
        ConstitutiveLaw::Parameters& rParameterValues,
        const Variable<Matrix>& rThisVariable,
        Matrix& rValue
        ) override;

    /**
     * @brief This function provides the place to perform checks on the completeness of the input.
     * @details It is designed to be called only once (or anyway, not often) typically at the beginning
     * of the calculations, so to verify that nothing is missing from the input or that no common error is found.
     * @param rMaterialProperties The properties of the material
     * @param rElementGeometry The geometry of the element
     * @param rCurrentProcessInfo The current process info instance
     * @return 0 if OK, 1 otherwise
     */
    int Check(
        const Properties& rMaterialProperties,
        const GeometryType& rElementGeometry,
        const ProcessInfo& rCurrentProcessInfo
        ) const override;

    ///@}
    ///@name Access
    ///@{

    ///@}
    ///@name Inquiry
    ///@{

    ///@}
    ///@name Input and output
    ///@{

    ///@}
    ///@name Friends
    ///@{

    ///@}

protected:
    ///@name Protected static Member Variables
    ///@{

    ///@}
    ///@name Protected member Variables
    ///@{

    ///@}
    ///@name Protected Operators
    ///@{

    ///@}
    ///@name Protected Operations
    ///@{

    ///@}
    ///@name Protected  Access
    ///@{

    ///@}
    ///@name Protected Inquiry
    ///@{

    ///@}
    ///@name Protected LifeCycle
    ///@{

    ///@}
  private:
    ///@name Static Member Variables
    ///@{

    ///@}
    ///@name Member Variables
    ///@{

    ///@}
    ///@name Private Operators
    ///@{

    ///@}
    ///@name Private Operations
    ///@{

    /**
     * @brief This method computes the tangent tensor
     * @param rValues The constitutive law parameters and flags
     * @param rStressMeasure The stress measure of the law
     */
    void CalculateTangentTensor(
        ConstitutiveLaw::Parameters &rValues,
        const ConstitutiveLaw::StressMeasure& rStressMeasure,
        const Vector& rPlasticStrain
        );

    ///@}
    ///@name Private  Access
    ///@{

    ///@}
    ///@name Private Inquiry
    ///@{

    ///@}
    ///@name Un accessible methods
    ///@{


    ///@}

}; // Class GenericYieldSurface

} // namespace Kratos

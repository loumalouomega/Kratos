//    |  /           |
//    ' /   __| _` | __|  _ \   __|
//    . \  |   (   | |   (   |\__ `
//   _|\_\_|  \__,_|\__|\___/ ____/
//                   Multi-Physics
//
//  License:		BSD License
//					Kratos default license: kratos/license.txt
//
//  Main authors:    Ilaria Iaconeta
//


#if !defined (KRATOS_HENCKY_PLASTIC_PLANE_STRAIN_2D_LAW_H_INCLUDED)
#define       KRATOS_HENCKY_PLASTIC_PLANE_STRAIN_2D_LAW_H_INCLUDED

// System includes

// External includes

// Project includes
#include "custom_constitutive/hencky_plastic_3D_law.hpp"


namespace Kratos
{
/**
 * Defines a hyperelastic-plastic isotropic constitutive law in plane strain 2D
 * With stress split in an isochoric and volumetric parts
 * This material law is defined by the parameters needed by the yield criterion:

 * The functionality is limited to large displacements
 */

class KRATOS_API(MPM_APPLICATION) HenckyElasticPlasticPlaneStrain2DLaw : public HenckyElasticPlastic3DLaw
{
public:
    /**
     * Type Definitions
     */
    typedef ProcessInfo      ProcessInfoType;
    typedef ConstitutiveLaw         BaseType;
    typedef std::size_t             SizeType;

    typedef MPMFlowRule::Pointer                FlowRulePointer;
    typedef MPMYieldCriterion::Pointer    YieldCriterionPointer;
    typedef MPMHardeningLaw::Pointer        HardeningLawPointer;
    typedef Properties::Pointer            PropertiesPointer;

    /**
     * Counted pointer of HyperElasticPlasticPlaneStrain2DLaw
     */

    KRATOS_CLASS_POINTER_DEFINITION(HenckyElasticPlasticPlaneStrain2DLaw);

    /**
     * Life Cycle
     */

    /**
     * Default constructor.
     */
    HenckyElasticPlasticPlaneStrain2DLaw();


    HenckyElasticPlasticPlaneStrain2DLaw(FlowRulePointer pMPMFlowRule, YieldCriterionPointer pYieldCriterion, HardeningLawPointer pHardeningLaw);

    /**
     * Copy constructor.
     */
    HenckyElasticPlasticPlaneStrain2DLaw (const HenckyElasticPlasticPlaneStrain2DLaw& rOther);


    /**
     * Assignment operator.
     */

    //HyperElasticPlasticPlaneStrain2DLaw& operator=(const HyperElasticPlasticPlaneStrain2DLaw& rOther);

    /**
     * Clone function (has to be implemented by any derived class)
     * @return a pointer to a new instance of this constitutive law
     */
    ConstitutiveLaw::Pointer Clone() const override;

    /**
     * Destructor.
     */
    ~HenckyElasticPlasticPlaneStrain2DLaw() override;

    /**
     * Operators
     */

    /**
     * Operations needed by the base class:
     */

    /**
     * Dimension of the law:
     */
    SizeType WorkingSpaceDimension() override
    {
        return 2;
    };

    /**
     * Voigt tensor size:
     */
    SizeType GetStrainSize() const override
    {
        return 3;
    };



    /**
     * This function is designed to be called once to perform all the checks needed
     * on the input provided. Checks can be "expensive" as the function is designed
     * to catch user's errors.
     * @param props
     * @param geom
     * @param CurrentProcessInfo
     * @return
     */
    //int Check(const Properties& rProperties, const GeometryType& rGeometry, const ProcessInfo& rCurrentProcessInfo);



    /**
     * Input and output
     */
    /**
     * Turn back information as a string.
     */
    //virtual String Info() const;
    /**
     * Print information about this object.
     */
    //virtual void PrintInfo(std::ostream& rOStream) const;
    /**
     * Print object's data.
     */
    //virtual void PrintData(std::ostream& rOStream) const;

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

    /**
     * Calculates the GreenLagrange strains
     * @param rRightCauchyGreen
     * @param rStrainVector
     */
    void CalculateGreenLagrangeStrain( const Matrix & rRightCauchyGreen,
                                       Vector& rStrainVector ) override;


    /**
     * Calculates the Almansi strains
     * @param rRightCauchyGreen
     * @param rStrainVector
     */
    void CalculateAlmansiStrain( const Matrix & rLeftCauchyGreen,
                                 Vector& rStrainVector ) override;


    //virtual void ConvertConstitutiveMatrixToAppropiateDimension(Matrix& rConstitutiveMatrix);
    Matrix SetConstitutiveMatrixToAppropiateDimension(Matrix& rConstitutiveMatrix,const Matrix& rElastoPlasticTangentMatrix) override;

    Vector SetStressMatrixToAppropiateVectorDimension(Vector& rStressVector, const Matrix& rStressMatrix) override;

    void CalculateHenckyMainStrain(const Matrix& rCauchyGreeMatrix,
                                   MPMFlowRule::RadialReturnVariables& rReturnMappingVariables,
                                   Vector& rMainStrain) override;
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


    ///@}
    ///@name Private  Access
    ///@{
    ///@}


    ///@}
    ///@name Serialization
    ///@{
    friend class Serializer;

    void save(Serializer& rSerializer) const override
    {
        KRATOS_SERIALIZE_SAVE_BASE_CLASS( rSerializer, HenckyElasticPlastic3DLaw )
    }

    void load(Serializer& rSerializer) override
    {
        KRATOS_SERIALIZE_LOAD_BASE_CLASS( rSerializer, HenckyElasticPlastic3DLaw )
    }



}; // Class HyperElasticPlasticPlaneStrain2DLaw
}  // namespace Kratos.
#endif // KRATOS_HENCKY_PLASTIC_PLANE_STRAIN_2D_LAW_H_INCLUDED defined


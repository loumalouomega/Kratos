// KRATOS    ______            __             __  _____ __                  __                   __
//          / ____/___  ____  / /_____ ______/ /_/ ___// /________  _______/ /___  ___________ _/ /
//         / /   / __ \/ __ \/ __/ __ `/ ___/ __/\__ \/ __/ ___/ / / / ___/ __/ / / / ___/ __ `/ / 
//        / /___/ /_/ / / / / /_/ /_/ / /__/ /_ ___/ / /_/ /  / /_/ / /__/ /_/ /_/ / /  / /_/ / /  
//        \____/\____/_/ /_/\__/\__,_/\___/\__//____/\__/_/   \__,_/\___/\__/\__,_/_/   \__,_/_/  MECHANICS
//
//  License:         BSD License
//                   license: ContactStructuralMechanicsApplication/license.txt
//
//  Main authors:    Vicente Mataix Ferrandiz
//

#pragma once

// System includes

// External includes

// Project includes
#include "utilities/table_stream_utility.h"
#include "solving_strategies/convergencecriterias/convergence_criteria.h"
#include "utilities/color_utilities.h"
#include "utilities/constraint_utilities.h"

namespace Kratos
{
///@addtogroup ContactStructuralMechanicsApplication
///@{

///@name Kratos Globals
///@{

///@}
///@name Type Definitions
///@{

///@}
///@name  Enum's
///@{

///@}
///@name  Functions
///@{

///@name Kratos Classes
///@{

/**
 * @class DisplacementLagrangeMultiplierMixedContactCriteria
 * @ingroup ContactStructuralMechanicsApplication
 * @brief Convergence criteria for contact problems
 * @details This class implements a convergence control based on nodal displacement and
 * lagrange multiplier values. The error is evaluated separately for each of them, and
 * relative and absolute tolerances for both must be specified.
 * @author Vicente Mataix Ferrandiz
 */
template<   class TSparseSpace,
            class TDenseSpace >
class DisplacementLagrangeMultiplierMixedContactCriteria
    : public ConvergenceCriteria< TSparseSpace, TDenseSpace >
{
public:

    ///@name Type Definitions
    ///@{

    /// Pointer definition of DisplacementLagrangeMultiplierMixedContactCriteria
    KRATOS_CLASS_POINTER_DEFINITION( DisplacementLagrangeMultiplierMixedContactCriteria );

    /// Local Flags
    KRATOS_DEFINE_LOCAL_FLAG( ENSURE_CONTACT );
    KRATOS_DEFINE_LOCAL_FLAG( PRINTING_OUTPUT );
    KRATOS_DEFINE_LOCAL_FLAG( TABLE_IS_INITIALIZED );
    KRATOS_DEFINE_LOCAL_FLAG( ROTATION_DOF_IS_CONSIDERED );
    KRATOS_DEFINE_LOCAL_FLAG( INITIAL_RESIDUAL_IS_SET );

    /// The base class definition
    using BaseType = ConvergenceCriteria<TSparseSpace, TDenseSpace>;

    /// The definition of the current class
    using ClassType = DisplacementLagrangeMultiplierMixedContactCriteria<TSparseSpace, TDenseSpace>;

    /// The dofs array type
    using DofsArrayType = typename BaseType::DofsArrayType;

    /// The sparse matrix type
    using TSystemMatrixType = typename BaseType::TSystemMatrixType;

    /// The dense vector type
    using TSystemVectorType = typename BaseType::TSystemVectorType;

    /// The sparse space used
    using SparseSpaceType = TSparseSpace;

    /// The table stream definition TODO: Replace by logger
    using TablePrinterPointerType = TableStreamUtility::Pointer;

    /// The index type definition
    using IndexType = std::size_t;

    /// The epsilon tolerance definition
    static constexpr double Tolerance = std::numeric_limits<double>::epsilon();

    ///@}
    ///@name Life Cycle
    ///@{

    /**
     * @brief Default constructor.
     */
    explicit DisplacementLagrangeMultiplierMixedContactCriteria()
        : BaseType()
    {
    }

    /**
     * @brief Default constructor. (with parameters)
     * @param ThisParameters The configuration parameters
     */
    explicit DisplacementLagrangeMultiplierMixedContactCriteria(Kratos::Parameters ThisParameters)
        : BaseType()
    {
        // Validate and assign defaults
        ThisParameters = this->ValidateAndAssignParameters(ThisParameters, this->GetDefaultParameters());
        this->AssignSettings(ThisParameters);
    }

    /**
     * @brief Default constructor
     * @param DispRatioTolerance Relative tolerance for displacement residual error
     * @param DispAbsTolerance Absolute tolerance for displacement residual error
     * @param RotRatioTolerance Relative tolerance for rotation residual error
     * @param RotAbsTolerance Absolute tolerance for rotation residual error
     * @param LMRatioTolerance Relative tolerance for lagrange multiplier residual  error
     * @param LMAbsTolerance Absolute tolerance for lagrange multiplier residual error
     * @param EnsureContact To check if the contact is lost
     * @param pTable The pointer to the output r_table
     * @param PrintingOutput If the output is going to be printed in a txt file
     */
    explicit DisplacementLagrangeMultiplierMixedContactCriteria(
        const double DispRatioTolerance,
        const double DispAbsTolerance,
        const double RotRatioTolerance,
        const double RotAbsTolerance,
        const double LMRatioTolerance,
        const double LMAbsTolerance,
        const bool EnsureContact = false,
        const bool PrintingOutput = false
        )
        : BaseType()
    {
        // Set local flags
        mOptions.Set(DisplacementLagrangeMultiplierMixedContactCriteria::ENSURE_CONTACT, EnsureContact);
        mOptions.Set(DisplacementLagrangeMultiplierMixedContactCriteria::PRINTING_OUTPUT, PrintingOutput);
        mOptions.Set(DisplacementLagrangeMultiplierMixedContactCriteria::TABLE_IS_INITIALIZED, false);
        mOptions.Set(DisplacementLagrangeMultiplierMixedContactCriteria::ROTATION_DOF_IS_CONSIDERED, false);
        mOptions.Set(DisplacementLagrangeMultiplierMixedContactCriteria::INITIAL_RESIDUAL_IS_SET, false);

        // The displacement solution
        mDispRatioTolerance = DispRatioTolerance;
        mDispAbsTolerance = DispAbsTolerance;

        // The rotation solution
        mRotRatioTolerance = RotRatioTolerance;
        mRotAbsTolerance = RotAbsTolerance;

        // The normal contact solution
        mLMRatioTolerance = LMRatioTolerance;
        mLMAbsTolerance = LMAbsTolerance;
    }

    //* Copy constructor.
    DisplacementLagrangeMultiplierMixedContactCriteria( DisplacementLagrangeMultiplierMixedContactCriteria const& rOther )
      :BaseType(rOther)
      ,mOptions(rOther.mOptions)
      ,mDispRatioTolerance(rOther.mDispRatioTolerance)
      ,mDispAbsTolerance(rOther.mDispAbsTolerance)
      ,mDispInitialResidualNorm(rOther.mDispInitialResidualNorm)
      ,mDispCurrentResidualNorm(rOther.mDispCurrentResidualNorm)
      ,mRotRatioTolerance(rOther.mRotRatioTolerance)
      ,mRotAbsTolerance(rOther.mRotAbsTolerance)
      ,mRotInitialResidualNorm(rOther.mRotInitialResidualNorm)
      ,mRotCurrentResidualNorm(rOther.mRotCurrentResidualNorm)
      ,mLMRatioTolerance(rOther.mLMRatioTolerance)
      ,mLMAbsTolerance(rOther.mLMAbsTolerance)
    {
    }

    /// Destructor.
    ~DisplacementLagrangeMultiplierMixedContactCriteria() override = default;

    ///@}
    ///@name Operators
    ///@{

    ///@}
    ///@name Operations
    ///@{

    /**
     * @brief Create method
     * @param ThisParameters The configuration parameters
     */
    typename BaseType::Pointer Create(Parameters ThisParameters) const override
    {
        return Kratos::make_shared<ClassType>(ThisParameters);
    }

    /**
     * @brief Compute relative and absolute error.
     * @param rModelPart Reference to the ModelPart containing the contact problem.
     * @param rDofSet Reference to the container of the problem's degrees of freedom (stored by the BuilderAndSolver)
     * @param rA System matrix (unused)
     * @param rDx Vector of results (variations on nodal variables)
     * @param rb RHS vector (residual)
     * @return true if convergence is achieved, false otherwise
     */
    bool PostCriteria(
        ModelPart& rModelPart,
        DofsArrayType& rDofSet,
        const TSystemMatrixType& rA,
        const TSystemVectorType& rDx,
        const TSystemVectorType& rb
        ) override
    {
        if (SparseSpaceType::Size(rb) != 0) { //if we are solving for something
            // Initialize
            double disp_residual_solution_norm = 0.0, rot_residual_solution_norm = 0.0, lm_solution_norm = 0.0, lm_increase_norm = 0.0;
            IndexType disp_dof_num(0),rot_dof_num(0),lm_dof_num(0);

            // Auxiliary values
            struct AuxValues {
                std::size_t dof_id = 0;
                double residual_dof_value = 0.0, dof_value = 0.0, dof_incr = 0.0;
            };

            // The number of active dofs
            const std::size_t number_active_dofs = rb.size();

            // Auxiliary displacement DoF check
            const std::function<bool(const VariableData&)> check_without_rot =
            [](const VariableData& rCurrVar) -> bool {return true;};
            const std::function<bool(const VariableData&)> check_with_rot =
            [](const VariableData& rCurrVar) -> bool {return ((rCurrVar == DISPLACEMENT_X) || (rCurrVar == DISPLACEMENT_Y) || (rCurrVar == DISPLACEMENT_Z));};
            const auto* p_check_disp = (mOptions.Is(DisplacementLagrangeMultiplierMixedContactCriteria::ROTATION_DOF_IS_CONSIDERED)) ? &check_with_rot : &check_without_rot;

            // Loop over Dofs
            using SevenReduction = CombinedReduction<SumReduction<double>, SumReduction<double>, SumReduction<double>, SumReduction<double>, SumReduction<IndexType>, SumReduction<IndexType>, SumReduction<IndexType>>;
            std::tie(disp_residual_solution_norm, rot_residual_solution_norm, lm_solution_norm, lm_increase_norm, disp_dof_num, rot_dof_num, lm_dof_num) = block_for_each<SevenReduction>(rDofSet, AuxValues(), [this,p_check_disp,&number_active_dofs,&rb,&rDx](Dof<double>& rDof, AuxValues& aux_values) {
                aux_values.dof_id = rDof.EquationId();

                // Check dof id is solved
                if (aux_values.dof_id < number_active_dofs) {
                    if (mActiveDofs[aux_values.dof_id] == 1) {
                        const auto& r_curr_var = rDof.GetVariable();
                        if ((r_curr_var == VECTOR_LAGRANGE_MULTIPLIER_X) || (r_curr_var == VECTOR_LAGRANGE_MULTIPLIER_Y) || (r_curr_var == VECTOR_LAGRANGE_MULTIPLIER_Z) || (r_curr_var == LAGRANGE_MULTIPLIER_CONTACT_PRESSURE)) {
                            aux_values.dof_value = rDof.GetSolutionStepValue(0);
                            aux_values.dof_incr = rDx[aux_values.dof_id];
                            return std::make_tuple(0.0,0.0,std::pow(aux_values.dof_value, 2),std::pow(aux_values.dof_incr, 2),0,0,1);
                        } else if ((*p_check_disp)(r_curr_var)) {
                            aux_values.residual_dof_value = rb[aux_values.dof_id];
                            return std::make_tuple(std::pow(aux_values.residual_dof_value, 2),0.0,0.0,0.0,1,0,0);
                        } else { // We will assume is rotation dof
                            KRATOS_DEBUG_ERROR_IF_NOT((r_curr_var == ROTATION_X) || (r_curr_var == ROTATION_Y) || (r_curr_var == ROTATION_Z)) << "Variable must be a ROTATION and it is: " << r_curr_var.Name() << std::endl;
                            aux_values.residual_dof_value = rb[aux_values.dof_id];
                            return std::make_tuple(0.0,std::pow(aux_values.residual_dof_value, 2),0.0,0.0,0,1,0);
                        }
                    }
                }
                return std::make_tuple(0.0,0.0,0.0,0.0,0,0,0);
            });

            if(lm_increase_norm < Tolerance) lm_increase_norm = 1.0;
            KRATOS_ERROR_IF(mOptions.Is(DisplacementLagrangeMultiplierMixedContactCriteria::ENSURE_CONTACT) && lm_solution_norm < Tolerance) << "ERROR::CONTACT LOST::ARE YOU SURE YOU ARE SUPPOSED TO HAVE CONTACT?" << std::endl;

            mDispCurrentResidualNorm = disp_residual_solution_norm;
            mRotCurrentResidualNorm = rot_residual_solution_norm;
            const double lm_ratio = lm_solution_norm > Tolerance ? std::sqrt(lm_increase_norm/lm_solution_norm) : 0.0;
            const double lm_abs = std::sqrt(lm_increase_norm)/static_cast<double>(lm_dof_num);

            double residual_disp_ratio, residual_rot_ratio;

            // We initialize the solution
            if (mOptions.IsNot(DisplacementLagrangeMultiplierMixedContactCriteria::INITIAL_RESIDUAL_IS_SET)) {
                mDispInitialResidualNorm = (disp_residual_solution_norm < Tolerance) ? 1.0 : disp_residual_solution_norm;
                residual_disp_ratio = 1.0;
                if (mOptions.Is(DisplacementLagrangeMultiplierMixedContactCriteria::ROTATION_DOF_IS_CONSIDERED)) {
                    mRotInitialResidualNorm = (rot_residual_solution_norm < Tolerance) ? 1.0 : rot_residual_solution_norm;
                    residual_rot_ratio = 1.0;
                }
                mOptions.Set(DisplacementLagrangeMultiplierMixedContactCriteria::INITIAL_RESIDUAL_IS_SET, true);
            }

            // We calculate the ratio of the displacements
            residual_disp_ratio = mDispCurrentResidualNorm/mDispInitialResidualNorm;

            // We calculate the ratio of the rotations
            residual_rot_ratio = mRotCurrentResidualNorm/mRotInitialResidualNorm;

            // We calculate the absolute norms
            double residual_disp_abs = mDispCurrentResidualNorm/disp_dof_num;
            double residual_rot_abs = mRotCurrentResidualNorm/rot_dof_num;

            // The process info of the model part
            ProcessInfo& r_process_info = rModelPart.GetProcessInfo();

            // We print the results // TODO: Replace for the new log
            if (rModelPart.GetCommunicator().MyPID() == 0 && this->GetEchoLevel() > 0) {
                if (r_process_info.Has(TABLE_UTILITY)) {
                    std::cout.precision(4);
                    TablePrinterPointerType p_table = r_process_info[TABLE_UTILITY];
                    auto& r_table = p_table->GetTable();
                    if (mOptions.Is(DisplacementLagrangeMultiplierMixedContactCriteria::ROTATION_DOF_IS_CONSIDERED)) {
                        r_table << residual_disp_ratio << mDispRatioTolerance << residual_disp_abs << mDispAbsTolerance << residual_rot_ratio << mRotRatioTolerance << residual_rot_abs << mRotAbsTolerance << lm_ratio << mLMRatioTolerance << lm_abs << mLMAbsTolerance;
                    } else {
                        r_table << residual_disp_ratio << mDispRatioTolerance << residual_disp_abs << mDispAbsTolerance << lm_ratio << mLMRatioTolerance << lm_abs << mLMAbsTolerance;
                    }
                } else {
                    std::cout.precision(4);
                    if (mOptions.IsNot(DisplacementLagrangeMultiplierMixedContactCriteria::PRINTING_OUTPUT)) {
                        KRATOS_INFO("DisplacementLagrangeMultiplierMixedContactCriteria") << BOLDFONT("MIXED CONVERGENCE CHECK") << "\tSTEP: " << r_process_info[STEP] << "\tNL ITERATION: " << r_process_info[NL_ITERATION_NUMBER] << std::endl << std::scientific;
                        KRATOS_INFO("DisplacementLagrangeMultiplierMixedContactCriteria") << BOLDFONT("\tDISPLACEMENT: RATIO = ") << residual_disp_ratio << BOLDFONT(" EXP.RATIO = ") << mDispRatioTolerance << BOLDFONT(" ABS = ") << residual_disp_abs << BOLDFONT(" EXP.ABS = ") << mDispAbsTolerance << std::endl;
                        if (mOptions.Is(DisplacementLagrangeMultiplierMixedContactCriteria::ROTATION_DOF_IS_CONSIDERED)) {
                            KRATOS_INFO("DisplacementLagrangeMultiplierMixedContactCriteria") << BOLDFONT("\tROTATION: RATIO = ") << residual_rot_ratio << BOLDFONT(" EXP.RATIO = ") << mRotRatioTolerance << BOLDFONT(" ABS = ") << residual_rot_abs << BOLDFONT(" EXP.ABS = ") << mRotAbsTolerance << std::endl;
                        }
                        KRATOS_INFO("DisplacementLagrangeMultiplierMixedContactCriteria") << BOLDFONT("\tLAGRANGE MUL: RATIO = ") << lm_ratio << BOLDFONT(" EXP.RATIO = ") << mLMRatioTolerance << BOLDFONT(" ABS = ") << lm_abs << BOLDFONT(" EXP.ABS = ") << mLMAbsTolerance << std::endl;
                    } else {
                        KRATOS_INFO("DisplacementLagrangeMultiplierMixedContactCriteria") << "MIXED CONVERGENCE CHECK" << "\tSTEP: " << r_process_info[STEP] << "\tNL ITERATION: " << r_process_info[NL_ITERATION_NUMBER] << std::endl << std::scientific;
                        KRATOS_INFO("DisplacementLagrangeMultiplierMixedContactCriteria") << "\tDISPLACEMENT: RATIO = " << residual_disp_ratio << " EXP.RATIO = " << mDispRatioTolerance << " ABS = " << residual_disp_abs << " EXP.ABS = " << mDispAbsTolerance << std::endl;
                        if (mOptions.Is(DisplacementLagrangeMultiplierMixedContactCriteria::ROTATION_DOF_IS_CONSIDERED)) {
                            KRATOS_INFO("DisplacementLagrangeMultiplierMixedContactCriteria") << "\tROTATION: RATIO = " << residual_rot_ratio << " EXP.RATIO = " << mRotRatioTolerance << " ABS = " << residual_rot_abs << " EXP.ABS = " << mRotAbsTolerance << std::endl;
                        }
                        KRATOS_INFO("DisplacementLagrangeMultiplierMixedContactCriteria") << "\tLAGRANGE MUL: RATIO = " << lm_ratio << " EXP.RATIO = " << mLMRatioTolerance << " ABS = " << lm_abs << " EXP.ABS = " << mLMAbsTolerance << std::endl;
                    }
                }
            }

            r_process_info[CONVERGENCE_RATIO] = (residual_disp_ratio > lm_ratio) ? residual_disp_ratio : lm_ratio;
            r_process_info[RESIDUAL_NORM] = (lm_abs > mLMAbsTolerance) ? lm_abs : mLMAbsTolerance;

            // We check if converged
            const bool disp_converged = (residual_disp_ratio <= mDispRatioTolerance || residual_disp_abs <= mDispAbsTolerance);
            const bool rot_converged = (mOptions.Is(DisplacementLagrangeMultiplierMixedContactCriteria::ROTATION_DOF_IS_CONSIDERED)) ? (residual_rot_ratio <= mRotRatioTolerance || residual_rot_abs <= mRotAbsTolerance) : true;
            const bool lm_converged = (mOptions.IsNot(DisplacementLagrangeMultiplierMixedContactCriteria::ENSURE_CONTACT) && lm_solution_norm < Tolerance) ? true : (lm_ratio <= mLMRatioTolerance || lm_abs <= mLMAbsTolerance);

            if ( disp_converged && rot_converged && lm_converged ) {
                if (rModelPart.GetCommunicator().MyPID() == 0 && this->GetEchoLevel() > 0) {
                    if (r_process_info.Has(TABLE_UTILITY)) {
                        TablePrinterPointerType p_table = r_process_info[TABLE_UTILITY];
                        auto& r_table = p_table->GetTable();
                        if (mOptions.IsNot(DisplacementLagrangeMultiplierMixedContactCriteria::PRINTING_OUTPUT))
                            r_table << BOLDFONT(FGRN("       Achieved"));
                        else
                            r_table << "Achieved";
                    } else {
                        if (mOptions.IsNot(DisplacementLagrangeMultiplierMixedContactCriteria::PRINTING_OUTPUT))
                            KRATOS_INFO("DisplacementLagrangeMultiplierMixedContactCriteria") << BOLDFONT("\tConvergence") << " is " << BOLDFONT(FGRN("achieved")) << std::endl;
                        else
                            KRATOS_INFO("DisplacementLagrangeMultiplierMixedContactCriteria") << "\tConvergence is achieved" << std::endl;
                    }
                }
                return true;
            } else {
                if (rModelPart.GetCommunicator().MyPID() == 0 && this->GetEchoLevel() > 0) {
                    if (r_process_info.Has(TABLE_UTILITY)) {
                        TablePrinterPointerType p_table = r_process_info[TABLE_UTILITY];
                        auto& r_table = p_table->GetTable();
                        if (mOptions.IsNot(DisplacementLagrangeMultiplierMixedContactCriteria::PRINTING_OUTPUT))
                            r_table << BOLDFONT(FRED("   Not achieved"));
                        else
                            r_table << "Not achieved";
                    } else {
                        if (mOptions.IsNot(DisplacementLagrangeMultiplierMixedContactCriteria::PRINTING_OUTPUT))
                            KRATOS_INFO("DisplacementLagrangeMultiplierMixedContactCriteria") << BOLDFONT("\tConvergence") << " is " << BOLDFONT(FRED(" not achieved")) << std::endl;
                        else
                            KRATOS_INFO("DisplacementLagrangeMultiplierMixedContactCriteria") << "\tConvergence is not achieved" << std::endl;
                    }
                }
                return false;
            }
        } else // In this case all the displacements are imposed!
            return true;
    }

    /**
     * @brief This function initialize the convergence criteria
     * @param rModelPart Reference to the ModelPart containing the contact problem. (unused)
     */
    void Initialize( ModelPart& rModelPart) override
    {
        // Initialize
        BaseType::mConvergenceCriteriaIsInitialized = true;

        // Check rotation dof
        mOptions.Set(DisplacementLagrangeMultiplierMixedContactCriteria::ROTATION_DOF_IS_CONSIDERED, ContactUtilities::CheckModelPartHasRotationDoF(rModelPart));

        // Initialize header
        ProcessInfo& r_process_info = rModelPart.GetProcessInfo();
        if (r_process_info.Has(TABLE_UTILITY) && mOptions.IsNot(DisplacementLagrangeMultiplierMixedContactCriteria::TABLE_IS_INITIALIZED)) {
            TablePrinterPointerType p_table = r_process_info[TABLE_UTILITY];
            auto& r_table = p_table->GetTable();
            r_table.AddColumn("DP RATIO", 10);
            r_table.AddColumn("EXP. RAT", 10);
            r_table.AddColumn("ABS", 10);
            r_table.AddColumn("EXP. ABS", 10);
            if (mOptions.Is(DisplacementLagrangeMultiplierMixedContactCriteria::ROTATION_DOF_IS_CONSIDERED)) {
                r_table.AddColumn("RT RATIO", 10);
                r_table.AddColumn("EXP. RAT", 10);
                r_table.AddColumn("ABS", 10);
                r_table.AddColumn("EXP. ABS", 10);
            }
            r_table.AddColumn("LM RATIO", 10);
            r_table.AddColumn("EXP. RAT", 10);
            r_table.AddColumn("ABS", 10);
            r_table.AddColumn("EXP. ABS", 10);
            r_table.AddColumn("CONVERGENCE", 15);
            mOptions.Set(DisplacementLagrangeMultiplierMixedContactCriteria::TABLE_IS_INITIALIZED, true);
        }
    }

    /**
     * @brief This function initializes the solution step
     * @param rModelPart Reference to the ModelPart containing the contact problem.
     * @param rDofSet Reference to the container of the problem's degrees of freedom (stored by the BuilderAndSolver)
     * @param rA System matrix (unused)
     * @param rDx Vector of results (variations on nodal variables)
     * @param rb RHS vector (residual)
     */
    void InitializeSolutionStep(
        ModelPart& rModelPart,
        DofsArrayType& rDofSet,
        const TSystemMatrixType& rA,
        const TSystemVectorType& rDx,
        const TSystemVectorType& rb
        ) override
    {
        // Initialize flag
        mOptions.Set(DisplacementLagrangeMultiplierMixedContactCriteria::INITIAL_RESIDUAL_IS_SET, false);

        // Filling mActiveDofs when MPC exist
        ConstraintUtilities::ComputeActiveDofs(rModelPart, mActiveDofs, rDofSet);
    }

    /**
     * @brief This method provides the defaults parameters to avoid conflicts between the different constructors
     * @return The default parameters
     */
    Parameters GetDefaultParameters() const override
    {
        Parameters default_parameters = Parameters(R"(
        {
            "name"                                    : "displacement_lagrange_multiplier_mixed_contact_criteria",
            "ensure_contact"                          : false,
            "print_convergence_criterion"             : false,
            "residual_relative_tolerance"             : 1.0e-4,
            "residual_absolute_tolerance"             : 1.0e-9,
            "rotation_residual_relative_tolerance"    : 1.0e-4,
            "rotation_residual_absolute_tolerance"    : 1.0e-9,
            "contact_displacement_relative_tolerance" : 1.0e-4,
            "contact_displacement_absolute_tolerance" : 1.0e-9
        })");

        // Getting base class default parameters
        const Parameters base_default_parameters = BaseType::GetDefaultParameters();
        default_parameters.RecursivelyAddMissingParameters(base_default_parameters);
        return default_parameters;
    }

    /**
     * @brief Returns the name of the class as used in the settings (snake_case format)
     * @return The name of the class
     */
    static std::string Name()
    {
        return "displacement_lagrange_multiplier_mixed_contact_criteria";
    }

    ///@}
    ///@name Access
    ///@{

    ///@}
    ///@name Inquiry
    ///@{

    ///@}
    ///@name Input and output
    ///@{

    /// Turn back information as a string.
    std::string Info() const override
    {
        return "DisplacementLagrangeMultiplierMixedContactCriteria";
    }

    /// Print information about this object.
    void PrintInfo(std::ostream& rOStream) const override
    {
        rOStream << Info();
    }

    /// Print object's data.
    void PrintData(std::ostream& rOStream) const override
    {
        rOStream << Info();
    }

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

    /**
     * @brief This method assigns settings to member variables
     * @param ThisParameters Parameters that are assigned to the member variables
     */
    void AssignSettings(const Parameters ThisParameters) override
    {
        BaseType::AssignSettings(ThisParameters);

        // The displacement solution
        mDispRatioTolerance = ThisParameters["residual_relative_tolerance"].GetDouble();
        mDispAbsTolerance = ThisParameters["residual_absolute_tolerance"].GetDouble();

        // The rotation solution
        mRotRatioTolerance = ThisParameters["rotation_residual_relative_tolerance"].GetDouble();
        mRotAbsTolerance = ThisParameters["rotation_residual_absolute_tolerance"].GetDouble();

        // The contact solution
        mLMRatioTolerance =  ThisParameters["contact_displacement_relative_tolerance"].GetDouble();
        mLMAbsTolerance =  ThisParameters["contact_displacement_absolute_tolerance"].GetDouble();

        // Set local flags
        mOptions.Set(DisplacementLagrangeMultiplierMixedContactCriteria::ENSURE_CONTACT, ThisParameters["ensure_contact"].GetBool());
        mOptions.Set(DisplacementLagrangeMultiplierMixedContactCriteria::PRINTING_OUTPUT, ThisParameters["print_convergence_criterion"].GetBool());
        mOptions.Set(DisplacementLagrangeMultiplierMixedContactCriteria::TABLE_IS_INITIALIZED, false);
        mOptions.Set(DisplacementLagrangeMultiplierMixedContactCriteria::ROTATION_DOF_IS_CONSIDERED, false);
        mOptions.Set(DisplacementLagrangeMultiplierMixedContactCriteria::INITIAL_RESIDUAL_IS_SET, false);
    }

    ///@}
private:
    ///@name Static Member Variables
    ///@{

    ///@}
    ///@name Member Variables
    ///@{

    Flags mOptions; /// Local flags

    double mDispRatioTolerance;      /// The ratio threshold for the norm of the displacement residual
    double mDispAbsTolerance;        /// The absolute value threshold for the norm of the displacement residual
    double mDispInitialResidualNorm; /// The reference norm of the displacement residual
    double mDispCurrentResidualNorm; /// The current norm of the displacement residual

    double mRotRatioTolerance;      /// The ratio threshold for the norm of the rotation residual
    double mRotAbsTolerance;        /// The absolute value threshold for the norm of the rotation residual
    double mRotInitialResidualNorm; /// The reference norm of the rotation residual
    double mRotCurrentResidualNorm; /// The current norm of the rotation residual

    double mLMRatioTolerance; /// The ratio threshold for the norm of the LM
    double mLMAbsTolerance;   /// The absolute value threshold for the norm of the LM

    std::vector<int> mActiveDofs;  /// This vector contains the dofs that are active

    ///@}
};  // Kratos DisplacementLagrangeMultiplierMixedContactCriteria

///@name Local flags creation
///@{

/// Local Flags
template<class TSparseSpace, class TDenseSpace>
const Kratos::Flags DisplacementLagrangeMultiplierMixedContactCriteria<TSparseSpace, TDenseSpace>::ENSURE_CONTACT(Kratos::Flags::Create(0));
template<class TSparseSpace, class TDenseSpace>
const Kratos::Flags DisplacementLagrangeMultiplierMixedContactCriteria<TSparseSpace, TDenseSpace>::PRINTING_OUTPUT(Kratos::Flags::Create(1));
template<class TSparseSpace, class TDenseSpace>
const Kratos::Flags DisplacementLagrangeMultiplierMixedContactCriteria<TSparseSpace, TDenseSpace>::TABLE_IS_INITIALIZED(Kratos::Flags::Create(2));
template<class TSparseSpace, class TDenseSpace>
const Kratos::Flags DisplacementLagrangeMultiplierMixedContactCriteria<TSparseSpace, TDenseSpace>::ROTATION_DOF_IS_CONSIDERED(Kratos::Flags::Create(3));
template<class TSparseSpace, class TDenseSpace>
const Kratos::Flags DisplacementLagrangeMultiplierMixedContactCriteria<TSparseSpace, TDenseSpace>::INITIAL_RESIDUAL_IS_SET(Kratos::Flags::Create(4));
}

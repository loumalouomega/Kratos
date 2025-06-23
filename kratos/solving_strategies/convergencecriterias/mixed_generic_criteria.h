//    |  /           |
//    ' /   __| _` | __|  _ \   __|
//    . \  |   (   | |   (   |\__ `
//   _|\_\_|  \__,_|\__|\___/ ____/
//                   Multi-Physics
//
//  License:         BSD License
//                   Kratos default license: kratos/license.txt
//
//  Main authors:    Jordi Cotela, Riccardo Rossi, Carlos Roig and Ruben Zorrilla
//

#pragma once

// System includes

// External includes

// Project includes
#include "includes/define.h"
#include "includes/model_part.h"
#include "convergence_criteria.h"
#include "utilities/convergence_criteria_utilities.h"

// Application includes

namespace Kratos
{
///@addtogroup KratosCore
///@{

///@name Kratos Classes
///@{

/// Convergence criteria for mixed vector-scalar problems.
/**
 This class implements a convergence control based on a nodal vector variable and
 a nodal scalar variable. The error is evaluated separately for each of them, and
 relative and absolute tolerances for both must be specified.
 */
template< class TSparseSpace, class TDenseSpace >
class MixedGenericCriteria : public ConvergenceCriteria< TSparseSpace, TDenseSpace >
{
public:
    ///@name Type Definitions
    ///@{

    KRATOS_CLASS_POINTER_DEFINITION(MixedGenericCriteria);

    typedef ConvergenceCriteria< TSparseSpace, TDenseSpace > BaseType;

    typedef MixedGenericCriteria< TSparseSpace, TDenseSpace > ClassType;

    typedef typename BaseType::TDataType TDataType;

    typedef typename BaseType::DofsArrayType DofsArrayType;

    typedef typename BaseType::TSystemMatrixType TSystemMatrixType;

    typedef typename BaseType::TSystemVectorType TSystemVectorType;

    typedef std::vector<std::tuple<const VariableData*, TDataType, TDataType>> ConvergenceVariableListType;

    typedef std::size_t KeyType;

    ///@}
    ///@name Life Cycle
    ///@{

    /// Constructor.

    explicit MixedGenericCriteria()
        : BaseType(),
          mVariableSize(0)
    {
    }

    /**
     * @brief Default constructor. (with parameters)
     * @param ThisParameters The configuration parameters
     */
    explicit MixedGenericCriteria(Kratos::Parameters ThisParameters)
        : MixedGenericCriteria(ConvergenceCriteriaUtilities::GenerateConvergenceVariableListFromParameters(ThisParameters))
    {
    }

    /**
     * @brief Construct a new Mixed Generic Criteria object
     * Construct the mixed generic convergence criteria from a convergence variables list.
     * The convergence variable list contains for each variable the variable itself as well as the corresponding relative and absolute tolerances.
     * @param rConvergenceVariablesList List containing tuples with the convergence variables to be checked. The tuples are set as <Variable, relative tolerance, absolute tolerance>
     */
    MixedGenericCriteria(const ConvergenceVariableListType& rConvergenceVariablesList)
        : BaseType()
        , mVariableSize([&] (const ConvergenceVariableListType& rList) -> int {return rList.size();} (rConvergenceVariablesList))
        , mVariableDataVector(ConvergenceCriteriaUtilities::GenerateVariableDataVector(rConvergenceVariablesList))
        , mRatioToleranceVector(ConvergenceCriteriaUtilities::GenerateRatioToleranceVector(rConvergenceVariablesList))
        , mAbsToleranceVector(ConvergenceCriteriaUtilities::GenerateAbsToleranceVector(rConvergenceVariablesList))
        , mLocalKeyMap(ConvergenceCriteriaUtilities::GenerateLocalKeyMap(rConvergenceVariablesList))
    {}

    /// Destructor.
    ~MixedGenericCriteria() override
    {}

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

    /// Compute relative and absolute error.
    /**
     * @param rModelPart Reference to the ModelPart containing the fluid problem.
     * @param rDofSet Reference to the container of the problem's degrees of freedom (stored by the BuilderAndSolver)
     * @param A System matrix (unused)
     * @param Dx Vector of results (variations on nodal variables)
     * @param b RHS vector (residual)
     * @return true if convergence is achieved, false otherwise
     */
    bool PostCriteria(
        ModelPart& rModelPart,
        DofsArrayType& rDofSet,
        const TSystemMatrixType& A,
        const TSystemVectorType& Dx,
        const TSystemVectorType& b) override
    {
        // Check if we are solving for something
        if (TSparseSpace::Size(Dx) != 0) {
            // Calculate the convergence ratio and absolute norms
            const auto convergence_norms = CalculateConvergenceNorms(rModelPart, rDofSet, Dx);

            // Output convergence status
            OutputConvergenceStatus(convergence_norms);

            // Check convergence
            return ConvergenceCriteriaUtilities::CheckConvergence(convergence_norms, mVariableSize, mVariableDataVector, mRatioToleranceVector, mAbsToleranceVector, mLocalKeyMap, this->GetEchoLevel());
        } else {
            // Case in which all the DOFs are constrained!
            return true;
        }
    }

    /**
     * @brief Returns the name of the class as used in the settings (snake_case format)
     * @return The name of the class
     */
    static std::string Name()
    {
        return "mixed_generic_criteria";
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
        return "MixedGenericCriteria";
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
    ///@name Protected Static Member Variables
    ///@{


    ///@}
    ///@name Protected Member Variables
    ///@{


    ///@}
    ///@name Protected Operators
    ///@{


    ///@}
    ///@name Protected Operations
    ///@{

    /**
     * @brief Get the Variable Size object
     * Get the number of variables to be checked
     * @return const int Number of variables to check
     */
    int GetVariableSize() const
    {
        return mVariableSize;
    }

    /**
     * @brief Get the Variable Data Vector object
     * Get the member vector that stores pointers to the variables to check
     * @return std::vector<VariableData*> Vector containing pointers to the variables to check
     */
    std::vector<const VariableData*> GetVariableDataVector() const
    {
        return mVariableDataVector;
    }

    /**
     * @brief Get the Ratio Tolerance Vector object
     * Get the member vector containing the ratio tolerances for each variable to check
     * @return std::vector<TDataType> Vector containing the ratio tolerances
     */
    std::vector<TDataType> GetRatioToleranceVector() const
    {
        return mRatioToleranceVector;
    }

    /**
     * @brief Get the Abs Tolerance Vector object
     * Get the member vector containing the absolute tolerances for each variable to check
     * @return std::vector<TDataType> Vector containing the absolute tolerances
     */
    std::vector<TDataType> GetAbsToleranceVector() const
    {
        return mAbsToleranceVector;
    }

    /**
     * @brief Get the Local Key Map object
     * Returns a reference to the variable key local map
     * @return std::unordered_map<KeyType, KeyType>& Reference to the local key map
     */
    std::unordered_map<KeyType, KeyType>& GetLocalKeyMap()
    {
        return mLocalKeyMap;
    }

    /**
     * @brief Calculate the convergence norms
     * This method calculates the convergence norms for all the variables to be checked
     * @param rModelPart Reference to the ModelPart containing the fluid problem.
     * @param rDofSet Reference to the container of the problem's degrees of freedom (stored by the BuilderAndSolver)
     * @param rDx Vector of results (variations on nodal variables)
     * @return std::tuple<std::vector<TDataType>, std::vector<TDataType>> Tuple containing the absolute and relative convergence values
     */
    std::tuple<std::vector<TDataType>, std::vector<TDataType>> CalculateConvergenceNorms(
        const ModelPart& rModelPart,
        const DofsArrayType& rDofSet,
        const TSystemVectorType& rDx)
    {
        // Initialize
        std::vector<int> dofs_count(mVariableSize, 0);
        std::vector<TDataType> solution_norms_vector(mVariableSize, 0.0);
        std::vector<TDataType> increase_norms_vector(mVariableSize, 0.0);

        // Accumulate the norm values
        GetNormValues(rModelPart, rDofSet, rDx, dofs_count, solution_norms_vector, increase_norms_vector);

        // Synchronize the norm values
        const auto& r_data_comm = rModelPart.GetCommunicator().GetDataCommunicator();
        auto global_solution_norms_vector = r_data_comm.SumAll(solution_norms_vector);
        auto global_increase_norms_vector = r_data_comm.SumAll(increase_norms_vector);
        auto global_dofs_count = r_data_comm.SumAll(dofs_count);

        // Check division by zero in global solution norms
        const double zero_tol = 1.0e-12;
        for(int i = 0; i < mVariableSize; i++) {
            if (global_solution_norms_vector[i] < zero_tol) {
                global_solution_norms_vector[i] = 1.0;
            }
        }

        // Calculate the norm values
        std::vector<TDataType> var_ratio(mVariableSize, 0.0);
        std::vector<TDataType> var_abs(mVariableSize, 0.0);
        for(int i = 0; i < mVariableSize; i++) {
            var_ratio[i] = std::sqrt(global_increase_norms_vector[i] / global_solution_norms_vector[i]);
            var_abs[i] = std::sqrt(global_increase_norms_vector[i]) / static_cast<TDataType>(global_dofs_count[i]);
        }

        // Output the ratio and absolute norms as a tuple
        return std::make_tuple(var_ratio, var_abs);
    }

    /**
     * @brief Method to output the convergence status
     * This method prints the convergence status to the screen for each one of the checked variables
     * @param rConvergenceNorms Tuple containing the absolute and relative convergence values
     */
    virtual void OutputConvergenceStatus(const std::tuple<std::vector<TDataType>,std::vector<TDataType>>& rConvergenceNorms)
    {
        ConvergenceCriteriaUtilities::OutputConvergenceStatus(rConvergenceNorms, mVariableSize, mVariableDataVector, mRatioToleranceVector, mAbsToleranceVector, mLocalKeyMap, this->GetEchoLevel());
    }

    /**
     * @brief Finds the var local key in the mLocalKeyMap for 
     * a gifen DOF. If the variable does not exist in mLocalKeyMap
     * this function returns false
     * @param itDof the DOF iterator
     * @param rVarLocalKey variable local key 
     * @return dof variable is found or not
     */
    bool FindVarLocalKey(
        typename DofsArrayType::const_iterator itDof,
        int& rVarLocalKey) const
    {
        const auto &r_current_variable = itDof->GetVariable();
        const KeyType key = r_current_variable.IsComponent() ? r_current_variable.GetSourceVariable().Key() : r_current_variable.Key();
        auto key_find = this->mLocalKeyMap.find(key);
        bool found = true;
        if (key_find == this->mLocalKeyMap.end()) {
            found = false;
        } else {
            rVarLocalKey = key_find->second;;
        }
        return found;
    }


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
    ///@name Private Static Member Variables
    ///@{


    ///@}
    ///@name Private Member Variables
    ///@{

    const int mVariableSize;
    const std::vector<const VariableData*> mVariableDataVector;
    const std::vector<TDataType> mRatioToleranceVector;
    const std::vector<TDataType> mAbsToleranceVector;
    std::unordered_map<KeyType, KeyType> mLocalKeyMap;

    ///@}
    ///@name Private Operators
    ///@{


    ///@}
    ///@name Private Operations
    ///@{

    /**
     * @brief Get the Norm Values
     * This function accumulates the solution and increment norm values in the provided arrays.
     * Note that these arrays are assumed to be already initialized to zero.
     * @param rModelPart Reference to the ModelPart containing the fluid problem.
     * @param rDofSet Reference to the container of the problem's degrees of freedom (stored by the BuilderAndSolver)
     * @param rDx Vector of results (variations on nodal variables)
     * @param rDofsCount Array containing the number of DOFs per variable
     * @param rSolutionNormsVector Array containing the solution norms accumulated values for each variable checked
     * @param rIncreaseNormsVector Array containing the correction norms accumulated values for each variable checked
     */
    virtual void GetNormValues(
        const ModelPart& rModelPart,
        const DofsArrayType& rDofSet,
        const TSystemVectorType& rDx,
        std::vector<int>& rDofsCount,
        std::vector<TDataType>& rSolutionNormsVector,
        std::vector<TDataType>& rIncreaseNormsVector) const
    {
        int n_dofs = rDofSet.size();

        // Loop over Dofs
#pragma omp parallel
        {
            // Local thread variables
            int dof_id;
            TDataType dof_dx;
            TDataType dof_value;

            // Local reduction variables
            std::vector<TDataType> var_solution_norm_reduction(mVariableSize);
            std::vector<TDataType> var_correction_norm_reduction(mVariableSize);
            std::vector<int> dofs_counter_reduction(mVariableSize);
            for (int i = 0; i < mVariableSize; i++) {
                var_solution_norm_reduction[i] = 0.0;
                var_correction_norm_reduction[i] = 0.0;
                dofs_counter_reduction[i] = 0;
            }

#pragma omp for
            for (int i = 0; i < n_dofs; i++) {
                auto it_dof = rDofSet.begin() + i;
                if (it_dof->IsFree()) {
                    dof_id = it_dof->EquationId();
                    dof_value = it_dof->GetSolutionStepValue(0);
                    dof_dx = TSparseSpace::GetValue(rDx, dof_id);

                    int var_local_key;
                    bool key_found = FindVarLocalKey(it_dof,var_local_key);
                    if (!key_found) {
                        // the dof does not belong to the list of variables
                        // we are checking for convergence, so we skip it
                        continue;
                    }
                    var_solution_norm_reduction[var_local_key] += dof_value * dof_value;
                    var_correction_norm_reduction[var_local_key] += dof_dx * dof_dx;
                    dofs_counter_reduction[var_local_key]++;
                }
            }

#pragma omp critical
            {
                for (int i = 0; i < mVariableSize; i++) {
                    rDofsCount[i] += dofs_counter_reduction[i];
                    rSolutionNormsVector[i] += var_solution_norm_reduction[i];
                    rIncreaseNormsVector[i] += var_correction_norm_reduction[i];
                }
            }
        }
    }

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
};
///@} // Kratos classes

///@} // Application group
}

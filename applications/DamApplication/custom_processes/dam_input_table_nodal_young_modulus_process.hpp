//    |  /           |
//    ' /   __| _` | __|  _ \   __|
//    . \  |   (   | |   (   |\__ `
//   _|\_\_|  \__,_|\__|\___/ ____/
//                   Multi-Physics
//
//  License:		 BSD License
//					 Kratos default license: kratos/license.txt
//
//  Main authors:    Lorenzo Gracia
//
//

#if !defined(KRATOS_DAM_INPUT_TABLE_NODAL_YOUNG_MODULUS_PROCESS )
#define  KRATOS_DAM_INPUT_TABLE_NODAL_YOUNG_MODULUS_PROCESS

#include <cmath>

// Project includes
#include "includes/kratos_flags.h"
#include "includes/kratos_parameters.h"
#include "processes/process.h"

// Application include
#include "dam_application_variables.h"

namespace Kratos
{

class DamInputTableNodalYoungModulusProcess : public Process
{

public:

    KRATOS_CLASS_POINTER_DEFINITION(DamInputTableNodalYoungModulusProcess);

    typedef Table<double,double> TableType;
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    /// Constructor
    DamInputTableNodalYoungModulusProcess(ModelPart& rModelPart, TableType& Table,
                                Parameters& rParameters
                                ) : Process(Flags()) , mrModelPart(rModelPart) , mrTable(Table)
    {
        KRATOS_TRY

        //only include validation with c++11 since raw_literals do not exist in c++03
        Parameters default_parameters( R"(
            {
                "model_part_name":"PLEASE_CHOOSE_MODEL_PART_NAME",
                "variable_name"      : "PLEASE_PRESCRIBE_VARIABLE_NAME",
                "initial_value"      : 0.0,
                "input_file_name"    : "",
                "interval":[
                0.0,
                0.0
                ]
            }  )" );

        // Some values need to be mandatorily prescribed since no meaningful default value exist. For this reason try accessing to them
        // So that an error is thrown if they don't exist
        rParameters["variable_name"];
        rParameters["model_part_name"];

        // Now validate against defaults -- this also ensures no type mismatch
        rParameters.ValidateAndAssignDefaults(default_parameters);

        mVariableName = rParameters["variable_name"].GetString();
        mInitialValue = rParameters["initial_value"].GetDouble();
        mInputFile = rParameters["input_file_name"].GetString();

        KRATOS_CATCH("");
    }

    ///------------------------------------------------------------------------------------

    /// Destructor
    virtual ~DamInputTableNodalYoungModulusProcess() {}


//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    void Execute() override
    {
    }

    //----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    void ExecuteBeforeSolutionLoop() override
    {

        KRATOS_TRY;

        const Variable<double>& var = KratosComponents<Variable<double>>::Get(mVariableName);
        const int nnodes = mrModelPart.GetMesh(0).Nodes().size();

        if(nnodes != 0)
        {
            ModelPart::NodesContainerType::iterator it_begin = mrModelPart.GetMesh(0).NodesBegin();

            if ((mInputFile == "") || (mInputFile == "- No file") || (mInputFile == "- Add new file"))
            {
                #pragma omp parallel for
                for(int i = 0; i<nnodes; i++)
                {
                    ModelPart::NodesContainerType::iterator it = it_begin + i;

                    it->FastGetSolutionStepValue(var) = mInitialValue;

                }
            }
            else
            {
                #pragma omp parallel for
                for(int i = 0; i<nnodes; i++)
                {
                    ModelPart::NodesContainerType::iterator it = it_begin + i;

                    it->FastGetSolutionStepValue(var) = mrTable.GetValue(it->Id());

                }
            }
        }

        KRATOS_CATCH("");
    }

//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    void ExecuteInitializeSolutionStep() override
    {

        KRATOS_TRY;

        const Variable<double>& var = KratosComponents<Variable<double>>::Get(mVariableName);
        const int nnodes = mrModelPart.GetMesh(0).Nodes().size();


        if(nnodes != 0)
        {
            ModelPart::NodesContainerType::iterator it_begin = mrModelPart.GetMesh(0).NodesBegin();

            if ((mInputFile == "") || (mInputFile == "- No file") || (mInputFile == "- Add new file"))
            {
                #pragma omp parallel for
                for(int i = 0; i<nnodes; i++)
                {
                    ModelPart::NodesContainerType::iterator it = it_begin + i;

                    it->FastGetSolutionStepValue(var) = mInitialValue;

                }
            }
            else
            {
                #pragma omp parallel for
                for(int i = 0; i<nnodes; i++)
                {
                    ModelPart::NodesContainerType::iterator it = it_begin + i;

                    it->FastGetSolutionStepValue(var) = mrTable.GetValue(it->Id());

                }
            }
        }

        KRATOS_CATCH("");
    }

//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    /// Turn back information as a string.
    std::string Info() const override
    {
        return "DamInputTableNodalYoungModulusProcess";
    }

    /// Print information about this object.
    void PrintInfo(std::ostream& rOStream) const override
    {
        rOStream << "DamInputTableNodalYoungModulusProcess";
    }

    /// Print object's data.
    void PrintData(std::ostream& rOStream) const override
    {
    }

///----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

protected:

    /// Member Variables

    ModelPart& mrModelPart;
    TableType& mrTable;
    std::string mVariableName;
    double mInitialValue;
    std::string mInputFile;

//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

private:

    /// Assignment operator.
    DamInputTableNodalYoungModulusProcess& operator=(DamInputTableNodalYoungModulusProcess const& rOther);

};//Class


/// input stream function
inline std::istream& operator >> (std::istream& rIStream,
                                    DamInputTableNodalYoungModulusProcess& rThis);

/// output stream function
inline std::ostream& operator << (std::ostream& rOStream,
                                  const DamInputTableNodalYoungModulusProcess& rThis)
{
    rThis.PrintInfo(rOStream);
    rOStream << std::endl;
    rThis.PrintData(rOStream);

    return rOStream;
}

} /* namespace Kratos.*/

#endif /* KRATOS_DAM_INPUT_TABLE_NODAL_YOUNG_MODULUS_PROCESS defined */


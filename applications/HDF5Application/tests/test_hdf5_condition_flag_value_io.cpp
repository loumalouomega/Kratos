//    |  /           |
//    ' /   __| _` | __|  _ \   __|
//    . \  |   (   | |   (   |\__ `
//   _|\_\_|  \__,_|\__|\___/ ____/
//                   Multi-Physics
//
//  License:         BSD License
//                   Kratos default license: kratos/license.txt
//
//  Main authors:    Michael Andre, https://github.com/msandre
//                   Suneth Warnakulasuriya, https://github.com/sunethwarna
//

// System includes

// External includes

// Project includes
#include "containers/model.h"
#include "includes/kratos_parameters.h"
#include "testing/testing.h"

// Application includes
#include "custom_io/hdf5_file.h"
#include "custom_io/hdf5_container_component_io.h"
#include "custom_utilities/container_io_utils.h"
#include "tests/test_utils.h"

namespace Kratos
{
namespace Testing
{
KRATOS_TEST_CASE_IN_SUITE(HDF5PointsData_ReadConditionFlags, KratosHDF5TestSuite)
{
    Parameters file_params(R"(
        {
            "file_name" : "test.h5",
            "file_access_mode": "exclusive",
            "file_driver": "core"
        })");
    Model this_model;
    ModelPart& r_read_model_part = this_model.CreateModelPart("test_read");
    ModelPart& r_write_model_part = this_model.CreateModelPart("test_write");

    auto p_test_file = Kratos::make_shared<HDF5::File>(r_read_model_part.GetCommunicator().GetDataCommunicator(), file_params);

    TestModelPartFactory::CreateModelPart(r_write_model_part, {{"Element2D3N"}},
                                          {{"LineCondition2D2N"}});
    TestModelPartFactory::CreateModelPart(r_read_model_part, {{"Element2D3N"}},
                                          {{"LineCondition2D2N"}});

    r_read_model_part.SetBufferSize(2);
    r_write_model_part.SetBufferSize(2);

    std::vector<std::string> variables_list = {"SLIP", "ACTIVE", "STRUCTURE"};

    // "shuffle" the list of variables to check whether it's handled
    // without deadlocks.
    std::rotate(
        variables_list.begin(),
        variables_list.begin() + (r_read_model_part.GetCommunicator().GetDataCommunicator().Rank() % variables_list.size()),
        variables_list.end()
    );

    for (auto& r_condition : r_write_model_part.Conditions())
    {
        TestModelPartFactory::AssignDataValueContainer(
            r_condition.GetData(), r_condition, variables_list);
    }

    Parameters io_params(R"(
        {
            "prefix": "/Step",
            "list_of_variables": []
        })");
    io_params["list_of_variables"].SetStringArray(variables_list);

    HDF5::ContainerComponentIO<ModelPart::ConditionsContainerType, HDF5::Internals::FlagIO, Flags> data_io(io_params, p_test_file);
    data_io.Write(r_write_model_part.Conditions(), HDF5::Internals::FlagIO{}, Parameters("""{}"""));
    data_io.Read(r_read_model_part.Conditions(), HDF5::Internals::FlagIO{}, r_read_model_part.GetCommunicator());

    for (auto& r_write_condition : r_write_model_part.Conditions())
    {
        HDF5::ConditionType& r_read_condition =
            r_read_model_part.Conditions()[r_write_condition.Id()];
        CompareDataValueContainers({"SLIP", "ACTIVE", "STRUCTURE"}, r_read_condition.GetData(), r_read_condition,
                                   r_write_condition.GetData(), r_write_condition);
    }
}

} // namespace Testing
} // namespace Kratos.

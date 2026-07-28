import json

import KratosMultiphysics
import KratosMultiphysics.ConvectionDiffusionApplication as KratosConvDiff

from KratosMultiphysics.json_utilities import write_external_json


def Factory(settings, Model):
    if(type(settings) != KratosMultiphysics.Parameters):
        raise Exception("expected input shall be a Parameters object, encapsulating a json string")
    return ComputeHtcTableProcess(Model, settings["Parameters"])


class ComputeHtcTableProcess(KratosMultiphysics.Process):
    """Drives the HeatTransferCoefficientUtility and writes the resulting table to a JSON file.

    The utility estimates the effective heat transfer coefficient of the interface between two
    materials, or between one material and air, and returns it as a temperature-HTC table in the
    schema understood by ReadMaterialsUtility. This process only takes care of running it and of
    dumping the result to disk, so that it can be pasted into a materials.json file.

    Every setting other than the ones listed in the defaults below is forwarded verbatim to the
    utility, whose own defaults are the authority on their meaning.
    """

    def __init__(self, Model, settings):
        KratosMultiphysics.Process.__init__(self)

        # The process owns two settings of its own, the rest belongs to the utility
        process_settings = KratosMultiphysics.Parameters("{}")
        for key in ["output_file_name", "wrap_in_tables_block", "table_name"]:
            if settings.Has(key):
                process_settings.AddValue(key, settings[key])
                settings.RemoveValue(key)

        default_settings = KratosMultiphysics.Parameters(r'''{
            "output_file_name"     : "htc_table.json",
            "wrap_in_tables_block" : false,
            "table_name"           : "Table1"
        }''')
        process_settings.ValidateAndAssignDefaults(default_settings)

        self.output_file_name = process_settings["output_file_name"].GetString()
        self.wrap_in_tables_block = process_settings["wrap_in_tables_block"].GetBool()
        self.table_name = process_settings["table_name"].GetString()

        if self.output_file_name == "":
            raise Exception("Empty 'output_file_name'. Set a valid file name.")

        # The utility validates the remaining settings against its own defaults
        self.utility = KratosConvDiff.HeatTransferCoefficientUtility(Model, settings)

        self.table = None

    def ExecuteInitialize(self):
        """Computes the table and writes it out.

        The estimation has no time dependency, so it is performed once at the very beginning.
        """
        self.table = self.utility.ComputeTable()

        if self.wrap_in_tables_block:
            output = KratosMultiphysics.Parameters("{}")
            output.AddValue("Tables", KratosMultiphysics.Parameters("{}"))
            output["Tables"].AddValue(self.table_name, self.table)
        else:
            output = self.table

        # write_external_json expects a plain python object, not a json string
        write_external_json(self.output_file_name, json.loads(output.WriteJsonString()))

        KratosMultiphysics.Logger.PrintInfo(
            "ComputeHtcTableProcess",
            "Heat transfer coefficient table written to '{}'.".format(self.output_file_name))

    def GetTable(self):
        """Returns the computed table as a Parameters object, or None if not computed yet."""
        return self.table

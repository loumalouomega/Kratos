import KratosMultiphysics
import KratosMultiphysics.KratosUnittest as KratosUnittest

import KratosMultiphysics.kratos_utilities as kratos_utils
from KratosMultiphysics.gid_output_process import GiDOutputProcess
from KratosMultiphysics import compare_two_files_check_process

import os
import sys
import subprocess

def GetFilePath(fileName):
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), fileName)

class TestGidIO(KratosUnittest.TestCase):

    def __WriteOutput(self, model_part, output_file):

        gid_output = GiDOutputProcess(model_part,
                                    output_file,
                                    KratosMultiphysics.Parameters("""
                                        {
                                            "result_file_configuration": {
                                                "gidpost_flags": {
                                                    "GiDPostMode": "GiD_PostAscii",
                                                    "WriteDeformedMeshFlag": "WriteUndeformed",
                                                    "WriteConditionsFlag": "WriteConditions",
                                                    "MultiFileFlag": "SingleFile"
                                                },
                                                "file_label": "time",
                                                "output_control_type": "step",
                                                "output_interval": 1.0,
                                                "body_output": true,
                                                "node_output": false,
                                                "skin_output": false,
                                                "plane_output": [],
                                                "nodal_results": ["DISPLACEMENT","VISCOSITY"],
                                                "nodal_nonhistorical_results": ["TEMPERATURE","INERTIA","ELEMENTAL_DISTANCES"],
                                                "nodal_flags_results": ["ISOLATED"],
                                                "gauss_point_results": [],
                                                "additional_list_files": []
                                            }
                                        }
                                        """)
                                    )

        gid_output.ExecuteInitialize()
        gid_output.ExecuteBeforeSolutionLoop()
        gid_output.ExecuteInitializeSolutionStep()
        gid_output.PrintOutput()
        gid_output.ExecuteFinalizeSolutionStep()
        gid_output.ExecuteFinalize()

    def __InitialRead(self, current_model):
        model_part = current_model.CreateModelPart("Main")
        model_part.AddNodalSolutionStepVariable(KratosMultiphysics.DISPLACEMENT)
        model_part.AddNodalSolutionStepVariable(KratosMultiphysics.VISCOSITY)
        model_part.AddNodalSolutionStepVariable(KratosMultiphysics.VELOCITY)
        KratosMultiphysics.ModelPartIO(GetFilePath("auxiliar_files_for_python_unittest/mdpa_files/test_model_part_io_read")).ReadModelPart(model_part)
        model_part.SetBufferSize(2)
        return model_part

    def __Check(self,output_file,reference_file):

        ## Settings string in json format
        params = KratosMultiphysics.Parameters("""
            {
                "reference_file_name"   : "",
                "output_file_name"      : ""
            }
        """)

        params["reference_file_name"].SetString(GetFilePath(reference_file))
        params["output_file_name"].SetString(output_file)

        cmp_process = compare_two_files_check_process.CompareTwoFilesCheckProcess(params)

        cmp_process.ExecuteInitialize()
        cmp_process.ExecuteBeforeSolutionLoop()
        cmp_process.ExecuteInitializeSolutionStep()
        cmp_process.ExecuteFinalizeSolutionStep()
        cmp_process.ExecuteBeforeOutputStep()
        cmp_process.ExecuteAfterOutputStep()
        cmp_process.ExecuteFinalize()

    def test_gid_io_all(self):
        current_model = KratosMultiphysics.Model()

        model_part = self.__InitialRead(current_model)

        self.__WriteOutput(model_part,"all_active_out")

        self.__Check("all_active_out.post.msh","auxiliar_files_for_python_unittest/reference_files/all_active_ref.ref")

    def test_gid_io_deactivation(self):
        current_model = KratosMultiphysics.Model()

        model_part = self.__InitialRead(current_model)

        model_part.Elements[3].Set(KratosMultiphysics.ACTIVE,False)
        model_part.Elements[1796].Set(KratosMultiphysics.ACTIVE,False)

        model_part.Conditions[1947].Set(KratosMultiphysics.ACTIVE,False)
        model_part.Conditions[1948].Set(KratosMultiphysics.ACTIVE,False)

        self.__WriteOutput(model_part,"deactivated_out")

        self.__Check("deactivated_out.post.msh","auxiliar_files_for_python_unittest/reference_files/deactivated_ref.ref")

    def test_gid_io_results(self):
        current_model = KratosMultiphysics.Model()

        model_part = self.__InitialRead(current_model)

        # Initialize flag
        for node in model_part.Nodes:
            node.Set(KratosMultiphysics.ISOLATED, False)

        model_part.Nodes[1].Set(KratosMultiphysics.ISOLATED, True)
        model_part.Nodes[2].Set(KratosMultiphysics.ISOLATED, True)
        model_part.Nodes[973].Set(KratosMultiphysics.ISOLATED, True)
        model_part.Nodes[974].Set(KratosMultiphysics.ISOLATED, True)

        # Initialize value
        for node in model_part.Nodes:
            node.SetValue(KratosMultiphysics.TEMPERATURE, 0.0)

        model_part.Nodes[1].SetValue(KratosMultiphysics.TEMPERATURE, 100.0)
        model_part.Nodes[2].SetValue(KratosMultiphysics.TEMPERATURE, 200.0)
        model_part.Nodes[973].SetValue(KratosMultiphysics.TEMPERATURE, 300.0)
        model_part.Nodes[974].SetValue(KratosMultiphysics.TEMPERATURE, 400.0)

        vector = KratosMultiphysics.Vector(3)
        vector[0] = 0.0
        vector[1] = 0.0
        vector[2] = 0.0
        # Initialize value
        for node in model_part.Nodes:
            node.SetValue(KratosMultiphysics.ELEMENTAL_DISTANCES, vector)

        vector[0] = 1.0
        vector[1] = 1.0
        model_part.Nodes[1].SetValue(KratosMultiphysics.ELEMENTAL_DISTANCES, vector)
        model_part.Nodes[2].SetValue(KratosMultiphysics.ELEMENTAL_DISTANCES, vector)
        model_part.Nodes[973].SetValue(KratosMultiphysics.ELEMENTAL_DISTANCES, vector)
        model_part.Nodes[974].SetValue(KratosMultiphysics.ELEMENTAL_DISTANCES, vector)

        matrix = KratosMultiphysics.Matrix(2, 2)
        matrix[0, 0] = 0.0
        matrix[0, 1] = 0.0
        matrix[1, 0] = 0.0
        matrix[1, 1] = 0.0
        # Initialize value
        for node in model_part.Nodes:
            node.SetValue(KratosMultiphysics.INERTIA, matrix)

        matrix[0, 0] = 1.0
        matrix[1, 1] = 1.0
        model_part.Nodes[1].SetValue(KratosMultiphysics.INERTIA, matrix)
        model_part.Nodes[2].SetValue(KratosMultiphysics.INERTIA, matrix)
        model_part.Nodes[973].SetValue(KratosMultiphysics.INERTIA, matrix)
        model_part.Nodes[974].SetValue(KratosMultiphysics.INERTIA, matrix)

        self.__WriteOutput(model_part,"results_out")

        self.__Check("results_out.post.res","auxiliar_files_for_python_unittest/reference_files/results_out_ref.ref")

    def test_gid_io_real_number_format(self):
        """Custom real_number_format must actually reach gidpost's ASCII output.

        gidpost caches each result type's number format the first time it is written in the
        process and never re-reads it afterwards (see the caveat documented on GidIO's constructor
        in kratos/includes/gid_io.h), so this can only be observed reliably as the very first GiD
        write of a process -- other tests in this same file already write ASCII output with the
        default "%g" format, which would otherwise make this test's own outcome depend on test
        execution order. The write therefore runs in its own subprocess via an auxiliary script.
        """
        script = GetFilePath("auxiliar_files_for_python_unittest/gid_io/write_gid_io_with_custom_number_format.py")
        output_file = "custom_number_format_out"

        # 1/3 differs under "%g" (6 significant digits, the default) vs "%.15g"
        subprocess.run([sys.executable, script, output_file, "%.15g"], check=True)

        with open(output_file + ".post.res") as f:
            content = f.read()

        self.assertIn("0.333333333333333", content)

        kratos_utils.DeleteFileIfExisting(output_file + ".post.msh")
        kratos_utils.DeleteFileIfExisting(output_file + ".post.res")

    def test_gid_io_invalid_number_format(self):
        """GidIO must reject number formats that would overflow gidpost's fixed-size buffers or
        that carry no printf conversion specifier, rather than passing them through."""
        gid_mode = KratosMultiphysics.GiDPostMode.GiD_PostAscii
        multifile = KratosMultiphysics.MultiFileFlag.SingleFile
        deformed_mesh_flag = KratosMultiphysics.WriteDeformedMeshFlag.WriteUndeformed
        write_conditions = KratosMultiphysics.WriteConditionsFlag.WriteConditions

        with self.assertRaisesRegex(RuntimeError, "must contain a printf conversion specifier"):
            KratosMultiphysics.GidIO("invalid_format_out", gid_mode, multifile,
                deformed_mesh_flag, write_conditions, True, "no_percent_sign", "%.16g")

        with self.assertRaisesRegex(RuntimeError, "must be non-empty"):
            KratosMultiphysics.GidIO("invalid_format_out", gid_mode, multifile,
                deformed_mesh_flag, write_conditions, True, "%g", "")

        with self.assertRaisesRegex(RuntimeError, "shorter than 32 characters"):
            KratosMultiphysics.GidIO("invalid_format_out", gid_mode, multifile,
                deformed_mesh_flag, write_conditions, True, "%" + "0"*40 + "g", "%.16g")

        kratos_utils.DeleteFileIfExisting("invalid_format_out.post.msh")

    @KratosUnittest.skipUnless(
        KratosMultiphysics.Registry.HasItem("libraries.gidpost_hdf5"),
        "This test requires Kratos to be built with -DKRATOS_GIDPOST_WITH_HDF5=ON")
    def test_gid_io_hdf5(self):
        """GiD_PostHDF5 must actually produce an HDF5 file when Kratos is built with HDF5 support."""
        current_model = KratosMultiphysics.Model()

        model_part = self.__InitialRead(current_model)

        gid_output = GiDOutputProcess(model_part,
                                    "hdf5_out",
                                    KratosMultiphysics.Parameters("""
                                        {
                                            "result_file_configuration": {
                                                "gidpost_flags": {
                                                    "GiDPostMode": "GiD_PostHDF5",
                                                    "WriteDeformedMeshFlag": "WriteUndeformed",
                                                    "WriteConditionsFlag": "WriteConditions",
                                                    "MultiFileFlag": "SingleFile"
                                                },
                                                "file_label": "time",
                                                "output_control_type": "step",
                                                "output_interval": 1.0,
                                                "body_output": true,
                                                "node_output": false,
                                                "skin_output": false,
                                                "plane_output": [],
                                                "nodal_results": ["DISPLACEMENT"],
                                                "gauss_point_results": [],
                                                "additional_list_files": []
                                            }
                                        }
                                        """)
                                    )

        gid_output.ExecuteInitialize()
        gid_output.ExecuteBeforeSolutionLoop()
        gid_output.ExecuteInitializeSolutionStep()
        gid_output.PrintOutput()
        gid_output.ExecuteFinalizeSolutionStep()
        gid_output.ExecuteFinalize()

        # GidIO names GiD_PostHDF5 output the same way as GiD_PostBinary (".post.bin"); gidpost
        # writes actual HDF5 content into it regardless of the extension.
        hdf5_file = "hdf5_out.post.bin"
        self.assertTrue(os.path.isfile(hdf5_file))

        # HDF5's signature: https://docs.hdfgroup.org/hdf5/develop/_f_m_t3.html
        hdf5_signature = b"\x89HDF\r\n\x1a\n"
        with open(hdf5_file, "rb") as f:
            self.assertEqual(f.read(len(hdf5_signature)), hdf5_signature)

        kratos_utils.DeleteFileIfExisting(hdf5_file)
        kratos_utils.DeleteFileIfExisting("python_scripts.post.lst")
        kratos_utils.DeleteFileIfExisting("tests.post.lst")

    def test_DoubleFreeError(self):
        current_model = KratosMultiphysics.Model()

        output_file_1 = "outFile"
        output_file_2 = "otherFile"

        gid_mode = KratosMultiphysics.GiDPostMode.GiD_PostAscii
        multifile = KratosMultiphysics.MultiFileFlag.MultipleFiles
        deformed_mesh_flag = KratosMultiphysics.WriteDeformedMeshFlag.WriteUndeformed
        write_conditions = KratosMultiphysics.WriteConditionsFlag.WriteConditions

        gid_io_1 = KratosMultiphysics.GidIO(output_file_1, gid_mode, multifile,
                         deformed_mesh_flag, write_conditions)

        gid_io_2 = KratosMultiphysics.GidIO(output_file_2, gid_mode, multifile,
                         deformed_mesh_flag, write_conditions)

        gid_io_1 = None
        gid_io_2 = None

    def tearDown(self):
        kratos_utils.DeleteFileIfExisting("all_active_out.post.msh")
        kratos_utils.DeleteFileIfExisting("all_active_out.post.res")
        kratos_utils.DeleteFileIfExisting("deactivated_out.post.msh")
        kratos_utils.DeleteFileIfExisting("deactivated_out.post.res")
        kratos_utils.DeleteFileIfExisting("results_out.post.msh")
        kratos_utils.DeleteFileIfExisting("results_out.post.res")
        kratos_utils.DeleteFileIfExisting("python_scripts.post.lst")
        kratos_utils.DeleteFileIfExisting("tests.post.lst")


if __name__ == '__main__':
    KratosUnittest.main()

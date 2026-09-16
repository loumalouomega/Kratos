"""Writes a single ASCII GiD result file with a caller-supplied real_number_format.

gidpost caches each result type's number format the first time it is written in the process and
never re-reads it afterwards (see the caveat documented on GidIO's constructor in
kratos/includes/gid_io.h), so a custom real_number_format can only be observed reliably as the
very first GiD write of a process. This script is therefore run in its own subprocess by
test_gid_io.py's test_gid_io_real_number_format, rather than sharing a process with the other GiD
tests in that file.

Usage: write_gid_io_with_custom_number_format.py <output_file_name_without_extension> <real_number_format>
"""
import sys

import KratosMultiphysics


def main(output_file_name, real_number_format):
    model = KratosMultiphysics.Model()
    model_part = model.CreateModelPart("Main")
    model_part.AddNodalSolutionStepVariable(KratosMultiphysics.DISPLACEMENT)
    node = model_part.CreateNewNode(1, 0.0, 0.0, 0.0)

    displacement = KratosMultiphysics.Vector(3)
    displacement[0] = 1.0 / 3.0
    displacement[1] = 0.0
    displacement[2] = 0.0
    node.SetSolutionStepValue(KratosMultiphysics.DISPLACEMENT, displacement)

    gid_io = KratosMultiphysics.GidIO(
        output_file_name,
        KratosMultiphysics.GiDPostMode.GiD_PostAscii,
        KratosMultiphysics.MultiFileFlag.SingleFile,
        KratosMultiphysics.WriteDeformedMeshFlag.WriteUndeformed,
        KratosMultiphysics.WriteConditionsFlag.WriteElementsOnly,
        False,
        real_number_format,
        "%.16g")

    gid_io.InitializeMesh(0.0)
    gid_io.WriteNodeMesh(model_part.GetMesh())
    gid_io.FinalizeMesh()
    gid_io.InitializeResults(0.0, model_part.GetMesh())
    gid_io.WriteNodalResults(KratosMultiphysics.DISPLACEMENT, model_part.Nodes, 0.0, 0)
    gid_io.FinalizeResults()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

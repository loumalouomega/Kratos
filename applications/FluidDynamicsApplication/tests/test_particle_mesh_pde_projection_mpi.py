import math

import KratosMultiphysics as KM
import KratosMultiphysics.KratosUnittest as KratosUnittest
import KratosMultiphysics.FluidDynamicsApplication as KratosCFD
import KratosMultiphysics.TrilinosApplication as KratosTrilinos
from KratosMultiphysics.FluidDynamicsApplication import particle_pde_projection_utilities
from KratosMultiphysics.FluidDynamicsApplication.fluid_dynamics_analysis import FluidDynamicsAnalysis
from KratosMultiphysics.FluidDynamicsApplication.trilinos_navier_stokes_two_fluids_solver import NavierStokesMPITwoFluidsSolver
from KratosMultiphysics.mpi.distributed_import_model_part_utility import DistributedImportModelPartUtility

from particle_mesh_mpi_test_utilities import PartitionInStripes
from test_particle_mesh_pde_projection import _SlottedDiskDistance

def _CreateUnitSquare(model, name, divisions, time_step):
    model_part = model.CreateModelPart(name, 2)
    model_part.AddNodalSolutionStepVariable(KM.DISTANCE)
    model_part.AddNodalSolutionStepVariable(KM.VELOCITY)
    model_part.AddNodalSolutionStepVariable(KM.PARTITION_INDEX)
    model_part.ProcessInfo[KM.DOMAIN_SIZE] = 2
    model_part.ProcessInfo[KM.DELTA_TIME] = time_step
    geometry = KM.Quadrilateral2D4(KM.Node(1, 0.0, 0.0, 0.0), KM.Node(2, 0.0, 1.0, 0.0), KM.Node(3, 1.0, 1.0, 0.0), KM.Node(4, 1.0, 0.0, 0.0))
    mesher_settings = KM.Parameters("""{
        "number_of_divisions"        : 1,
        "element_name"               : "Element2D3N",
        "condition_name"             : "LineCondition2D2N",
        "create_skin_sub_model_part" : false
    }""")
    mesher_settings["number_of_divisions"].SetInt(divisions)
    KM.StructuredMeshGeneratorProcess(geometry, model_part, mesher_settings).Execute()
    for node in model_part.Nodes:
        node.SetSolutionStepValue(KM.DISTANCE, _SlottedDiskDistance(node.X, node.Y))
        # Divergence-free vortex with zero normal velocity on the walls (closed domain)
        velocity = [math.sin(math.pi * node.X) * math.cos(math.pi * node.Y), -math.cos(math.pi * node.X) * math.sin(math.pi * node.Y), 0.0]
        node.SetSolutionStepValue(KM.VELOCITY, 0, velocity)
        node.SetSolutionStepValue(KM.VELOCITY, 1, velocity)
    return model_part

class TestParticleMeshPdeProjectionMPI(KratosUnittest.TestCase):
    """Particle level set transport in MPI: particle migration, distributed facet system and global reductions."""

    def testSlottedDiskInClosedVortexMPI2D(self):
        data_communicator = KM.ParallelEnvironment.GetDefaultDataCommunicator()
        model = KM.Model()
        distributed_model_part = _CreateUnitSquare(model, "Distributed", 24, 0.02)
        PartitionInStripes(distributed_model_part, data_communicator)
        self.assertTrue(distributed_model_part.IsDistributed())

        # Reference: the same problem solved in every rank without partitioning
        serial_model_part = _CreateUnitSquare(model, "Serial", 24, 0.02)
        self.assertFalse(serial_model_part.IsDistributed())

        settings = KM.Parameters("""{
            "linear_solver_settings" : {
                "solver_type"         : "amgcl",
                "krylov_type"         : "cg",
                "preconditioner_type" : "relaxation",
                "smoother_type"       : "damped_jacobi",
                "tolerance"           : 1e-12,
                "max_iteration"       : 2000,
                "block_size"          : 1,
                "verbosity"           : 0
            }
        }""")
        epetra_communicator = KratosTrilinos.CreateEpetraCommunicator(data_communicator)
        distributed_process = particle_pde_projection_utilities.CreateParticlePdeProjectionProcess(distributed_model_part, 2, settings, epetra_communicator)
        serial_process = particle_pde_projection_utilities.CreateParticlePdeProjectionProcess(serial_model_part, 2, settings)
        distributed_process.ExecuteInitialize()
        serial_process.ExecuteInitialize()

        # The seeding does not depend on the partitioning
        initial_volume = distributed_process.GetIndicatorIntegral()
        self.assertAlmostEqual(initial_volume, serial_process.GetIndicatorIntegral(), delta=1.0e-12)
        self.assertEqual(distributed_process.GetNumberOfParticles(), serial_process.GetNumberOfParticles())

        for _ in range(8):
            distributed_process.Execute()
            serial_process.Execute()
            info = distributed_process.GetLastProjectionInfo()
            # Exact local and global conservation (closed domain)
            self.assertLess(info["max_cell_residual"], 1.0e-12)
            self.assertAlmostEqual(info["net_boundary_outflow"], 0.0, delta=1.0e-14)
            self.assertAlmostEqual(info["integral"], info["previous_integral"], delta=1.0e-13)
            self.assertLess(info["facet_system_residual"], 1.0e-10)
            # No particle is lost when crossing the partitions (the RK stage points beyond a partition use the clipped local
            # velocity, so a particle may end up in a different element than in the serial run and be reseeded differently)
            n_distributed = distributed_process.GetNumberOfParticles()
            n_serial = serial_process.GetNumberOfParticles()
            self.assertLess(abs(n_distributed - n_serial), 1.0e-3 * n_serial)

        self.assertAlmostEqual(distributed_process.GetIndicatorIntegral(), initial_volume, delta=1.0e-12)
        negative_volume = KratosCFD.FluidAuxiliaryUtilities.CalculateFluidNegativeVolume(distributed_model_part)
        self.assertAlmostEqual(negative_volume, distributed_process.GetIndicatorIntegral(), delta=1.0e-8)

        # The distributed solution matches the serial one (differences come from the RK stage points beyond the partitions)
        rank = data_communicator.Rank()
        max_difference = 0.0
        for node in distributed_model_part.Nodes:
            if node.GetSolutionStepValue(KM.PARTITION_INDEX) == rank:
                serial_node = serial_model_part.GetNode(node.Id)
                max_difference = max(max_difference, abs(node.GetSolutionStepValue(KM.DISTANCE) - serial_node.GetSolutionStepValue(KM.DISTANCE)))
        max_difference = data_communicator.MaxAll(max_difference)
        self.assertLess(max_difference, 1.0e-3)

class _StripePartitionedTwoFluidsSolver(NavierStokesMPITwoFluidsSolver):
    """MPI two-fluids solver whose mesh is read in every rank and partitioned in stripes (no METIS needed)."""

    def ImportModelPart(self):
        KM.ModelPartIO(self.settings["model_import_settings"]["input_filename"].GetString()).ReadModelPart(self.main_model_part)
        PartitionInStripes(self.main_model_part, KM.ParallelEnvironment.GetDefaultDataCommunicator(), fill_communicator=False)
        import_settings = self.settings.Clone()
        import_settings["model_import_settings"]["input_type"].SetString("use_input_model_part")
        self.distributed_model_part_importer = DistributedImportModelPartUtility(self.main_model_part, import_settings)
        self.distributed_model_part_importer.ImportModelPart()

class _ParticleTransportSloshingMPIAnalysis(FluidDynamicsAnalysis):

    def __init__(self, model, project_parameters, test_case):
        super().__init__(model, project_parameters)
        self.test_case = test_case

    def _CreateSolver(self):
        return _StripePartitionedTwoFluidsSolver(self.model, self.project_parameters["solver_settings"])

    def ModifyInitialGeometry(self):
        for node in self._GetSolver().GetComputingModelPart().Nodes:
            distance = node.Y - (0.5 + 0.1 * math.cos(math.pi * (1.0 - node.X)))
            node.SetSolutionStepValue(KM.DISTANCE, distance)
            node.SetSolutionStepValue(KM.DISTANCE, 1, distance)

    def InitializeSolutionStep(self):
        super().InitializeSolutionStep()
        process = self._GetSolver()._GetLevelSetConvectionProcess()
        info = process.GetLastProjectionInfo()
        volume = info["integral"]
        self.test_case.assertLess(info["max_cell_residual"], 1.0e-12)
        self.test_case.assertAlmostEqual(info["integral"] - info["previous_integral"] + info["net_boundary_outflow"], 0.0, delta=1.0e-12 * volume)
        negative_volume = KratosCFD.FluidAuxiliaryUtilities.CalculateFluidNegativeVolume(self._GetSolver().GetComputingModelPart())
        self.test_case.assertAlmostEqual(negative_volume, volume, delta=1.0e-4 * volume)
        self.test_case.volumes.append(volume)

class TestTrilinosTwoFluidsParticleLevelSetTransport(KratosUnittest.TestCase):

    def testSloshingTankMPI2D(self):
        self.volumes = []
        with KratosUnittest.WorkFolderScope("TwoFluidMassConservationSourceTest", __file__):
            with open("ProjectParameters2D.json", 'r') as parameter_file:
                parameters = KM.Parameters(parameter_file.read())
            parameters["problem_data"]["parallel_type"].SetString("MPI")
            parameters["problem_data"]["end_time"].SetDouble(0.5)
            parameters["problem_data"]["echo_level"].SetInt(0)
            solver_settings = parameters["solver_settings"]
            solver_settings["echo_level"].SetInt(0)
            solver_settings["formulation"]["mass_source"].SetBool(False)
            solver_settings.AddEmptyValue("levelset_transport_type").SetString("particle_pde_projection")
            solver_settings.AddValue("particle_pde_projection_settings", KM.Parameters("""{
                "particles_per_element" : {"initial" : 20, "minimum" : 12, "maximum" : 40},
                "zeta" : 20.0
            }"""))
            parameters["processes"]["json_check_process_list"] = KM.Parameters("[]")

            _ParticleTransportSloshingMPIAnalysis(KM.Model(), parameters, self).Run()

        # The tank is closed: the water volume does not change
        self.assertEqual(len(self.volumes), 5)
        for volume in self.volumes:
            self.assertAlmostEqual(volume, self.volumes[0], delta=1.0e-12 * self.volumes[0])

if __name__ == '__main__':
    KratosUnittest.main()

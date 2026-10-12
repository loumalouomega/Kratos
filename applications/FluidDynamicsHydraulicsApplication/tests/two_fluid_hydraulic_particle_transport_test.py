import math

import KratosMultiphysics
import KratosMultiphysics.FluidDynamicsApplication as KratosCFD
from KratosMultiphysics.FluidDynamicsHydraulicsApplication.fluid_dynamics_hydraulic_analysis import FluidDynamicsHydraulicsAnalysis

import KratosMultiphysics.KratosUnittest as UnitTest
import KratosMultiphysics.kratos_utilities as KratosUtilities

class _ParticleTransportHydraulicAnalysis(FluidDynamicsHydraulicsAnalysis):
    """Sloshing tank solved with the particle level set transport (LEoPart PDE-constrained projection)."""

    def __init__(self, model, project_parameters, test_case):
        super().__init__(model, project_parameters)
        self.test_case = test_case

    def ModifyInitialGeometry(self):
        for node in self._GetSolver().GetComputingModelPart().Nodes:
            distance = node.Y - (0.5 + 0.1 * math.cos(math.pi * (1.0 - node.X)))
            node.SetSolutionStepValue(KratosMultiphysics.DISTANCE, distance)
            node.SetSolutionStepValue(KratosMultiphysics.DISTANCE, 1, distance)

    def InitializeSolutionStep(self):
        super().InitializeSolutionStep()
        process = self._GetSolver()._GetLevelSetConvectionProcess()
        info = process.GetLastProjectionInfo()
        volume = info["integral"]
        self.test_case.assertLess(info["max_cell_residual"], 1.0e-12)
        self.test_case.assertAlmostEqual(info["integral"] - info["previous_integral"] + info["net_boundary_outflow"], 0.0, delta=1.0e-12 * volume)
        self.test_case.assertGreater(info["min_value"], -0.15)
        self.test_case.assertLess(info["max_value"], 1.15)
        # The distance is shifted to the conserved water volume (the distance modification process then moves it slightly)
        negative_volume = KratosCFD.FluidAuxiliaryUtilities.CalculateFluidNegativeVolume(self._GetSolver().GetComputingModelPart())
        self.test_case.assertAlmostEqual(negative_volume, volume, delta=1.0e-4 * volume)
        self.test_case.volumes.append(volume)

class TwoFluidHydraulicParticleTransportTest(UnitTest.TestCase):

    def setUp(self):
        self.work_folder = "two_fluid_hydraulic_solver_test"
        self.volumes = []

    def tearDown(self):
        with UnitTest.WorkFolderScope(self.work_folder, __file__):
            KratosUtilities.DeleteFileIfExisting('.time')

    def testTwoFluidHydraulicParticleTransport2D(self):
        self._RunTest("ProjectParameters2D.json", 5)

    def testTwoFluidHydraulicParticleTransport3D(self):
        self._RunTest("ProjectParameters3D.json", 2)

    def _RunTest(self, project_parameters_name, number_of_steps):
        with UnitTest.WorkFolderScope(self.work_folder, __file__):
            with open(project_parameters_name, 'r') as parameter_file:
                parameters = KratosMultiphysics.Parameters(parameter_file.read())

            time_step = parameters["solver_settings"]["time_stepping"]["time_step"].GetDouble()
            parameters["problem_data"]["end_time"].SetDouble(number_of_steps * time_step)
            parameters["solver_settings"]["echo_level"].SetInt(0)
            parameters["solver_settings"].AddEmptyValue("levelset_transport_type").SetString("particle_pde_projection")
            parameters["processes"]["json_check_process_list"] = KratosMultiphysics.Parameters("[]")

            _ParticleTransportHydraulicAnalysis(KratosMultiphysics.Model(), parameters, self).Run()

        # The tank is closed: the water volume does not change
        self.assertEqual(len(self.volumes), number_of_steps)
        for volume in self.volumes:
            self.assertAlmostEqual(volume, self.volumes[0], delta=1.0e-12 * self.volumes[0])

if __name__ == "__main__":
    UnitTest.main()

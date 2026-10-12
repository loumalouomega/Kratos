import math

import KratosMultiphysics
import KratosMultiphysics.KratosUnittest as KratosUnittest
import KratosMultiphysics.FluidDynamicsApplication as KratosCFD
from KratosMultiphysics.FluidDynamicsApplication.fluid_dynamics_analysis import FluidDynamicsAnalysis
from KratosMultiphysics.FluidDynamicsApplication.navier_stokes_two_fluids_solver import NavierStokesTwoFluidsSolver

def _SlottedDiskDistance(x, y):
    """Signed distance (approximate inside the slot corners) of a Zalesak slotted disk."""
    center_x, center_y, radius = 0.5, 0.65, 0.2
    slot_half_width, slot_bottom, slot_top = 0.05, 0.4, 0.75
    disk = math.hypot(x - center_x, y - center_y) - radius
    # Box distance of the slot (negative inside)
    q_x = abs(x - center_x) - slot_half_width
    q_y = abs(y - 0.5 * (slot_bottom + slot_top)) - 0.5 * (slot_top - slot_bottom)
    slot = math.hypot(max(q_x, 0.0), max(q_y, 0.0)) + min(max(q_x, q_y), 0.0)
    return max(disk, -slot)

class TestParticleMeshPdeProjection(KratosUnittest.TestCase):
    """Particle-based level set transport with the LEoPart PDE-constrained projection."""

    def _CreateUnitSquare(self, divisions):
        self.model = KratosMultiphysics.Model()
        model_part = self.model.CreateModelPart("Square", 2)
        model_part.AddNodalSolutionStepVariable(KratosMultiphysics.DISTANCE)
        model_part.AddNodalSolutionStepVariable(KratosMultiphysics.VELOCITY)
        model_part.ProcessInfo[KratosMultiphysics.DOMAIN_SIZE] = 2
        geometry = KratosMultiphysics.Quadrilateral2D4(
            KratosMultiphysics.Node(1, 0.0, 0.0, 0.0),
            KratosMultiphysics.Node(2, 0.0, 1.0, 0.0),
            KratosMultiphysics.Node(3, 1.0, 1.0, 0.0),
            KratosMultiphysics.Node(4, 1.0, 0.0, 0.0))
        mesher_settings = KratosMultiphysics.Parameters("""{
            "number_of_divisions"        : 1,
            "element_name"               : "Element2D3N",
            "condition_name"             : "LineCondition2D2N",
            "create_skin_sub_model_part" : false
        }""")
        mesher_settings["number_of_divisions"].SetInt(divisions)
        KratosMultiphysics.StructuredMeshGeneratorProcess(geometry, model_part, mesher_settings).Execute()
        return model_part

    def testSlottedDiskInClosedVortex2D(self):
        model_part = self._CreateUnitSquare(32)
        model_part.ProcessInfo[KratosMultiphysics.DELTA_TIME] = 0.02
        for node in model_part.Nodes:
            node.SetSolutionStepValue(KratosMultiphysics.DISTANCE, _SlottedDiskDistance(node.X, node.Y))
            # Divergence-free vortex with zero normal velocity on the walls (closed domain)
            velocity = [math.sin(math.pi * node.X) * math.cos(math.pi * node.Y), -math.cos(math.pi * node.X) * math.sin(math.pi * node.Y), 0.0]
            node.SetSolutionStepValue(KratosMultiphysics.VELOCITY, 0, velocity)
            node.SetSolutionStepValue(KratosMultiphysics.VELOCITY, 1, velocity)

        settings = KratosMultiphysics.Parameters("""{
            "linear_solver_settings" : {"solver_type" : "skyline_lu_factorization"}
        }""")
        process = CreateParticlePdeProjectionProcess(model_part, settings)
        process.ExecuteInitialize()
        initial_volume = process.GetIndicatorIntegral()
        self.assertGreater(process.GetNumberOfParticles(), 0)

        for _ in range(10):
            process.Execute()
            info = process.GetLastProjectionInfo()
            # Exact local and global conservation (closed domain, no boundary flux)
            self.assertLess(info["max_cell_residual"], 1.0e-12)
            self.assertAlmostEqual(info["net_boundary_outflow"], 0.0, delta=1.0e-14)
            self.assertAlmostEqual(info["integral"], info["previous_integral"], delta=1.0e-13)
            # The gradient penalty keeps the over/undershoots small
            self.assertGreater(info["min_value"], -0.1)
            self.assertLess(info["max_value"], 1.1)

        self.assertAlmostEqual(process.GetIndicatorIntegral(), initial_volume, delta=1.0e-12)

        # The nodal distance has been shifted to match the conserved phase volume
        negative_volume = KratosCFD.FluidAuxiliaryUtilities.CalculateFluidNegativeVolume(model_part)
        self.assertAlmostEqual(negative_volume, process.GetIndicatorIntegral(), delta=1.0e-8)
        for node in model_part.Nodes:
            self.assertTrue(node.Has(KratosMultiphysics.PHASE_FRACTION))

    def testDefaultSettingsValidation(self):
        model_part = self._CreateUnitSquare(2)
        wrong_settings = KratosMultiphysics.Parameters("""{
            "particles_per_element" : {"initial" : 10, "minimum" : 2, "maximum" : 20},
            "linear_solver_settings" : {"solver_type" : "skyline_lu_factorization"}
        }""")
        with self.assertRaisesRegex(RuntimeError, "At least 3 particles per element"):
            CreateParticlePdeProjectionProcess(model_part, wrong_settings)

def CreateParticlePdeProjectionProcess(model_part, settings):
    from KratosMultiphysics.FluidDynamicsApplication import particle_pde_projection_utilities
    return particle_pde_projection_utilities.CreateParticlePdeProjectionProcess(model_part, model_part.ProcessInfo[KratosMultiphysics.DOMAIN_SIZE], settings)

class _ParticleTransportSloshingAnalysis(FluidDynamicsAnalysis):
    """Sloshing tank with the particle level set transport (the two-fluids solver is created directly)."""

    def __init__(self, model, project_parameters, test_case):
        super().__init__(model, project_parameters)
        self.test_case = test_case

    def _CreateSolver(self):
        return NavierStokesTwoFluidsSolver(self.model, self.project_parameters["solver_settings"])

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
        # The distance is shifted to the conserved volume (the distance modification process then moves it slightly)
        negative_volume = KratosCFD.FluidAuxiliaryUtilities.CalculateFluidNegativeVolume(self._GetSolver().GetComputingModelPart())
        self.test_case.assertAlmostEqual(negative_volume, volume, delta=1.0e-4 * volume)
        self.test_case.volumes.append(volume)

class TestTwoFluidsParticleLevelSetTransport(KratosUnittest.TestCase):

    def testSloshingTank2D(self):
        self.volumes = []
        with KratosUnittest.WorkFolderScope("TwoFluidMassConservationSourceTest", __file__):
            with open("ProjectParameters2D.json", 'r') as parameter_file:
                parameters = KratosMultiphysics.Parameters(parameter_file.read())
            parameters["problem_data"]["end_time"].SetDouble(0.5)
            parameters["problem_data"]["echo_level"].SetInt(0)
            solver_settings = parameters["solver_settings"]
            solver_settings["echo_level"].SetInt(0)
            solver_settings["formulation"]["mass_source"].SetBool(False)
            solver_settings.AddEmptyValue("levelset_transport_type").SetString("particle_pde_projection")
            solver_settings.AddValue("particle_pde_projection_settings", KratosMultiphysics.Parameters("""{
                "particles_per_element" : {"initial" : 20, "minimum" : 12, "maximum" : 40},
                "zeta" : 20.0
            }"""))
            parameters["processes"]["json_check_process_list"] = KratosMultiphysics.Parameters("[]")

            _ParticleTransportSloshingAnalysis(KratosMultiphysics.Model(), parameters, self).Run()

        # The tank is closed: the water volume does not change
        self.assertEqual(len(self.volumes), 5)
        for volume in self.volumes:
            self.assertAlmostEqual(volume, self.volumes[0], delta=1.0e-12 * self.volumes[0])

if __name__ == '__main__':
    KratosUnittest.main()

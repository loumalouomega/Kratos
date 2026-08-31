from pathlib import Path

import numpy

import KratosMultiphysics as Kratos
import KratosMultiphysics.KratosUnittest as KratosUnittest
import KratosMultiphysics.kratos_utilities as KratosUtilities

from KratosMultiphysics.PhysicsNeMoApplication.deployment import model_registry
from KratosMultiphysics.PhysicsNeMoApplication.utilities import nvtx_utils

try:
    import torch
    have_torch = True
except ImportError:
    have_torch = False

try:
    import physicsnemo
    have_physicsnemo = True
except ImportError:
    have_physicsnemo = False


@KratosUnittest.skipUnless(have_torch, "Missing required python module: torch.")
class TestModelRegistry(KratosUnittest.TestCase):
    def setUp(self):
        self.checkpoint = Path("test_model_registry_model.pt")
        model = torch.nn.Linear(3, 2).double()
        torch.jit.script(model).save(str(self.checkpoint))

    def tearDown(self):
        KratosUtilities.DeleteFileIfExisting(str(self.checkpoint))

    def test_TorchScriptCheckpoint(self):
        settings = Kratos.Parameters("""{
            "checkpoint_file" : "test_model_registry_model.pt",
            "checkpoint_type" : "torchscript",
            "device"          : "cpu"
        }""")
        model, device = model_registry.LoadModel(settings)
        self.assertEqual(device.type, "cpu")
        with torch.no_grad():
            out = model(torch.zeros(5, 3, dtype=torch.float64))
        self.assertEqual(list(out.shape), [5, 2])

    def test_AutoDevice(self):
        settings = Kratos.Parameters("""{
            "checkpoint_file" : "test_model_registry_model.pt"
        }""")
        model, device = model_registry.LoadModel(settings)
        expected = "cuda" if torch.cuda.is_available() else "cpu"
        self.assertEqual(device.type, expected)

    def test_UnknownCheckpointTypeRaises(self):
        settings = Kratos.Parameters("""{
            "checkpoint_file" : "test_model_registry_model.pt",
            "checkpoint_type" : "pickle"
        }""")
        with self.assertRaisesRegex(ValueError, "Unsupported checkpoint type"):
            model_registry.LoadModel(settings)

    def test_TorchCompileRaisesForTorchScript(self):
        settings = Kratos.Parameters("""{
            "checkpoint_file" : "test_model_registry_model.pt",
            "checkpoint_type" : "torchscript",
            "device"          : "cpu",
            "torch_compile"   : true
        }""")
        with self.assertRaisesRegex(ValueError, "torch_compile"):
            model_registry.LoadModel(settings)

    def test_NvtxRangesSettingEnablesRanges(self):
        nvtx_utils.DisableNvtxRanges()
        self.addCleanup(nvtx_utils.DisableNvtxRanges)
        settings = Kratos.Parameters("""{
            "checkpoint_file" : "test_model_registry_model.pt",
            "device"          : "cpu",
            "nvtx_ranges"     : true
        }""")
        model_registry.LoadModel(settings)
        self.assertTrue(nvtx_utils.NvtxRangesEnabled())


@KratosUnittest.skipUnless(have_torch and have_physicsnemo,
                           "Missing required python modules: torch and physicsnemo.")
class TestTorchCompile(KratosUnittest.TestCase):
    def setUp(self):
        from physicsnemo.models.mlp.fully_connected import FullyConnected
        self.checkpoint = Path("test_model_registry_compiled.mdlus")
        torch.manual_seed(0)
        model = FullyConnected(in_features=3, out_features=2, num_layers=1, layer_size=8)
        model.save(str(self.checkpoint))

    def tearDown(self):
        KratosUtilities.DeleteFileIfExisting(str(self.checkpoint))

    def test_CompiledModelMatchesEager(self):
        settings = Kratos.Parameters("""{
            "checkpoint_file" : "test_model_registry_compiled.mdlus",
            "checkpoint_type" : "physicsnemo",
            "device"          : "cpu"
        }""")
        eager, _ = model_registry.LoadModel(settings.Clone())
        settings.AddBool("torch_compile", True)
        compiled, _ = model_registry.LoadModel(settings)

        x = torch.linspace(-1.0, 1.0, 15).reshape(5, 3)
        with torch.no_grad():
            expected = eager(x)
            actual = compiled(x)
        self.assertLess(float((expected - actual).abs().max()), 1e-6)


class TestNvtxUtils(KratosUnittest.TestCase):
    def tearDown(self):
        nvtx_utils.DisableNvtxRanges()

    def test_DisabledRangeIsNoOp(self):
        nvtx_utils.DisableNvtxRanges()
        with nvtx_utils.NvtxRange("PhysicsNeMo::Test") as r:
            self.assertFalse(r._pushed)

    @KratosUnittest.skipUnless(have_torch, "Missing required python module: torch.")
    def test_PushPopPairing(self):
        events = []
        originals = (torch.cuda.is_available, torch.cuda.nvtx.range_push, torch.cuda.nvtx.range_pop)
        torch.cuda.is_available = lambda: True
        torch.cuda.nvtx.range_push = lambda name: events.append(("push", name))
        torch.cuda.nvtx.range_pop = lambda: events.append(("pop",))
        try:
            nvtx_utils.EnableNvtxRanges()
            with nvtx_utils.NvtxRange("A"):
                with nvtx_utils.NvtxRange("B"):
                    pass
            # disabling mid-run must not unbalance an already-pushed range
            with nvtx_utils.NvtxRange("C"):
                nvtx_utils.DisableNvtxRanges()
        finally:
            torch.cuda.is_available, torch.cuda.nvtx.range_push, torch.cuda.nvtx.range_pop = originals
        self.assertEqual(events, [
            ("push", "A"), ("push", "B"), ("pop",), ("pop",), ("push", "C"), ("pop",)])


@KratosUnittest.skipUnless(have_torch, "Missing required python module: torch.")
class TestModelCards(KratosUnittest.TestCase):
    def setUp(self):
        self.checkpoint = Path("test_model_card_model.pt")
        torch.jit.script(torch.nn.Linear(3, 3).double()).save(str(self.checkpoint))

    def tearDown(self):
        KratosUtilities.DeleteFileIfExisting(str(self.checkpoint))
        KratosUtilities.DeleteFileIfExisting(str(self.checkpoint) + ".card.json")

    def test_CardRoundTrip(self):
        self.assertIsNone(model_registry.LoadModelCard(self.checkpoint))
        card = {
            "input_fields": [{"variable_name": "VELOCITY", "data_location": "node_historical"}],
            "output_fields": [{"variable_name": "ACCELERATION", "data_location": "node_historical"}],
            "notes": "trained on run 42",
        }
        model_registry.SaveModelCard(self.checkpoint, card)
        self.assertEqual(model_registry.LoadModelCard(self.checkpoint), card)

    def test_ValidateFieldsAgainstCard(self):
        card = {
            "input_fields": [{"variable_name": "VELOCITY", "data_location": "node_historical"}],
            "output_fields": [{"variable_name": "ACCELERATION", "data_location": "node_historical"}],
        }
        matching_in = [("VELOCITY", "node_historical")]
        matching_out = [("ACCELERATION", "node_historical")]
        self.assertTrue(model_registry.ValidateFieldsAgainstCard(card, matching_in, matching_out, "Test"))
        self.assertTrue(model_registry.ValidateFieldsAgainstCard(None, [], [], "Test"))
        self.assertFalse(model_registry.ValidateFieldsAgainstCard(
            card, [("PRESSURE", "node_historical")], matching_out, "Test"))
        # a card without field keys does not constrain anything
        self.assertTrue(model_registry.ValidateFieldsAgainstCard({"notes": "x"}, matching_in, [], "Test"))

    def test_MismatchedCardIsAdvisoryInInferenceProcess(self):
        from KratosMultiphysics.PhysicsNeMoApplication.processes.inference import inference_process
        # Card claims different fields than the process config: must warn but run.
        model_registry.SaveModelCard(self.checkpoint, {
            "input_fields": [{"variable_name": "PRESSURE", "data_location": "node_historical"}],
        })

        model = Kratos.Model()
        model_part = model.CreateModelPart("Main")
        model_part.AddNodalSolutionStepVariable(Kratos.VELOCITY)
        model_part.AddNodalSolutionStepVariable(Kratos.ACCELERATION)
        for i in range(3):
            node = model_part.CreateNewNode(i + 1, float(i), 0.0, 0.0)
            node.SetSolutionStepValue(Kratos.VELOCITY, [1.0, 2.0, 3.0])

        settings = Kratos.Parameters("""{
            "Parameters": {
                "model_part_name" : "Main",
                "model_settings"  : { "checkpoint_file" : "test_model_card_model.pt", "device" : "cpu" },
                "input_fields"    : [ { "variable_name" : "VELOCITY",     "data_location" : "node_historical" } ],
                "output_fields"   : [ { "variable_name" : "ACCELERATION", "data_location" : "node_historical" } ]
            }
        }""")
        process = inference_process.Factory(settings, model)
        model_part.ProcessInfo[Kratos.STEP] = 1
        process.ExecuteFinalizeSolutionStep()  # advisory: must not raise
        self.assertNotEqual(
            model_part.GetNode(1).GetSolutionStepValue(Kratos.ACCELERATION)[0], 0.0)

    def test_StrictPolicyRaisesOnMismatch(self):
        card = {
            "input_fields": [{"variable_name": "VELOCITY", "data_location": "node_historical"}],
        }
        mismatched = [("PRESSURE", "node_historical")]
        with self.assertRaisesRegex(RuntimeError, "strict"):
            model_registry.ValidateFieldsAgainstCard(card, mismatched, [], "Test", policy="strict")
        # matching specs pass under strict
        self.assertTrue(model_registry.ValidateFieldsAgainstCard(
            card, [("VELOCITY", "node_historical")], [], "Test", policy="strict"))

    def test_IgnorePolicyIsSilent(self):
        card = {
            "input_fields": [{"variable_name": "VELOCITY", "data_location": "node_historical"}],
        }
        self.assertFalse(model_registry.ValidateFieldsAgainstCard(
            card, [("PRESSURE", "node_historical")], [], "Test", policy="ignore"))

    def test_UnknownPolicyRaises(self):
        with self.assertRaisesRegex(ValueError, "model card policy"):
            model_registry.ValidateFieldsAgainstCard(None, [], [], "Test", policy="loose")

    def test_LoadModelWithCardCheckStrict(self):
        model_registry.SaveModelCard(self.checkpoint, {
            "input_fields": [{"variable_name": "PRESSURE", "data_location": "node_historical"}],
        })
        settings = Kratos.Parameters("""{
            "checkpoint_file"   : "test_model_card_model.pt",
            "device"            : "cpu",
            "model_card_policy" : "strict"
        }""")
        with self.assertRaisesRegex(RuntimeError, "strict"):
            model_registry.LoadModelWithCardCheck(
                settings, [("VELOCITY", "node_historical")], [], "Test")

        settings["model_card_policy"].SetString("ignore")
        model, device = model_registry.LoadModelWithCardCheck(
            settings, [("VELOCITY", "node_historical")], [], "Test")
        self.assertIsNotNone(model)


class TestOutputNormalization(KratosUnittest.TestCase):
    """The card-carried inverse of a training normalization.

    Pure numpy - no torch, no checkpoint - so it runs everywhere including
    the torch-free CI.
    """

    def test_MeanStdInverse(self):
        normalization = {"type": "mean_std", "mean": [1.0, 10.0], "std": [2.0, 4.0]}
        raw = numpy.array([[0.0, 1.0], [-1.0, 0.5]])
        numpy.testing.assert_allclose(
            model_registry.ApplyOutputNormalization(raw, normalization),
            [[1.0, 14.0], [-1.0, 12.0]], rtol=1e-12)

    def test_MinMaxInverseForBothRanges(self):
        unit = {"type": "min_max", "min": [0.0], "max": [10.0]}
        numpy.testing.assert_allclose(
            model_registry.ApplyOutputNormalization(
                numpy.array([[0.0], [0.5], [1.0]]), unit).ravel(),
            [0.0, 5.0, 10.0], rtol=1e-12)
        # DoMINO's convention
        symmetric = {"type": "min_max", "min": [0.0], "max": [10.0],
                     "range": [-1.0, 1.0]}
        numpy.testing.assert_allclose(
            model_registry.ApplyOutputNormalization(
                numpy.array([[-1.0], [0.0], [1.0]]), symmetric).ravel(),
            [0.0, 5.0, 10.0], rtol=1e-12)

    def test_AScalarBroadcastsOverChannels(self):
        normalization = {"type": "mean_std", "mean": [1.0], "std": [2.0]}
        numpy.testing.assert_allclose(
            model_registry.ApplyOutputNormalization(numpy.ones((2, 3)), normalization),
            numpy.full((2, 3), 3.0), rtol=1e-12)

    def test_SpreadsAreScaledButNotShifted(self):
        # THE trap. A standard deviation shifted by the training mean is
        # meaningless; a shared write hook must distinguish the two.
        normalization = {"type": "mean_std", "mean": [100.0], "std": [2.0]}
        std = numpy.array([[1.0], [3.0]])
        numpy.testing.assert_allclose(
            model_registry.ApplyOutputNormalization(std, normalization, scale_only=True),
            [[2.0], [6.0]], rtol=1e-12)
        # and the mean path does shift, so the two really differ
        numpy.testing.assert_allclose(
            model_registry.ApplyOutputNormalization(std, normalization),
            [[102.0], [106.0]], rtol=1e-12)

    def test_NoNormalizationIsTheIdentityObject(self):
        # the property that keeps every pre-existing configuration and test
        # untouched: nothing is copied, cast or computed
        raw = numpy.array([[1.0, 2.0]])
        self.assertIs(model_registry.ApplyOutputNormalization(raw, None), raw)

    def test_DegenerateScaleIsRejected(self):
        for bad in ({"type": "mean_std", "mean": [0.0], "std": [0.0]},
                    {"type": "min_max", "min": [1.0], "max": [1.0]}):
            with self.subTest(normalization=bad["type"]):
                with self.assertRaisesRegex(ValueError, "zero scale"):
                    model_registry.ApplyOutputNormalization(numpy.ones((2, 1)), bad)

    def test_WrongChannelCountIsRejected(self):
        normalization = {"type": "mean_std", "mean": [0.0, 0.0, 0.0],
                         "std": [1.0, 1.0, 1.0]}
        with self.assertRaisesRegex(ValueError, "does not belong to this model"):
            model_registry.ApplyOutputNormalization(numpy.ones((4, 2)), normalization)


class TestLoadOutputNormalization(KratosUnittest.TestCase):
    def setUp(self):
        self.checkpoint = Path("test_normalization_card.pt")
        self.settings = Kratos.Parameters("""{
            "checkpoint_file"   : "",
            "model_card_policy" : "advisory"
        }""")
        self.settings["checkpoint_file"].SetString(str(self.checkpoint))

    def tearDown(self):
        KratosUtilities.DeleteFileIfExisting(str(self.checkpoint) + ".card.json")

    def test_NoCardOrNoKeyGivesNone(self):
        self.assertIsNone(model_registry.LoadOutputNormalization(self.settings))
        model_registry.SaveModelCard(self.checkpoint, {"notes": "no normalization here"})
        self.assertIsNone(model_registry.LoadOutputNormalization(self.settings))
        model_registry.SaveModelCard(self.checkpoint, {"output_normalization": {"type": "none"}})
        self.assertIsNone(model_registry.LoadOutputNormalization(self.settings))

    def test_ReadRegardlessOfIgnorePolicy(self):
        # "ignore" means do not validate the FIELD LISTS. Dropping the
        # de-normalization there would reintroduce the bug this prevents.
        model_registry.SaveModelCard(self.checkpoint, {
            "output_normalization": {"type": "mean_std", "mean": [1.0], "std": [2.0]}})
        self.settings["model_card_policy"].SetString("ignore")
        normalization = model_registry.LoadOutputNormalization(self.settings)
        self.assertIsNotNone(normalization)
        self.assertEqual(normalization["type"], "mean_std")

    def test_MalformedEntriesAreRejected(self):
        model_registry.SaveModelCard(self.checkpoint, {
            "output_normalization": {"type": "voxel_scaling"}})
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            model_registry.LoadOutputNormalization(self.settings)
        model_registry.SaveModelCard(self.checkpoint, {
            "output_normalization": {"type": "mean_std", "mean": [1.0]}})
        with self.assertRaisesRegex(ValueError, "std"):
            model_registry.LoadOutputNormalization(self.settings)


if __name__ == '__main__':
    KratosUnittest.main()

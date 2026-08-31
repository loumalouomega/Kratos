"""Process deploying point-cloud models on a model part's nodes.

Point-cloud transformers (physicsnemo.models.transolver.Transolver and
friends) consume per-point features with coordinates, batched as
(1, N, C) - no tessellation, graph or grid required. This process gathers
the nodal input fields plus the (optionally normalized) node coordinates,
runs one forward pass, and writes the (1, N, C_out) prediction back through
the same field-splitting contract as InferenceProcess (whose settings it
extends).

Model interfaces:
- "generic": model(x) with x = (1, N, 3 + C_in), coordinates prepended to
  the features - anything from an MLP to a scripted trunk.
- "transolver": model(fx, embedding) with fx = (1, N, C_in) functional
  features and embedding = (1, N, 3) coordinates, matching
  Transolver.forward.
- "flare": alias of the "transolver" call contract - matching
  physicsnemo.experimental.models.flare.FLARE.forward(fx, embedding)
  (verified signature-compatible). Listed separately for discoverability
  and model-card clarity; FLARE lives in physicsnemo.experimental (no API
  stability guarantee).
- "geotransolver": model(local_embedding, local_positions=..., geometry=...)
  with local_embedding = (1, N, C_in) features, local_positions = (1, N, 3)
  coordinates and geometry = (1, N, 3) coordinates (or None when the model
  was built with geometry_dim=None - set "pass_geometry" to false then),
  matching physicsnemo.experimental.models.geotransolver.GeoTransolver.
  Construct with use_te=False unless transformer_engine is installed.
  Experimental namespace - no API stability guarantee.
- "figconvnet": model(vertices, features) with vertices = (1, N, 3) and
  features = (1, N, C_in), matching FIGConvUNet.forward, which returns a
  TUPLE (point features (1, N, C_out), scalar (1, 1) or None). The point
  features are written back as usual; the scalar (a drag-style global
  output) is stashed as ``last_scalar_prediction`` and logged. Notes:
  FIGConvUNet's warp backend is float32-only (the parameter-dtype cast
  covers stock models), construct it with ``has_input_features=True`` and
  ``in_channels`` equal to the total gathered input width, and its default
  aabb of (0,0,0)-(1,1,1) matches ``normalize_coordinates=True``.

DoMINO remains served through the CAE datapipes (see CaeDatasetExportProcess
and torch_dataset.CreateDoMINODataPipe).

torch is imported lazily on first execution.
"""

import numpy

import KratosMultiphysics as Kratos
from KratosMultiphysics.PhysicsNeMoApplication import torch_bridge
from KratosMultiphysics.PhysicsNeMoApplication.inference_process import InferenceProcess
from KratosMultiphysics.PhysicsNeMoApplication.utilities.nvtx_utils import NvtxRange

_MODEL_INTERFACES = ("generic", "transolver", "flare", "geotransolver", "figconvnet")


def GatherPointCloudCoordinates(model_part, normalize: bool = True,
                                local_only: bool = False):
    """Returns the (N, 3) current node coordinates as a numpy array,
    optionally min-max normalized to [0, 1] per axis (degenerate axes
    left at 0).

    With local_only the owned nodes only (the communicator's LocalMesh) are
    read. Note that normalize then uses each rank's *own* bounding box, so a
    distributed caller wanting rank-independent coordinates must either pass
    normalize=False or normalize against a globally reduced box itself.
    """
    nodes = model_part.GetCommunicator().LocalMesh().Nodes if local_only else model_part.Nodes
    position_ta = Kratos.TensorAdaptors.NodePositionTensorAdaptor(
        nodes, Kratos.Configuration.Current)
    position_ta.CollectData()
    coordinates = numpy.array(position_ta.data, dtype=numpy.float64)  # (N, 3)
    if normalize:
        low = coordinates.min(axis=0)
        extent = coordinates.max(axis=0) - low
        extent[extent == 0.0] = 1.0  # planar/linear clouds: leave that axis at 0
        coordinates = (coordinates - low) / extent
    return coordinates


def RunPointCloudForward(model, device, model_interface, features, coordinates,
                         pass_geometry: bool = True, enable_grad: bool = False):
    """One point-cloud forward pass through a model interface.

    Shared by PointCloudInferenceProcess and the CoSimulation surrogate
    wrapper. features/coordinates are (N, C_in)/(N, 3) torch tensors; both
    are cast to the model's parameter dtype before the call. Runs under
    no_grad by default; enable_grad=True keeps the autograd graph (for
    sensitivity computations - see sensitivity_utils).

    Returns:
        (prediction, scalar): the (N, C_out) float64 prediction and the
        figconvnet-style global scalar (None for the other interfaces).
    """
    torch = torch_bridge._TryImportTorch()

    parameter = next(model.parameters(), None)
    dtype = parameter.dtype if parameter is not None else features.dtype
    features = features.to(dtype)
    coordinates = coordinates.to(dtype)

    scalar_prediction = None
    grad_context = torch.enable_grad() if enable_grad else torch.no_grad()
    with grad_context, NvtxRange("PhysicsNeMo::Forward"):
        if model_interface in ("transolver", "flare"):
            prediction = model(
                features[None].to(device),
                coordinates[None].to(device))
        elif model_interface == "geotransolver":
            geometry = coordinates[None].to(device) if pass_geometry else None
            prediction = model(
                features[None].to(device),
                local_positions=coordinates[None].to(device),
                geometry=geometry)
        elif model_interface == "figconvnet":
            prediction, scalar = model(
                coordinates[None].to(device),
                features[None].to(device))
            scalar_prediction = (
                float(scalar.reshape(-1)[0]) if scalar is not None else None)
        elif model_interface == "generic":  # coordinates prepended to the features
            prediction = model(
                torch.cat([coordinates, features], dim=-1)[None].to(device))
        else:
            raise ValueError(
                f"Unsupported model interface \"{model_interface}\". "
                f"Supported: {', '.join(_MODEL_INTERFACES)}.")
        prediction = prediction.cpu()

    if prediction.ndim != 3 or prediction.shape[0] != 1:
        raise ValueError(
            f"The model must return a (1, N, C_out) prediction; got shape "
            f"{list(prediction.shape)}.")
    return prediction[0].to(torch.float64), scalar_prediction


def Factory(settings: Kratos.Parameters, model: Kratos.Model) -> "PointCloudInferenceProcess":
    if not isinstance(settings, Kratos.Parameters):
        raise TypeError("Expected input shall be a Parameters object, encapsulating a json string")
    return PointCloudInferenceProcess(model, settings["Parameters"])


class PointCloudInferenceProcess(InferenceProcess):
    """Runs point-cloud model inference each output_interval steps."""

    def __init__(self, model: Kratos.Model, settings: Kratos.Parameters) -> None:
        # Split the subclass keys off (with their defaults) before the parent
        # validates the shared InferenceProcess settings.
        self.model_interface = "generic"
        self.normalize_coordinates = True
        self.pass_geometry = True  # geotransolver only: forward coordinates as geometry
        if settings.Has("model_interface"):
            self.model_interface = settings["model_interface"].GetString()
            settings.RemoveValue("model_interface")
        if settings.Has("normalize_coordinates"):
            self.normalize_coordinates = settings["normalize_coordinates"].GetBool()
            settings.RemoveValue("normalize_coordinates")
        if settings.Has("pass_geometry"):
            self.pass_geometry = settings["pass_geometry"].GetBool()
            settings.RemoveValue("pass_geometry")
        self.last_scalar_prediction = None  # figconvnet-style global output
        super().__init__(model, settings)

        if self.model_interface not in _MODEL_INTERFACES:
            raise ValueError(
                f"Unsupported model interface \"{self.model_interface}\". "
                f"Supported: {', '.join(_MODEL_INTERFACES)}.")

    def RunInference(self) -> None:
        torch = torch_bridge._TryImportTorch()

        with NvtxRange("PhysicsNeMo::GatherInputs"):
            inputs, n_entities = self._GatherInputs()
            features = torch.cat(inputs, dim=-1)  # (N, C_in)
            coordinates = torch.from_numpy(GatherPointCloudCoordinates(
                self.model_part, self.normalize_coordinates))  # (N, 3)
        self._CheckOOD(features)

        def forward(model):
            prediction, scalar = RunPointCloudForward(
                model, self._device, self.model_interface, features, coordinates,
                self.pass_geometry)
            if self.model_interface == "figconvnet":
                self.last_scalar_prediction = scalar
                if scalar is not None:
                    Kratos.Logger.PrintInfo(
                        type(self).__name__,
                        f"figconvnet scalar prediction: {scalar:.6e}")
            return prediction

        prediction, std = self._PredictWithUncertainty(forward)
        with NvtxRange("PhysicsNeMo::WriteOutputs"):
            self._WriteOutputs(prediction, n_entities)
            self._WriteUncertainty(std, n_entities)

"""Model checkpoint loading and device placement.

Single source of truth for how this application loads trained models, so
every process and strategy resolves checkpoints and devices identically.

torch and physicsnemo are optional runtime dependencies, imported lazily
inside LoadModel only.
"""

import numpy

import KratosMultiphysics as Kratos


def _TryImportTorch():
    try:
        import torch
        return torch
    except ImportError as e:
        raise ImportError(
            "PhysicsNeMoApplication.model_registry requires torch, which could not be "
            "imported. Install it with e.g. 'pip install torch'.") from e


def _TryImportPhysicsNemo():
    try:
        import physicsnemo
        return physicsnemo
    except ImportError as e:
        raise ImportError(
            "Loading a \"physicsnemo\" checkpoint requires physicsnemo, which could not be "
            "imported. Install it with e.g. 'pip install nvidia-physicsnemo'.") from e


def ResolveDevice(device_name: str):
    """Resolves a device string ("auto"/"cpu"/"cuda"[:i]) to a torch.device."""
    torch = _TryImportTorch()
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device_name)


def LoadModel(settings: Kratos.Parameters):
    """Loads a trained model from a checkpoint and places it on a device.

    Settings:
        checkpoint_file: Path to the checkpoint.
        checkpoint_type: "torchscript" (torch.jit.load, default) or
            "physicsnemo" (physicsnemo.Module.from_checkpoint, .mdlus files).
        device: "auto" (default; cuda when available, else cpu), "cpu" or
            "cuda"[:index].
        model_card_policy: "advisory" (default), "strict" or "ignore" - how
            deployment processes treat model-card mismatches (see
            LoadModelWithCardCheck; LoadModel itself never reads the card).
        torch_compile: false (default). When true, the loaded model is
            wrapped with torch.compile(fullgraph=True). Only "physicsnemo"
            checkpoints are compilable - TorchScript modules do not compose
            with dynamo, so "torchscript" + torch_compile raises.
        nvtx_ranges: false (default). When true, enables the NVTX ranges
            around the deployment hot paths (see utilities.nvtx_utils);
            they emit only when CUDA is available.

    Returns:
        (model, device): The model in eval mode on the device, and the
        torch.device it was placed on.
    """
    default_settings = Kratos.Parameters("""{
        "checkpoint_file"   : "PLEASE_SPECIFY_CHECKPOINT_FILE",
        "checkpoint_type"   : "torchscript",
        "device"            : "auto",
        "model_card_policy" : "advisory",
        "torch_compile"     : false,
        "nvtx_ranges"       : false
    }""")
    settings.ValidateAndAssignDefaults(default_settings)

    torch = _TryImportTorch()
    device = ResolveDevice(settings["device"].GetString())

    if settings["nvtx_ranges"].GetBool():
        from KratosMultiphysics.PhysicsNeMoApplication.utilities import nvtx_utils
        nvtx_utils.EnableNvtxRanges()

    checkpoint_file = settings["checkpoint_file"].GetString()
    checkpoint_type = settings["checkpoint_type"].GetString()
    if checkpoint_type == "torchscript":
        model = torch.jit.load(checkpoint_file, map_location=device)
    elif checkpoint_type == "physicsnemo":
        physicsnemo = _TryImportPhysicsNemo()
        model = physicsnemo.Module.from_checkpoint(checkpoint_file)
    else:
        raise ValueError(
            f"Unsupported checkpoint type \"{checkpoint_type}\". "
            "Use \"torchscript\" or \"physicsnemo\".")

    model = model.eval().to(device)
    if settings["torch_compile"].GetBool():
        if checkpoint_type == "torchscript":
            raise ValueError(
                "\"torch_compile\" is not supported for \"torchscript\" checkpoints: "
                "TorchScript modules do not compose with torch.compile. Save the model "
                "as a physicsnemo checkpoint (.mdlus) to compile it.")
        model = torch.compile(model, fullgraph=True)
    return model, device


def _CardPath(checkpoint_file):
    import pathlib
    return pathlib.Path(str(checkpoint_file) + ".card.json")


def SaveModelCard(checkpoint_file, card: dict) -> None:
    """Writes a model card sidecar ("<checkpoint_file>.card.json").

    The card is a free-form JSON dict describing what the checkpoint was
    trained for. Recommended keys, checked by the deployment processes:
    "input_fields" / "output_fields" as [{"variable_name", "data_location"}],
    and "output_normalization" (see LoadOutputNormalization) when the model
    was trained on normalized targets. Anything else ("grid_shape",
    "history_size", training provenance, ...) travels along untouched.

    This function never synthesizes keys - the card written is exactly the
    dict given. Anything auto-derived belongs in whatever builds that dict.
    """
    import json
    with open(_CardPath(checkpoint_file), "w") as f:
        json.dump(card, f, indent=4)


def LoadModelCard(checkpoint_file):
    """Returns the model card dict for a checkpoint, or None if there is none."""
    import json
    path = _CardPath(checkpoint_file)
    if not path.is_file():
        return None
    with open(path) as f:
        return json.load(f)


def _FieldList(card_entry):
    return [(entry.get("variable_name"), entry.get("data_location")) for entry in card_entry]


_MODEL_CARD_POLICIES = ("advisory", "strict", "ignore")


def ValidateFieldsAgainstCard(card, input_specs, output_specs, tag: str,
                              policy: str = "advisory") -> bool:
    """Compares configured field specs with a model card's.

    Policies:
        "advisory" (default): a mismatch produces one detailed
            KRATOS_WARNING (the card may be stale, or the user may
            deliberately remap fields) and never raises.
        "strict": a mismatch raises RuntimeError with the same message.
        "ignore": mismatches are silent.

    Returns True when everything matches (or the card doesn't constrain the
    fields), False when a mismatch was found (and the policy let it pass).

    Args:
        card: The dict from LoadModelCard (None is accepted and passes).
        input_specs/output_specs: [(variable_name, data_location)] as
            configured in the deployment process.
        tag: Logger tag of the calling process.
        policy: "advisory", "strict" or "ignore".
    """
    if policy not in _MODEL_CARD_POLICIES:
        raise ValueError(
            f"Unsupported model card policy \"{policy}\". Use one of {_MODEL_CARD_POLICIES}.")
    if card is None:
        return True
    ok = True
    for key, configured in (("input_fields", list(input_specs)), ("output_fields", list(output_specs))):
        if key not in card:
            continue
        expected = _FieldList(card[key])
        if expected != configured:
            message = (
                f"Configured {key} {configured} do not match the model card's {expected}. "
                "The model may have been trained for different fields")
            if policy == "strict":
                raise RuntimeError(
                    f"{tag}: {message}; the model card policy is \"strict\", so execution stops.")
            if policy == "advisory":
                Kratos.Logger.PrintWarning(
                    tag, f"{message}; the card is advisory, so execution continues.")
            ok = False
    return ok


_NORMALIZATION_TYPES = ("none", "mean_std", "min_max")


def LoadOutputNormalization(model_settings: Kratos.Parameters, checkpoint_file=None):
    """The card's "output_normalization" entry, or None when there is none.

    A model trained on normalized targets emits normalized predictions, and
    writing those onto Kratos variables as if they were physical is wrong
    by whatever the scaling was. The card is where that scaling travels
    with the checkpoint.

    Schema:
        {"type": "mean_std", "mean": [...], "std":  [...]}
        {"type": "min_max",  "min":  [...], "max":  [...],
                             "range": [0.0, 1.0]}     // optional
        {"type": "none"}

    Each array is length 1 (broadcast over all channels) or length
    total_width (per channel, in the concatenated output_fields order).
    "range" is the interval the training normalization mapped onto -
    [0, 1] by default, though DoMINO's convention is [-1, 1].

    Deliberately read regardless of "model_card_policy": "ignore" means
    "do not validate the field lists", and silently dropping the
    de-normalization would reintroduce exactly the bug this exists to
    prevent. Configurations written before this key existed are unaffected,
    since a card without it yields None.

    Returns:
        The entry dict, or None (the identity path).
    """
    if checkpoint_file is None:
        # Not every process keys its card off "checkpoint_file": ONNX uses
        # "onnx_file", Triton "card_file", and an ensemble names its members
        # in "checkpoint_files". Those pass their own path; a settings block
        # with none of them simply has no card.
        if model_settings.Has("checkpoint_file"):
            checkpoint_file = model_settings["checkpoint_file"].GetString()
        elif model_settings.Has("checkpoint_files"):
            members = model_settings["checkpoint_files"].GetStringArray()
            checkpoint_file = members[0] if members else None
        if not checkpoint_file:
            return None

    card = LoadModelCard(checkpoint_file)
    if not card:
        return None
    normalization = card.get("output_normalization")
    if not normalization:
        return None

    kind = normalization.get("type", "none")
    if kind not in _NORMALIZATION_TYPES:
        raise ValueError(
            f"Unsupported \"output_normalization\" type \"{kind}\" in the model card. "
            f"Supported: {', '.join(_NORMALIZATION_TYPES)}.")
    if kind == "none":
        return None

    required = ("mean", "std") if kind == "mean_std" else ("min", "max")
    for key in required:
        if key not in normalization:
            raise ValueError(
                f"\"output_normalization\" of type \"{kind}\" needs \"{key}\".")
    return normalization


def ApplyOutputNormalization(prediction, normalization, scale_only: bool = False):
    """Inverts a training normalization on a model's prediction.

    Args:
        prediction: (n_entities, total_width) array-like. Returned
            unchanged - the same object - when normalization is None, so
            the identity path costs nothing and preserves dtype.
        normalization: A LoadOutputNormalization entry.
        scale_only: Apply the scale but NOT the offset. Required for a
            standard deviation or any other spread: shifting a spread by
            the mean is meaningless, and doing so is the mistake a single
            shared hook most easily makes.

    Returns:
        The de-normalized prediction, with the same type as the input - a
        torch tensor in, a torch tensor out. WriteOutputFields hands the
        result straight to torch_bridge, which needs a tensor; returning
        numpy there raised AttributeError on .detach(). Returns the
        untouched input object when there is nothing to do.
    """
    if normalization is None:
        return prediction

    is_tensor = hasattr(prediction, "detach") and hasattr(prediction, "cpu")
    source = prediction.detach().cpu().numpy() if is_tensor else prediction
    values = numpy.asarray(source, dtype=numpy.float64)
    n_channels = values.shape[-1] if values.ndim > 1 else 1

    def _Vector(name):
        vector = numpy.asarray(normalization[name], dtype=numpy.float64).reshape(-1)
        if vector.size not in (1, n_channels):
            raise ValueError(
                f"\"output_normalization\" has {vector.size} entries for \"{name}\" but "
                f"the prediction has {n_channels} channels; the card does not belong to "
                "this model.")
        return vector

    if normalization["type"] == "mean_std":
        scale, offset = _Vector("std"), _Vector("mean")
    else:
        low, high = _Vector("min"), _Vector("max")
        interval = normalization.get("range", [0.0, 1.0])
        span = float(interval[1]) - float(interval[0])
        if span == 0.0:
            raise ValueError("\"output_normalization\" range must not be degenerate.")
        scale = (high - low) / span
        offset = low - float(interval[0]) * scale

    if numpy.any(scale == 0.0):
        raise ValueError(
            "\"output_normalization\" has a zero scale, which would make the inverse "
            "undefined (a constant channel needs a scale of 1, not 0).")

    values = values * scale
    if not scale_only:
        values = values + offset

    if is_tensor:
        import torch
        return torch.as_tensor(values, dtype=prediction.dtype)
    return values


def LoadModelWithCardCheck(model_settings: Kratos.Parameters, input_specs, output_specs, tag: str):
    """LoadModel plus model-card validation - the single entry point every
    deployment process uses at (lazy) model-load time.

    The policy comes from model_settings["model_card_policy"] ("advisory"
    by default, "strict" to refuse mismatched deployments, "ignore" to
    silence the check).

    Returns:
        (model, device), as LoadModel.
    """
    model, device = LoadModel(model_settings)
    policy = model_settings["model_card_policy"].GetString()
    if policy != "ignore":
        card = LoadModelCard(model_settings["checkpoint_file"].GetString())
        ValidateFieldsAgainstCard(card, input_specs, output_specs, tag, policy)
    return model, device

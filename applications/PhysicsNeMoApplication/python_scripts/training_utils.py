"""Parameters-driven training loop and checkpoint saving.

Removes the boilerplate every surrogate needs: TrainModel runs a standard
supervised loop over any (inputs, targets) Dataset (CreateNpzDataset output,
a TensorDataset, ...) configured through Kratos Parameters, and
SaveTrainedModel writes the checkpoint in one of the two formats
model_registry.LoadModel reads (physicsnemo .mdlus or TorchScript),
optionally with a model card sidecar.

torch is imported lazily; module import stays ML-free.
"""

import KratosMultiphysics as Kratos
from KratosMultiphysics.PhysicsNeMoApplication import model_registry


def _TryImportTorch():
    try:
        import torch
        return torch
    except ImportError as e:
        raise ImportError(
            "PhysicsNeMoApplication.training_utils requires torch, which could not be "
            "imported. Install it with e.g. 'pip install torch'.") from e


def _GuardCalibrationRows(inputs):
    """A training batch as (rows, channels) guard features.

    Must mirror what the deployment processes feed GuardCheck.Check, or the
    calibrated guard scores a different quantity than it later checks:

    * nodal batches (B, N, C) - channels LAST - are checked as (N, C), so
      calibration rows are the points: (B*N, C);
    * grid batches (B, C, *spatial) - channels FIRST, ndim >= 4 - are
      checked as (prod(spatial), C) by the grid processes, so the channel
      axis is moved last before flattening;
    * plain (B, C) batches are their own rows.

    Grids used to be pooled over everything but the LAST axis, which for a
    (B, C, D, H, W) batch is the W axis - a guard calibrated on that flags
    every deployment input, in-distribution ones included.
    """
    if inputs.ndim >= 4:
        return inputs.movedim(1, -1).reshape(-1, inputs.shape[1])
    return inputs.reshape(-1, inputs.shape[-1])


def TrainModel(model, dataset, settings: Kratos.Parameters, epoch_callbacks=None,
               extra_loss_terms=None):
    """Trains a model on an (inputs, targets) dataset.

    Args:
        model: A torch.nn.Module (physicsnemo Modules included).
        dataset: A torch.utils.data.Dataset yielding (inputs, targets).
        settings: Kratos Parameters; defaults:
            epochs (100), batch_size (32), learning_rate (1e-3),
            optimizer ("adam"|"sgd"), loss ("mse"|"l1"),
            device ("auto"|"cpu"|"cuda"), shuffle (true),
            echo_interval (0 = silent, N = log every N epochs),
            seed (-1 = leave the RNG alone),
            concrete_dropout_reg_weight (0.0 = off; > 0 adds that multiple
                of the summed physicsnemo.nn.ConcreteDropout regularization
                losses to the objective, making the dropout rates learnable),
            ood_guard ({} = off; {"guard_file": "...", "buffer_size": 0 =
                len(dataset), "knn_k", "sensitivity"} calibrates an OOD
                guard on the training inputs during the first epoch and
                saves it to guard_file - the sidecar the deployment
                processes' "ood_guard" blocks load; see ood_guard_utils).
        epoch_callbacks: Optional iterable of callables
            cb(epoch, model, history) invoked after every epoch (plain
            Python argument - callables do not serialize into Parameters).
            The canonical physics-informed monitor: run the model on a
            held-out case, write the prediction into the case's model part,
            and log solver_residuals.ResidualEvaluator.ComputeResidualNorm()
            - the real PDE residual of the current surrogate. The plain
            ResidualEvaluator is assembled outside the autodiff graph
            (logging/ranking/early stopping only); for a gradient-carrying
            EXACT-residual loss term, use
            differentiable_residual.MakeExactResidualLossTerm via
            extra_loss_terms instead.
        extra_loss_terms: Optional iterable of callables
            term(model, inputs, prediction) -> scalar tensor, ADDED to the
            data loss every batch (gradient-carrying, unlike the plain
            evaluator above). Canonical sources:
            physics_informed.MakePhysicsLossTerm (analytic strong-form
            residuals via physicsnemo.sym) and
            differentiable_residual.MakeExactResidualLossTerm (the exact
            discrete residual through the real FEM assembly).

    Returns:
        list[float]: mean training loss per epoch. The model ends up on the
        resolved device, in eval mode.
    """
    torch = _TryImportTorch()

    default_settings = Kratos.Parameters("""{
        "epochs"                      : 100,
        "batch_size"                  : 32,
        "learning_rate"               : 1e-3,
        "optimizer"                   : "adam",
        "loss"                        : "mse",
        "device"                      : "auto",
        "shuffle"                     : true,
        "echo_interval"               : 0,
        "seed"                        : -1,
        "concrete_dropout_reg_weight" : 0.0,
        "streaming"                   : false,
        "warm_restart"                : {},
        "ood_guard"                   : {}
    }""")
    settings.ValidateAndAssignDefaults(default_settings)

    # optional concrete-dropout regularization (learnable dropout rates)
    concrete_reg_weight = settings["concrete_dropout_reg_weight"].GetDouble()

    # optional OOD-guard calibration: collect the training inputs and save
    # the guard sidecar for the deployment processes' "ood_guard" blocks
    guard_settings = settings["ood_guard"]
    guard_settings.ValidateAndAssignDefaults(Kratos.Parameters("""{
        "guard_file"  : "",
        "buffer_size" : 0,
        "knn_k"       : 10,
        "sensitivity" : 1.5
    }"""))
    guard_file = guard_settings["guard_file"].GetString()
    guard = None

    seed = settings["seed"].GetInt()
    if seed >= 0:
        torch.manual_seed(seed)

    device = model_registry.ResolveDevice(settings["device"].GetString())
    model = model.to(device)

    # Re-initialization goes here deliberately: after the device move so the
    # noise is drawn on the right device, and BEFORE the optimizer exists so
    # its moment estimates are never stale with respect to perturbed weights.
    warm_restart = settings["warm_restart"]
    if warm_restart.Has("shrink") or warm_restart.Has("perturb"):
        _ApplyWarmRestart(model, warm_restart, seed)

    optimizer_name = settings["optimizer"].GetString()
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"].GetDouble())
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=settings["learning_rate"].GetDouble())
    else:
        raise ValueError(f"Unsupported optimizer \"{optimizer_name}\". Use \"adam\" or \"sgd\".")

    loss_name = settings["loss"].GetString()
    if loss_name == "mse":
        loss_fn = torch.nn.functional.mse_loss
    elif loss_name == "l1":
        loss_fn = torch.nn.functional.l1_loss
    else:
        raise ValueError(f"Unsupported loss \"{loss_name}\". Use \"mse\" or \"l1\".")

    streaming = settings["streaming"].GetBool()
    if streaming:
        # A live stream is single-pass and has no length: shuffling needs a
        # sampler (impossible), and a second epoch would drain an exhausted
        # queue and record a spurious zero-loss epoch.
        if settings["shuffle"].GetBool():
            raise ValueError(
                "\"shuffle\" is not possible with \"streaming\": an iterable dataset "
                "has no sampler. Set \"shuffle\": false (samples arrive in solver order).")
        if settings["epochs"].GetInt() != 1:
            raise ValueError(
                f"\"streaming\" requires \"epochs\": 1, got "
                f"{settings['epochs'].GetInt()}: the stream is consumed once, so later "
                "epochs would see an empty queue and log a false zero loss.")
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=None)   # the dataset already emits whole batches
    else:
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=settings["batch_size"].GetInt(),
            shuffle=settings["shuffle"].GetBool())
    echo_interval = settings["echo_interval"].GetInt()

    history = []
    model.train()
    for epoch in range(settings["epochs"].GetInt()):
        epoch_loss = 0.0
        batches = 0
        for inputs, targets in loader:
            optimizer.zero_grad()
            inputs = inputs.to(device)
            prediction = model(inputs)
            loss = loss_fn(prediction, targets.to(device))
            if extra_loss_terms:
                for loss_term in extra_loss_terms:
                    loss = loss + loss_term(model, inputs, prediction)
            if concrete_reg_weight > 0.0:
                from KratosMultiphysics.PhysicsNeMoApplication import uncertainty_utils
                loss = loss + concrete_reg_weight * uncertainty_utils.CollectConcreteDropoutLosses(model)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batches += 1
            if guard_file and epoch == 0:  # calibrate on the first pass over the data
                from KratosMultiphysics.PhysicsNeMoApplication import ood_guard_utils
                with torch.no_grad():
                    # the guard is a CPU-resident deployment artifact (the
                    # inference processes score CPU features against it), so
                    # feed it CPU tensors regardless of the training device -
                    # collect() otherwise fails on CUDA with a device mismatch
                    rows = _GuardCalibrationRows(
                        inputs.detach().to("cpu", torch.float32))
                if guard is None:
                    feature_width = int(rows.shape[-1])
                    rows_per_sample = max(1, rows.shape[0] // max(1, inputs.shape[0]))
                    buffer_size = guard_settings["buffer_size"].GetInt()
                    if buffer_size <= 0:
                        if streaming:
                            raise ValueError(
                                "The OOD guard needs an explicit \"buffer_size\" when "
                                "\"streaming\" is set: a live stream has no len().")
                        buffer_size = min(4096, len(dataset) * rows_per_sample)
                    guard = ood_guard_utils.CreateOODGuard(
                        buffer_size, feature_width,
                        guard_settings["knn_k"].GetInt(),
                        guard_settings["sensitivity"].GetDouble())
                    guard_quota = max(1, -(-buffer_size * int(inputs.shape[0]) //
                                           max(1, len(dataset))))
                with torch.no_grad():
                    if rows.shape[0] > guard_quota:  # spread the buffer over the epoch
                        rows = rows[torch.randperm(rows.shape[0])[:guard_quota]]
                    sample_latents = rows.mean(dim=0, keepdim=True)
                    guard.collect(rows, sample_latents)
        history.append(epoch_loss / max(batches, 1))
        if echo_interval > 0 and (epoch + 1) % echo_interval == 0:
            Kratos.Logger.PrintInfo(
                "TrainModel", f"epoch {epoch + 1}/{settings['epochs'].GetInt()}: loss = {history[-1]:.6e}")
        if epoch_callbacks:
            model.eval()
            with torch.no_grad():
                for callback in epoch_callbacks:
                    callback(epoch, model, history)
            model.train()
    model.eval()
    if guard is not None:
        from KratosMultiphysics.PhysicsNeMoApplication import ood_guard_utils
        guard.compute_threshold()
        ood_guard_utils.SaveGuard(guard, guard_file)
        Kratos.Logger.PrintInfo("TrainModel", f"Saved OOD guard to \"{guard_file}\".")
    return history


def _GatherShardedStateDict(model):
    """Full, unsharded state dict for an FSDP2 model - or None if not sharded.

    FSDP2 replaces parameters with DTensors. Serializing those directly
    produces a checkpoint that reports success and then cannot be loaded
    ("aten.copy_.default got mixed torch.Tensor and DTensor"), and across
    ranks each writes only its own shard. Gathering first is the fix.

    Two details that matter:
    - a forward whose backward never ran leaves the parameters unsharded as
      plain Parameters, so reshard() must run before inspecting them, or a
      rank can silently contribute full tensors while another contributes
      shards;
    - full_state_dict=True does not by itself materialize every DTensor, so
      any survivor gets an explicit full_tensor().
    """
    torch = _TryImportTorch()
    try:
        from torch.distributed.tensor import DTensor
        from torch.distributed.checkpoint.state_dict import (
            get_model_state_dict, StateDictOptions)
    except ImportError:
        return None

    if hasattr(model, "reshard"):
        model.reshard()
    if not any(isinstance(parameter, DTensor) for parameter in model.parameters()):
        return None

    state_dict = get_model_state_dict(
        model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
    return {name: (value.full_tensor() if isinstance(value, DTensor) else value)
            for name, value in state_dict.items()}


def SaveTrainedModel(model, checkpoint_file, card=None) -> str:
    """Saves a model in a format model_registry.LoadModel can read.

    physicsnemo Modules save natively (.mdlus, checkpoint_type
    "physicsnemo"); anything else is scripted to TorchScript
    (checkpoint_type "torchscript"). When a card dict is given, the model
    card sidecar is written alongside (see model_registry.SaveModelCard).

    FSDP2-sharded models are gathered to full tensors first and written by
    rank 0 only; saving their DTensors directly yields a checkpoint that
    reports success and cannot be loaded. The gather is collective, so
    every rank must call this.

    Returns:
        The checkpoint_type string to use when loading.
    """
    torch = _TryImportTorch()
    checkpoint_file = str(checkpoint_file)

    checkpoint_type = None
    try:
        import physicsnemo
        if isinstance(model, physicsnemo.Module):
            if not checkpoint_file.endswith(".mdlus"):
                raise ValueError(
                    f"physicsnemo modules must be saved with a \".mdlus\" extension "
                    f"(got \"{checkpoint_file}\").")
            full_state_dict = _GatherShardedStateDict(model)
            if full_state_dict is None:
                model.save(checkpoint_file)
            else:
                # sharded: every rank must reach the gather above (it is
                # collective), but only rank 0 writes, and it writes the
                # gathered full tensors rather than its own DTensors
                import torch.distributed as distributed
                is_writer = (not distributed.is_initialized()
                             or distributed.get_rank() == 0)
                if is_writer:
                    model.save(checkpoint_file, _state_dict=full_state_dict)
                if distributed.is_initialized():
                    distributed.barrier()
            checkpoint_type = "physicsnemo"
    except ImportError:
        pass

    if checkpoint_type is None:
        if _GatherShardedStateDict(model) is not None:
            raise RuntimeError(
                f"Cannot save \"{checkpoint_file}\": the model has sharded (DTensor) "
                "parameters but is not a physicsnemo Module, and TorchScript cannot "
                "represent them. Unshard it first, or make it a physicsnemo Module.")
        try:
            torch.jit.script(model).save(checkpoint_file)
        except Exception as e:
            raise RuntimeError(
                f"Could not save the model to \"{checkpoint_file}\": it is not a physicsnemo "
                "Module and TorchScript scripting failed. Make the model scriptable or save "
                f"it manually. Original error: {e}") from e
        checkpoint_type = "torchscript"

    if card is not None:
        model_registry.SaveModelCard(checkpoint_file, card)
    return checkpoint_type


def ExportOnnxModel(model, sample_inputs, onnx_file, card=None) -> str:
    """Exports a trained model to an .onnx file for OnnxInferenceProcess.

    The export runs physicsnemo.deploy.onnx.export_to_onnx_stream on the
    model with the given sample inputs (a torch tensor or tuple of tensors
    with the deployment-time shapes, e.g. one gathered input batch), and
    writes the resulting byte stream. When a card dict is given, the model
    card sidecar ("<onnx_file>.card.json") is written alongside - the same
    sidecar format every deployment process validates.

    Note: some operators (e.g. the FFTs inside FNO-style models) are not
    supported by the CPU ONNX Runtime; MLP/conv models export and run
    everywhere.

    Returns:
        The onnx_file path as a string.
    """
    onnx_file = str(onnx_file)
    if not onnx_file.endswith(".onnx"):
        raise ValueError(
            f"ONNX models must be exported with a \".onnx\" extension (got \"{onnx_file}\").")

    from KratosMultiphysics.PhysicsNeMoApplication import onnx_bridge
    export_to_onnx_stream = onnx_bridge._TryImportOnnxExport()

    try:
        stream = export_to_onnx_stream(model, sample_inputs)
    except ModuleNotFoundError as e:
        if "onnxscript" in str(e):
            raise ImportError(
                "torch's ONNX exporter additionally requires onnxscript, which could not "
                "be imported. Install it with e.g. 'pip install onnxscript'.") from e
        raise
    with open(onnx_file, "wb") as f:
        f.write(stream)

    if card is not None:
        model_registry.SaveModelCard(onnx_file, card)
    return onnx_file


def _ApplyWarmRestart(model, settings: Kratos.Parameters, seed: int):
    """Shrink-and-perturb re-initialization before a warm restart.

    Re-seeding a trained model part-way toward its initialization keeps what
    it learned while restoring plasticity - the fix for a surrogate that has
    to absorb Kratos data from a new geometry family without forgetting the
    old one, and without the pathologies of training a converged network on
    a shifted distribution.

    Settings:
        {
            "shrink"  : 0.5,   // theta <- shrink * theta + perturb * noise
            "perturb" : 0.1,
            "noise"   : "scaled_normal",   // or "normal"
            "include_all_parameters" : false
        }

    Note:
        `perturb` is RELATIVE under the default "scaled_normal" (the noise is
        scaled by each tensor's own standard deviation) and ABSOLUTE under
        "normal", where 0.1 is enormous for typical weights.

        By default only float parameters with more than one dimension are
        touched. Upstream applies itself to everything, which crashes on
        integer parameters and halves LayerNorm/BatchNorm gains toward zero;
        "include_all_parameters" restores that behaviour if you want it.
    """
    torch = _TryImportTorch()
    try:
        from physicsnemo.nn import shrink_and_perturb_
    except ImportError as e:
        raise ImportError(
            "TrainModel's \"warm_restart\" requires physicsnemo >= 2.2 "
            "(shrink_and_perturb_ landed in 2.2), which could not be imported. Install "
            "it with e.g. 'pip install -U nvidia-physicsnemo'.") from e

    defaults = Kratos.Parameters("""{
        "shrink"                 : 0.5,
        "perturb"                : 0.1,
        "noise"                  : "scaled_normal",
        "include_all_parameters" : false
    }""")
    settings = settings.Clone()
    settings.ValidateAndAssignDefaults(defaults)

    shrink = settings["shrink"].GetDouble()
    perturb = settings["perturb"].GetDouble()
    noise = settings["noise"].GetString()
    if noise not in ("scaled_normal", "normal"):
        raise ValueError(
            f"Unsupported warm-restart noise \"{noise}\". Use \"scaled_normal\" or \"normal\".")
    if shrink < 0.0 or perturb < 0.0:
        raise ValueError(
            f"\"shrink\" and \"perturb\" must be >= 0 [ shrink = {shrink}, perturb = {perturb} ].")
    if shrink > 1.0:
        # upstream accepts this and silently amplifies every weight
        raise ValueError(
            f"\"shrink\" must be <= 1 [ shrink = {shrink} ]: values above 1 amplify the "
            "weights instead of shrinking them toward initialization.")

    generator = None
    if seed >= 0:
        generator = torch.Generator(device=next(model.parameters()).device)
        generator.manual_seed(seed)

    include = None
    if not settings["include_all_parameters"].GetBool():
        include = lambda name, parameter: (parameter.is_floating_point()
                                           and parameter.dim() > 1)
    return shrink_and_perturb_(model, shrink, perturb, noise=noise,
                               include=include, generator=generator)

"""Training and sampling helpers for conditional diffusion field models.

Bridges physicsnemo's diffusion stack (CorrDiff-style downscaling) to the
grid data this application exports:

- TrainDiffusionModel: an EDM loss loop over (condition, target) grid pairs
  (CreateGridPairDataset output) for a preconditioned denoiser such as
  physicsnemo.diffusion.preconditioners.EDMPrecondSuperResolution wrapping a
  SongUNet - a physicsnemo Module, so SaveTrainedModel writes a regular
  .mdlus checkpoint.
- GenerateEnsemble: repeated reverse-diffusion sampling conditioned on one
  grid, returning an (S, C_out, *spatial) ensemble whose mean is the
  prediction and whose spread is a calibrated uncertainty field.
- The CorrDiff TWO-STAGE recipe: TrainDiffusionModel's "regression" loss
  trains a deterministic mean stage (CorrDiffRegressionUNet), "residual"
  trains the denoiser on target - regression(condition) with the frozen
  stage-1 model (physicsnemo's RegressionLoss/ResidualLoss);
  TrainCorrDiffPair runs both, and RunRegressionMean +
  DiffusionInferenceProcess's "regression_settings" add the mean back at
  inference (mean shifts, ensemble spread untouched).

The same machinery covers the documented variations: TopoDiff-style
generative design (condition = constraint masks) and flow reconstruction
from sparse data (condition = masked observations) differ only in what the
condition channels contain.

torch and physicsnemo are optional runtime dependencies, imported lazily.
"""

import numpy

import KratosMultiphysics as Kratos
from KratosMultiphysics.PhysicsNeMoApplication import model_registry


def _TryImportTorch():
    try:
        import torch
        return torch
    except ImportError as e:
        raise ImportError(
            "PhysicsNeMoApplication.diffusion_utils requires torch, which could not be "
            "imported. Install it with e.g. 'pip install torch'.") from e


def _TryImportPhysicsNemoDiffusion():
    try:
        from physicsnemo.diffusion.metrics.legacy_losses import (
            EDMLossSR, RegressionLoss, ResidualLoss)
        from physicsnemo.diffusion.samplers import deterministic_sampler
        return EDMLossSR, RegressionLoss, ResidualLoss, deterministic_sampler
    except ImportError as e:
        raise ImportError(
            "PhysicsNeMoApplication.diffusion_utils requires physicsnemo, which could not "
            "be imported. Install it with e.g. 'pip install nvidia-physicsnemo'.") from e


def WrapDenoiser(model, interface: str = "dit", out_channels: int = 0):
    """Adapts a denoiser to the EDM sampler/loss contract net(x, img_lr, sigma).

    The shipped samplers and EDMLossSR drive super-resolution-style
    denoisers as ``net(x, img_lr, sigma)``. physicsnemo.models.dit.DiT
    speaks ``dit(x, t, condition=None)`` instead; the "dit" wrapper maps
    one onto the other by concatenating the conditioning grid into the
    input channels (construct the DiT with in_channels = C_out + C_cond and
    out_channels = C_out) and broadcasting sigma to the per-sample timestep
    tensor. The wrapped module exposes img_out_channels, so GenerateEnsemble
    and DiffusionInferenceProcess work unchanged; note the raw DiT acts as
    the denoiser D(x, sigma) directly (no EDM pre/post-scaling), so train it
    through the same wrapper (TrainDiffusionModel accepts it as-is).

    RoPE, invalid-region masking and alternative attention backends are DiT
    construction choices (block_kwargs/attn_kwargs/attention_backend) - the
    wrapper only standardizes the forward call.

    Args:
        model: The denoiser to wrap (a DiT for interface "dit").
        interface: Only "dit" currently.
        out_channels: The number of predicted channels; 0 reads the DiT's
            out_channels attribute.

    Returns:
        A torch.nn.Module with the net(x, img_lr, sigma) interface.
    """
    torch = _TryImportTorch()
    if interface != "dit":
        raise ValueError(
            f"Unsupported denoiser interface \"{interface}\". Only \"dit\" needs wrapping - "
            "EDM-preconditioned denoisers already speak net(x, img_lr, sigma).")

    resolved_out_channels = out_channels or getattr(model, "out_channels", 0)
    if not resolved_out_channels:
        raise ValueError(
            "\"out_channels\" is 0 and the model exposes no out_channels; set it explicitly.")

    class DitDenoiser(torch.nn.Module):
        # EDM interface attributes the samplers read from the net
        sigma_min = 0.0
        sigma_max = float("inf")

        def __init__(self, dit):
            super().__init__()
            self.dit = dit
            self.img_out_channels = resolved_out_channels

        @staticmethod
        def round_sigma(sigma):
            return torch.as_tensor(sigma)

        def forward(self, x, img_lr, sigma, class_labels=None, **kwargs):
            # the samplers run their loop in float64; cast to the DiT's
            # parameter dtype at the boundary and back
            parameter = next(self.dit.parameters(), None)
            dtype = parameter.dtype if parameter is not None else x.dtype
            t = torch.atleast_1d(torch.as_tensor(sigma, device=x.device, dtype=dtype))
            if t.numel() == 1:
                t = t.expand(x.shape[0])
            denoised = self.dit(
                torch.cat([x, img_lr], dim=1).to(dtype), t.reshape(x.shape[0]))
            return denoised.to(x.dtype)

    return DitDenoiser(model)


def TrainDiffusionModel(model, dataset, settings: Kratos.Parameters, regression_model=None):
    """Trains a conditional diffusion (or CorrDiff-stage) model on
    (condition, target) pairs.

    Args:
        model: The trainable model. For "edm_sr" and "residual": a
            preconditioned denoiser with the super-resolution interface
            net(x, img_lr, sigma) - e.g. EDMPrecondSuperResolution wrapping
            a SongUNet. For "regression": a deterministic mean predictor
            with the (x_zeros, img_lr) interface -
            physicsnemo.models.diffusion_unets.CorrDiffRegressionUNet.
            2D spatial layout (C, H, W): planar Kratos cases use the
            thin-axis idiom (CreateGridPairDataset squeeze_axis).
        dataset: A torch Dataset yielding (condition_grid, target_grid)
            float pairs, e.g. CreateGridPairDataset output.
        settings: Kratos Parameters; defaults:
            epochs (100), batch_size (8), learning_rate (1e-4),
            device ("auto"), shuffle (true), echo_interval (0), seed (-1),
            loss ("edm_sr" | "regression" | "residual"),
            P_mean (-1.2 for edm_sr, 0.0 for residual - upstream's
            defaults; explicit values always win), P_std (1.2),
            sigma_data (0.5).
        regression_model: REQUIRED for loss "residual": the trained
            CorrDiff regression stage. It is moved to the device, frozen
            (.eval() + requires_grad_(False) - upstream's ResidualLoss
            does NOT no_grad it) and used to compute the residual targets
            target - regression(condition) the denoiser learns. NOTE:
            ResidualLoss passes embedding_selector/global_index kwargs to
            the denoiser, so the "residual" model must wrap
            SongUNetPosEmbd (model_type="SongUNetPosEmbd"; its
            N_grid_channels, default 4, count toward img_in_channels -
            upstream CorrDiff's own sizing).

    Returns:
        list[float]: mean training loss per epoch. The model ends up on the
        resolved device, in eval mode.
    """
    torch = _TryImportTorch()
    EDMLossSR, RegressionLoss, ResidualLoss, _ = _TryImportPhysicsNemoDiffusion()

    user_set_p_mean = settings.Has("P_mean")
    default_settings = Kratos.Parameters("""{
        "epochs"        : 100,
        "batch_size"    : 8,
        "learning_rate" : 1e-4,
        "device"        : "auto",
        "shuffle"       : true,
        "echo_interval" : 0,
        "seed"          : -1,
        "loss"          : "edm_sr",
        "P_mean"        : -1.2,
        "P_std"         : 1.2,
        "sigma_data"    : 0.5
    }""")
    settings.ValidateAndAssignDefaults(default_settings)
    loss_name = settings["loss"].GetString()
    if loss_name not in ("edm_sr", "regression", "residual"):
        raise ValueError(
            f"Unsupported diffusion loss \"{loss_name}\". Use \"edm_sr\" (conditional "
            "EDM), \"regression\" (CorrDiff mean stage) or \"residual\" (CorrDiff "
            "denoiser stage, needs regression_model).")

    seed = settings["seed"].GetInt()
    if seed >= 0:
        torch.manual_seed(seed)

    device = model_registry.ResolveDevice(settings["device"].GetString())
    model = model.to(device)
    parameter = next(model.parameters(), None)
    dtype = parameter.dtype if parameter is not None else torch.float32

    # upstream's per-loss P_mean defaults differ (edm_sr -1.2, residual 0.0)
    p_mean = settings["P_mean"].GetDouble()
    if not user_set_p_mean and loss_name == "residual":
        p_mean = 0.0

    if loss_name == "edm_sr":
        loss_fn = EDMLossSR(
            P_mean=p_mean,
            P_std=settings["P_std"].GetDouble(),
            sigma_data=settings["sigma_data"].GetDouble())
    elif loss_name == "regression":
        loss_fn = RegressionLoss()
    else:  # residual
        if regression_model is None:
            raise ValueError(
                "Loss \"residual\" needs the trained CorrDiff regression stage via "
                "the regression_model argument.")
        regression_model = regression_model.to(device).eval()
        regression_model.requires_grad_(False)  # upstream does not no_grad it
        loss_fn = ResidualLoss(
            regression_net=regression_model,
            P_mean=p_mean,
            P_std=settings["P_std"].GetDouble(),
            sigma_data=settings["sigma_data"].GetDouble())
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"].GetDouble())
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
        for conditions, targets in loader:
            optimizer.zero_grad()
            loss = loss_fn(
                net=model,
                img_clean=targets.to(device, dtype),
                img_lr=conditions.to(device, dtype)).mean()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batches += 1
        history.append(epoch_loss / max(batches, 1))
        if echo_interval > 0 and (epoch + 1) % echo_interval == 0:
            Kratos.Logger.PrintInfo(
                "TrainDiffusionModel",
                f"epoch {epoch + 1}/{settings['epochs'].GetInt()}: loss = {history[-1]:.6e}")
    model.eval()
    return history


def TrainCorrDiffPair(regression_model, diffusion_model, dataset, settings: Kratos.Parameters):
    """The CorrDiff two-stage recipe in one call.

    Stage 1 trains the deterministic regression mean (loss "regression",
    "regression_epochs"/"regression_learning_rate" overriding the shared
    values when given); stage 2 trains the denoiser on the RESIDUAL
    target - regression(condition) with the frozen stage-1 model (loss
    "residual"). Inference then adds the regression mean back - see
    RunRegressionMean and DiffusionInferenceProcess's "regression_settings".

    Args:
        regression_model: e.g. CorrDiffRegressionUNet (trained in place).
        diffusion_model: the EDM-preconditioned denoiser (trained in place).
        dataset: (condition, target) pairs shared by both stages.
        settings: TrainDiffusionModel settings for the DIFFUSION stage,
            plus optional "regression_epochs" and
            "regression_learning_rate".

    Returns:
        (regression_history, diffusion_history)
    """
    settings = settings.Clone()
    regression_epochs = None
    if settings.Has("regression_epochs"):
        regression_epochs = settings["regression_epochs"].GetInt()
        settings.RemoveValue("regression_epochs")
    regression_learning_rate = None
    if settings.Has("regression_learning_rate"):
        regression_learning_rate = settings["regression_learning_rate"].GetDouble()
        settings.RemoveValue("regression_learning_rate")

    def with_overrides(loss, epochs=None, learning_rate=None):
        stage = settings.Clone()
        for key, value in (("loss", loss), ("epochs", epochs), ("learning_rate", learning_rate)):
            if value is None:
                continue
            if stage.Has(key):
                stage.RemoveValue(key)
            entry = stage.AddEmptyValue(key)
            if isinstance(value, str):
                entry.SetString(value)
            elif isinstance(value, int):
                entry.SetInt(value)
            else:
                entry.SetDouble(value)
        return stage

    regression_history = TrainDiffusionModel(
        regression_model, dataset,
        with_overrides("regression", regression_epochs, regression_learning_rate))
    diffusion_history = TrainDiffusionModel(
        diffusion_model, dataset, with_overrides("residual"),
        regression_model=regression_model)
    return regression_history, diffusion_history


def RunRegressionMean(model, condition_grid, output_channels: int = 0):
    """The CorrDiff regression stage's mean prediction for one condition.

    Args:
        model: The trained regression model ((x_zeros, img_lr) interface,
            e.g. CorrDiffRegressionUNet).
        condition_grid: (C_in, *spatial) array-like condition.
        output_channels: 0 reads the model's img_out_channels attribute.

    Returns:
        (C_out, *spatial) float64 numpy array.
    """
    torch = _TryImportTorch()

    if output_channels == 0:
        output_channels = getattr(model, "img_out_channels", 0)
        if not output_channels:
            raise ValueError(
                "output_channels is 0 and the model exposes no img_out_channels; "
                "set it explicitly.")

    parameter = next(model.parameters(), None)
    device = parameter.device if parameter is not None else torch.device("cpu")
    dtype = parameter.dtype if parameter is not None else torch.float32

    condition = torch.as_tensor(numpy.asarray(condition_grid)).to(device, dtype)[None]
    zeros = torch.zeros((1, output_channels) + tuple(condition.shape[2:]),
                        device=device, dtype=dtype)
    with torch.no_grad():
        mean = model(zeros, condition)
    return mean[0].cpu().to(torch.float64).numpy()


def GenerateEnsemble(model, condition_grid, settings: Kratos.Parameters):
    """Samples an ensemble of fields conditioned on one grid.

    Args:
        model: The trained preconditioned denoiser (net(x, img_lr, sigma)).
        condition_grid: (C_in, *spatial) array-like condition (e.g. the
            coarse field sampled by grid_bridge, thin axis squeezed).
        settings: Kratos Parameters; defaults:
            num_samples (8), num_steps (18), solver ("heun"),
            output_channels (0 = read the model's img_out_channels),
            seed (-1 = leave the RNG alone).

    Returns:
        (num_samples, C_out, *spatial) float64 numpy array.
    """
    torch = _TryImportTorch()
    _, _, _, deterministic_sampler = _TryImportPhysicsNemoDiffusion()

    default_settings = Kratos.Parameters("""{
        "num_samples"     : 8,
        "num_steps"       : 18,
        "solver"          : "heun",
        "output_channels" : 0,
        "seed"            : -1
    }""")
    settings.ValidateAndAssignDefaults(default_settings)

    num_samples = settings["num_samples"].GetInt()
    if num_samples < 1:
        raise ValueError(f"\"num_samples\" must be >= 1 [ num_samples = {num_samples} ].")
    output_channels = settings["output_channels"].GetInt()
    if output_channels == 0:
        output_channels = getattr(model, "img_out_channels", 0)
        if not output_channels:
            raise ValueError(
                "\"output_channels\" is 0 and the model exposes no img_out_channels; "
                "set it explicitly.")

    seed = settings["seed"].GetInt()
    if seed >= 0:
        torch.manual_seed(seed)

    parameter = next(model.parameters(), None)
    device = parameter.device if parameter is not None else torch.device("cpu")
    dtype = parameter.dtype if parameter is not None else torch.float32

    condition = torch.as_tensor(numpy.asarray(condition_grid)).to(device, dtype)[None]
    spatial = tuple(condition.shape[2:])

    samples = []
    with torch.no_grad():
        for _ in range(num_samples):
            latents = torch.randn((1, output_channels) + spatial, device=device, dtype=dtype)
            sample = deterministic_sampler(
                net=model,
                latents=latents,
                img_lr=condition,
                num_steps=settings["num_steps"].GetInt(),
                solver=settings["solver"].GetString())
            samples.append(sample[0].cpu().to(torch.float64).numpy())
    return numpy.stack(samples)

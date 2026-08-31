"""Process validating ML-predicted fields against reference Kratos fields.

Per configured interval, computes physicsnemo.metrics.general metrics between
a predicted and a reference field, logs them, and writes the accumulated
history as a JSON report in ExecuteFinalize. This provides model-vs-solver
benchmarking with core physicsnemo only.

physicsnemo/torch are imported lazily at first evaluation.
"""

import json
from pathlib import Path

import KratosMultiphysics as Kratos
from KratosMultiphysics.PhysicsNeMoApplication import torch_bridge
from KratosMultiphysics.PhysicsNeMoApplication.utilities.tensor_adaptor_dataset_utils import GetTensorAdaptor

SUPPORTED_METRICS = ("mse", "rmse", "max_abs_error", "wasserstein",
                     "relative_l2", "weighted_mse", "weighted_rmse")
WEIGHTED_METRICS = ("weighted_mse", "weighted_rmse")
ENSEMBLE_METRICS = ("crps", "kcrps")
CALIBRATION_METRICS = ("coverage", "nll", "sharpness", "calibration_error")


def _TryImportPhysicsNemoMetrics():
    try:
        from physicsnemo.metrics.general import mse as mse_module
        from physicsnemo.metrics.general import wasserstein as wasserstein_module
        return mse_module, wasserstein_module
    except ImportError as e:
        raise ImportError(
            "PhysicsNeMoApplication.validation_metrics_process requires physicsnemo, which "
            "could not be imported. Install it with e.g. 'pip install nvidia-physicsnemo'.") from e


def _TryImportPhysicsNemoCrps():
    try:
        from physicsnemo.metrics.general import crps as crps_module
        return crps_module
    except ImportError as e:
        raise ImportError(
            "PhysicsNeMoApplication.validation_metrics_process requires physicsnemo, which "
            "could not be imported. Install it with e.g. 'pip install nvidia-physicsnemo'.") from e


def ComputeMetricValues(predicted, reference, metric_names, weights=None) -> dict:
    """Computes SUPPORTED_METRICS values between two same-shape torch tensors.

    The single metric implementation shared by ValidationMetricsProcess and
    the active-learning metrology strategy. Returns {metric_name: float}.

    "relative_l2" is ||predicted - reference|| / ||reference|| over the
    flattened fields (implemented locally - physicsnemo.metrics has no
    relative-L2). The "weighted_*" metrics need a weights tensor whose first
    dimension matches the fields' (e.g. nodal areas): the squared error is
    averaged over each entity's components and weighted-averaged over
    entities.
    """
    mse_module, wasserstein_module = _TryImportPhysicsNemoMetrics()
    torch = torch_bridge._TryImportTorch()

    if predicted.shape != reference.shape:
        raise ValueError(
            f"Shape mismatch between predicted {list(predicted.shape)} and reference "
            f"{list(reference.shape)}.")

    weighted_error2 = None
    if any(metric in WEIGHTED_METRICS for metric in metric_names):
        if weights is None:
            raise ValueError(
                f"Metrics {WEIGHTED_METRICS} need a weights tensor (e.g. nodal areas); "
                "none was given.")
        weights = weights.reshape(-1)
        if weights.shape[0] != predicted.shape[0]:
            raise ValueError(
                f"weights has {weights.shape[0]} entries but the fields have "
                f"{predicted.shape[0]} entities.")
        entity_error2 = (predicted - reference).reshape(predicted.shape[0], -1).square().mean(dim=1)
        weighted_error2 = float((weights * entity_error2).sum() / weights.sum())

    values = {}
    for metric in metric_names:
        if metric == "mse":
            values[metric] = float(mse_module.mse(predicted, reference))
        elif metric == "rmse":
            values[metric] = float(mse_module.rmse(predicted, reference))
        elif metric == "max_abs_error":
            values[metric] = float(torch.max(torch.abs(predicted - reference)))
        elif metric == "wasserstein":
            values[metric] = float(wasserstein_module.wasserstein_from_samples(
                predicted.reshape(-1), reference.reshape(-1)))
        elif metric == "relative_l2":
            values[metric] = float(
                torch.linalg.vector_norm((predicted - reference).reshape(-1))
                / torch.linalg.vector_norm(reference.reshape(-1)))
        elif metric == "weighted_mse":
            values[metric] = weighted_error2
        elif metric == "weighted_rmse":
            values[metric] = weighted_error2 ** 0.5
        else:
            raise ValueError(
                f"Unsupported metric \"{metric}\". Supported: {', '.join(SUPPORTED_METRICS)}.")
    return values


def ComputeCalibrationMetricValues(mean, std, reference, metric_names,
                                   confidence_z: float = 1.96) -> dict:
    """Computes CALIBRATION_METRICS for a predicted (mean, std) against truth.

    These answer the question an error metric cannot: is the model's stated
    uncertainty honest? A surrogate can have an excellent RMSE and still be
    badly calibrated, which is what makes its error bars unusable for
    decisions.

    - "coverage": fraction of references inside mean +/- confidence_z * std.
      For a well-calibrated Gaussian at z = 1.96 this is ~0.95.
    - "calibration_error": |coverage - nominal|, the signed miss turned
      absolute (0 is perfect; large means over- or under-confident).
    - "nll": mean Gaussian negative log-likelihood, which unlike coverage
      punishes both over- and under-confidence continuously.
    - "sharpness": mean predicted std. Meaningful only next to coverage -
      a model can be arbitrarily sharp by being wrong, or trivially
      well-covered by being vague.

    Returns {metric_name: float}.
    """
    torch = torch_bridge._TryImportTorch()

    if tuple(mean.shape) != tuple(reference.shape) or tuple(std.shape) != tuple(reference.shape):
        raise ValueError(
            f"Shape mismatch between mean {list(mean.shape)}, std {list(std.shape)} and "
            f"reference {list(reference.shape)}.")
    if float(std.min()) < 0.0:
        raise ValueError("std carries negative entries; it must be a standard deviation.")

    safe_std = std.clamp_min(1e-12)
    nominal = float(torch.special.erf(torch.tensor(
        confidence_z / (2.0 ** 0.5), dtype=torch.float64)))

    values = {}
    for metric in metric_names:
        if metric == "coverage":
            inside = ((reference >= mean - confidence_z * std)
                      & (reference <= mean + confidence_z * std))
            values[metric] = float(inside.to(torch.float64).mean())
        elif metric == "calibration_error":
            inside = ((reference >= mean - confidence_z * std)
                      & (reference <= mean + confidence_z * std))
            values[metric] = float(abs(float(inside.to(torch.float64).mean()) - nominal))
        elif metric == "nll":
            residual = (reference - mean) / safe_std
            values[metric] = float((0.5 * residual.square()
                                    + safe_std.log()
                                    + 0.5 * float(torch.log(torch.tensor(
                                        2.0 * torch.pi, dtype=torch.float64)))).mean())
        elif metric == "sharpness":
            values[metric] = float(std.mean())
        else:
            raise ValueError(
                f"Unsupported calibration metric \"{metric}\". "
                f"Supported: {', '.join(CALIBRATION_METRICS)}.")
    return values


def ComputeEnsembleMetricValues(ensemble, reference, metric_names) -> dict:
    """Computes ENSEMBLE_METRICS between an (M, ...) prediction ensemble and
    a (...)-shaped reference, via physicsnemo.metrics.general.crps.

    "crps" is the kernel CRPS estimator, "kcrps" the (biased) kernel form;
    both are averaged over all field entries. Returns {metric_name: float}.
    """
    crps_module = _TryImportPhysicsNemoCrps()

    if ensemble.ndim < 2 or ensemble.shape[0] < 2:
        raise ValueError(
            f"ensemble must be (M >= 2, ...); got shape {list(ensemble.shape)}.")
    if tuple(ensemble.shape[1:]) != tuple(reference.shape):
        raise ValueError(
            f"Shape mismatch between ensemble members {list(ensemble.shape[1:])} and "
            f"reference {list(reference.shape)}.")

    values = {}
    for metric in metric_names:
        if metric == "crps":
            values[metric] = float(crps_module.crps(ensemble, reference, dim=0).mean())
        elif metric == "kcrps":
            values[metric] = float(crps_module.kcrps(ensemble, reference, dim=0).mean())
        else:
            raise ValueError(
                f"Unsupported ensemble metric \"{metric}\". "
                f"Supported: {', '.join(ENSEMBLE_METRICS)}.")
    return values


def Factory(settings: Kratos.Parameters, model: Kratos.Model) -> "ValidationMetricsProcess":
    if not isinstance(settings, Kratos.Parameters):
        raise TypeError("Expected input shall be a Parameters object, encapsulating a json string")
    return ValidationMetricsProcess(model, settings["Parameters"])


class ValidationMetricsProcess(Kratos.Process):
    """Evaluates configured metrics every output_interval steps."""

    def __init__(self, model: Kratos.Model, settings: Kratos.Parameters) -> None:
        super().__init__()

        default_settings = Kratos.Parameters("""{
            "model_part_name"     : "PLEASE_SPECIFY_MODEL_PART_NAME",
            "list_of_comparisons" : [
                {
                    "predicted_variable" : "PLEASE_SPECIFY_VARIABLE_NAME",
                    "predicted_location" : "node_historical",
                    "reference_variable" : "PLEASE_SPECIFY_VARIABLE_NAME",
                    "reference_location" : "node_historical",
                    "weight_variable"    : "",
                    "weight_location"    : "node_non_historical",
                    "metrics"            : ["mse", "rmse", "max_abs_error"]
                }
            ],
            "cfd_metrics"         : [],
            "uncertainty_comparisons" : [],
            "ensemble_comparisons"    : [],
            "output_interval"     : 1,
            "output_file"         : "validation_metrics.json"
        }""")
        settings.ValidateAndAssignDefaults(default_settings)
        for i in range(settings["list_of_comparisons"].size()):
            settings["list_of_comparisons"][i].ValidateAndAssignDefaults(default_settings["list_of_comparisons"][0])

        self.model_part = model[settings["model_part_name"].GetString()]
        self.comparisons = []
        for i in range(settings["list_of_comparisons"].size()):
            comparison = settings["list_of_comparisons"][i]
            metrics = comparison["metrics"].GetStringArray()
            for metric in metrics:
                if metric not in SUPPORTED_METRICS:
                    raise ValueError(
                        f"Unsupported metric \"{metric}\". Supported: {', '.join(SUPPORTED_METRICS)}.")
            weight_variable = comparison["weight_variable"].GetString()
            if any(metric in WEIGHTED_METRICS for metric in metrics) and not weight_variable:
                raise ValueError(
                    f"Metrics {WEIGHTED_METRICS} need a \"weight_variable\" "
                    "(e.g. NODAL_AREA) in the comparison.")
            self.comparisons.append({
                "predicted": (comparison["predicted_variable"].GetString(),
                              comparison["predicted_location"].GetString()),
                "reference": (comparison["reference_variable"].GetString(),
                              comparison["reference_location"].GetString()),
                "weight": ((weight_variable, comparison["weight_location"].GetString())
                           if weight_variable else None),
                "metrics": metrics,
            })
        # Optional physicsnemo-cfd registry metrics: each entry names a
        # registered metric plus the Kratos fields feeding its SEMANTIC keys
        # ("pressure", "velocity", ...). The "fields" sub-block is free-form
        # (keyed by those semantic names), so it is validated manually.
        self.cfd_metric_specs = []
        for i in range(settings["cfd_metrics"].size()):
            entry = settings["cfd_metrics"][i]
            field_pairs = {}
            if entry.Has("fields"):
                pair_defaults = Kratos.Parameters("""{
                    "predicted_variable" : "PLEASE_SPECIFY_VARIABLE_NAME",
                    "predicted_location" : "node_historical",
                    "reference_variable" : "PLEASE_SPECIFY_VARIABLE_NAME",
                    "reference_location" : "node_historical"
                }""")
                for semantic_name in entry["fields"].keys():
                    pair = entry["fields"][semantic_name]
                    pair.ValidateAndAssignDefaults(pair_defaults)
                    field_pairs[semantic_name] = (
                        (pair["predicted_variable"].GetString(), pair["predicted_location"].GetString()),
                        (pair["reference_variable"].GetString(), pair["reference_location"].GetString()))
                entry.RemoveValue("fields")
            entry.ValidateAndAssignDefaults(Kratos.Parameters("""{
                "name"   : "PLEASE_SPECIFY_METRIC_NAME",
                "domain" : "surface"
            }"""))
            if not field_pairs:
                raise ValueError(
                    f"cfd_metrics entry \"{entry['name'].GetString()}\" needs a \"fields\" block "
                    "mapping the metric's semantic keys to Kratos variables.")
            self.cfd_metric_specs.append(
                (entry["name"].GetString(), entry["domain"].GetString(), field_pairs))

        uncertainty_comparison_defaults = Kratos.Parameters("""{
            "mean_variable"      : "PLEASE_SPECIFY_VARIABLE_NAME",
            "mean_location"      : "node_historical",
            "std_variable"       : "PLEASE_SPECIFY_VARIABLE_NAME",
            "std_location"       : "node_non_historical",
            "reference_variable" : "PLEASE_SPECIFY_VARIABLE_NAME",
            "reference_location" : "node_historical",
            "confidence_z"       : 1.96,
            "metrics"            : ["coverage", "nll"]
        }""")
        self.uncertainty_comparisons = []
        for i in range(settings["uncertainty_comparisons"].size()):
            entry = settings["uncertainty_comparisons"][i]
            entry.ValidateAndAssignDefaults(uncertainty_comparison_defaults)
            metrics = entry["metrics"].GetStringArray()
            for metric in metrics:
                if metric not in CALIBRATION_METRICS:
                    raise ValueError(
                        f"Unsupported calibration metric \"{metric}\". "
                        f"Supported: {', '.join(CALIBRATION_METRICS)}.")
            self.uncertainty_comparisons.append({
                "mean": (entry["mean_variable"].GetString(),
                         entry["mean_location"].GetString()),
                "std": (entry["std_variable"].GetString(),
                        entry["std_location"].GetString()),
                "reference": (entry["reference_variable"].GetString(),
                              entry["reference_location"].GetString()),
                "confidence_z": entry["confidence_z"].GetDouble(),
                "metrics": metrics,
            })

        ensemble_comparison_defaults = Kratos.Parameters("""{
            "member_variables"   : [],
            "member_location"    : "node_non_historical",
            "reference_variable" : "PLEASE_SPECIFY_VARIABLE_NAME",
            "reference_location" : "node_historical",
            "metrics"            : ["crps"]
        }""")
        self.ensemble_comparisons = []
        for i in range(settings["ensemble_comparisons"].size()):
            entry = settings["ensemble_comparisons"][i]
            entry.ValidateAndAssignDefaults(ensemble_comparison_defaults)
            members = entry["member_variables"].GetStringArray()
            # crps/kcrps are scoring rules over the FULL ensemble and cannot be
            # recovered from a (mean, std) pair, so the members are named
            # explicitly rather than reduced first
            if len(members) < 2:
                raise ValueError(
                    f"\"ensemble_comparisons\" needs at least 2 \"member_variables\", got "
                    f"{len(members)}: an ensemble metric is undefined for a single member.")
            metrics = entry["metrics"].GetStringArray()
            for metric in metrics:
                if metric not in ENSEMBLE_METRICS:
                    raise ValueError(
                        f"Unsupported ensemble metric \"{metric}\". "
                        f"Supported: {', '.join(ENSEMBLE_METRICS)}.")
            self.ensemble_comparisons.append({
                "members": [(name, entry["member_location"].GetString()) for name in members],
                "reference": (entry["reference_variable"].GetString(),
                              entry["reference_location"].GetString()),
                "metrics": metrics,
            })

        self.output_interval = settings["output_interval"].GetInt()
        self.output_file = Path(settings["output_file"].GetString())
        if self.output_interval < 1:
            raise ValueError(f"\"output_interval\" must be >= 1 [ output_interval = {self.output_interval} ].")

        self.history = []

    def ExecuteFinalizeSolutionStep(self) -> None:
        step = self.model_part.ProcessInfo[Kratos.STEP]
        if step % self.output_interval != 0:
            return
        self.Evaluate()

    def Evaluate(self) -> None:
        record = {
            "STEP": self.model_part.ProcessInfo[Kratos.STEP],
            "TIME": self.model_part.ProcessInfo[Kratos.TIME],
        }
        for comparison in self.comparisons:
            predicted = self._Gather(*comparison["predicted"])
            reference = self._Gather(*comparison["reference"])
            weights = (self._Gather(*comparison["weight"])
                       if comparison["weight"] is not None else None)
            values = ComputeMetricValues(predicted, reference, comparison["metrics"], weights)

            name = f"{comparison['predicted'][0]}_vs_{comparison['reference'][0]}"
            record[name] = values
            Kratos.Logger.PrintInfo(
                "ValidationMetricsProcess",
                f"step {record['STEP']}: {name}: " +
                ", ".join(f"{k}={v:.6e}" for k, v in values.items()))
        for comparison in self.uncertainty_comparisons:
            mean = self._Gather(*comparison["mean"])
            std = self._Gather(*comparison["std"])
            reference = self._Gather(*comparison["reference"])
            values = ComputeCalibrationMetricValues(
                mean, std, reference, comparison["metrics"], comparison["confidence_z"])

            name = f"calibration_{comparison['mean'][0]}_vs_{comparison['reference'][0]}"
            record[name] = values
            Kratos.Logger.PrintInfo(
                "ValidationMetricsProcess",
                f"step {record['STEP']}: {name}: " +
                ", ".join(f"{k}={v:.6e}" for k, v in values.items()))

        for comparison in self.ensemble_comparisons:
            torch = torch_bridge._TryImportTorch()
            ensemble = torch.stack([self._Gather(*member) for member in comparison["members"]])
            reference = self._Gather(*comparison["reference"])
            values = ComputeEnsembleMetricValues(ensemble, reference, comparison["metrics"])

            name = f"ensemble_{comparison['reference'][0]}"
            record[name] = values
            Kratos.Logger.PrintInfo(
                "ValidationMetricsProcess",
                f"step {record['STEP']}: {name}: " +
                ", ".join(f"{k}={v:.6e}" for k, v in values.items()))

        if self.cfd_metric_specs:
            from KratosMultiphysics.PhysicsNeMoApplication import cfd_bridge
            for name, domain, field_pairs in self.cfd_metric_specs:
                ground_truth = {}
                predictions = {}
                for semantic_name, (predicted, reference) in field_pairs.items():
                    predictions[semantic_name] = self._Gather(*predicted).numpy()
                    ground_truth[semantic_name] = self._Gather(*reference).numpy()
                value = cfd_bridge.EvaluateCfdMetrics(
                    [(name, domain)], ground_truth, predictions)[name]
                record[f"cfd_{name}"] = value
                Kratos.Logger.PrintInfo(
                    "ValidationMetricsProcess",
                    f"step {record['STEP']}: cfd {name} ({domain}): {value:.6e}")
        self.history.append(record)

    def _Gather(self, variable_name: str, data_location: str):
        variable = Kratos.KratosGlobals.GetVariable(variable_name)
        tensor_adaptor = GetTensorAdaptor(self.model_part, data_location, variable)
        return torch_bridge.KratosTensorToTorch(tensor_adaptor).clone()

    def ExecuteFinalize(self) -> None:
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_file, "w") as f:
            json.dump(self.history, f, indent=4)
        Kratos.Logger.PrintInfo(
            "ValidationMetricsProcess",
            f"Wrote {len(self.history)} record(s) to \"{self.output_file}\".")

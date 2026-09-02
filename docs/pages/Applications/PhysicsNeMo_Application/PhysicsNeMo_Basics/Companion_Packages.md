---
title: Companion packages
keywords: physicsnemo cfd curator active learning experimental guardrails lora
tags: [Companion_Packages.md]
sidebar: physicsnemo_application
summary: active_learning, physicsnemo-cfd, physicsnemo-curator and the experimental namespace - what each needs installed.
---

# Companion packages

Four things ship alongside the core library. Two are inside the `physicsnemo` package; two are separate installs.

## `physicsnemo.active_learning` (bundled)

Active learning inverts the usual order: instead of generating a dataset and then training, the model trains, decides where it is least certain, and asks for labels **there**.

- `Driver`, `DriverConfig` — the loop.
- `StrategiesConfig` — the four pluggable strategies: **query** (which samples to label next), **label** (how a label is produced), **training**, **metrology** (how progress is measured).
- `DefaultTrainingLoop`, `TrainingConfig`, `OptimizerConfig`.
- `protocols` — the interfaces you implement.

The label strategy is where Kratos goes: the "oracle" that produces ground truth is a real solve. See [Active Learning](../Active_Learning/Active_Learning.html).

## `physicsnemo.experimental` (bundled)

Not stable API, but several things here are the *only* implementation of what they do:

| Module | What it holds |
|---|---|
| `experimental.guardrails.embedded` | `OODGuard`, `OODGuardConfig` — out-of-distribution detection embedded next to a model |
| `experimental.uq` | variational GP heads (`variational_gp_head`, `field_variational_gp_head`) for calibrated posterior variance |
| `experimental.peft` | LoRA adapters — `apply`, `merge`, `io` |
| `experimental.models` | `flare`, `geotransolver`, `aerojepa`, `globe`, `healda`, `strata`, `xdeeponet`, `diffusion`, `diffusion_unets` |
| `experimental.nn` | FLARE attention, point tokenizers, RoPE, symmetry layers, 3-D diffusion U-Net blocks |
| `experimental.datapipes`, `experimental.metrics`, `experimental.utils` | HealDA pipeline, diffusion metrics, caching and prefetch |

**Note for the diffusion bridge.** `physicsnemo.models.diffusion_unets` is 2-D-image oriented, but `experimental.models.diffusion_unets.DiffusionUNet3D` is a genuine volumetric denoiser: it implements the `physicsnemo.diffusion.base.DiffusionModel` protocol, so it composes with the same preconditioners, losses and samplers the shipped bridge already uses, and takes optional volume (`(B, C, D, H, W)`) and vector conditioning. The bridge deploys it through `denoiser_interface: "unet3d"` (see the [Diffusion](../Diffusion/Diffusion.html) page).

## `physicsnemo-cfd` (separate install, source only)

```bash
pip install git+https://github.com/NVIDIA/physicsnemo-cfd
```

Not on PyPI. Provides `physicsnemo.cfd`:

- `cfd.postprocessing_tools.metric_registry` — a domain-aware CFD metric registry (relative-L2, drag and lift, physics residuals, UQ metrics);
- `cfd.hybrid_initialization_tools` — blending a prediction into a solver's initial condition;
- `cfd.evaluation` — checkpoint-driven evaluation wrappers, benchmarks, datasets, reports and NIM clients.

Kratos side: `bridges.cfd_bridge` — Kratos ↔ pyvista `Flowfield` conversion, the metric registry reachable through `processes.validation_metrics_process`'s `cfd_metrics` block, and hybrid-initialization blending.

## `physicsnemo-curator` (separate install, git only)

An ETL framework for turning raw simulation output into AI-ready datasets (Zarr stores, VTU grids). Its **sinks** ship upstream; the **source** side is what a solver has to supply.

Its build pulls a Rust toolchain, so nothing here requires it: without it installed `bridges.curator_bridge` still imports and only its entry points raise.

Kratos side: `bridges.curator_bridge` supplies the source, `processes.export.curator_export_process` writes Zarr or VTU straight from a running solution loop.

## Optional dependencies at a glance

| Package | Needed for | Without it |
|---|---|---|
| `torch` | everything ML | the application still imports; ML entry points raise with an install hint |
| `nvidia-physicsnemo` | everything ML | same |
| `torch_geometric`, `torch_scatter` | the graph processes | those tests skip |
| `torch_sparse` or `dgl` | GraphCast | that recipe skips |
| `onnxruntime` / `onnxruntime-gpu` | ONNX inference | ONNX tests skip |
| `tritonclient` | Triton serving | Triton tests skip |
| `gpytorch` | the GP uncertainty head | that head raises |
| `pyvista` | mesh visualization, `cfd_bridge` | those paths raise |
| `nvidia-physicsnemo-cfd` | the CFD metric registry | `cfd_metrics` raises |
| `physicsnemo-curator` | Zarr/VTU AI-ready export | `curator_bridge` entry points raise |
| `nbclient`, `nbformat` | running the example notebooks | notebook tests skip |

Every one of these is checked lazily. `import KratosMultiphysics.PhysicsNeMoApplication` succeeds with none of them installed — guaranteed by `tests/test_import_contract.py`.

Back to [PhysicsNeMo Basics](Overview.html), or on to [Where things live](../General/Module_Map.html).

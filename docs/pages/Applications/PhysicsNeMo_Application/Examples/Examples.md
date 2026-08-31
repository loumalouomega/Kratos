---
title: Examples
keywords: examples notebooks
tags: [Examples.md]
sidebar: physicsnemo_application
summary: 
---

# Example notebooks

Runnable, CPU-only notebooks under
`applications/PhysicsNeMoApplication/examples/notebooks/`, ordered as a learning
path. Each states its prerequisites up front; outputs go to a local `./output/`.

| Notebook | What it shows |
|---|---|
| 01 — Tensor bridge and dataset export | `ModelPart` ↔ `torch.Tensor` (zero-copy in, checked copy out), `DatasetExportProcess`, `CreateNpzDataset` + `DataLoader` |
| 02 — Mesh bridge round trip | Tessellation with provenance, volume conservation, `DomainMesh` with named boundaries, exact nodal scatter-back, native save/load, both meshes rendered via `KratosMultiphysics.pyvista_utilities` |
| 03 — Training and deploying a surrogate | Parameter sweep → dataset export → MLP training → TorchScript checkpoint → `InferenceProcess` and `HybridInitializationProcess` |
| 04 — Active learning with Kratos as the oracle | `KratosALSample`, execution backends, `CreateKratosLabelStrategy`, one real `physicsnemo.active_learning.Driver` step |
| 05 — Superresolution | Coarse/fine meshes → voxel grids, `SRResNet` training, `.mdlus` checkpointing, `SuperResolutionProcess` deployment |
| 06 — Mesh graph nets and transient surrogates | Element-edge graph extraction, `MeshGraphNet` + `GraphInferenceProcess` (needs `torch_geometric`/`torch_scatter`), autoregressive rollout via `TimeSeriesInferenceProcess` |
| 07 — Real-solver thermal surrogate | The Darcy-FNO recipe on **real `ConvectionDiffusionApplication` solves**, `GridInferenceProcess` deployment, `ValidationMetricsProcess` numbers, mesh-aware field comparison via `pyvista_utilities`, availability-gated on compiled applications |
| 08 — RNN grid sequences | Transient field → grid series (`GridDatasetExportProcess`), `CreateGridSequenceDataset` + `One2ManyRNN` training, seed-once/roll-forward deployment with `SequenceInferenceProcess` |
| 09 — Diffusion downscaling | CorrDiff-style conditional diffusion over matched grid pairs, ensemble mean + per-node uncertainty via `DiffusionInferenceProcess` |
| 10 — Structural deformation surrogate | The crash/deformation-surrogate pattern on **real `StructuralMechanicsApplication` solves**, load → displacement-field surrogate, `InferenceProcess` deployment, warped-mesh comparison via `pyvista_utilities` (availability-gated) |
| 11 — ROM surrogate and temporal attention | Neural-augmented reduced bases on **real `RomApplication` POD output**, `RomSurrogateProcess` deployment at an unseen parameter, reconstructed-field comparison via `pyvista_utilities`, `Sequence_Model` temporal attention over ROM trajectories (availability-gated, numpy-SVD fallback) |
| 12 — Co-simulation surrogate coupling | A trained surrogate as a **solver** in Kratos co-simulation: `cosim_surrogate_solver_wrapper` by module path, weak Gauss-Seidel coupling through `kratos_mapping` transfer, Aitken-accelerated strong-coupling fixed point (needs `CoSimulationApplication` + `MappingApplication`) |
| 13 — Lagrangian particle surrogate | Learning-to-Simulate on a particle cloud: trajectory windowing, velocity-history features, autoregressive `ParticleInferenceProcess` rollout with the proximity graph rebuilt every step |
| 14 — PINN forward and inverse solves | Data-free Laplace solve from Dirichlet fixities via `PinnSolveProcess` + inverse diffusion-coefficient recovery (physicsnemo.sym, bundled) |
| 15 — CorrDiff two-stage downscaling | The regression + residual-diffusion split: `TrainCorrDiffPair` (upstream `RegressionLoss`/`ResidualLoss`, `SongUNetPosEmbd` sizing) and `DiffusionInferenceProcess` with `"regression_settings"` adding the regression mean to the ensemble — sharper mean, calibrated spread |
| 16 — Transient structural surrogate | Temporal schemes on a real implicit-dynamic (Bossak) cantilever solve: `RunTransientAnalysis` trajectories, single-step window training, `TrainAutoregressive` BPTT with gradient checkpointing (2.7x lower rollout error), and deployment through `TimeSeriesInferenceProcess` |
| 17 — Exact shape gradients and shape optimization | The discretely exact `dJ/dX` of a real FEM solve from one element-local pass (`ComputeShapeSensitivityField`), the measured crossover against the per-parameter global path, `ComputeControlSensitivities` chaining it onto an FFD lattice, and a gradient descent driving the shape onto a target — validated against re-solve finite differences to ten digits |

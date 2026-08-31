---
title: Models
keywords: physicsnemo models architectures fno meshgraphnet transolver domino
tags: [Models.md]
sidebar: physicsnemo_application
summary: The 25 architecture families under physicsnemo.models, and which ones this application deploys.
---

# Models

`physicsnemo.models` holds 25 architecture families. You do not need to know
them all; you need to know **which shape of data you have**, because that is
what picks the family.

## Pick by data shape

| Your data is | Use | Deployed here by |
|---|---|---|
| A regular grid, same resolution in and out | FNO, AFNO, UNet, DPOT | `grid_inference_process` |
| A coarse grid in, a fine grid out | SRResNet (`srrn`) | `superresolution_process` |
| An unstructured mesh with connectivity | MeshGraphNet family | `graph_inference_process` |
| An unordered point cloud | Transolver, FIGConvNet, FLARE | `point_cloud_inference_process` |
| A CAD surface plus a volume (external aero) | DoMINO | `domino_inference_process` |
| A time series of states | RNN, `mesh_reduced` | `sequence_inference_process`, `time_series_inference_process` |
| Particles with trajectories | MeshGraphNet, VFGN | `particle_inference_process` |
| A distribution, not a single answer | diffusion U-Nets, DiT | `diffusion_inference_process` |
| The globe | GraphCast, Pangu, FengWu, DLWP | `grid_inference_process` (GraphCast) |

## The families

### Neural operators on grids

| Family | Classes | Notes |
|---|---|---|
| `fno` | `FNO` | The reference neural operator. `dimension=1..4`; the 4-D case is a spatio-temporal block operator |
| `afno` | `AFNO`, `ModAFNO` | Fourier operator with an adaptive token mixer. `ModAFNO` takes a *timestep* input, so the solver's TIME conditions it |
| `dpot` | `DPOTNet` | Denoising pre-trained operator transformer, 2-D and 3-D |
| `unet` | `UNet` | A plain 3-D convolutional U-Net |
| `pix2pix` | `Pix2Pix` | Image-to-image translation |
| `srrn` | `SRResNet` | Super-resolution residual network — coarse in, fine out |

### Graphs and meshes

| Family | Classes | Notes |
|---|---|---|
| `meshgraphnet` | `MeshGraphNet`, `BiStrideMeshGraphNet`, `HybridMeshGraphNet`, `MeshGraphKAN` | Encode–process–decode on an edge graph. The variants add a multiscale hierarchy, proximity "world" edges, and KAN layers |
| `mesh_reduced` | `Mesh_Reduced`, `Sequence_Model` | Reduce a mesh, then learn the dynamics in the reduced space with temporal attention |
| `graphcast` | `GraphCastNet` | The weather architecture, on an icosahedral grid |
| `vfgn` | `VFGNLearnedSimulator` and its encode/process/decode parts | Virtual Foundry GraphNet, for sintering and additive manufacturing |

### Point clouds and transformers

| Family | Classes | Notes |
|---|---|---|
| `transolver` | `Transolver` | Attention over learned "physics slices" of a point cloud |
| `geotransolver` | `GeoTransolver` | Geometry-aware successor |
| `figconvnet` | `FIGConvUNet` | Factorized implicit grids; per-point fields plus a scalar (drag-style) head |
| `flare` | `FLARE` | Experimental attention successor (also under `experimental.models`) |
| `domino` | `DoMINO` | External aerodynamics: CAD surface plus volume, with pretrained checkpoints |

### Generative

| Family | Classes | Notes |
|---|---|---|
| `diffusion_unets` | `SongUNet`, `SongUNetPosEmbd`, `DhariwalUNet`, `CorrDiffRegressionUNet`, `UNet`, `StormCastUNet` | The denoiser backbones. **2-D image oriented** |
| `dit` | `DiT` | Diffusion transformer — the denoiser this application wraps for non-image data |
| `topodiff` | `TopoDiff` | Diffusion for topology optimization |

A **volumetric** 3-D denoiser exists too, but under
`physicsnemo.experimental.models.diffusion_unets.DiffusionUNet3D` — see
[Companion packages](Companion_Packages.html).

### Sequences and weather

`rnn` (`One2ManyRNN`, `Seq2SeqRNN`), `dlwp` (`DLWP`), `dlwp_healpix`
(`HEALPixUNet`, `HEALPixRecUNet`), `pangu` (`Pangu`), `fengwu` (`Fengwu`),
`swinvrnn` (`SwinRNN`).

### The plain one

`mlp` (`FullyConnected`) — a multilayer perceptron. It is the right first model
far more often than it looks: if your input is a handful of case parameters and
your output is a field, you want this, not a neural operator.

## Building blocks

`physicsnemo.nn` holds the layers the models are made of — around 150 of them.
Two are worth knowing by name because this application uses them directly:

- `ConcreteDropout` — dropout with a *learned* rate, which makes MC-dropout
  uncertainty estimates meaningfully calibrated instead of arbitrary;
- `physicsnemo.nn.functional.derivatives` — differential operators used by the
  physics-informed path.

## What this application uses it for

| Architecture | Kratos-side entry | Status here |
|---|---|---|
| `FNO`, `AFNO`, `ModAFNO` | `processes.inference.grid_inference_process` | test-pinned |
| `SRResNet` | `processes.inference.superresolution_process` | test-pinned |
| `MeshGraphNet` + all three variants | `processes.inference.graph_inference_process` | test-pinned, one on a real fluid solve |
| `Transolver`, `GeoTransolver`, `FLARE`, `FIGConvUNet` | `processes.inference.point_cloud_inference_process` | test-pinned |
| `DoMINO` | `processes.inference.domino_inference_process` | test-pinned, pretrained checkpoints de-normalized |
| `One2ManyRNN` | `processes.inference.sequence_inference_process` | test-pinned |
| `Sequence_Model` | `training.rom_temporal` | test-pinned |
| `VFGNLearnedSimulator` | `bridges.vfgn_bridge` | driven by a real sintering solve |
| `DiT`, `CorrDiffRegressionUNet` | `processes.inference.diffusion_inference_process` | test-pinned |
| `GraphCastNet` | `processes.inference.grid_inference_process` | documented, needs `torch_sparse` or `dgl` |
| `FullyConnected` | anything | the default in most examples |

2-D UNets and DLWP go through the grid process mechanically, by the same
thin-axis squeeze idiom, but are not covered by a test.

Next: [Data and datapipes](Data_And_Datapipes.html) — feeding them.

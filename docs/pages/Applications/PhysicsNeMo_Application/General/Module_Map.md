---
title: Where things live
keywords: module map layout python_scripts navigation
tags: [Module_Map.md]
sidebar: physicsnemo_application
summary: Every module in the application, by folder, with a one-line purpose - and an index from what you want to do.
---

# Where things live

The application is a hundred-odd Python modules. This page is the map: what each
folder is for, what is in it, and how to get from "I want to do X" to the module
that does X.

## The folders

`python_scripts/` is a tree of packages, and the folder a module is in tells you
what kind of thing it is.

| Folder | Contains | Import as |
|---|---|---|
| `processes/inference/` | run a trained model in the solution loop | `...PhysicsNeMoApplication.processes.inference` |
| `processes/export/` | write solver data out as training data | `...PhysicsNeMoApplication.processes.export` |
| `processes/` | adaptive remeshing, validation metrics | `...PhysicsNeMoApplication.processes` |
| `bridges/` | convert Kratos data to and from PhysicsNeMo data | `...PhysicsNeMoApplication.bridges` |
| `training/` | training loops, datasets, schemes | `...PhysicsNeMoApplication.training` |
| `physics/` | residuals, PINN machinery, sensitivities | `...PhysicsNeMoApplication.physics` |
| `deployment/` | checkpoints, cards, export, serving, uncertainty | `...PhysicsNeMoApplication.deployment` |
| `distributed/` | MPI and multi-rank | `...PhysicsNeMoApplication.distributed` |
| `active_learning/` | Kratos as the labeling oracle | `...PhysicsNeMoApplication.active_learning` |
| `utilities/` | small shared helpers | `...PhysicsNeMoApplication.utilities` |

**The rule that keeps it navigable:** everything under `processes/` has a
`Factory`, so it can be attached from `ProjectParameters.json`. Nothing outside
`processes/` does. If you are looking for something to put in a process list,
you only have to look in one place.

Each package's `__init__.py` carries a docstring saying what belongs in it —
read that before adding a module.

## Attaching a process

```json
{
    "python_module" : "inference_process",
    "kratos_module" : "KratosMultiphysics.PhysicsNeMoApplication.processes.inference",
    "Parameters"    : { }
}
```

`kratos_module` names the **package**, `python_module` the module inside it.
Getting the package wrong fails at run time, so the suite resolves every
documented pair against the real tree.

## I want to...

### Get data out of a solve

| Goal | Module |
|---|---|
| Dump nodal/elemental/Gauss fields as `.npz` | `processes.export.dataset_export_process` |
| The same, on a regular voxel grid | `processes.export.grid_dataset_export_process` |
| External-aero cases for DoMINO/Transolver | `processes.export.cae_dataset_export_process` |
| A `.pmsh` mesh series | `processes.export.mesh_export_process` |
| AI-ready Zarr or VTU | `processes.export.curator_export_process` |
| Train while the solve is still running | `processes.export.streaming_dataset_export_process` + `training.streaming_dataset` |
| Raw tensors, no process | `bridges.torch_bridge` |

### Train something

| Goal | Module |
|---|---|
| Fit a model to exported data | `training.training_utils.TrainModel` |
| Build a dataset from what was exported | `training.torch_dataset` |
| Backpropagate through a rollout | `training.temporal_training.TrainAutoregressive` |
| Train a diffusion model | `training.diffusion_utils` |
| Adapt a pretrained DoMINO | `training.domino_finetune` |
| Learn dynamics in ROM space | `training.rom_temporal` |
| Add a physics term to the loss | `physics.physics_informed`, `physics.differentiable_residual` |
| Measure multi-step error growth | `training.rollout_utils.EvaluateRollout` |
| Save it | `training.training_utils.SaveTrainedModel` |

### Deploy it

| Goal | Module |
|---|---|
| Run it in the solution loop | `processes.inference.inference_process` |
| ...on the mesh graph | `processes.inference.graph_inference_process` |
| ...on a grid | `processes.inference.grid_inference_process` |
| ...on the nodes as a point cloud | `processes.inference.point_cloud_inference_process` |
| ...autoregressively in time | `processes.inference.time_series_inference_process`, `...sequence_inference_process` |
| ...on particles | `processes.inference.particle_inference_process` |
| ...as an ensemble with uncertainty | `processes.inference.diffusion_inference_process` |
| Warm-start the solver instead | `processes.inference.hybrid_initialization_process` |
| Upscale a coarse solve | `processes.inference.superresolution_process` |
| Run it as a CoSimulation solver | `deployment.cosim_surrogate_solver_wrapper` |
| Ship it without physicsnemo | `training.training_utils.ExportOnnxModel` + `processes.inference.onnx_inference_process` |
| Ship it to a server | `deployment.triton_export` + `processes.inference.triton_inference_process` |

### Trust it

| Goal | Module |
|---|---|
| Compare predictions against a reference | `processes.validation_metrics_process` |
| Attach error bars | `deployment.uncertainty_utils` |
| Catch out-of-distribution inputs | `deployment.ood_guard_utils` |
| Record what a checkpoint's fields mean | `deployment.model_registry` (model cards) |
| Score with the real PDE residual | `physics.solver_residuals` |
| Let the model choose its own training data | `active_learning` |

### Work with geometry

| Goal | Module |
|---|---|
| Tessellate a Kratos mesh | `bridges.mesh_bridge.tessellation` |
| Map predictions back to the original entities | `bridges.mesh_bridge.provenance` |
| Generate a mesh from an SDF | `bridges.mesh_bridge.generate` |
| Signed distances as node features | `bridges.mesh_bridge.spatial` |
| Deform a shape differentiably | `bridges.mesh_bridge.deformation` |
| Refine where the surrogate is wrong | `bridges.mesh_bridge.adaptive_remeshing` + `processes.adaptive_remesh_process` |
| Exact shape gradients | `physics.sensitivity_utils` |
| Differential operators on the mesh | `bridges.calculus_bridge` |
| Move fields between non-matching meshes | `bridges.mapping_bridge` |

### Scale it

| Goal | Module |
|---|---|
| Align physicsnemo's ranks with Kratos's | `distributed.distributed_utils` |
| Train a graph model across ranks | `distributed.graph_partition_utils` |

## Tests and examples

- `tests/` mirrors the sources by name: `test_<module>.py`. Subdirectories
  (`tests/bridges/mesh_bridge/`, `tests/active_learning/`) are discovered
  automatically by the suite runner.
- `examples/notebooks/` — eighteen notebooks, executed by `tests/test_notebooks.py`
  so a changed signature breaks a test rather than rotting.
- The [Examples repository](https://github.com/KratosMultiphysics/Examples/tree/master/physics_nemo_application)
  holds eighteen fully documented use cases against real solves.

New here? [From scratch](From_Scratch.html) walks one path end to end. New to
PhysicsNeMo itself? [PhysicsNeMo Basics](../PhysicsNeMo_Basics/Overview.html).

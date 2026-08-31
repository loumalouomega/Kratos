---
title: Overview
keywords: physicsnemo machine learning surrogate
tags: [Overview.md]
sidebar: physicsnemo_application
summary: 
---

# PhysicsNeMo Application

The *PhysicsNeMo Application* bridges *Kratos Multiphysics* with [NVIDIA PhysicsNeMo](https://docs.nvidia.com/physicsnemo/latest/index.html), a PyTorch-based framework for physics-ML. It covers the full loop between a finite-element solver and a machine-learning workflow:

- **Data out**: export simulation fields as training datasets ([Tensor Bridge](../Tensor_Bridge/Tensor_Bridge.html)).
- **Geometry out**: convert arbitrary Kratos meshes into PhysicsNeMo's simplicial mesh representation, with provenance to map predictions back ([Mesh Bridge](../Mesh_Bridge/Mesh_Bridge.html)).
- **Predictions in**: run trained models inside a Kratos solution loop, including warm-starting solves ([Inference](../Inference/Inference.html)).
- **Kratos as oracle**: use Kratos as the ground-truth solver inside a `physicsnemo.active_learning` loop ([Active Learning](../Active_Learning/Active_Learning.html)).

Built on those four, the rest of the application covers:

- **Training** ([Training](../Training/Training.html)) — datasets, schedules, streaming and warm restarts; **Reduced-order models** ([ROM](../Reduced_Order_Models/Reduced_Order_Models.html)); **Sequence models** ([Sequence Models](../Sequence_Models/Sequence_Models.html)); **Super-resolution** ([Super-Resolution](../Super_Resolution/Super_Resolution.html)).
- **Model families**: graph networks ([GNNs](../Graph_Neural_Networks/Graph_Neural_Networks.html)), point clouds and transformers ([Point Clouds](../Point_Clouds/Point_Clouds.html)), diffusion and downscaling ([Diffusion](../Diffusion/Diffusion.html)), Lagrangian particles ([Particle Methods](../Particle_Methods/Particle_Methods.html)), and external-aero datapipes ([CAE Datapipes](../CAE_Datapipes/CAE_Datapipes.html)).
- **Physics in the loss** ([Physics-Informed](../Physics_Informed/Physics_Informed.html)) — strong-form residual terms, PINN solves, the exact discrete residual through the real FEM assembly, and adjoint sensitivities cross-validated against Kratos's own adjoint stack.
- **Trust and scale**: prediction uncertainty and governance ([Uncertainty](../Uncertainty/Uncertainty.html)), MPI-distributed training and inference ([Distributed](../Distributed/Distributed.html)), and deploying a surrogate as a co-simulation solver ([CoSimulation](../CoSimulation/CoSimulation.html)).

Serving (ONNX export and Triton) is covered in [Inference](../Inference/Inference.html); adaptive remeshing in [Mesh Bridge](../Mesh_Bridge/Mesh_Bridge.html); streaming datasets and warm restarts in [Training](../Training/Training.html).

## Dependency policy

`torch` and `nvidia-physicsnemo` are **optional, pure-Python runtime dependencies**:

- `import KratosMultiphysics.PhysicsNeMoApplication` always succeeds without them; only the specific submodules that need them fail — lazily, at call time, with an actionable error message.
- There is intentionally **no CMake gating** (no `find_package(Torch)`, no `USE_PHYSICSNEMO` option): nothing in the C++ core links against torch or CUDA. Do not add such a gate.
- Contributions must keep all `import torch` / `import physicsnemo` statements inside lazy `_TryImport*()` helpers.
- **Packaging**: the application ships like any per-application Kratos wheel (`scripts/wheels/build_wheel.py` derives app wheels automatically from what was compiled). `torch`/`nvidia-physicsnemo` are never wheel dependencies — they stay optional runtime `pip install`s, and the import-contract test guarantees a wheel installed without them remains fully importable.

## Working with other Kratos applications

Some tests and examples run **real Kratos solves** (e.g. a stationary heat-conduction
case through `ConvectionDiffusionApplication`). They are conditioned on the compiled
applications following the standard pattern (as used by CoSimulationApplication):

```python
import KratosMultiphysics.kratos_utilities as kratos_utils
have_convection_diffusion = kratos_utils.CheckIfApplicationsAvailable(
    "ConvectionDiffusionApplication", "LinearSolversApplication")

@KratosUnittest.skipUnless(have_convection_diffusion, "Missing required applications: ...")
class TestRealSolver(...): ...
```

Tests are always registered and skip cleanly (with an explanatory message) on builds
that lack the required applications. The reusable in-memory thermal case powering
these tests lives in `tests/kratos_solver_cases/thermal_case.py` (meshed with the
core `StructuredMeshGeneratorProcess`, no `.mdpa` fixtures, solver fed through
`"input_type": "use_input_model_part"`).

## Examples

Runnable example notebooks (tensor bridge, mesh bridge, surrogate training and deployment, active learning) live in `applications/PhysicsNeMoApplication/examples/notebooks/`.

## Installation

Compile like any other Kratos application:

```bash
add_app ${KRATOS_APP_DIR}/PhysicsNeMoApplication
```

To use the ML features additionally install the runtime dependencies:

```bash
pip install torch             # CPU or CUDA build, your choice
pip install nvidia-physicsnemo
pip install nvidia-physicsnemo-cfd   # optional: hybrid-initialization recipes + CFD metric registry
```

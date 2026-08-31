# PhysicsNeMo Application

|      **Application**      |                                                                              **Description**                                                                              |                                       **Status**                                        |                       **Authors**                        |
|:--------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------:|:-----------------------------------------------------------:|
| `PhysicsNeMoApplication` | The *PhysicsNeMo Application* bridges *Kratos Multiphysics* with [NVIDIA PhysicsNeMo](https://docs.nvidia.com/physicsnemo/latest/index.html), a PyTorch-based framework for physics-ML: surrogate models, neural operators, and active learning with an external ground-truth solver. | <img src="https://img.shields.io/badge/Status-%F0%9F%9A%80%20Actively%20developed-Green"  width="200px"> | [*Vicente Mataix Ferrándiz*](mailto:vicente.mataix-ferrandiz@siemens.com) |

The application includes tests to check the proper functioning of the application.

## 😎 Features:

- **Tensor bridge**: zero-copy `ModelPart` data ↔ `torch.Tensor` built on the core `Kratos.TensorAdaptors`, plus a dataset-export process producing `.npz` training samples from any combination of nodal, elemental and Gauss-point fields, and a `torch.utils.data.Dataset` factory over the exported samples for direct training — plus a **streaming path** (`streaming_dataset`) that trains directly out of a running solve with no file round trip, yielding samples byte-identical to the dumped ones (asserted by running one case both ways), and **shrink-and-perturb warm restarts** in `TrainModel` for when Kratos data drifts to a new geometry family. Dataset curation for the `.pmsh` mesh series: **coherent** random rotate/scale/translate augmentations (`MakeMeshAugmentations`/`CreateAugmentedMeshDataset` — vector and rank-2 tensor fields transform *with* the coordinates, which upstream's defaults silently skip, plus the dtype cast its transforms require and reproducible per-epoch seeding) and `MultiDataset` mixing of several series (`CreateMultiMeshDataset`).

- **Mesh bridge**: tessellation of arbitrary Kratos meshes (hexahedra, prisms, pyramids, quadrilaterals, higher-order elements) into the simplicial representation required by `physicsnemo.mesh` — watertight on unstructured meshes via the smallest-node-id diagonal rule (Dompierre et al.), with optional subdivision of quadratic geometries through their real mid-side nodes and an opt-in **curved (isoparametric) mode** that samples the exact quadratic geometry on a configurable refinement lattice with synthetic points (interpolated on gather, dropped on scatter-back), watertight across curved neighbours — plus a field-provenance map that round-trips predictions back onto the original Kratos entities; `DomainMesh` export with named boundaries from sub-model-parts, and native save/load of PhysicsNeMo's memory-mapped mesh format. On top of it:
    - **Discrete calculus** (`calculus_bridge`) — LSQ/DEC gradient, divergence, curl, Laplacian and integrals on the tessellated mesh, autograd-differentiable, with the backend-validity guards and boundary masks the upstream operators lack.
    - **Mesh generation from implicit geometry** (`mesh_bridge/generate`) — SDF primitives and combinators into a meshed volume, marching-cubes level sets, quality-guaranteed 2D loop filling, **tetrahedral filling of watertight 3D surfaces** (winding-number-carved, so non-convex solids fill correctly, and self-validating against the input's own volume and boundary area), differentiable refitting, and `PopulateModelPartFromMesh`, the direction the bridge never had: generated geometry materialized as real Kratos entities and handed straight to MMG or a solver.
    - **Grid-stencil counterparts** in `grid_bridge.ComputeGridDerivatives`/`ComputeGridVectorOperator` — gradients, divergence, curl and Laplacian, with the float64-precision backend choice upstream gets wrong by default.
    - **Signed distance fields as features** (`mesh_bridge/spatial`) — written into an ordinary nodal variable so every existing gather picks them up, on a boundary surface this bridge re-orients because upstream's extraction winds it inconsistently.
    - **Surrogate-error-driven adaptive remeshing** (`adaptive_remeshing`/`AdaptiveRemeshProcess`) — residual scoring → equidistributed size field → `MeshingApplication` MMG scalar-metric remesh, plus error-weighted `partition_cells` surface clustering and a Warp-backed `remesh` wrapper.
    - **A differentiable shape-deformation layer** (`mesh_bridge/deformation`) — FFD/RBF/morph/displace parameterizations, mesh-quality energies including the inversion term that keeps an optimizer from tearing the mesh, and coordinate write-back through the position tensor adaptor; the matching `sensitivity_utils.ComputeShapeSensitivities`, which gives `dJ/d(control)` in one backward pass, lives with the other adjoints.

- **In-loop inference**: an `InferenceProcess` running a trained model (TorchScript or physicsnemo checkpoint) at a configurable point of the solution loop, writing predictions into existing Kratos variables, and a `HybridInitializationProcess` warm-starting solves from a model prediction. Deployment extras: **ONNX export and inference** (`ExportOnnxModel` via `physicsnemo.deploy.onnx` plus an `OnnxInferenceProcess` running a cached ONNX Runtime session — a portable deployment artifact needing neither torch-the-model-runtime nor physicsnemo), opt-in **`torch.compile(fullgraph=True)`** wrapping of loaded physicsnemo checkpoints, and opt-in **NVTX ranges** around every deployment process's gather/forward/scatter hot paths for Nsight Systems profiling. **Serving**: `triton_export` writes a Triton Inference Server model repository (dynamic-entity-axis ONNX or TorchScript plus a generated `config.pbtxt`, tensor names taken from the model card) and `TritonInferenceProcess` calls the running server from inside the solution loop, so the solver host needs neither weights nor a GPU.

- **Superresolution**: a grid bridge sampling unstructured-mesh fields onto regular voxel grids (FE shape-function interpolation, exact for linear fields) and a `SuperResolutionProcess` deploying `physicsnemo.models.srrn.SRResNet`-style models: coarse-mesh solve in, superresolved fine-mesh field out.

- **Graph neural networks on the mesh**: a graph bridge extracting the true element-edge graph of a `ModelPart` (bidirectional, MeshGraphNet edge encoding) and a `GraphInferenceProcess` deploying `physicsnemo.models.meshgraphnet.MeshGraphNet`-style models directly on unstructured meshes (requires the optional `torch_geometric`/`torch_scatter`), plus a `TimeSeriesInferenceProcess` for autoregressive transient surrogates.
    - The **scalable external-aero variants** deploy through the same process via `model_interface`: `BiStrideMeshGraphNet`'s multiscale U-Net with a scipy-based hierarchy builder replacing an upstream one blocked on `sparse_dot_mkl`, `HybridMeshGraphNet`'s mesh+proximity "world" edges, and `MeshGraphKAN` — validated on a real in-memory lid-driven-cavity FluidDynamics solve.
    - **Temporal training schemes** (`temporal_training`) — single-step, time-conditional and one-shot window datasets sharing the deployment window convention, plus `TrainAutoregressive` backpropagating through a self-fed rollout with optional per-step gradient checkpointing, driven by real transient Kratos solves.
    - The **GraphCast** grid surrogate recipe (`physicsnemo.models.graphcast.GraphCastNet` through `GridInferenceProcess`'s squeeze idiom, batch-size-1 training, a numpy-only shallow-water reference integrator generating the trajectories) is documented and test-pinned, gated on the extra `torch_sparse` (or `dgl`) dependency.

- **Grid-to-grid models and real-solver integration**: a `GridInferenceProcess` deploying FNO/UNet-style same-resolution grid models — 3D and, via the thin-axis squeeze idiom, the 2D operator zoo (`FNO(dimension=2)` and AFNO are test-pinned; 2D UNets and DLWP go through the same layout mechanically but are not covered by a test) plus time-modulated ModAFNO (`model_interface: "modafno"` feeds the solver TIME as the timestep input) and spatiotemporal block operators (`FNO(dimension=4)` through `SequenceInferenceProcess`'s window-as-time-axis mode) — and integration tests/examples driven by **real Kratos solves** — ConvectionDiffusionApplication stationary heat conduction and a StructuralMechanicsApplication cantilever (crash/deformation-surrogate pattern) — conditioned on the compiled applications via the standard `CheckIfApplicationsAvailable` pattern.

- **Non-matching transfer and MPI scale**: a `mapping_bridge` moving fields between the solver mesh and ML grids/meshes that match no tessellation, via MappingApplication's mappers (`nearest_element`, `barycentric`, ..., MPI-capable), with structured background-grid model parts and `(C, D, H, W)` gathering.
    - MPI-aware `DatasetExportProcess` **and** `MeshExportProcess` — ghost-free `DataCommunicator` gathers of fields, and full mesh **topology** reconstructed on a rank-0 shadow part via the reusable `GatherModelPartToRank0` primitive, with rank 0 writing the exact serial file layout.
    - Matched **process groups and device meshes** (`CreateMatchedProcessGroup(s)`, `InitializeDeviceMesh`) pairing physicsnemo subgroups with registered Kratos sub-communicators over the same ranks.
    - **Halo-partitioned distributed graph training** (`graph_partition_utils`) — per-rank subgraphs whose owned sets partition the global node set exactly and whose one-hop neighbourhoods match a serial run, fixing a real interface truncation in the plain graph bridge, plus gloo `DistributedDataParallel` gradient sync asserted bit-identical across ranks.
    - An MPI test suite (`tests/test_PhysicsNeMoApplication_mpi.py`, no Metis required).

- **Sequence, point-cloud and diffusion models**: three deployment families sharing the same gather/scatter contract.
    - **Sequences**: a grid-series exporter (`GridDatasetExportProcess`) with sequence/pair dataset factories feeding `physicsnemo.models.rnn` one-to-many surrogates, deployed by a seed-once/roll-forward `SequenceInferenceProcess`; and a **Virtual Foundry GraphNet** bridge for sintering/AM (`vfgn_bridge` — position-sequence rollout through the public `inference()`, driven by a **real thermo-mechanical sintering solve** (ConvectionDiffusion's `CoupledThermoMechanicalSolver`: cooling-driven thermal contraction — the test asserts the body actually contracts and that the shrinkage tracks the cooling rate, rather than merely running), a training path that routes around the upstream `forward()` being unusable in 2.2, and the normalization-statistics card round trip the particle path had been missing).
    - **Point clouds**: a `PointCloudInferenceProcess` running Transolver-style transformers, **GeoTransolver and FLARE** (the experimental FLARE-attention successors, also reachable through `physicsnemo-cfd`'s checkpoint-driven evaluation wrappers via `cfd_bridge`), FIGConvNet (per-point fields plus its drag-style scalar head) and generic point models directly on the mesh nodes.
    - **Diffusion**: a bridge (`TrainDiffusionModel` with the conditional EDM loss, `DiffusionInferenceProcess`) doing CorrDiff-style downscaling with ensemble-mean predictions and per-node **uncertainty fields** — including the full **CorrDiff two-stage recipe** (`"regression"`/`"residual"` losses on upstream's `RegressionLoss`/`ResidualLoss`, `TrainCorrDiffPair`, and a `"regression_settings"` block on `DiffusionInferenceProcess` adding the regression mean to the generated ensemble) and a test-pinned **FWI-style subsurface inversion** recipe (sparse borehole observations + mask as conditioning, ensemble std as inversion uncertainty) — with **DiT denoisers** (`physicsnemo.models.dit.DiT`) plugging into the same sampler/loss machinery through `denoiser_interface: "dit"` / `WrapDenoiser` (conditioning concatenated into the input channels, sigma broadcast to timesteps).

- **Pretrained DoMINO, correctly de-normalized, and fine-tuning on top of it**: the public ungated `nvidia/domino_drivaerml` checkpoints load straight through `model_registry`, and `DominoInferenceProcess` now applies the de-normalization they need — a pretrained DoMINO emits *dimensionless* fields, so writing its raw output onto Kratos entities is wrong by roughly three orders of magnitude (a raw `0.1386` is really −609 Pa).
    - It guards the three ways such a checkpoint is silently misfed: a `grid_resolution` disagreeing with the checkpoint's own is now named rather than surfacing as an opaque reshape error, and `normalize_coordinates`/`bounding_box_surface` defaults that produce no error at all are warned about.
    - `domino_finetune` ships both adaptation recipes — the predictor-corrector decomposition with the frozen predictor's output cached (so its cost is independent of the predictor's size) and LoRA adapters that merge back into an ordinary `.mdlus` needing no configuration change to deploy.
- **Sharded (FSDP2) checkpoints**: `SaveTrainedModel` gathers an FSDP2 model's DTensors to full tensors and writes them from rank 0, so a sharded model produces one ordinary `.mdlus` that `model_registry` loads. Saving the DTensors directly — what it used to do — yielded a checkpoint that reported success and could not be loaded, and across ranks each wrote only its own shard. Asserted at np=2 and np=3 over gloo, with the parameters confirmed genuinely split (`local != global`) before the save; only the multi-GPU NCCL transport is untested here.
- **Card-carried output de-normalization**: a model trained on normalized targets emits normalized predictions, and writing those onto Kratos variables as if they were physical is wrong by the training scaling — silently, since the values stay finite. The model card's `"output_normalization"` key travels that scaling with the checkpoint and `WriteOutputFields` inverts it, covering `InferenceProcess` and everything on its write path; `ParticleInferenceProcess` reads it too, where it matters most because the prediction is integrated twice into node positions. Spreads are scaled but never shifted, an absent key is exactly the identity, and a degenerate scale or wrong channel count raises rather than producing silent NaNs.
- **GPU ONNX inference**: `"device" : "cuda"`/`"cuda:N"` on a CUDA build of ONNX Runtime, verified against `onnxruntime-gpu` 1.29.0 (a CUDA-13 build) — the CUDA provider genuinely instantiated, GPU and CPU agreeing to ~2.5e-4 relative under TF32. The bridge guards the two failures ONNX Runtime reports as success: a silent fall back to CPU (a missing CUDA build, or a nonexistent device index, both yield a working CPU session — `require_device` turns that into an error) and a dropped device index (`"cuda:1"` previously ran on device 0).
- **Vertex-morphing comparison**: the shipped deformation layer pinned against `ShapeOptimizationApplication`'s `MapperVertexMorphing`, whose reference fields were generated once in a wheel-only environment and committed as a fixture — no rebuild, and no mixing of a GCC-built wheel with this Clang-built core. The two are close relatives, not the same operator: Kratos normalizes by `sum(w)` (an exact partition of unity), physicsnemo by `1 + sum(a)` (a regularized compact Shepard field), so they agree in the dense-control limit and visibly differ when controls are sparse — and `morph` interpolates at a control point where morphing damps.
- **Exact shape gradients and shape optimization**: `sensitivity_utils.ComputeShapeSensitivityField` gives the discretely exact `dJ/dX` at *every* node from **one** pass over the mesh — perturbing each entity's own nodes and re-evaluating only that entity's local right-hand side, since moving a node perturbs only its adjacent entities. That replaces the per-parameter path's `6N` global assemblies with a cost linear in the mesh (measured ~15x faster at 3200 2-D triangles, ~100x at 24k 3-D tetrahedra, independent of the number of design parameters), and `ComputeControlSensitivities` then pushes the field back through the differentiable deformers to give `dJ/d(control)` for FFD/RBF/morph/displace parameterizations — the FEM-exact counterpart of the surrogate-side `ComputeShapeSensitivities`. Notebook 17 drives a gradient descent with it onto a target.
- **Adjoint cross-validation**: the shipped shape sensitivities (`sensitivity_utils` + `differentiable_residual`) are pinned against `StructuralMechanicsApplication`'s entirely separate adjoint stack — `AdjointFiniteDifferencing*` elements, `AdjointNodalDisplacementResponseFunction`, `ResidualBasedAdjointStaticScheme` and `SensitivityBuilder` producing `SHAPE_SENSITIVITY` — with full finite differences through real solves as a third opinion. All three agree to eight significant digits, and the deformation layer's boundary displacement is smoothed into the interior through `MeshMovingApplication` without folding elements.
- **Co-simulation surrogates**: a `cosim_surrogate_solver_wrapper` letting a trained model participate as a *solver* in Kratos co-simulation — referenced by module path in `CoSimulationApplication`'s `ProjectParameters`, exchanging interface fields through its data-transfer layer (MappingApplication mappers included) and convergence accelerators, with the flat `InferenceProcess` contract or any point-cloud interface, optional time ownership, and lazy model loading through the shared model registry (weak and strong Gauss-Seidel coupling both exercised in the tests). `"distributed" : true` runs the surrogate across MPI ranks — Metis-partitioned or, for element-free clouds Metis cannot graph, partition-free — inferring on owned nodes only and synchronizing ghosts, honouring `mpi_settings` so the wrapper can live on a subset of the ranks.

- **CAE datapipes**: a `CaeDatasetExportProcess` writing per-case `.npz` files in the exact layout `physicsnemo.datapipes.cae` consumes (triangulated STL from the mesh bridge, surface/volume fields, DoMINO- and Transolver-style global parameters), with `CreateDoMINODataPipe`/`CreateTransolverDataPipe`/`CreateCaeDataset` factories — DoMINO and Transolver pipelines run on Kratos data out of the box, MPI included. **DoMINO deploys in-loop** through `DominoInferenceProcess`: the current state is exported as a single preprocessed datapipe case per execution, and the model's per-node volume / per-triangle surface predictions are written back onto the Kratos entities (surface triangles collapse onto their parent conditions/elements via the mesh-bridge provenance).

- **Lagrangian particle surrogates**: a `particle_bridge` building proximity graphs (radius/kNN, warp-accelerated with an exact numpy fallback) over particle node clouds in `graph_bridge`'s exact contract, a `ParticleInferenceProcess` running the Learning-to-Simulate loop (velocity history + node-type one-hots in, per-particle acceleration out, semi-implicit Euler advancing and moving the nodes, graph rebuilt every step) for `physicsnemo.models.meshgraphnet.MeshGraphNet`-style models — the process reads any particle node cloud, so `MPMApplication`/`SPHApplication`/`DEMApplication`/`PfemFluidDynamicsApplication` fit the contract, though none of those applications is compiled in the reference environment and the tests drive synthetic clouds, and a `CreateParticleTrajectoryDataset` factory windowing position trajectories into training pairs with normalization statistics.

- **RomApplication interoperability**: `rom_bridge` consumes `CalculateRomBasisOutputProcess`'s numpy POD bases (the exact interleaved row-order contract; no compiled RomApplication needed to consume), a `RomSurrogateProcess` deploys **neural-augmented reduced bases** in the solution loop (case parameters → modal coefficients → full-field reconstruction `u = Φq`, with the model card validating against the basis's nodal unknowns automatically), and `rom_temporal` brings **mesh-reduced temporal attention in ROM space** — physicsnemo's decoder-only `Sequence_Model` trained teacher-forced on reduced trajectories with autoregressive rollout.

- **Physics-informed residuals, PINN solves, exact residual losses and adjoints**: three residual notions.
    1. `solver_residuals.ResidualEvaluator` assembles the real PDE residual of any (ML-predicted) field through the solver's own builder machinery — the cheap non-differentiable *score* feeding query strategies, epoch callbacks and validation.
    2. `physics_informed`: SymPy-defined strong-form residuals evaluated by `physicsnemo.sym`'s `PhysicsInformer` (bundled — no extra install) as real **training loss terms** (`TrainModel(..., extra_loss_terms=[...])`; autodiff at point coordinates, least-squares on mesh/particle graphs, finite-difference/spectral on grids), with builtin diffusion, convection-diffusion, **linear elasticity** and **incompressible Navier–Stokes** PDEs (vector fields auto-split into per-component sympy functions), plus a `PinnSolveProcess` for pure-PINN forward solves from the model part's Dirichlet data and inverse coefficient recovery.
    3. `differentiable_residual`: the **exact discrete residual through the real FEM assembly** as a `torch.autograd.Function` — forward = `BuildRHS`, backward = the consistent tangent's transpose (a matvec, no solve), gradcheck-pinned on real ConvectionDiffusion and StructuralMechanics cases — with `MakeExactResidualLossTerm` making the physics' own verdict a gradient-carrying loss, now covering **transients** too (element-integrated time stepping unchanged; Bossak/Newmark displacement schemes via `scheme=` plus a per-step `InitializeSolutionStep`, assembling the effective tangent K + M(1-alpha)c0 + D c1).

    Alongside them, `sensitivity_utils` provides cheap surrogate `dJ/dx` (autograd through any point-cloud interface) plus **exact adjoint parameter sensitivities** (`Kᵀλ = ∂J/∂u`, one solve for all parameters, validated against full finite differences through real solves).

- **Training utilities and model governance**: a `Parameters`-driven training loop (`TrainModel`) and checkpoint saving (`SaveTrainedModel`, physicsnemo `.mdlus` or TorchScript) with optional **model cards** — JSON sidecars describing a checkpoint's fields, validated (advisorily) by every deployment process — plus autoregressive **rollout evaluation** (`EvaluateRollout`) exposing multi-step error growth of time-series surrogates.

- **Validation, mesh datapipes and distributed alignment**: a `ValidationMetricsProcess` benchmarking predictions against reference fields with `physicsnemo.metrics` (JSON reports), extended by `cfd_bridge` with delegation to the optional `physicsnemo-cfd` package — its domain-aware CFD metric registry (relative-L2, drag/lift, physics residuals, UQ metrics) via the process's `cfd_metrics` block, plus Kratos↔pyvista `Flowfield` conversion and hybrid-initialization blending; a `MeshExportProcess` producing `.pmsh` mesh series directly consumable by `physicsnemo.datapipes.mesh_dataset` for training mesh-based models; a `curator_bridge` turning any Kratos solve into a **`physicsnemo-curator` ETL source** (the sinks ship upstream) with a `CuratorExportProcess` writing AI-ready Zarr stores or VTU grids per step straight from the solution loop; and `distributed_utils` aligning `physicsnemo.distributed.DistributedManager` with Kratos's `DataCommunicator`, with a loud consistency check when the two disagree.

- **Uncertainty and governance**: guardrails, calibrated error bars and probabilistic validation across every deployment process.
    - **Out-of-distribution guardrails** (`ood_guard_utils` bridging `physicsnemo.experimental.guardrails.embedded.OODGuard`) — calibrated on the training inputs by `TrainModel`, saved as a sidecar, checked per inference under an advisory/strict/ignore policy.
    - **Predictive uncertainty for any deployed model** via the `"uncertainty"` block: MC dropout over dropout-like layers including learnable `physicsnemo.nn.ConcreteDropout`, checkpoint ensembles, or a **GP head** giving calibrated posterior variance whose *epistemic* term rises where the surrogate extrapolates — fitted post-training by `FitGpHead`, which implements the inducing-point/auxiliary-MSE/KL-ramp recipe upstream requires but does not ship, and saved as a sidecar since gpytorch modules cannot be TorchScripted. Mean goes to the output fields, per-node std to dedicated uncertainty fields, generalizing the diffusion bridge's pattern.
    - **Calibration metrics** (coverage, NLL, sharpness) answering whether those error bars are honest at all, and **ensemble scoring** (`crps`/`kcrps` over explicitly named members, with `retain_ensemble` keeping the members an inference process would otherwise reduce away).
    - **Probabilistic validation**: `relative_l2`/`weighted_mse`/`weighted_rmse` comparisons, ensemble `crps`/`kcrps` via `physicsnemo.metrics`, and multi-step rollout UQ in `EvaluateRollout` (per-step metric curves and MC-dropout std growth).

- **Active learning**: Kratos as the ground-truth solver in a `physicsnemo.active_learning` loop, via a `LabelStrategy` implementation with two execution backends — an in-process backend for small problems and a subprocess backend (recommended) that keeps Kratos MPI ranks and `torch.distributed` ranks in separate OS processes, fans labeling batches out over `max_parallel_jobs` concurrent solves, and supports HPC job submission. Completed by a query-strategy library (ensemble disagreement, predictive entropy, and **solver-residual scoring** — the real PDE residual assembled by Kratos ranks the surrogate's weak spots), a `MetrologyStrategy` backed by the validation-metrics machinery, and optional strict model-card enforcement at deployment.

  ```python
  import KratosMultiphysics as Kratos
  from KratosMultiphysics.PhysicsNeMoApplication.active_learning.kratos_label_strategy import CreateKratosLabelStrategy
  from KratosMultiphysics.PhysicsNeMoApplication.active_learning.execution_backends.subprocess_backend import SubprocessBackend

  backend = SubprocessBackend(Kratos.Parameters("""{
      "template_directory": "my_case_template",
      "run_command": ["python3", "MainKratos.py"]
  }"""))
  label_strategy = CreateKratosLabelStrategy(backend, provides_fields={"VELOCITY__node_historical"})
  # pass label_strategy to physicsnemo.active_learning.config.StrategiesConfig
  ```

  The case template must run this application's `DatasetExportProcess` so results come back as `.npz` files (a `srun`/`sbatch --wait` prefix in `run_command` turns the subprocess backend into an HPC job submission).

## 🗺️ Roadmap:

Candidate extensions, grouped by theme and ordered roughly by value/feasibility, each naming the concrete PhysicsNeMo API and its Kratos-side counterpart. Every remaining item is `(blocked: …)`, and the parenthesis names the gate — verified against the currently exercised **physicsnemo 2.2.0** and the reference build, not assumed. The gates fall into four kinds: **hardware** (one GPU here), **upstream** (the API does not exist yet, or raises `NotImplementedError`), **external access** (NIM/NGC checkpoints, Omniverse), and **the build** (an application that is not compiled — see the installation note above on why adding one is not a local operation).

- **Reorganize the scripts**
   * Properly classify the scripts, like utils, processes, bridges, ... in the corresponding folders. Otherwise is easy to get lost. 
   * Tests and examples hould be check for proper path.
   * Even better documentation of everything. Is super easy to get lost, so many features and utilities. Specially thinking if someone wants to do something from scratch.

- **Using CuPy insteadof numpy when possible**
   * An idea to be chekced that may enhance performance if CuPy is available.
   * numpy should be always the default fallback just in case.

- **Adjoint integration**
   * In the same manner there is an integration witrh ROM application, an integration with adjoint computation.  

- **Infrastructure and CI**
    * GPU CI for the device-dependent paths (the app builds and runs its torch-free tests in the Linux CI; the ML-dependent tests self-skip there — see `benchmarks/benchmark_bridges.py` for the profiling that settled the former C++-acceleration item. On 196k tets / 32k hexes the nodal gather/scatter paths cost 0.11–0.83 µs per entity, but not everything per-entity is sub-µs: element scatter-back is 2.1, grid sampling 3.5, and provenance construction 4.2 on tets and ~39 on hexes. No custom C++ adaptors are warranted even so — the dominant cost was provenance being rebuilt every step, and caching it in Python removed that)

- **Model architectures and foundation models**
    * Multi-GPU validation of the shipped halo-partitioned graph training — the partitioner, halo exchange and DDP gradient sync ship and are asserted at np=2/3, but only over CPU/gloo: NCCL rejects multiple ranks on one GPU, so the accelerator transport is untested here (blocked: needs a multi-GPU machine)
    * Volumetric (3D) diffusion U-Nets for the diffusion bridge — upstream renamed `physicsnemo.models.diffusion` to `diffusion_unets` and its U-Nets are 2D-image oriented; the shipped DiT interface covers transformers, but a true 3D U-Net denoiser is pending upstream (`physicsnemo.models.topodiff` still pairs naturally with `TopologyOptimizationApplication`)
    * The PyG-based molecular-dynamics / Lennard-Jones GNN for force and potential prediction, pairing with `DEMApplication` particle interactions (blocked: no molecular architecture exists in physicsnemo 2.2 — none of the 25 models under `physicsnemo.models`, nor `experimental.models`/`datapipes`, is molecular; `DEMApplication` is also not compiled here)

- **Mesh, geometry and shape optimization**
    * **Exact** boundary recovery in the tetrahedral fill — `FillSurfaceWithTetrahedra` ships and fills watertight 3D surfaces (upstream first, then a winding-number-carved Delaunay), but it retriangulates planar facets with its own diagonals instead of preserving them, and cannot fill Schönhardt-class solids, which need Steiner points (blocked: `fill_interior`'s `n = 3` still raises `NotImplementedError` in 2.2; a tetgen-enabled MeshingApplication build would also close it)

- **Deployment and serving**
    * A NIM microservice client backend for `InferenceProcess`, calling a running PhysicsNeMo NIM (e.g. DoMINO-Automotive-Aero) over HTTP/gRPC instead of loading a local checkpoint — note the generic Triton HTTP/gRPC transport already ships in `triton_inference_process`, so what is missing is the NIM-specific request schema, not the client (blocked: running a NIM needs an NGC API key and `docker login nvcr.io`; `nvcr.io` returns 401 anonymously and the public hosted catalog carries no CFD model)

- **Scale and distributed**
    * True domain-parallel inference and training with `ShardTensor` (now `physicsnemo.domain_parallel`, not `distributed`), sharding one large Kratos mesh or point cloud across GPUs — distinct from the shipped data-parallel halo partitioning (blocked on **hardware**, not on the release: NCCL rejects two ranks on a single GPU, and forcing a CPU mesh trips physicsnemo's `DistributedManager`, which requires `init_process_group`'s `device_id` to be an accelerator with an index)

- **New physics domains**
    * Exact NURBS geometry sampling for `IgaApplication` meshes, the isogeometric analogue of the shipped curved/isoparametric mode (blocked: needs a NURBS-aware sampler; low demand so far)
    * Digital-twin / Omniverse export of deployed-surrogate predictions for interactive visualization, on top of `HDF5Application`/XDMF output (external tooling; mostly format glue)

**Where to start — the roadmap is now exhausted.** Every bullet above is gated on something outside this repository. Treat the list as a record of what is blocked, not as a queue of work.

*What has shipped.* The formerly "unblocked wins" are all in — GeoTransolver/FLARE, ONNX, `torch.compile`/NVTX, the CoSimulation `solver_wrapper`, the uncertainty-and-governance layer, the grid-operator zoo, DoMINO deployment and the `domino_finetune` predictor-corrector and LoRA recipes, DiT denoisers, the Lagrangian particle surrogate, and the complete physics-informed/differentiability layer: `PhysicsInformer` training terms with elasticity/Navier–Stokes builtins, `PinnSolveProcess`, and — the item long marked *blocked* — the `torch.autograd.Function` through the real FEM assembly plus exact adjoint sensitivities (`differentiable_residual`/`sensitivity_utils`).

The `physicsnemo-curator` ETL bridge has also landed — `curator_bridge` supplies the source side (the sinks ship upstream) and `CuratorExportProcess` writes Zarr/VTU straight from a running solve. Curator stays an optional, git-only dependency whose build pulls a Rust toolchain, so nothing here requires it: without it installed the bridge still imports and only its entry points raise.

That last gate moved without waiting for `kratos/future/`: the classic `CompressedMatrix` already exposes the CSR triple as (value-)zero-copy views consumed by core `scipy_conversion_tools.to_csr`, and the block builder's `Build`/`BuildRHS`/`ResizeAndInitializeVectors` are pybound (the native `CsrMatrix` with `SpMV`/`TransposeSpMV` is the forward-looking path).

So are the rest: the new-physics-domain recipes (GraphCast grid surrogates, test-pinned and gated only on `torch_sparse`/`dgl`; the CorrDiff two-stage regression+residual pipeline; FWI-style diffusion inversion); the mesh-and-data rounds (discrete calculus and grid derivatives, residual-driven MMG adaptive remeshing, coherent dataset augmentation/mixing, and the transient layer — real time-loop solver cases, temporal training schemes with BPTT, and the differentiable residual extended to dynamic schemes); the scalable external-aero GNN variants (bistride multiscale, mesh+world hybrid, KAN) and the Triton serving export; and the distributed extensions of the serial machinery — halo-partitioned graph training and the MPI-distributed co-simulation surrogate. The application now tracks **physicsnemo 2.2**, which unblocked and shipped the differentiable shape-deformation layer, mesh signed distance fields and the grid divergence/curl/Laplacian operators — and whose mesh-calculus gradient-layout flip is absorbed behind the bridge's own stable contract.

*How the gates were checked.* Re-verifying them once network access returned found three stale — the CUDA-13 ONNX wheel now exists, Kratos publishes application wheels that make an offline vertex-morphing reference possible, and the DoMINO checkpoint is public and ungated; all three are reflected above.

*What is blocked, and on what.*

| Item | Gate | Specifically |
|---|---|---|
| `ShardTensor` domain parallelism, FSDP2 round-trips, multi-GPU validation of the shipped halo partitioning, GPU CI | hardware | one GPU here; NCCL rejects two ranks on a single device. Note `ShardTensor` lives in `physicsnemo.domain_parallel`, not `distributed` |
| Exact boundary recovery in the tetrahedral fill (the fill itself ships) | upstream | `fill_interior`'s `n = 3` raises `NotImplementedError` in 2.2 |
| Volumetric (3D) diffusion U-Nets | upstream | `diffusion_unets` are 2D-image oriented |
| Lennard-Jones / molecular GNN | upstream | no molecular architecture exists among the 25 `physicsnemo.models` submodules |
| NIM client backend, Omniverse export | external access | needs an NGC API key and external tooling |
| IGA sampling | the build | `IgaApplication` is not compiled here — see the installation note above on why adding one is not a local operation |

Beyond that, what remains is modeling work rather than engineering: richer crash architectures, VFGN/AM datasets, and the recipes a user brings their own data to.

*Where the real work came from.* The engineering that actually remained was found by auditing the code rather than by reading this list — the de-normalization bug class, of which three instances are now fixed (DoMINO, the shared write path, and the particle path, where the prediction is integrated twice into node positions) and whose remaining exposure is documented per process.

Contributions and prioritization requests are welcome — please open an issue on the Kratos repository mentioning `PhysicsNeMoApplication`.

## ⚠️ Dependency policy — read before contributing:

`torch` and `nvidia-physicsnemo` (and the extras: `nvidia-physicsnemo-cfd` with `pyvista`, `torch_geometric`, ...) are **optional, pure-Python runtime dependencies**:

- `import KratosMultiphysics.PhysicsNeMoApplication` always succeeds without them; only the specific submodules that need them fail — lazily, at call time, with an actionable error message.
- There is intentionally **no CMake gating** (no `find_package(Torch)`, no `USE_PHYSICSNEMO` option): nothing in the C++ core links against torch or CUDA. Do not add such a gate.
- New Python modules must keep all `import torch` / `import physicsnemo` statements inside lazy `_TryImport*()` helpers, never at module scope of anything imported eagerly.

**Packaging policy** (wheels / the standard `KratosMultiphysics` distribution): the application ships like any other per-application Kratos wheel — `scripts/wheels/build_wheel.py` derives the app wheels automatically from whatever was compiled, so no wheel-side registration exists or is needed. `torch` / `nvidia-physicsnemo` are **never** declared as wheel dependencies (they are large, CUDA-variant-specific, and the user's choice of build); they remain the optional runtime `pip install`s above, and the lazy-import contract (`tests/test_import_contract.py`) is exactly what guarantees a wheel installed without them stays fully importable.

## ⚙️ Installation:

Compile like any other Kratos application (add `add_app ${KRATOS_APP_DIR}/PhysicsNeMoApplication` to your configure script). To use the ML features additionally install the runtime dependencies:

```bash
pip install torch             # CPU or CUDA build, your choice
pip install nvidia-physicsnemo
pip install git+https://github.com/NVIDIA/physicsnemo-cfd   # optional: hybrid-initialization recipes + CFD metric registry (source only - not on PyPI)
```

> **Before adding applications to an existing build — a trap worth knowing.** Kratos reads `KRATOS_APPLICATIONS` from the *environment* and never writes it to the CMake cache, and `make install` never prunes the install tree. An install directory therefore accumulates every application ever built, while the build graph covers only the ones in the current configure script — and those two sets drift apart silently. Applications outside the graph are not rebuilt, so the next `libKratosCore.so` relink leaves them ABI-stale: they keep *existing*, and `CheckIfApplicationsAvailable` keeps reporting them as present, but importing them fails with an undefined symbol or segfaults. Because several Kratos applications import their optional companions on a *presence* check rather than a working one, one stale library can take down an application that was never touched.
>
> So: reconfiguring to add an application is not a local operation. Restore the **full** historical application list in the same run, and treat the install tree as unreproducible from the cache alone — nothing records what produced it.

## ⚙️ Examples:

Eighteen example notebooks covering the tensor bridge, mesh bridge, surrogate training/deployment, active learning, superresolution, transient rollouts, exact shape gradients and DoMINO fine-tuning can be found in [`examples/notebooks/`](examples/notebooks/).

They are **executed** by `tests/test_notebooks.py`, one test per notebook, so a
changed signature or return type breaks a test rather than rotting silently. Each
runs in a throwaway copy of the tree, because the notebooks write artifacts next to themselves and two of them load solver cases from `tests/kratos_solver_cases`. At roughly five minutes for all eighteen they sit on the **validation** suite, so they run only when that suite is asked for — not in CI, and not nightly.

Executing them proves no exception was raised, which is weaker than proving the right answer: the five notebooks making a reproducible numeric claim (03, 05, 07, 17 and 18) assert it, and the unseeded ones deliberately assert nothing.

### Worked examples

Eighteen fully documented use cases live in the [Examples repository](https://github.com/KratosMultiphysics/Examples/tree/master/physics_nemo_application): self-contained scripts against real Kratos solves, with every figure regenerated by the code. Highlights (each image links to its case):

**[Thermal surrogate lifecycle](https://github.com/KratosMultiphysics/Examples/tree/master/physics_nemo_application/use_cases/thermal_surrogate_lifecycle)** — dataset from real solves, FNO ensemble with model cards and an OOD guard, governed in-loop deployment, validation, ONNX distillation and a Triton serving layout:

<p align="center">
  <a href="https://github.com/KratosMultiphysics/Examples/tree/master/physics_nemo_application/use_cases/thermal_surrogate_lifecycle"><img src="https://raw.githubusercontent.com/KratosMultiphysics/Examples/master/physics_nemo_application/use_cases/thermal_surrogate_lifecycle/data/deployment_fields.png" alt="Solver vs FNO ensemble, error vs predicted error bar." width="560"/></a>
</p>

**[Transient thermo-mechanical surrogate](https://github.com/KratosMultiphysics/Examples/tree/master/physics_nemo_application/use_cases/transient_thermomechanical_surrogate)** — coupled cooling-contraction ground truth, single-step vs BPTT training, self-fed rollout on an unseen cooling schedule:

<p align="center">
  <a href="https://github.com/KratosMultiphysics/Examples/tree/master/physics_nemo_application/use_cases/transient_thermomechanical_surrogate"><img src="https://raw.githubusercontent.com/KratosMultiphysics/Examples/master/physics_nemo_application/use_cases/transient_thermomechanical_surrogate/data/rollout.gif" alt="Coupled solver vs surrogate rollout." width="620"/></a>
</p>

**[GNN + exact shape optimization](https://github.com/KratosMultiphysics/Examples/tree/master/physics_nemo_application/use_cases/gnn_and_exact_shape_optimization)** — MeshGraphNet on the mesh's own graph, exact adjoint shape gradients verified against re-solved finite differences to ~1e-10, an FFD optimization hitting its target exactly:

<p align="center">
  <a href="https://github.com/KratosMultiphysics/Examples/tree/master/physics_nemo_application/use_cases/gnn_and_exact_shape_optimization"><img src="https://raw.githubusercontent.com/KratosMultiphysics/Examples/master/physics_nemo_application/use_cases/gnn_and_exact_shape_optimization/data/shape_optimization.png" alt="Objective convergence and optimized domain." width="700"/></a>
</p>

**[Superresolution + CorrDiff diffusion](https://github.com/KratosMultiphysics/Examples/tree/master/physics_nemo_application/use_cases/superresolution_and_diffusion)** — two-stage diffusion downscaling of genuinely under-resolved solves, with calibrated ensemble uncertainty:

<p align="center">
  <a href="https://github.com/KratosMultiphysics/Examples/tree/master/physics_nemo_application/use_cases/superresolution_and_diffusion"><img src="https://raw.githubusercontent.com/KratosMultiphysics/Examples/master/physics_nemo_application/use_cases/superresolution_and_diffusion/data/corrdiff_downscaling.png" alt="Coarse condition, regression mean, truth and ensemble spread." width="760"/></a>
</p>

**[Transient in-loop superresolution](https://github.com/KratosMultiphysics/Examples/tree/master/physics_nemo_application/use_cases/transient_superresolution)** — the process attached to a running transient analysis, upscaling every step; the learned upsampler halves bilinear interpolation's error:

<p align="center">
  <a href="https://github.com/KratosMultiphysics/Examples/tree/master/physics_nemo_application/use_cases/transient_superresolution"><img src="https://raw.githubusercontent.com/KratosMultiphysics/Examples/master/physics_nemo_application/use_cases/transient_superresolution/data/transient_sr.gif" alt="Coarse, superresolved and fine transient fields." width="700"/></a>
</p>

**[Physics-refined superresolution](https://github.com/KratosMultiphysics/Examples/tree/master/physics_nemo_application/use_cases/physics_informed_superresolution)** — the differentiable PDE residual grades the upsampler's output and refines it below the fine solve's own residual:

<p align="center">
  <a href="https://github.com/KratosMultiphysics/Examples/tree/master/physics_nemo_application/use_cases/physics_informed_superresolution"><img src="https://raw.githubusercontent.com/KratosMultiphysics/Examples/master/physics_nemo_application/use_cases/physics_informed_superresolution/data/pisr_metrics.png" alt="RMSE and PDE residual per field." width="640"/></a>
</p>

**[Lid-driven cavity MeshGraphNet](https://github.com/KratosMultiphysics/Examples/tree/master/physics_nemo_application/use_cases/fluid_cavity_gnn)** — the first fluid case: Navier–Stokes cavity solves, a GNN propagating boundary conditions through the mesh graph to the recirculating flow it never sees in its input:

<p align="center">
  <a href="https://github.com/KratosMultiphysics/Examples/tree/master/physics_nemo_application/use_cases/fluid_cavity_gnn"><img src="https://raw.githubusercontent.com/KratosMultiphysics/Examples/master/physics_nemo_application/use_cases/fluid_cavity_gnn/data/cavity_fields.png" alt="VMS solver vs MeshGraphNet cavity flow." width="760"/></a>
</p>

**[Surrogate-driven adaptive remeshing](https://github.com/KratosMultiphysics/Examples/tree/master/physics_nemo_application/use_cases/surrogate_driven_remeshing)** — the residual-scoring → MMG loop: elements concentrate exactly where the surrogate's physics error lives, with no reference solution needed:

<p align="center">
  <a href="https://github.com/KratosMultiphysics/Examples/tree/master/physics_nemo_application/use_cases/surrogate_driven_remeshing"><img src="https://raw.githubusercontent.com/KratosMultiphysics/Examples/master/physics_nemo_application/use_cases/surrogate_driven_remeshing/data/remeshing.png" alt="Surrogate error, driving residual, adapted mesh." width="760"/></a>
</p>

**[Lagrangian particle surrogate](https://github.com/KratosMultiphysics/Examples/tree/master/physics_nemo_application/use_cases/lagrangian_particle_surrogate)** — meshless: a learned acceleration integrated into particle positions by the process, de-normalized by the model card:

<p align="center">
  <a href="https://github.com/KratosMultiphysics/Examples/tree/master/physics_nemo_application/use_cases/lagrangian_particle_surrogate"><img src="https://raw.githubusercontent.com/KratosMultiphysics/Examples/master/physics_nemo_application/use_cases/lagrangian_particle_surrogate/data/particles.gif" alt="Falling particle cloud, surrogate vs closed form." width="480"/></a>
</p>

**[PINN forward + inverse](https://github.com/KratosMultiphysics/Examples/tree/master/physics_nemo_application/use_cases/pinn_forward_and_inverse)** — mesh-free solves against the FEM reference and inverse conductivity recovery from field observations:

<p align="center">
  <a href="https://github.com/KratosMultiphysics/Examples/tree/master/physics_nemo_application/use_cases/pinn_forward_and_inverse"><img src="https://raw.githubusercontent.com/KratosMultiphysics/Examples/master/physics_nemo_application/use_cases/pinn_forward_and_inverse/data/pinn_inverse.png" alt="PINN training and recovered conductivity." width="640"/></a>
</p>

**[Animated adaptive remeshing](https://github.com/KratosMultiphysics/Examples/tree/master/physics_nemo_application/use_cases/animated_adaptive_remeshing)** — the mesh chases a moving heat source (surrogate-residual MMG adaptation), recorded by core Kratos' new `TransientPlotter` with the topology changing every frame:

<p align="center">
  <a href="https://github.com/KratosMultiphysics/Examples/tree/master/physics_nemo_application/use_cases/animated_adaptive_remeshing"><img src="https://raw.githubusercontent.com/KratosMultiphysics/Examples/master/physics_nemo_application/use_cases/animated_adaptive_remeshing/data/chasing_refinement.gif" alt="The adapted mesh following the moving source." width="520"/></a>
</p>

**[The sintering solve, recorded live](https://github.com/KratosMultiphysics/Examples/tree/master/physics_nemo_application/use_cases/transient_thermomechanical_surrogate)** — the coupled thermo-mechanical transient rendered by `PyVistaAnimationOutputProcess` attached to the running analysis (temperature-colored, displacement-warped):

<p align="center">
  <a href="https://github.com/KratosMultiphysics/Examples/tree/master/physics_nemo_application/use_cases/transient_thermomechanical_surrogate"><img src="https://raw.githubusercontent.com/KratosMultiphysics/Examples/master/physics_nemo_application/use_cases/transient_thermomechanical_surrogate/data/sintering_solve.gif" alt="The cooling, shrinking body recorded by the animation output process." width="440"/></a>
</p>

**[Co-simulation surrogate](https://github.com/KratosMultiphysics/Examples/tree/master/physics_nemo_application/use_cases/cosim_surrogate)** — a trained checkpoint as a first-class CoSimulation solver, Aitken-accelerated to the learned operator's fixed point; **[uncertainty and trust](https://github.com/KratosMultiphysics/Examples/tree/master/physics_nemo_application/use_cases/uncertainty_and_trust)** — spread vs actual error, calibration catching over-confident bars, the OOD guard; **[active learning](https://github.com/KratosMultiphysics/Examples/tree/master/physics_nemo_application/use_cases/active_learning)** — Kratos as the labeler, with its honest result:

<p align="center">
  <a href="https://github.com/KratosMultiphysics/Examples/tree/master/physics_nemo_application/use_cases/uncertainty_and_trust"><img src="https://raw.githubusercontent.com/KratosMultiphysics/Examples/master/physics_nemo_application/use_cases/uncertainty_and_trust/data/uncertainty_sweep.png" alt="Ensemble spread vs actual error across the sweep." width="620"/></a>
</p>


**[Hybrid initialization](https://github.com/KratosMultiphysics/Examples/tree/master/physics_nemo_application/use_cases/hybrid_initialization)** — the surrogate accelerating the solver itself: `HybridInitializationProcess` warm-starts Newton–Raphson on a geometrically nonlinear cantilever (25 % of the iterations saved, identical solutions), with the measured lesson that Newton rewards *smooth* seed accuracy — a rough seed closer to the solution loses to a cold start:

<p align="center">
  <a href="https://github.com/KratosMultiphysics/Examples/tree/master/physics_nemo_application/use_cases/hybrid_initialization"><img src="https://raw.githubusercontent.com/KratosMultiphysics/Examples/master/physics_nemo_application/use_cases/hybrid_initialization/data/newton_acceleration.png" alt="Newton iterations cold vs warm-started, and residual traces." width="640"/></a>
</p>

**[Implicit geometry + SDF surrogates](https://github.com/KratosMultiphysics/Examples/tree/master/physics_nemo_application/use_cases/implicit_geometry_sdf_surrogate)** — meshes generated straight from signed distance functions (`mesh_bridge.generate`), solved by ConvectionDiffusion, and learned through a ladder of autograd-computed SDF encodings that generalizes across shapes:

<p align="center">
  <a href="https://github.com/KratosMultiphysics/Examples/tree/master/physics_nemo_application/use_cases/implicit_geometry_sdf_surrogate"><img src="https://raw.githubusercontent.com/KratosMultiphysics/Examples/master/physics_nemo_application/use_cases/implicit_geometry_sdf_surrogate/data/generated_family.png" alt="Six SDF-generated geometries with their solved fields." width="620"/></a>
</p>

**[Sparse-sensor field inversion](https://github.com/KratosMultiphysics/Examples/tree/master/physics_nemo_application/use_cases/sparse_sensor_inversion)** — the FWI-style recipe: a mask-conditioned diffusion model reconstructs the full temperature field from ten sensor readings, ensemble std as the inversion uncertainty; **[ROM-space temporal surrogate](https://github.com/KratosMultiphysics/Examples/tree/master/physics_nemo_application/use_cases/rom_temporal_surrogate)** — RomApplication's `CalculateRomBasisOutputProcess` writes the POD basis, physicsnemo's temporal attention learns the dynamics in eight coefficients:

<p align="center">
  <a href="https://github.com/KratosMultiphysics/Examples/tree/master/physics_nemo_application/use_cases/sparse_sensor_inversion"><img src="https://raw.githubusercontent.com/KratosMultiphysics/Examples/master/physics_nemo_application/use_cases/sparse_sensor_inversion/data/sensor_inversion.png" alt="Field inversion from ten sensors with ensemble uncertainty." width="700"/></a>
</p>
<p align="center">
  <a href="https://github.com/KratosMultiphysics/Examples/tree/master/physics_nemo_application/use_cases/rom_temporal_surrogate"><img src="https://raw.githubusercontent.com/KratosMultiphysics/Examples/master/physics_nemo_application/use_cases/rom_temporal_surrogate/data/coefficient_rollout.png" alt="Reduced coordinates, truth vs rollout at an unseen conductivity." width="700"/></a>
</p>

## 🗎 Documentation:

User documentation is available in the [Kratos documentation pages](https://kratosmultiphysics.github.io/Kratos/pages/Applications/PhysicsNeMo_Application/General/Overview.html) (sources under `docs/pages/Applications/PhysicsNeMo_Application/`). Further information regarding NVIDIA PhysicsNeMo can be found in the [official documentation](https://docs.nvidia.com/physicsnemo/latest/index.html).

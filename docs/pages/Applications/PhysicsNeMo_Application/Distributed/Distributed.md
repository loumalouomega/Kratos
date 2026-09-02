---
title: Distributed
keywords: distributed torch mpi data communicator
tags: [Distributed.md]
sidebar: physicsnemo_application
summary: 
---

# Aligning torch.distributed with Kratos MPI

`physicsnemo.distributed.DistributedManager` and Kratos's `DataCommunicator` are two independent views of the same set of processes. When a single job needs both — e.g. distributed inference inside an MPI-parallel analysis — they must agree on rank and world size, or data ends up exchanged between the wrong processes.

`distributed_utils.InitializeDistributedManager` initializes the torch/physicsnemo side consistently with the Kratos side:

```python
from KratosMultiphysics.PhysicsNeMoApplication.distributed.distributed_utils import InitializeDistributedManager

manager = InitializeDistributedManager()      # backend defaults to "gloo"; pass "nccl" for GPU jobs
print(manager.rank, manager.world_size, manager.device)
```

Behavior:

- **Serial Kratos run**: explicit single-process setup (`rank 0 / world_size 1`).
- **Distributed run under a recognized launcher** (torch env vars, SLURM, OpenMPI — detected via `RANK`/`SLURM_PROCID`/`OMPI_COMM_WORLD_RANK`): physicsnemo's own `DistributedManager.initialize()` runs, followed by a **consistency check** against the `DataCommunicator`. A mismatch raises a `RuntimeError` explaining that torch was initialized from a different launcher environment than the MPI ranks Kratos runs under — the classic silent-corruption scenario, caught loudly instead.
- **Distributed run without those environment variables**: explicit `DistributedManager.setup(rank=Rank(), world_size=Size(), addr=..., port=...)`; the rendezvous `addr`/`port` must be reachable by all ranks.
- Already initialized: only the consistency check runs (idempotent).

Manual verification on a workstation (CPU, two ranks):

```bash
mpiexec -n 2 python3 -c "
import KratosMultiphysics
from KratosMultiphysics.PhysicsNeMoApplication.distributed.distributed_utils import InitializeDistributedManager
m = InitializeDistributedManager(backend='gloo')
print('rank', m.rank, 'of', m.world_size)"
```

## Scope

This covers the "one job, both worlds" case. For **active learning**, keeping training and solving in separate OS processes — the `SubprocessBackend` of the [Active Learning](../Active_Learning/Active_Learning.html) page — remains the recommended architecture: it avoids this alignment problem entirely.

## MPI-aware dataset export

`DatasetExportProcess` is MPI-aware: on a distributed model part it gathers the ghost-free per-rank field blocks through `DataCommunicator.Gatherv` and rank 0 writes **one** `.npz` file with the entities **sorted by global id** — the exact layout a serial run produces, so `CreateNpzDataset` and the active-learning harvesting are rank-count-agnostic. Serial runs are untouched.

The underlying helper is public:

```python
from KratosMultiphysics.PhysicsNeMoApplication.distributed import distributed_utils
ids, values = distributed_utils.GatherFieldToRank0(model_part, "VELOCITY", "node_historical")
# rank 0: (n_global,) int64 ids ascending + (n_global, *entity_shape) float64
# other ranks: (None, None)
```

Ghost entities are excluded (communicator `LocalMesh` containers — every entity is owned by exactly one rank), so nothing is duplicated.

## MPI-aware mesh topology export

`MeshExportProcess` is MPI-aware too: on a distributed model part the whole topology is gathered onto rank 0 and the ordinary serial tessellation/export runs on a reconstructed "shadow" part, writing a `.pmsh` identical to a serial run's. The primitives are public:

```python
mesh = distributed_utils.GatherMeshToRank0(model_part)          # nodes + entities
# GatheredMesh: node_ids/coordinates/entity_ids/geometry_codes/connectivity, id-sorted

gathered = distributed_utils.GatherModelPartToRank0(
    model_part, [("PRESSURE", "node_historical")], "Elements")
# rank 0: gathered.model_part is a serial shadow part (owned by gathered.model -
# keep it alive), entities recreated with generic registered names per geometry
# type, fields attached; other ranks: model_part is None. Serial parts pass
# through unchanged. Gauss-point specs are gathered, mean-collapsed and
# re-labeled "element"/"condition" (gathered.field_specs holds the effective
# specs), matching the serial mesh-bridge collection exactly.
```

`GatherModelPartToRank0` is the general "run serial-only machinery on distributed data" primitive — the CAE datapipe exporter uses it for its surface pipeline. All gathers are collective: call them on **every** rank, never inside a rank guard.

## Shared process groups and device meshes

Beyond the rank-consistency alignment, `distributed_utils` mirrors Kratos communicators into physicsnemo's process groups and device meshes (requires an initialized `DistributedManager`; helpers raise otherwise):

```python
distributed_utils.InitializeDistributedManager(backend="gloo")   # or nccl on GPU jobs

# a physicsnemo subgroup AND a registered Kratos sub-communicator over the SAME ranks
sub_dc = distributed_utils.CreateMatchedProcessGroup("model_parallel", size=2)
# -> DistributedManager().group("model_parallel") on the torch side,
#    ParallelEnvironment.GetDataCommunicator("model_parallel") on the Kratos side,
#    consistency-checked against each other

groups = distributed_utils.CreateMatchedProcessGroups(Kratos.Parameters("""{
    "process_groups" : [ { "name" : "model_parallel", "size" : 2 },
                         { "name" : "mp_inner", "size" : 1, "parent" : "model_parallel" } ]
}"""))

mesh = distributed_utils.InitializeDeviceMesh((-1, 2), ("data", "model"))
# validated against the Kratos communicator size before touching torch
```

Group sizes must divide the parent group size (physicsnemo builds blocks of consecutive ranks). On serial runs only the torch-side group is created and the parent communicator is returned unchanged, so the same script runs serially.

## The MPI test suite

`tests/test_PhysicsNeMoApplication_mpi.py` assembles the `mpi_small` / `mpi_nightly` / `mpi_all` suites. Most of it needs no Metis: each rank builds its own slab of a structured mesh (with consistent `PARTITION_INDEX` values) and `ParallelFillCommunicator` wires the communication meshes. The distributed co-simulation tests are the exception — they exercise the Metis import path and self-skip when `MetisApplication` is not compiled. Run the suite with:

```bash
OMP_NUM_THREADS=1 mpiexec -np 2 python3 tests/test_PhysicsNeMoApplication_mpi.py --using-mpi
```

Everything is asserted at both `np=2` and `np=3`; the co-simulation subgroup case needs at least three ranks and self-skips below that.

## Halo-partitioned graph training

`graph_bridge.BuildGraph` is correct on a serial model part and quietly wrong on a distributed one, in two ways that both show up at partition interfaces:

- **rank subgraphs overlap** — each rank's `model_part.Nodes` is local *plus* ghost, so interface nodes and their edges appear on both sides and the per-rank node counts sum to more than the global count;
- **neighbourhoods are truncated** — a rank holds only its own elements, so an interface node is missing the edges contributed by elements on the far side. Measured against a serial reference, a one-hop aggregation is wrong at every interface node **including at nodes the rank owns**.

`graph_partition_utils.BuildHaloSubgraph` fixes both: each rank takes the nodes it owns plus a halo of the elements touching them. Owned sets then partition the global node set exactly, and a one-hop aggregation matches the serial reference at every owned node — both asserted in the MPI suite at np=2 and np=3, with today's `BuildGraph` kept as the negative control. An L-layer message-passing network needs L halo rings.

Two rules the implementation encodes because the obvious alternatives fail:

- **Edge ownership follows element ownership, never `PARTITION_INDEX`.** Elements are uniquely owned; nodes are not. A `min(PARTITION_INDEX)` rule silently *drops* edges whose endpoints are owned by different ranks, because such an edge may not exist on the lower-ranked owner at all.
- **Kratos provides no element halo** (`GhostMesh()` carries ghost nodes but zero ghost elements), and a halo deeper than one ring reaches nodes outside the ghost layer entirely — so connectivity *and* node coordinates/features are exchanged explicitly. The shipped exchange is a whole-communicator all-gather: exact and simple; a production-scale version would trade only with `NeighbourIndices()` colours.

`InitializeTorchProcessGroup` sets up `torch.distributed` from Kratos's rank/size, deliberately bypassing physicsnemo's `DistributedManager` — that helper passes a CUDA `device_id` to `init_process_group`, which requires an accelerator per rank and therefore fails on a single-GPU machine. Going direct works with gloo on CPU and coexists with a live Kratos MPI run in the same process. `WrapForDataParallel` then wraps the model in `DistributedDataParallel`; the MPI suite asserts the gradients come out bit-identical across ranks after a backward over per-rank subgraphs.

`CheckPartitionSafeInterface` refuses the two graph interfaces that cannot be partitioned: `"bistride"` builds its coarse levels from a global connected-components pass and a geometric-centre seed, and `"hybrid"` adds proximity edges through a partition-blind radius search, so per-rank subgraphs would silently differ from the serial graph. Use `"meshgraphnet"` or `"meshgraphkan"` for distributed runs.

**What this is and is not.** This is **data parallelism** — every rank trains the same model on its own subgraph and gradients are averaged. It is *not* domain-parallel sharding of one tensor's activations across devices; that is physicsnemo's `ShardTensor` story, and it ships separately in `domain_parallel_utils` (next section). Both are asserted over CPU/gloo only: the mathematics is proven, the NCCL transport is not, and two or three ranks on one box measures correctness rather than throughput.

## Domain parallelism with `ShardTensor`

`distributed.domain_parallel_utils` puts ONE tensor across the ranks — each holds the rows it owns, or its slab of a grid — and lets physicsnemo's registered handlers make the model's own ops mesh-aware: a pointwise layer runs on the local rows, a convolution exchanges the halo it needs, and a reduction becomes a mesh-wide quantity.

```python
from KratosMultiphysics.PhysicsNeMoApplication.distributed import domain_parallel_utils

mesh = domain_parallel_utils.InitializeDomainMesh()            # once per program
shard, owned_ids = domain_parallel_utils.ShardKratosField(
    model_part, [("PRESSURE", "node_historical")], mesh)       # owned rows, ghost-free
prediction = model(shard)                                      # a ShardTensor
loss = (prediction - target).square().mean()                   # mesh-wide
loss.backward()                                                # full gradient on EVERY rank
domain_parallel_utils.WriteShardedField(
    prediction, model_part, [("TEMPERATURE", "node_non_historical")])   # owners, then ghosts
```

Rules the module encodes, each pinned by the MPI suite at np=2 and np=3:

- **Uneven shards are the normal case.** A Kratos partition owns however many nodes it owns, so per-rank shapes are exchanged with `all_gather_object` and handed to `from_local` explicitly — upstream's `sharding_shapes="infer"` does that exchange on CUDA unconditionally.
- **The mesh is built with torch, not with `DistributedManager`.** `initialize_mesh` puts the mesh on CUDA whenever a GPU is visible, even under gloo; `InitializeDomainMesh` goes to `init_device_mesh("cpu", ...)` directly, over `InitializeTorchProcessGroup`. Build it once: a mesh creates sub-groups, and creating them repeatedly while other groups are live deadlocks.
- **Do not all-reduce the gradients.** The backward of a mean over a `ShardTensor` already delivers the serial gradient on every rank (measured to 3e-8); summing again would be wrong by the rank count. `float(loss)` reads the rank's *partial* — `MeshWideValue` returns the global value.
- **Grids split by `torch.chunk`** (`ShardGridAlongAxis`, replicated input in, slab out), which is the layout the halo convolution and pooling handlers reassemble exactly: a 3×3 convolution on a W-sharded grid matches the serial result to machine precision.

CUDA-only upstream, and therefore not exercised here: `scatter_tensor` (needs the `DistributedManager`), `sharding_shapes="infer"`, ring attention, the sharded kNN/radius search, and the NCCL transport itself.

---
title: Distributed and scale
keywords: physicsnemo distributed manager shard tensor domain parallel mpi
tags: [Distributed_And_Scale.md]
sidebar: physicsnemo_application
summary: DistributedManager, process groups, ShardTensor - and which of them a one-GPU machine can actually run.
---

# Distributed and scale

Two different subpackages, two different kinds of parallelism, and confusing
them is the usual mistake.

## `physicsnemo.distributed` — data parallelism

Every rank holds the **whole model** and a **different slice of the data**.
Gradients are averaged across ranks; the model is replicated.

- `DistributedManager` — the singleton that owns rank, world size, device and
  the backend. Everything else asks it.
- `ProcessGroupConfig`, `ProcessGroupNode` — subgroups of ranks, for models that
  want more than one axis of parallelism.
- Collectives with autograd support: `gather_v`, `all_gather_v`, `scatter_v`,
  `indexed_all_to_all_v`, `reduce_loss`, `fused_all_reduce`.
- `mark_module_as_shared` — tells the gradient machinery a module's parameters
  are replicated rather than sharded.
- `distributed.fft` — distributed FFTs, for spectral models.

### The alignment problem

Kratos has its own communicator abstraction (`DataCommunicator`), and torch has
`torch.distributed`. If they disagree about which rank you are, the failure is
silent and the results are garbage.

`distributed.distributed_utils` aligns the two and **checks loudly** when they
disagree. `CreateMatchedProcessGroup(s)` and `InitializeDeviceMesh` pair
physicsnemo subgroups with registered Kratos sub-communicators over the same
ranks.

### Halo partitioning

Splitting a mesh graph across ranks naively truncates the interfaces: a node
near a partition boundary loses the neighbours that live on another rank, so its
one-hop neighbourhood no longer matches a serial run and the message-passing
result is wrong at exactly the places that matter.

`distributed.graph_partition_utils` builds per-rank subgraphs whose owned sets
partition the global node set exactly and whose one-hop neighbourhoods do match,
then syncs gradients with `DistributedDataParallel` — asserted bit-identical
across ranks over gloo.

## `physicsnemo.domain_parallel` — domain parallelism

Every rank holds **part of one tensor**. The mesh itself is split, not the batch.
This is what you want when a single sample does not fit on one GPU.

- `ShardTensor` — a tensor sharded across a device mesh, with `shard_tensor`,
  `scatter_tensor` and `sync_module_over_mesh`.

Note it moved: `ShardTensor` lives in `physicsnemo.domain_parallel`, **not** in
`physicsnemo.distributed`, as of 2.2.

### Why this application does not ship it

Not an upstream gap — hardware. NCCL rejects two ranks on a single GPU, and
forcing a CPU mesh trips `DistributedManager`, which requires
`init_process_group`'s `device_id` to be an accelerator with an index. There is
one GPU on the reference machine, so the path cannot be exercised, let alone
asserted. It stays on the roadmap for that reason and no other.

## Sharded checkpoints

An FSDP2 model's parameters are `DTensor`s — each rank holds a shard. Saving
them directly produces a checkpoint that reports success and cannot be loaded,
with each rank having written only its own piece.

`training.training_utils.SaveTrainedModel` gathers them to full tensors and
writes from rank 0, so a sharded model yields one ordinary `.mdlus`. Asserted at
np=2 and np=3 over gloo, with the parameters confirmed genuinely split
(`local != global`) before the save. Only the multi-GPU NCCL transport is
untested here.

## What this application uses it for

| PhysicsNeMo API | Kratos-side module | Gives you |
|---|---|---|
| `DistributedManager` | `distributed.distributed_utils` | alignment with Kratos's `DataCommunicator`, with a loud check |
| `ProcessGroupConfig` / `ProcessGroupNode` | `distributed.distributed_utils` | `CreateMatchedProcessGroup(s)`, `InitializeDeviceMesh` |
| `DistributedDataParallel` (torch) | `distributed.graph_partition_utils` | halo-partitioned graph training |
| FSDP2 `DTensor` | `training.training_utils` | one loadable `.mdlus` from a sharded model |
| `domain_parallel.ShardTensor` | — | not shipped; blocked on hardware |

MPI-aware export also lives here in spirit: `processes.export.dataset_export_process`
and `processes.export.mesh_export_process` do ghost-free gathers and reconstruct
full mesh **topology** on a rank-0 shadow part via `GatherModelPartToRank0`,
with rank 0 writing the exact serial file layout.

See [Distributed](../Distributed/Distributed.html).

Next: [Companion packages](Companion_Packages.html).

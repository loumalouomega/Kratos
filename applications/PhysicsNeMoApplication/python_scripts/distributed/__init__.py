"""Running across MPI ranks.

``distributed_utils``
    Aligns ``physicsnemo.distributed.DistributedManager`` with Kratos's
    ``DataCommunicator`` (with a loud check when the two disagree), matched
    process groups and device meshes, and ``GatherModelPartToRank0`` - the
    primitive the MPI-aware export processes reconstruct topology with.
``graph_partition_utils``
    Halo-partitioned graphs for distributed graph training: per-rank subgraphs
    whose owned sets partition the global node set exactly and whose one-hop
    neighbourhoods match a serial run.

This is *data* parallelism. Domain parallelism over ``ShardTensor``
(``physicsnemo.domain_parallel``) is not shipped - see the roadmap.
"""

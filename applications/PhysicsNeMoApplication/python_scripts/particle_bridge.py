"""Particle-graph construction for Lagrangian (Learning-to-Simulate) surrogates.

Particle methods (MPM, SPH, DEM, PFEM) have no persistent element-edge
graph - connectivity is proximity, rebuilt every step. BuildParticleGraph
connects a model part's nodes by radius or kNN and returns exactly
graph_bridge.BuildGraph's contract, so ScatterNodeFeatures, ToPyGGraph and
every downstream idiom work unchanged:

    (node_features (N, F) float64,
     edge_index    (2, E) int64, bidirectional,
     edge_features (E, 4) float64: relative position + distance,
     node_ids      (N,)  int64)

Neighbor search runs through physicsnemo.nn.functional (warp-backed
radius_search/knn) when available and falls back to an exact numpy
brute-force path otherwise ("backend": "numpy" forces it - also the
reference the accelerated path is tested against).

BuildKinematicFeatures assembles the standard Learning-to-Simulate node
features: the last K velocity states from the historical buffer, oldest
first (matching the TimeSeriesInferenceProcess window convention).

torch/physicsnemo stay optional; the numpy path needs neither.
"""

import numpy

import KratosMultiphysics as Kratos
from KratosMultiphysics.PhysicsNeMoApplication.utilities.tensor_adaptor_dataset_utils import GetTensorAdaptor

_CONNECTIVITY_TYPES = ("radius", "knn")
_BACKENDS = ("auto", "numpy")


def _TryImportNeighborSearch():
    try:
        from physicsnemo.nn.functional import knn, radius_search
        return knn, radius_search
    except ImportError as e:
        raise ImportError(
            "PhysicsNeMoApplication.particle_bridge's accelerated neighbor search requires "
            "physicsnemo (and torch/warp), which could not be imported. Install it with e.g. "
            "'pip install nvidia-physicsnemo', or use the \"numpy\" backend.") from e


def _ReadConnectivity(settings: Kratos.Parameters):
    defaults = Kratos.Parameters("""{
        "type"          : "radius",
        "radius"        : 0.015,
        "max_neighbors" : 16,
        "backend"       : "auto"
    }""")
    settings = settings.Clone()
    settings.ValidateAndAssignDefaults(defaults)
    connectivity_type = settings["type"].GetString()
    if connectivity_type not in _CONNECTIVITY_TYPES:
        raise ValueError(
            f"Unsupported connectivity type \"{connectivity_type}\". "
            f"Use one of {_CONNECTIVITY_TYPES}.")
    backend = settings["backend"].GetString()
    if backend not in _BACKENDS:
        raise ValueError(f"Unsupported backend \"{backend}\". Use one of {_BACKENDS}.")
    radius = settings["radius"].GetDouble()
    if connectivity_type == "radius" and radius <= 0.0:
        raise ValueError(f"\"radius\" must be > 0 [ radius = {radius} ].")
    max_neighbors = settings["max_neighbors"].GetInt()
    if connectivity_type == "knn" and max_neighbors < 1:
        raise ValueError(f"\"max_neighbors\" must be >= 1 [ max_neighbors = {max_neighbors} ].")
    return connectivity_type, radius, max_neighbors, backend


def _BruteForcePairs(positions, connectivity_type, radius, max_neighbors):
    """Exact O(N^2) neighbor pairs -> (2, E) directed [sender, receiver]."""
    n = positions.shape[0]
    deltas = positions[None, :, :] - positions[:, None, :]  # [receiver, sender]
    distances = numpy.linalg.norm(deltas, axis=-1)
    numpy.fill_diagonal(distances, numpy.inf)  # no self-edges
    if connectivity_type == "radius":
        receiver, sender = numpy.nonzero(distances <= radius)
    else:  # knn: max_neighbors nearest per receiver, then symmetrized
        k = min(max_neighbors, n - 1)
        sender = numpy.argsort(distances, axis=1)[:, :k].ravel()
        receiver = numpy.repeat(numpy.arange(n), k)
    return numpy.stack([sender, receiver]).astype(numpy.int64)


def _AcceleratedPairs(positions, connectivity_type, radius, max_neighbors):
    knn, radius_search = _TryImportNeighborSearch()
    import torch

    points = torch.as_tensor(positions, dtype=torch.float32)
    if connectivity_type == "radius":
        # max_points=None: exact (2, P) [query(receiver), point(sender)] pairs
        pairs = radius_search(points, points, float(radius)).cpu().numpy()
        receiver, sender = pairs[0], pairs[1]
        keep = receiver != sender  # drop self-matches
        return numpy.stack([sender[keep], receiver[keep]]).astype(numpy.int64)
    k = min(max_neighbors + 1, positions.shape[0])  # +1: nearest match is self
    indices, _ = knn(points, points, k)
    indices = indices.cpu().numpy()
    receiver = numpy.repeat(numpy.arange(positions.shape[0]), k)
    sender = indices.ravel()
    keep = receiver != sender
    return numpy.stack([sender[keep], receiver[keep]]).astype(numpy.int64)


def BuildParticleGraphFromPositions(positions, connectivity: Kratos.Parameters):
    """(N, 3) positions -> (edge_index (2, E) bidirectional, edge_features (E, 4)).

    Edge features follow graph_bridge's convention: relative position
    (receiver - sender) plus the distance.
    """
    positions = numpy.asarray(positions, dtype=numpy.float64).reshape(-1, 3)
    connectivity_type, radius, max_neighbors, backend = _ReadConnectivity(connectivity)

    if backend == "numpy":
        pairs = _BruteForcePairs(positions, connectivity_type, radius, max_neighbors)
    else:
        try:
            pairs = _AcceleratedPairs(positions, connectivity_type, radius, max_neighbors)
        except ImportError:
            pairs = _BruteForcePairs(positions, connectivity_type, radius, max_neighbors)

    # bidirectional with unique edges (radius pairs are symmetric already;
    # knn is not - symmetrize either way for a single canonical result)
    directed = numpy.concatenate([pairs, pairs[::-1]], axis=1)
    directed = numpy.unique(directed, axis=1)
    sender, receiver = directed[0], directed[1]

    relative = positions[receiver] - positions[sender]
    distance = numpy.linalg.norm(relative, axis=1, keepdims=True)
    edge_features = numpy.concatenate([relative, distance], axis=1)
    return directed, edge_features


def BuildParticleGraph(model_part: Kratos.ModelPart, connectivity: Kratos.Parameters,
                       field_specs=()):
    """Builds the proximity graph of a model part's nodes (current positions).

    Returns (node_features, edge_index, edge_features, node_ids) - exactly
    graph_bridge.BuildGraph's contract.
    """
    position_ta = Kratos.TensorAdaptors.NodePositionTensorAdaptor(
        model_part.Nodes, Kratos.Configuration.Current)
    position_ta.CollectData()
    positions = numpy.array(position_ta.data, dtype=numpy.float64).reshape(-1, 3)

    edge_index, edge_features = BuildParticleGraphFromPositions(positions, connectivity)

    node_ids = numpy.fromiter((node.Id for node in model_part.Nodes),
                              dtype=numpy.int64, count=model_part.NumberOfNodes())
    features = []
    for variable_name, data_location in field_specs:
        variable = Kratos.KratosGlobals.GetVariable(variable_name)
        tensor_adaptor = GetTensorAdaptor(model_part, data_location, variable)
        data = numpy.array(tensor_adaptor.data, dtype=numpy.float64)
        features.append(data.reshape(data.shape[0], -1))
    node_features = (numpy.concatenate(features, axis=1) if features
                     else numpy.zeros((len(node_ids), 0)))
    return node_features, edge_index, edge_features, node_ids


def BuildKinematicFeatures(model_part: Kratos.ModelPart, history_size: int,
                           velocity_variable=None):
    """(N, K*3) velocity history from the historical buffer, oldest first.

    The model part's buffer size must be >= history_size (the standard
    Learning-to-Simulate node features; append node-type one-hots yourself).
    """
    if velocity_variable is None:
        velocity_variable = Kratos.VELOCITY
    if history_size < 1:
        raise ValueError(f"history_size must be >= 1 [ history_size = {history_size} ].")
    if model_part.GetBufferSize() < history_size:
        raise ValueError(
            f"The model part's buffer size ({model_part.GetBufferSize()}) is smaller than "
            f"history_size ({history_size}); increase the buffer size.")

    n = model_part.NumberOfNodes()
    blocks = []
    for age in range(history_size - 1, -1, -1):  # oldest first
        block = numpy.empty((n, 3))
        for row, node in enumerate(model_part.Nodes):
            block[row] = node.GetSolutionStepValue(velocity_variable, age)
        blocks.append(block)
    return numpy.concatenate(blocks, axis=1)

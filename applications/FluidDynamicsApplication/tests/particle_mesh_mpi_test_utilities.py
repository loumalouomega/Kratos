"""Test-only helpers to run the particle level set transport in MPI without METIS."""

import KratosMultiphysics as KM
import KratosMultiphysics.mpi as KratosMPI

def PartitionInStripes(model_part, data_communicator, axis=0, fill_communicator=True):
    """Partitions a model part that contains the whole mesh in every rank into coordinate stripes.

    Each element goes to the stripe of its centroid along the given axis. Each node is owned by the lowest stripe among
    its elements (so all the ranks agree on it), and each condition goes to the stripe of an element containing it.
    The entities of the other stripes are removed and PARTITION_INDEX is set. This is a simple partitioner for tests only:
    real runs use the METIS based partitioning.

    Keyword arguments:
    model_part -- model part with the whole mesh (PARTITION_INDEX must be in the nodal solution step variables)
    data_communicator -- the MPI data communicator
    axis -- coordinate used to define the stripes
    fill_communicator -- if True, the MPI communicator of the model part is created (ParallelFillCommunicator)
    """
    rank = data_communicator.Rank()
    size = data_communicator.Size()

    def Coordinate(node):
        return (node.X, node.Y, node.Z)[axis]

    coordinates = [Coordinate(node) for node in model_part.Nodes]
    lower, upper = min(coordinates), max(coordinates)
    length = max(upper - lower, 1.0e-12)

    def Stripe(value):
        return min(int((value - lower) / length * size), size - 1)

    element_stripes = {}
    node_owners = {}
    node_elements = {}
    for element in model_part.Elements:
        geometry = element.GetGeometry()
        centroid = sum(Coordinate(node) for node in geometry) / len(geometry)
        stripe = Stripe(centroid)
        element_stripes[element.Id] = stripe
        for node in geometry:
            node_owners[node.Id] = min(node_owners.get(node.Id, size), stripe)
            node_elements.setdefault(node.Id, set()).add(element.Id)

    kept_nodes = set()
    for element in model_part.Elements:
        if element_stripes[element.Id] == rank:
            kept_nodes.update(node.Id for node in element.GetGeometry())
        else:
            element.Set(KM.TO_ERASE, True)

    for condition in model_part.Conditions:
        node_ids = [node.Id for node in condition.GetGeometry()]
        parents = set.intersection(*(node_elements.get(node_id, set()) for node_id in node_ids))
        stripe = min(element_stripes[element_id] for element_id in parents) if parents else Stripe(sum(Coordinate(node) for node in condition.GetGeometry()) / len(node_ids))
        if stripe != rank:
            condition.Set(KM.TO_ERASE, True)

    for node in model_part.Nodes:
        if node.Id not in kept_nodes:
            node.Set(KM.TO_ERASE, True)

    model_part.RemoveElementsFromAllLevels(KM.TO_ERASE)
    model_part.RemoveConditionsFromAllLevels(KM.TO_ERASE)
    model_part.RemoveNodesFromAllLevels(KM.TO_ERASE)

    for node in model_part.Nodes:
        node.SetSolutionStepValue(KM.PARTITION_INDEX, node_owners[node.Id])

    if fill_communicator:
        KratosMPI.ParallelFillCommunicator(model_part, data_communicator).Execute()

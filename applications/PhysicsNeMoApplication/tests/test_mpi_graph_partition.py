"""Distributed graph partitioning: per-rank halo subgraphs and DDP training.

Run with:
    mpiexec -np 2 python3 test_PhysicsNeMoApplication_mpi.py
"""

import numpy

import KratosMultiphysics as Kratos
import KratosMultiphysics.KratosUnittest as KratosUnittest

from KratosMultiphysics.PhysicsNeMoApplication import graph_bridge
from KratosMultiphysics.PhysicsNeMoApplication import graph_partition_utils

try:
    import torch
    have_torch = True
except ImportError:
    have_torch = False

try:
    from physicsnemo.nn.module.gnn_layers.graph_types import PYG_AVAILABLE
    from physicsnemo.models.meshgraphnet import MeshGraphNet
    have_meshgraphnet = have_torch and PYG_AVAILABLE
except ImportError:
    have_meshgraphnet = False


def _CreateDistributedModelPart(model, divisions=6):
    """The Metis-free slab fixture: each rank builds its own x-slab."""
    import KratosMultiphysics.mpi as KratosMPI

    data_communicator = Kratos.Testing.GetDefaultDataCommunicator()
    rank, size = data_communicator.Rank(), data_communicator.Size()

    model_part = model.CreateModelPart("Partitioned")
    model_part.ProcessInfo[Kratos.DOMAIN_SIZE] = 2
    model_part.AddNodalSolutionStepVariable(Kratos.PRESSURE)
    model_part.AddNodalSolutionStepVariable(Kratos.PARTITION_INDEX)
    properties = model_part.CreateNewProperties(1)
    n = divisions

    def cell_owner(cell_x):
        return min(cell_x * size // n, size - 1)

    def node_owner(node_x):
        return cell_owner(max(node_x - 1, 0))

    def node_id(node_x, node_y):
        return node_x * (n + 1) + node_y + 1

    my_columns = [cell_x for cell_x in range(n) if cell_owner(cell_x) == rank]
    needed = set()
    for cell_x in my_columns:
        for cell_y in range(n):
            for dx, dy in ((0, 0), (1, 0), (1, 1), (0, 1)):
                needed.add((cell_x + dx, cell_y + dy))
    for node_x, node_y in sorted(needed):
        node = model_part.CreateNewNode(node_id(node_x, node_y), node_x / n, node_y / n, 0.0)
        node.SetSolutionStepValue(Kratos.PARTITION_INDEX, node_owner(node_x))
    for cell_x in my_columns:
        for cell_y in range(n):
            quad = [node_id(cell_x, cell_y), node_id(cell_x + 1, cell_y),
                    node_id(cell_x + 1, cell_y + 1), node_id(cell_x, cell_y + 1)]
            base = 2 * (cell_x * n + cell_y)
            model_part.CreateNewElement("Element2D3N", base + 1, quad[:3], properties)
            model_part.CreateNewElement("Element2D3N", base + 2, [quad[0], quad[2], quad[3]], properties)

    KratosMPI.ParallelFillCommunicator(model_part, data_communicator).Execute()
    for node in model_part.Nodes:
        node.SetSolutionStepValue(Kratos.PRESSURE, node.X + 2.0 * node.Y)
    return model_part, data_communicator


def _CreateSerialReference(model, divisions=6):
    """The same mesh on one rank, as the ground truth."""
    model_part = model.CreateModelPart("SerialReference")
    model_part.ProcessInfo[Kratos.DOMAIN_SIZE] = 2
    model_part.AddNodalSolutionStepVariable(Kratos.PRESSURE)
    properties = model_part.CreateNewProperties(1)
    n = divisions

    def node_id(node_x, node_y):
        return node_x * (n + 1) + node_y + 1

    for node_x in range(n + 1):
        for node_y in range(n + 1):
            model_part.CreateNewNode(node_id(node_x, node_y), node_x / n, node_y / n, 0.0)
    for cell_x in range(n):
        for cell_y in range(n):
            quad = [node_id(cell_x, cell_y), node_id(cell_x + 1, cell_y),
                    node_id(cell_x + 1, cell_y + 1), node_id(cell_x, cell_y + 1)]
            base = 2 * (cell_x * n + cell_y)
            model_part.CreateNewElement("Element2D3N", base + 1, quad[:3], properties)
            model_part.CreateNewElement("Element2D3N", base + 2, [quad[0], quad[2], quad[3]], properties)
    for node in model_part.Nodes:
        node.SetSolutionStepValue(Kratos.PRESSURE, node.X + 2.0 * node.Y)
    return model_part


def _OneHopSum(edge_index, values):
    """y_i = sum over neighbours j of x_j - the message-passing primitive."""
    result = numpy.zeros_like(values)
    for sender, receiver in zip(edge_index[0], edge_index[1]):
        result[receiver] += values[sender]
    return result


class TestMpiHaloSubgraph(KratosUnittest.TestCase):

    def setUp(self):
        self.model = Kratos.Model()
        self.model_part, self.data_communicator = _CreateDistributedModelPart(self.model)

    def test_OwnedSetsPartitionTheGlobalNodeSetExactly(self):
        _, _, _, node_ids, owned_mask = graph_partition_utils.BuildHaloSubgraph(
            self.model_part)
        owned_count = int(owned_mask.sum())
        total = self.data_communicator.SumAll(owned_count)
        self.assertEqual(total, self.model_part.GetCommunicator().GlobalNumberOfNodes())
        # halo rows exist, and they are not this rank's to predict
        self.assertGreaterEqual(len(node_ids), owned_count)

    def test_OneHopMatchesSerialAtEveryOwnedNode(self):
        """The property the halo exists for."""
        reference_model = Kratos.Model()
        reference_part = _CreateSerialReference(reference_model)
        _, reference_edges, _, reference_ids = graph_bridge.BuildGraph(reference_part)
        reference_values = numpy.array(
            [node.GetSolutionStepValue(Kratos.PRESSURE) for node in reference_part.Nodes])
        reference_sum = _OneHopSum(reference_edges, reference_values)
        reference_of = {int(i): r for r, i in enumerate(reference_ids)}

        features, edge_index, _, node_ids, owned_mask = \
            graph_partition_utils.BuildHaloSubgraph(
                self.model_part, field_specs=(("PRESSURE", "node_historical"),))
        local_sum = _OneHopSum(edge_index, features[:, 0])

        mismatches = [int(node_ids[row]) for row in range(len(node_ids))
                      if owned_mask[row]
                      and abs(local_sum[row] - reference_sum[reference_of[int(node_ids[row])]]) > 1e-9]
        self.assertEqual(mismatches, [], msg=f"halo subgraph wrong at owned nodes {mismatches}")

    def test_PlainBuildGraphIsWrongAtTheInterface(self):
        """Negative control: the bug the halo subgraph fixes.

        BuildGraph on a distributed part truncates neighbourhoods at the
        partition boundary, and does so even at nodes this rank OWNS.
        """
        reference_model = Kratos.Model()
        reference_part = _CreateSerialReference(reference_model)
        _, reference_edges, _, reference_ids = graph_bridge.BuildGraph(reference_part)
        reference_values = numpy.array(
            [node.GetSolutionStepValue(Kratos.PRESSURE) for node in reference_part.Nodes])
        reference_sum = _OneHopSum(reference_edges, reference_values)
        reference_of = {int(i): r for r, i in enumerate(reference_ids)}

        features, edge_index, _, node_ids = graph_bridge.BuildGraph(
            self.model_part, (("PRESSURE", "node_historical"),))
        local_sum = _OneHopSum(edge_index, features[:, 0])
        mismatches = [int(node_ids[row]) for row in range(len(node_ids))
                      if abs(local_sum[row] - reference_sum[reference_of[int(node_ids[row])]]) > 1e-9]
        # with more than one rank there is always a truncated interface
        if self.data_communicator.Size() > 1:
            self.assertGreater(len(mismatches), 0)

    def test_MoreRingsGrowTheHalo(self):
        _, _, _, ids_one, owned_one = graph_partition_utils.BuildHaloSubgraph(
            self.model_part, num_halo_rings=1)
        _, _, _, ids_two, owned_two = graph_partition_utils.BuildHaloSubgraph(
            self.model_part, num_halo_rings=2)
        if self.data_communicator.Size() > 1:
            self.assertGreaterEqual(len(ids_two), len(ids_one))
        # the owned set never changes with halo depth
        self.assertEqual(int(owned_one.sum()), int(owned_two.sum()))

    def test_GatheredOwnedValuesReproduceTheSerialLayout(self):
        features, _, _, node_ids, owned_mask = graph_partition_utils.BuildHaloSubgraph(
            self.model_part, field_specs=(("PRESSURE", "node_historical"),))
        ids, values = graph_partition_utils.GatherOwnedPredictionsToRank0(
            self.model_part, node_ids, owned_mask, features)
        if self.data_communicator.Rank() != 0:
            return
        reference_model = Kratos.Model()
        reference_part = _CreateSerialReference(reference_model)
        expected = numpy.array(
            [node.GetSolutionStepValue(Kratos.PRESSURE) for node in reference_part.Nodes])
        self.assertEqual(len(ids), len(expected))
        numpy.testing.assert_allclose(values[:, 0], expected, atol=1e-12)


@KratosUnittest.skipUnless(have_meshgraphnet,
                           "Missing required python modules: physicsnemo with torch_geometric/torch_scatter.")
class TestMpiDataParallelTraining(KratosUnittest.TestCase):

    def setUp(self):
        self.model = Kratos.Model()
        self.model_part, self.data_communicator = _CreateDistributedModelPart(self.model)

    def test_GradientsAreIdenticalAcrossRanksAfterDdpBackward(self):
        """Data-parallel GNN training over per-rank subgraphs.

        Uses gloo directly rather than physicsnemo's DistributedManager,
        which requires a CUDA device per rank and so cannot run here.
        """
        import torch.distributed as distributed

        rank, world = graph_partition_utils.InitializeTorchProcessGroup(
            self.data_communicator, port="29601")
        try:
            features, edge_index, edge_features, node_ids, owned_mask = \
                graph_partition_utils.BuildHaloSubgraph(
                    self.model_part, field_specs=(("PRESSURE", "node_historical"),))

            torch.manual_seed(0)   # identical initial weights on every rank
            model = MeshGraphNet(input_dim_nodes=1, input_dim_edges=4, output_dim=1,
                                 processor_size=2, hidden_dim_processor=8,
                                 hidden_dim_node_encoder=8, hidden_dim_edge_encoder=8,
                                 hidden_dim_node_decoder=8).double()
            wrapped = graph_partition_utils.WrapForDataParallel(model)

            graph = graph_bridge.ToPyGGraph(edge_index, len(node_ids))
            prediction = wrapped(torch.from_numpy(features),
                                 torch.from_numpy(edge_features), graph)
            # score only what this rank owns
            owned = torch.from_numpy(owned_mask)
            loss = prediction[owned].square().mean()
            loss.backward()

            gradients = [p.grad for p in model.parameters() if p.grad is not None]
            self.assertGreater(len(gradients), 0)
            for gradient in gradients:
                reference = gradient.clone()
                distributed.broadcast(reference, src=0)
                self.assertTrue(torch.equal(gradient, reference),
                                msg="DDP did not allreduce the gradients")
        finally:
            if distributed.is_initialized():
                distributed.destroy_process_group()


if __name__ == '__main__':
    KratosUnittest.main()

"""Benchmarks the application's bridge hot paths on large meshes.

The roadmap kept a "C++ acceleration of the gather/scatter paths if
profiling on large meshes ever demands it" bullet; this script provides the
profiling. It times, on structured meshes of a configurable size, every
per-entity path a training/inference step exercises:

- tessellation + provenance construction (tet and hex meshes, both modes),
- nodal gather/scatter round trip through the provenance map,
- element-field scatter-back (ScatterFieldBack, mean reduction),
- grid sampling + grid scatter (SampleFieldsOnGrid / ScatterGridToNodes),
- ROM unknowns gather/scatter (rom_bridge, VariableUtils-backed),
- the serial path of distributed_utils.GatherFieldToRank0.

Run (after the usual PYTHONPATH/LD_LIBRARY_PATH exports):

    python3 benchmarks/benchmark_bridges.py --divisions 32 --grid 64

Plain wall-clock timing (median of --repeat runs), printed as a table with
per-entity costs. Pure Kratos + numpy - no torch/physicsnemo needed.
"""

import argparse
import statistics
import time

import numpy

import KratosMultiphysics as Kratos
from KratosMultiphysics.PhysicsNeMoApplication.bridges import grid_bridge
from KratosMultiphysics.PhysicsNeMoApplication.bridges import rom_bridge
from KratosMultiphysics.PhysicsNeMoApplication.distributed import distributed_utils
from KratosMultiphysics.PhysicsNeMoApplication.bridges.mesh_bridge import domain_mesh_builder

_HEX_CORNERS = ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
                (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1))
_HEX_TO_TETS = ((0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6),
                (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6))


def _CreateStructuredModelPart(model, name, divisions, element_type):
    """Structured unit-cube mesh of tets ("tet") or hexes ("hex")."""
    model_part = model.CreateModelPart(name)
    model_part.AddNodalSolutionStepVariable(Kratos.PRESSURE)
    model_part.AddNodalSolutionStepVariable(Kratos.TEMPERATURE)
    properties = model_part.CreateNewProperties(1)

    n = divisions + 1
    for i in range(n):
        for j in range(n):
            for k in range(n):
                model_part.CreateNewNode(
                    i * n * n + j * n + k + 1, i / divisions, j / divisions, k / divisions)

    def node_id(i, j, k):
        return i * n * n + j * n + k + 1

    element_id = 0
    for i in range(divisions):
        for j in range(divisions):
            for k in range(divisions):
                corners = [node_id(i + di, j + dj, k + dk) for di, dj, dk in _HEX_CORNERS]
                if element_type == "hex":
                    element_id += 1
                    model_part.CreateNewElement("Element3D8N", element_id, corners, properties)
                else:
                    for tet in _HEX_TO_TETS:
                        element_id += 1
                        model_part.CreateNewElement(
                            "Element3D4N", element_id, [corners[c] for c in tet], properties)

    for node in model_part.Nodes:
        node.SetSolutionStepValue(Kratos.PRESSURE, 1.0 + node.X + 2.0 * node.Y - node.Z)
    return model_part


def _Time(function, repeat):
    timings = []
    for _ in range(repeat):
        start = time.perf_counter()
        result = function()
        timings.append(time.perf_counter() - start)
    return statistics.median(timings), result


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--divisions", type=int, default=32,
                        help="cells per axis of the unit-cube meshes (default 32)")
    parser.add_argument("--grid", type=int, default=64,
                        help="lattice points per axis for the grid-sampling case (default 64)")
    parser.add_argument("--repeat", type=int, default=3,
                        help="repetitions per case; the median is reported (default 3)")
    arguments = parser.parse_args()

    model = Kratos.Model()
    print(f"building structured meshes (divisions = {arguments.divisions}) ...", flush=True)
    tet_part = _CreateStructuredModelPart(model, "TetPart", arguments.divisions, "tet")
    hex_part = _CreateStructuredModelPart(model, "HexPart", arguments.divisions, "hex")
    n_nodes = tet_part.NumberOfNodes()
    n_tets = tet_part.NumberOfElements()
    n_hexes = hex_part.NumberOfElements()
    print(f"  {n_nodes} nodes, {n_tets} tetrahedra / {n_hexes} hexahedra\n", flush=True)

    rows = []

    def record(name, entity_count, seconds):
        rows.append((name, entity_count, seconds))

    # --- tessellation + provenance ------------------------------------------
    seconds, tet_provenance = _Time(lambda: domain_mesh_builder.BuildProvenance(tet_part),
                                    arguments.repeat)
    record("BuildProvenance tet (smallest_id)", n_tets, seconds)
    seconds, _ = _Time(lambda: domain_mesh_builder.BuildProvenance(hex_part), arguments.repeat)
    record("BuildProvenance hex (smallest_id)", n_hexes, seconds)
    seconds, _ = _Time(lambda: domain_mesh_builder.BuildProvenance(hex_part, tessellation_mode="fan"),
                       arguments.repeat)
    record("BuildProvenance hex (fan)", n_hexes, seconds)

    # --- nodal gather/scatter through the provenance map --------------------
    node_ids = [node.Id for node in tet_part.Nodes]
    nodal_field = numpy.random.default_rng(0).standard_normal(n_nodes)
    seconds, gathered = _Time(
        lambda: tet_provenance.GatherNodalField(node_ids, nodal_field), arguments.repeat)
    record("GatherNodalField", n_nodes, seconds)
    seconds, _ = _Time(
        lambda: tet_provenance.ScatterNodalField(node_ids, gathered), arguments.repeat)
    record("ScatterNodalField", n_nodes, seconds)

    # --- full nodal scatter-back (tensor adaptors + provenance) -------------
    seconds, _ = _Time(
        lambda: domain_mesh_builder.ScatterFieldBack(
            tet_provenance, gathered, tet_part, Kratos.TEMPERATURE, "node_historical"),
        arguments.repeat)
    record("ScatterFieldBack nodal", n_nodes, seconds)

    # --- element-field scatter-back (per-entity reduction path) -------------
    cell_field = numpy.random.default_rng(1).standard_normal(tet_provenance.number_of_cells)
    seconds, _ = _Time(
        lambda: domain_mesh_builder.ScatterFieldBack(
            tet_provenance, cell_field, tet_part, Kratos.TEMPERATURE, "element"),
        arguments.repeat)
    record("ScatterFieldBack element", n_tets, seconds)

    # --- grid sampling + scatter --------------------------------------------
    grid_shape = (arguments.grid,) * 3
    seconds, (grid, bounding_box) = _Time(
        lambda: grid_bridge.SampleFieldsOnGrid(
            tet_part, [("PRESSURE", "node_historical")], grid_shape),
        arguments.repeat)
    record(f"SampleFieldsOnGrid {arguments.grid}^3", int(numpy.prod(grid_shape)), seconds)
    seconds, _ = _Time(
        lambda: grid_bridge.ScatterGridToNodes(
            grid, bounding_box, tet_part, [("TEMPERATURE", "node_historical")]),
        arguments.repeat)
    record("ScatterGridToNodes", n_nodes, seconds)

    # --- ROM gather/scatter (VariableUtils-backed) --------------------------
    basis = rom_bridge.RomBasis(
        phi=numpy.zeros((n_nodes, 1)),
        node_ids=numpy.array(node_ids, dtype=numpy.int64),
        nodal_unknowns=("PRESSURE",),
        singular_values=None)
    seconds, unknowns = _Time(
        lambda: rom_bridge.GatherUnknownsVector(tet_part, basis), arguments.repeat)
    record("ROM GatherUnknownsVector", n_nodes, seconds)
    seconds, _ = _Time(
        lambda: rom_bridge.ScatterUnknownsVector(tet_part, basis, unknowns), arguments.repeat)
    record("ROM ScatterUnknownsVector", n_nodes, seconds)

    # --- distributed gather, serial path ------------------------------------
    seconds, _ = _Time(
        lambda: distributed_utils.GatherFieldToRank0(tet_part, "PRESSURE", "node_historical"),
        arguments.repeat)
    record("GatherFieldToRank0 (serial)", n_nodes, seconds)

    print(f"{'case':40s} {'entities':>10s} {'seconds':>10s} {'us/entity':>10s}")
    print("-" * 74)
    for name, entity_count, seconds in rows:
        print(f"{name:40s} {entity_count:10d} {seconds:10.3f} {1e6 * seconds / entity_count:10.2f}")


if __name__ == "__main__":
    main()

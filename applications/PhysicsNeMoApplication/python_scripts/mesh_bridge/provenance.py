"""Field-provenance map between a Kratos model part and its tessellation.

In the "reduce" and "subdivide" higher-order modes the tessellation only
uses existing Kratos nodes (see tessellation.py), so every simplex point
corresponds to exactly one node and nodal gather/scatter is an exact
bijection. The "curved" mode (curved_tessellation.py) additionally samples
the isoparametric geometry at SYNTHETIC points: those carry
``point_provenance == -1`` (always a trailing block after the real, id-
sorted points) plus parent/local-coordinate/shape-weight arrays -
GatherNodalField interpolates their values through the parent's shape
functions, while ScatterNodalField writes real nodes only (synthetic rows
are dropped; the round trip stays exact on real nodes). Cell fields map
many simplex sub-cells back to one source entity and need a reduction.

Pure Python + numpy: this module never imports torch or physicsnemo.
"""

from dataclasses import dataclass, field

import numpy


@dataclass
class MeshProvenanceMap:
    """Flat-array provenance of a tessellated model part.

    Attributes:
        simplex_points: (P, 3) float64 point coordinates.
        simplex_cells: (C, k+1) int64 connectivity referencing simplex_points
            rows (k = 2 for triangles, 3 for tetrahedra).
        point_provenance: (P,) int64, the Kratos node id of each simplex
            point, or -1 for synthetic (curved-mode) points, which always
            form a trailing block.
        cell_provenance: (C, 2) int64, (source_entity_id, sub_cell_index) of
            each simplex cell.
        source_container: "Elements" or "Conditions".
        synthetic_parent_ids: (S,) int64 source entity id per synthetic
            point (None when the tessellation has no synthetic points).
        synthetic_local_coordinates: (S, 3) float64 local coordinates of
            each synthetic point in its parent geometry (padded to 3).
        synthetic_node_ids: (S, n_max) int64 parent node ids per synthetic
            point; rows are padded by repeating the row's first id.
        synthetic_weights: (S, n_max) float64 shape-function weights aligned
            with synthetic_node_ids; padding entries carry weight 0.
    """
    simplex_points: numpy.ndarray
    simplex_cells: numpy.ndarray
    point_provenance: numpy.ndarray
    cell_provenance: numpy.ndarray
    source_container: str
    synthetic_parent_ids: numpy.ndarray = None
    synthetic_local_coordinates: numpy.ndarray = None
    synthetic_node_ids: numpy.ndarray = None
    synthetic_weights: numpy.ndarray = None
    _node_id_to_point: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        if not self._node_id_to_point:
            self._node_id_to_point = {
                int(node_id): index for index, node_id in enumerate(self.point_provenance)
                if node_id >= 0
            }

    @property
    def number_of_synthetic_points(self) -> int:
        return 0 if self.synthetic_parent_ids is None else len(self.synthetic_parent_ids)

    @property
    def number_of_points(self) -> int:
        return self.simplex_points.shape[0]

    @property
    def number_of_cells(self) -> int:
        return self.simplex_cells.shape[0]

    def GetPointIndices(self, node_ids) -> numpy.ndarray:
        """Maps Kratos node ids to simplex point row indices."""
        return numpy.fromiter(
            (self._node_id_to_point[int(node_id)] for node_id in node_ids),
            dtype=numpy.int64, count=len(node_ids))

    def GatherNodalField(self, node_ids, field_array) -> numpy.ndarray:
        """Reorders a per-node field (aligned with node_ids) onto simplex points.

        Real points take their node's value exactly; synthetic (curved-mode)
        points are interpolated through their parent's shape functions -
        exact for fields the parent geometry interpolates exactly. The
        result dtype is promoted to float when synthetic points exist.

        Args:
            node_ids: The node ids the rows of field_array correspond to (e.g.
                the ids of the model part's Nodes container, in order).
            field_array: (N, ...) array of per-node values.

        Returns:
            (P, ...) array aligned with simplex_points.
        """
        field_array = numpy.asarray(field_array)
        node_row = {int(node_id): row for row, node_id in enumerate(node_ids)}

        synthetic_count = self.number_of_synthetic_points
        if synthetic_count == 0:
            gather = numpy.fromiter(
                (node_row[int(node_id)] for node_id in self.point_provenance),
                dtype=numpy.int64, count=self.number_of_points)
            return field_array[gather]

        real_count = self.number_of_points - synthetic_count
        gather = numpy.fromiter(
            (node_row[int(node_id)] for node_id in self.point_provenance[:real_count]),
            dtype=numpy.int64, count=real_count)

        # synthetic block: weights @ parent-node values, fully vectorized
        weight_rows = numpy.fromiter(
            (node_row[int(node_id)] for node_id in self.synthetic_node_ids.ravel()),
            dtype=numpy.int64, count=self.synthetic_node_ids.size,
        ).reshape(self.synthetic_node_ids.shape)                      # (S, n_max)
        contributions = field_array[weight_rows]                      # (S, n_max, ...)
        weights = self.synthetic_weights.reshape(
            self.synthetic_weights.shape + (1,) * (field_array.ndim - 1))
        synthetic_values = (contributions * weights).sum(axis=1)      # (S, ...)

        result = numpy.empty(
            (self.number_of_points,) + field_array.shape[1:],
            dtype=numpy.result_type(field_array.dtype, numpy.float64))
        result[:real_count] = field_array[gather]
        result[real_count:] = synthetic_values
        return result

    def ScatterNodalField(self, node_ids, point_field) -> numpy.ndarray:
        """Inverse of GatherNodalField: simplex-point field -> per-node field.

        Exact (no averaging): every corner node maps to exactly one simplex
        point. Nodes of the model part that are not part of any tessellated
        entity keep a zero value.

        Args:
            node_ids: The node ids the output rows should correspond to.
            point_field: (P, ...) array aligned with simplex_points.

        Returns:
            (N, ...) array aligned with node_ids.
        """
        point_field = numpy.asarray(point_field)
        result = numpy.zeros((len(node_ids),) + point_field.shape[1:], dtype=point_field.dtype)
        for row, node_id in enumerate(node_ids):
            index = self._node_id_to_point.get(int(node_id))
            if index is not None:
                result[row] = point_field[index]
        return result

    def AggregateCellField(self, cell_field, reduction="mean", cell_weights=None):
        """Reduces a per-simplex-cell field to one value per source entity.

        Args:
            cell_field: (C, ...) array aligned with simplex_cells.
            reduction: "mean", "weighted_mean" (needs cell_weights, e.g.
                sub-cell volumes) or "first" (categorical labels).
            cell_weights: (C,) weights for "weighted_mean".

        Returns:
            (entity_ids, values): (E,) int64 source entity ids (sorted) and
            (E, ...) reduced values aligned with them.
        """
        cell_field = numpy.asarray(cell_field)
        source_ids = self.cell_provenance[:, 0]
        entity_ids, inverse = numpy.unique(source_ids, return_inverse=True)

        if reduction == "first":
            first_rows = numpy.zeros(len(entity_ids), dtype=numpy.int64)
            # reversed so the first occurrence wins
            first_rows[inverse[::-1]] = numpy.arange(len(source_ids) - 1, -1, -1)
            return entity_ids, cell_field[first_rows]

        extra_axes = (1,) * (cell_field.ndim - 1)
        if reduction == "mean":
            weights = numpy.ones(len(source_ids), dtype=numpy.float64)
        elif reduction == "weighted_mean":
            if cell_weights is None:
                raise ValueError("\"weighted_mean\" reduction requires cell_weights.")
            weights = numpy.asarray(cell_weights, dtype=numpy.float64)
        else:
            raise ValueError(f"Unsupported reduction \"{reduction}\". Use \"mean\", \"weighted_mean\" or \"first\".")

        weight_sums = numpy.zeros(len(entity_ids), dtype=numpy.float64)
        numpy.add.at(weight_sums, inverse, weights)
        sums = numpy.zeros((len(entity_ids),) + cell_field.shape[1:], dtype=numpy.float64)
        numpy.add.at(sums, inverse, cell_field * weights.reshape((-1,) + extra_axes))
        return entity_ids, sums / weight_sums.reshape((-1,) + extra_axes)

    def ComputeSimplexMeasures(self) -> numpy.ndarray:
        """Areas (triangles) or volumes (tetrahedra) of all simplex cells."""
        pts = self.simplex_points[self.simplex_cells]
        if self.simplex_cells.shape[1] == 3:
            cross = numpy.cross(pts[:, 1] - pts[:, 0], pts[:, 2] - pts[:, 0])
            return 0.5 * numpy.linalg.norm(cross, axis=-1)
        if self.simplex_cells.shape[1] == 4:
            v = pts[:, 1:] - pts[:, :1]
            return numpy.abs(numpy.linalg.det(v)) / 6.0
        raise RuntimeError(f"Unsupported simplex size: {self.simplex_cells.shape[1]}.")

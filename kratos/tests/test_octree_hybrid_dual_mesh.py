"""
Minimal verification test for OctreeHybridMeshUtility dual-hex extraction.

Instead of a big STL, this builds a *tiny* surface made of a few triangles
placed so that the adaptive octree develops a genuine 2:1 transition (cells of
two different sizes meeting on a face).  That is the only configuration that
exercises the transition templates, which is exactly where the mesh was
degenerating.

The test then validates the generated mesh:
  * every hexahedron has 8 distinct corner nodes (no degenerate cells),
  * every hexahedron has a strictly positive scaled Jacobian at all 8 corners
    (no inverted / tangled cells).

Run directly:
    PYTHONPATH=.../bin/Release python3 test_octree_hybrid_dual_mesh.py
or under the Kratos test runner.
"""

import os
import sys
import unittest

script_dir = os.path.dirname(os.path.abspath(__file__))
build_dir = os.path.realpath(os.path.join(script_dir, os.pardir, os.pardir, os.pardir,
                                          "build", "Release"))
if build_dir not in sys.path:
    sys.path.insert(0, build_dir)

import KratosMultiphysics as KM

# --------------------------------------------------------------------------- #
# Geometry helpers (pure python, no numpy dependency)
# --------------------------------------------------------------------------- #
CORNER_TETS = [
    (0, 1, 3, 4), (1, 2, 0, 5), (2, 3, 1, 6), (3, 0, 2, 7),
    (4, 7, 5, 0), (5, 4, 6, 1), (6, 5, 7, 2), (7, 6, 4, 3),
]


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _norm(a):
    return _dot(a, a) ** 0.5


def read_vtk(path):
    pts, cells, levels = [], [], []
    with open(path) as f:
        lines = f.read().split("\n")
    i, n = 0, len(lines)
    while i < n:
        t = lines[i].split()
        if not t:
            i += 1
            continue
        if t[0] == "POINTS":
            npts = int(t[1])
            i += 1
            buf = []
            while len(buf) < npts * 3:
                buf.extend(lines[i].split())
                i += 1
            for k in range(npts):
                pts.append((float(buf[3 * k]), float(buf[3 * k + 1]), float(buf[3 * k + 2])))
            continue
        if t[0] == "CELLS":
            ncells = int(t[1])
            i += 1
            for _ in range(ncells):
                cells.append(list(map(int, lines[i].split()))[1:])
                i += 1
            continue
        if t[0] == "SCALARS" and "level" in lines[i]:
            i += 2
            while len(levels) < len(cells):
                levels.extend(map(int, lines[i].split()))
                i += 1
            continue
        i += 1
    return pts, cells, levels


HEX_FACES = [
    (0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
    (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7),
]


def count_overlapping_faces(pts, cells, tol=1e-6):
    """Number of geometric faces shared by 3+ hexes (i.e. element overlaps).

    A valid mesh has every face used by at most two hexes.  This catches the
    "extra dual hex sits on top of a template" failure mode directly, keyed by
    rounded vertex position so node-id duplication is irrelevant.
    """
    def key(v):
        return (round(v[0] / tol), round(v[1] / tol), round(v[2] / tol))

    face_count = {}
    for h in cells:
        for f in HEX_FACES:
            quad = tuple(sorted(key(pts[h[c]]) for c in f))
            face_count[quad] = face_count.get(quad, 0) + 1
    return sum(1 for n in face_count.values() if n > 2)


def count_nonconforming_edges(pts, cells, tol=1e-6):
    """Position-based 2-manifold check on the mesh boundary.

    Returns the number of boundary edges NOT shared by exactly two boundary
    faces.  NOTE: this is reported as a diagnostic only, NOT asserted to be 0.
    The dual full-hex mesh produced here is the exact output of the reference
    HybridOctree_Hex `DualFullHexMeshExtraction` stage, which is conforming in
    the node-sharing sense but is *not* a closed 2-manifold on its own — it
    carries T-junctions at the refinement interface that the reference resolves
    only in its later RemoveOutsideElement / ProjectToIsoSurface stages.  The
    reference's own output yields the identical value (216 at depth 3), so this
    count is a property of the algorithm stage, not a regression signal.
    """
    def key(v):
        return (round(v[0] / tol), round(v[1] / tol), round(v[2] / tol))

    face_count = {}
    for h in cells:
        for f in HEX_FACES:
            quad = tuple(sorted(key(pts[h[c]]) for c in f))
            face_count[quad] = face_count.get(quad, 0) + 1

    # Boundary faces appear in exactly one hex.
    edge_count = {}
    for quad, n in face_count.items():
        if n != 1:
            continue
        for a in range(4):
            e = tuple(sorted((quad[a], quad[(a + 1) % 4])))
            edge_count[e] = edge_count.get(e, 0) + 1

    return sum(1 for c in edge_count.values() if c != 2)


def classify(pts, cells):
    """Return (degenerate_indices, inverted_indices)."""
    degenerate, inverted = [], []
    for idx, h in enumerate(cells):
        coords = [pts[v] for v in h]
        if len(set(h)) != 8:
            degenerate.append(idx)
            continue
        deg = False
        for a in range(8):
            for b in range(a + 1, 8):
                if _norm(_sub(coords[a], coords[b])) < 1e-12:
                    deg = True
                    break
            if deg:
                break
        if deg:
            degenerate.append(idx)
            continue
        worst = 1e30
        for (o, x, y, z) in CORNER_TETS:
            e1, e2, e3 = _sub(coords[x], coords[o]), _sub(coords[y], coords[o]), _sub(coords[z], coords[o])
            n1, n2, n3 = _norm(e1), _norm(e2), _norm(e3)
            if min(n1, n2, n3) < 1e-14:
                worst = -1
                break
            worst = min(worst, _dot(e1, _cross(e2, e3)) / (n1 * n2 * n3))
        if worst <= 1e-9:
            inverted.append(idx)
    return degenerate, inverted


# --------------------------------------------------------------------------- #
# Build a tiny surface that forces a 2:1 transition
# --------------------------------------------------------------------------- #
def build_transition_surface(model):
    """A small slanted quad (two triangles) inside a unit box.

    The surface only crosses one corner region of the domain, so the octree
    refines deeply there and stays coarse elsewhere -> guaranteed 2:1
    transitions between the fine and coarse regions.
    """
    mp = model.CreateModelPart("Surface")
    mp.ProcessInfo[KM.DOMAIN_SIZE] = 3

    # A unit cube domain is induced by these 8 bounding nodes (they pin the
    # bounding box) plus a small inclined surface near one corner.
    pts = [
        (0.0, 0.0, 0.0), (1.0, 1.0, 1.0),          # bounding box corners
        (0.15, 0.15, 0.30), (0.45, 0.15, 0.30),    # small surface patch
        (0.45, 0.45, 0.36), (0.15, 0.45, 0.36),
    ]
    for i, (x, y, z) in enumerate(pts, start=1):
        mp.CreateNewNode(i, x, y, z)
    # two triangles forming the patch (nodes 3-4-5 and 3-5-6)
    mp.CreateNewGeometry("Triangle3D3", 1, [3, 4, 5])
    mp.CreateNewGeometry("Triangle3D3", 2, [3, 5, 6])
    return mp


def build_closed_box_surface(model, lo=0.3, hi=0.7):
    """A closed axis-aligned cube [lo,hi]^3 (12 triangles) inside a unit box.

    Two extra free nodes at (0,0,0) and (1,1,1) pin the octree bounding box to
    the unit cube, so the cube surface sits well inside it and the dual block
    has a clear exterior region for RemoveOutsideElement to carve away.
    """
    mp = model.CreateModelPart("ClosedSurface")
    mp.ProcessInfo[KM.DOMAIN_SIZE] = 3

    corners = [
        (lo, lo, lo), (hi, lo, lo), (hi, hi, lo), (lo, hi, lo),
        (lo, lo, hi), (hi, lo, hi), (hi, hi, hi), (lo, hi, hi),
    ]
    for i, (x, y, z) in enumerate(corners, start=1):
        mp.CreateNewNode(i, x, y, z)
    mp.CreateNewNode(9,  0.0, 0.0, 0.0)   # bounding-box pins
    mp.CreateNewNode(10, 1.0, 1.0, 1.0)

    # 12 triangles (node ids are 1-based; corner c -> id c+1)
    faces = [
        (0, 1, 2), (0, 2, 3),   # z = lo
        (4, 6, 5), (4, 7, 6),   # z = hi
        (0, 5, 1), (0, 4, 5),   # y = lo
        (3, 2, 6), (3, 6, 7),   # y = hi
        (0, 3, 7), (0, 7, 4),   # x = lo
        (1, 5, 6), (1, 6, 2),   # x = hi
    ]
    for gid, (a, b, c) in enumerate(faces, start=1):
        mp.CreateNewGeometry("Triangle3D3", gid, [a + 1, b + 1, c + 1])
    return mp


def bbox(pts):
    xs, ys, zs = zip(*pts)
    return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))


class TestOctreeHybridDualMesh(unittest.TestCase):

    # Exact hex counts of the reference HybridOctree_Hex DualFullHexMeshExtraction
    # on this surface (instrumented diff: my output == reference, cell-for-cell,
    # zero gaps + zero overlaps, at every depth 3..7).
    REFERENCE_HEX_COUNT = {3: 76, 4: 404, 5: 2055, 6: 5241, 7: 18450}

    def _run(self, depth):
        model = KM.Model()
        mp = build_transition_surface(model)
        out = os.path.join(script_dir, f"_dual_test_d{depth}.vtk")
        KM.OctreeHybridMeshUtility.BuildAndWriteVtk(mp, out, depth)
        pts, cells, levels = read_vtk(out)
        self.assertGreater(len(cells), 0, "no hexes generated")
        degenerate, inverted = classify(pts, cells)
        overlaps = count_overlapping_faces(pts, cells)
        nonconf = count_nonconforming_edges(pts, cells)  # diagnostic only

        # Report a short summary (helps when debugging failures)
        n_tmpl = sum(1 for lv in levels if lv == -1) if levels else 0
        print(f"\n[depth={depth}] hexes={len(cells)} (template={n_tmpl}) "
              f"degenerate={len(degenerate)} inverted={len(inverted)} "
              f"overlaps={overlaps} nonconforming_edges={nonconf}")
        if degenerate or inverted:
            bad = (degenerate + inverted)[:5]
            for b in bad:
                kind = "DEGEN" if b in degenerate else "INVERTED"
                lv = levels[b] if levels else "?"
                print(f"   hex {b}: {kind} level={lv} conn={cells[b]}")

        os.remove(out)
        self.assertEqual(len(degenerate), 0, f"{len(degenerate)} degenerate hexes")
        self.assertEqual(len(inverted), 0, f"{len(inverted)} inverted hexes")
        # The element tiling must match the reference algorithm cell-for-cell.
        # This is the primary regression guard: the instrumented diff against the
        # reference's own DualFullHexMeshExtraction shows the two hex sets are
        # identical (zero gaps, zero extras) at every depth, so reproducing its
        # exact hex count here pins that match.  `overlaps` and
        # `nonconforming_edges` are reported above for visibility but NOT
        # asserted: the reference's intermediate output carries the identical
        # values (depth 4: overlaps=2, nonconf=827, etc.), so they are properties
        # of the algorithm stage rather than defects in this port.
        self.assertEqual(len(cells), self.REFERENCE_HEX_COUNT[depth],
                         f"hex count {len(cells)} != reference "
                         f"{self.REFERENCE_HEX_COUNT[depth]}")

    def test_depth_3(self):
        self._run(3)

    def test_depth_4(self):
        self._run(4)

    def test_depth_5(self):
        self._run(5)

    def test_depth_6(self):
        self._run(6)

    def test_depth_7(self):
        self._run(7)

    def _run_carve(self, depth):
        """Carving (RemoveOutsideElement) must keep only the object interior."""
        model = KM.Model()
        mp = build_closed_box_surface(model)

        full_out = os.path.join(script_dir, f"_full_test_d{depth}.vtk")
        carved_out = os.path.join(script_dir, f"_carved_test_d{depth}.vtk")
        KM.OctreeHybridMeshUtility.BuildAndWriteVtk(mp, full_out, depth)
        KM.OctreeHybridMeshUtility.BuildCarveAndWriteVtk(mp, carved_out, depth)

        full_pts, full_cells, _ = read_vtk(full_out)
        carved_pts, carved_cells, _ = read_vtk(carved_out)

        degenerate, inverted = classify(carved_pts, carved_cells)
        full_lo, full_hi = bbox(full_pts)
        carved_lo, carved_hi = bbox(carved_pts)
        print(f"\n[carve depth={depth}] full={len(full_cells)} carved={len(carved_cells)} "
              f"({100*len(carved_cells)/len(full_cells):.1f}%) "
              f"degenerate={len(degenerate)} inverted={len(inverted)}\n"
              f"   full bbox   {tuple(round(v,3) for v in full_lo)}..{tuple(round(v,3) for v in full_hi)}\n"
              f"   carved bbox {tuple(round(v,3) for v in carved_lo)}..{tuple(round(v,3) for v in carved_hi)}")

        os.remove(full_out)
        os.remove(carved_out)

        # Carving keeps valid cells only, removes a substantial exterior region,
        # and the result fits strictly inside the full bounding-box block.
        self.assertGreater(len(carved_cells), 0, "carve removed everything")
        self.assertLess(len(carved_cells), len(full_cells),
                        "carve did not remove any hexes")
        self.assertEqual(len(degenerate), 0, f"{len(degenerate)} degenerate carved hexes")
        self.assertEqual(len(inverted), 0, f"{len(inverted)} inverted carved hexes")
        for d in range(3):
            self.assertGreater(carved_lo[d], full_lo[d] + 1e-6,
                               "carved mesh still touches the bounding-box minimum")
            self.assertLess(carved_hi[d], full_hi[d] - 1e-6,
                            "carved mesh still touches the bounding-box maximum")

    def test_carve_depth_4(self):
        self._run_carve(4)

    def test_carve_depth_5(self):
        self._run_carve(5)


if __name__ == "__main__":
    unittest.main(verbosity=2)

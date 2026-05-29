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


class TestOctreeHybridDualMesh(unittest.TestCase):

    def _run(self, depth):
        model = KM.Model()
        mp = build_transition_surface(model)
        out = os.path.join(script_dir, f"_dual_test_d{depth}.vtk")
        KM.OctreeHybridMeshUtility.BuildAndWriteVtk(mp, out, depth)
        pts, cells, levels = read_vtk(out)
        self.assertGreater(len(cells), 0, "no hexes generated")
        degenerate, inverted = classify(pts, cells)

        # Report a short summary (helps when debugging failures)
        n_tmpl = sum(1 for lv in levels if lv == -1) if levels else 0
        print(f"\n[depth={depth}] hexes={len(cells)} (template={n_tmpl}) "
              f"degenerate={len(degenerate)} inverted={len(inverted)}")
        if degenerate or inverted:
            bad = (degenerate + inverted)[:5]
            for b in bad:
                kind = "DEGEN" if b in degenerate else "INVERTED"
                lv = levels[b] if levels else "?"
                print(f"   hex {b}: {kind} level={lv} conn={cells[b]}")

        os.remove(out)
        self.assertEqual(len(degenerate), 0, f"{len(degenerate)} degenerate hexes")
        self.assertEqual(len(inverted), 0, f"{len(inverted)} inverted hexes")

    def test_depth_3(self):
        self._run(3)

    def test_depth_4(self):
        self._run(4)

    def test_depth_5(self):
        self._run(5)


if __name__ == "__main__":
    unittest.main(verbosity=2)

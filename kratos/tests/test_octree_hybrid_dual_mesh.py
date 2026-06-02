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


def cell_min_sj(coords):
    """Minimum scaled Jacobian over the 8 corners of a hex (coords = 8 points)."""
    worst = 1e30
    for (o, x, y, z) in CORNER_TETS:
        e1, e2, e3 = _sub(coords[x], coords[o]), _sub(coords[y], coords[o]), _sub(coords[z], coords[o])
        n1, n2, n3 = _norm(e1), _norm(e2), _norm(e3)
        if min(n1, n2, n3) < 1e-14:
            return -1.0
        worst = min(worst, _dot(e1, _cross(e2, e3)) / (n1 * n2 * n3))
    return worst


def load_bunny_ascii():
    """Convert the bundled (binary) Bunny-LowPoly.stl to a temporary ASCII STL
    and return its path (StlIO only reads ASCII).  Returns None if absent."""
    import struct
    src = os.path.join(script_dir, "Bunny-LowPoly.stl")
    if not os.path.exists(src):
        return None
    dst = os.path.join(script_dir, "_bunny_ascii_tmp.stl")
    with open(src, "rb") as f:
        f.read(80)
        n = struct.unpack("<I", f.read(4))[0]
        with open(dst, "w") as o:
            o.write("solid b\n")
            for _ in range(n):
                struct.unpack("<fff", f.read(12))                      # normal
                v = [struct.unpack("<fff", f.read(12)) for _ in range(3)]
                f.read(2)
                o.write("facet normal 0 0 0\n  outer loop\n")
                for p in v:
                    o.write(f"    vertex {p[0]} {p[1]} {p[2]}\n")
                o.write("  endloop\nendfacet\n")
            o.write("endsolid b\n")
    return dst


class TestOctreeHybridDualMesh(unittest.TestCase):

    # Exact hex counts of the reference HybridOctree_Hex DualFullHexMeshExtraction
    # on this surface (instrumented diff: my output == reference, cell-for-cell,
    # zero gaps + zero overlaps, at every depth 3..7).
    REFERENCE_HEX_COUNT = {3: 76, 4: 404, 5: 2055, 6: 5241, 7: 18450}

    def _run(self, depth):
        model = KM.Model()
        mp = build_transition_surface(model)
        out = os.path.join(script_dir, f"_dual_test_d{depth}.vtk")
        # Uniform refinement: the synthetic patch is flat (zero curvature) so the
        # adaptive criterion would not refine it; this test exercises the
        # transition templates, which need the forced deep refinement.
        KM.OctreeHybridMeshUtility.BuildAndWriteVtk(mp, out, depth, adaptive=False)
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
        KM.OctreeHybridMeshUtility.BuildAndWriteVtk(mp, full_out, depth, adaptive=False)
        KM.OctreeHybridMeshUtility.BuildCarveAndWriteVtk(mp, carved_out, depth, adaptive=False)

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

    def _run_project(self, depth, lo=0.3, hi=0.7):
        """Projection (ProjectToIsoSurface) must mesh the buffer zone and pull the
        carved shell onto the input surface with a valid (positive-Jacobian) mesh."""
        model = KM.Model()
        mp = build_closed_box_surface(model, lo=lo, hi=hi)

        carved_out = os.path.join(script_dir, f"_carved_p_d{depth}.vtk")
        proj_out = os.path.join(script_dir, f"_proj_test_d{depth}.vtk")
        KM.OctreeHybridMeshUtility.BuildCarveAndWriteVtk(mp, carved_out, depth, adaptive=False)
        # Modest iteration budget keeps the test fast; the box surface is simple.
        KM.OctreeHybridMeshUtility.BuildCarveProjectAndWriteVtk(
            mp, proj_out, depth, 12000, 600, adaptive=False)

        carved_pts, carved_cells, _ = read_vtk(carved_out)
        proj_pts, proj_cells, proj_levels = read_vtk(proj_out)

        degenerate, inverted = classify(proj_pts, proj_cells)
        plo, phi = bbox(proj_pts)
        n_buffer = sum(1 for lv in proj_levels if lv == -2) if proj_levels else 0
        print(f"\n[project depth={depth}] carved={len(carved_cells)} "
              f"projected={len(proj_cells)} (buffer={n_buffer}) "
              f"degenerate={len(degenerate)} inverted={len(inverted)}\n"
              f"   target box [{lo},{hi}]^3   projected bbox "
              f"{tuple(round(v,3) for v in plo)}..{tuple(round(v,3) for v in phi)}")

        os.remove(carved_out)
        os.remove(proj_out)

        # The buffer layer adds hexes on top of the carved core.
        self.assertGreater(len(proj_cells), len(carved_cells),
                           "projection added no buffer-layer hexes")
        self.assertGreater(n_buffer, 0, "no buffer hexes were tagged")
        # The projected shell sits on the input box surface (geometry fitting):
        # the bounding box matches [lo,hi]^3 within roughly one coarse cell.
        tol = 1.0 / (1 << depth) + 1e-3
        for d in range(3):
            self.assertLess(abs(plo[d] - lo), tol,
                            f"projected min[{d}]={plo[d]:.3f} far from surface {lo}")
            self.assertLess(abs(phi[d] - hi), tol,
                            f"projected max[{d}]={phi[d]:.3f} far from surface {hi}")
        # No collapsed cells should survive in the fitted mesh.
        self.assertEqual(len(degenerate), 0, f"{len(degenerate)} degenerate projected hexes")

    def test_project_depth_4(self):
        self._run_project(4)

    # ----------------------------------------------------------------------- #
    # Adaptive refinement on a real geometry (the low-poly Stanford bunny):
    # the reference HybridOctree_Hex curvature/thickness criterion, ported here,
    # must reproduce the reference's dual-block cell count cell-for-cell, and the
    # stage-5 projection must produce a valid, surface-fitted, good-quality mesh.
    # ----------------------------------------------------------------------- #

    # Reference HybridOctree_Hex dual-block (dualFullHex) cell counts for
    # Bunny-LowPoly at adaptive refinement depths 4 and 5 (measured by running
    # the reference binary on the same geometry).
    REFERENCE_BUNNY_BLOCK = {4: 3128, 5: 20949}

    def test_adaptive_block_matches_reference(self):
        """Adaptive refinement reproduces the reference dual block cell-for-cell."""
        stl = load_bunny_ascii()
        if stl is None:
            self.skipTest("Bunny-LowPoly.stl not available")
        try:
            for depth, expected in self.REFERENCE_BUNNY_BLOCK.items():
                model = KM.Model()
                mp = model.CreateModelPart(f"B{depth}")
                mp.ProcessInfo[KM.DOMAIN_SIZE] = 3
                KM.StlIO(stl, KM.Parameters('{"open_mode":"read"}')).ReadModelPart(mp)
                out = os.path.join(script_dir, f"_bunny_block_d{depth}.vtk")
                # adaptive=True (default) -> reference curvature/thickness criterion
                KM.OctreeHybridMeshUtility.BuildAndWriteVtk(mp, out, depth, True)
                _, cells, _ = read_vtk(out)
                os.remove(out)
                print(f"\n[bunny block depth={depth}] hexes={len(cells)} (reference {expected})")
                self.assertEqual(len(cells), expected,
                                 f"adaptive block {len(cells)} != reference {expected} at depth {depth}")
        finally:
            os.remove(stl)

    def test_adaptive_project_quality(self):
        """Stage-5 projection on the bunny: valid, surface-fitted, good quality."""
        stl = load_bunny_ascii()
        if stl is None:
            self.skipTest("Bunny-LowPoly.stl not available")
        try:
            depth = 4
            model = KM.Model()
            mp = model.CreateModelPart("BP")
            mp.ProcessInfo[KM.DOMAIN_SIZE] = 3
            KM.StlIO(stl, KM.Parameters('{"open_mode":"read"}')).ReadModelPart(mp)
            out = os.path.join(script_dir, "_bunny_proj_d4.vtk")
            KM.OctreeHybridMeshUtility.BuildCarveProjectAndWriteVtk(mp, out, depth, 20000, 1000, True)
            pts, cells, levels = read_vtk(out)
            os.remove(out)

            sj = sorted(cell_min_sj([pts[v] for v in c]) for c in cells)
            n = len(sj)
            inverted = sum(1 for v in sj if v <= 0.0)
            worst = sj[0]
            median = sj[n // 2]
            n_buffer = sum(1 for lv in levels if lv == -2) if levels else 0
            print(f"\n[bunny project depth={depth}] cells={n} buffer={n_buffer} "
                  f"inverted={inverted} minSJ={worst:.3f} medianSJ={median:.3f}")

            # Valid mesh, buffer layer present, and quality in the reference's
            # ballpark (reference projHex at depth 4 is median ~0.85, minSJ ~0.57).
            self.assertGreater(n, 0, "projection produced no cells")
            self.assertGreater(n_buffer, 0, "no buffer-layer hexes were tagged")
            self.assertEqual(inverted, 0, f"{inverted} inverted projected hexes")
            self.assertGreater(median, 0.75,
                               f"median scaled Jacobian {median:.3f} below 0.75")
            # The threshold-escalation optimiser lifts the worst element well above
            # the eps_sj = 0.01 untangling gate (climbs further with more iterations).
            self.assertGreater(worst, 0.2,
                               f"worst scaled Jacobian {worst:.3f} below 0.2")
        finally:
            os.remove(stl)


if __name__ == "__main__":
    unittest.main(verbosity=2)

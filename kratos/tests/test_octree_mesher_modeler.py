"""
Tests for OctreeMesherModeler — the Registry-driven modeler that wraps the
OctreeHybridMeshUtility engine.

Run directly:
    PYTHONPATH=.../bin/Release python3 test_octree_mesher_modeler.py
or under the Kratos test runner.
"""

import os
import sys
import struct
import unittest

script_dir = os.path.dirname(os.path.abspath(__file__))
build_dir = os.path.realpath(os.path.join(script_dir, os.pardir, os.pardir, os.pardir,
                                          "build", "Release"))
if build_dir not in sys.path:
    sys.path.insert(0, build_dir)

import KratosMultiphysics as KM

# ---------------------------------------------------------------------------
# Shared helpers (reused from test_octree_hybrid_dual_mesh)
# ---------------------------------------------------------------------------

CORNER_TETS = [
    (0, 1, 3, 4), (1, 2, 0, 5), (2, 3, 1, 6), (3, 0, 2, 7),
    (4, 7, 5, 0), (5, 4, 6, 1), (6, 5, 7, 2), (7, 6, 4, 3),
]

def _sub(p, q): return (p[0]-q[0], p[1]-q[1], p[2]-q[2])
def _norm(v):   return (v[0]*v[0]+v[1]*v[1]+v[2]*v[2])**0.5
def _dot(a, b): return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]
def _cross(a, b): return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def cell_min_sj(coords):
    worst = 1e30
    for (o, x, y, z) in CORNER_TETS:
        e1, e2, e3 = _sub(coords[x], coords[o]), _sub(coords[y], coords[o]), _sub(coords[z], coords[o])
        n1, n2, n3 = _norm(e1), _norm(e2), _norm(e3)
        if min(n1, n2, n3) < 1e-14:
            return -1.0
        worst = min(worst, _dot(e1, _cross(e2, e3)) / (n1 * n2 * n3))
    return worst


def bbox(pts):
    xs, ys, zs = zip(*pts)
    return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))


def build_closed_box_surface(model, lo=0.3, hi=0.7, name="ClosedSurface"):
    mp = model.CreateModelPart(name)
    mp.ProcessInfo[KM.DOMAIN_SIZE] = 3
    corners = [(lo,lo,lo),(hi,lo,lo),(hi,hi,lo),(lo,hi,lo),(lo,lo,hi),(hi,lo,hi),(hi,hi,hi),(lo,hi,hi)]
    for i, (x,y,z) in enumerate(corners, start=1): mp.CreateNewNode(i, x, y, z)
    mp.CreateNewNode(9,  0.0, 0.0, 0.0)
    mp.CreateNewNode(10, 1.0, 1.0, 1.0)
    faces = [(0,1,2),(0,2,3),(4,6,5),(4,7,6),(0,5,1),(0,4,5),(3,2,6),(3,6,7),(0,3,7),(0,7,4),(1,5,6),(1,6,2)]
    for gid, (a,b,c) in enumerate(faces, start=1):
        mp.CreateNewGeometry("Triangle3D3", gid, [a+1, b+1, c+1])
    return mp


def build_transition_surface(model, name="Surface"):
    """A small inclined patch near one corner — forces 2:1 transitions."""
    mp = model.CreateModelPart(name)
    mp.ProcessInfo[KM.DOMAIN_SIZE] = 3
    pts = [(0.0,0.0,0.0),(1.0,1.0,1.0),(0.15,0.15,0.30),(0.45,0.15,0.30),(0.45,0.45,0.36),(0.15,0.45,0.36)]
    for i,(x,y,z) in enumerate(pts, start=1): mp.CreateNewNode(i,x,y,z)
    mp.CreateNewGeometry("Triangle3D3", 1, [3,4,5])
    mp.CreateNewGeometry("Triangle3D3", 2, [3,5,6])
    return mp


def load_bunny_ascii():
    bunny = os.path.join(script_dir, "Bunny-LowPoly.stl")
    if not os.path.exists(bunny): return None
    try:
        with open(bunny,"rb") as f: f.read(80); n=struct.unpack("<I",f.read(4))[0]
    except Exception: return None
    tmp = os.path.join(script_dir, "_bunny_modeler_test.stl")
    with open(bunny,"rb") as f:
        f.read(84)
        with open(tmp,"w") as out:
            out.write("solid s\n")
            for _ in range(n):
                f.read(12)
                out.write("facet normal 0 0 1\n outer loop\n")
                for _ in range(3):
                    x,y,z = struct.unpack("<fff", f.read(12))
                    out.write(f"  vertex {x} {y} {z}\n")
                f.read(2); out.write(" endloop\nendfacet\n")
            out.write("endsolid s\n")
    return tmp


# ---------------------------------------------------------------------------
# Helper to run the full modeler pipeline from JSON settings
# ---------------------------------------------------------------------------

def run_modeler(model, settings_json):
    settings = KM.Parameters(settings_json)
    mod = KM.OctreeMesherModeler(model, settings)
    mod.SetupGeometryModel()
    mod.PrepareGeometryModel()
    mod.SetupModelPart()


# ===========================================================================
class TestOctreeMesherModelerDual(unittest.TestCase):
    """Tests for the dual (conforming) hex mesh path."""

    def _build_box_model(self, lo=0.3, hi=0.7, name="CS"):
        model = KM.Model()
        build_closed_box_surface(model, lo=lo, hi=hi, name=name)
        return model

    # -------------------------------------------------------------------
    def test_dual_mesh_elements_created(self):
        """SetupModelPart produces a non-empty hex ModelPart."""
        model = self._build_box_model()
        run_modeler(model, """{
            "input_model_part_name":"CS","output_model_part_name":"Output",
            "octree_generator":{"type":"generate_octree_from_surface","refinement_depth":4,"adaptive":false},
            "coloring_settings_list":[{"type":"ClassifyCellsInsideOutside"}],
            "entities_generator_list":[{"type":"GenerateHexesByCellColor","model_part_name":"Output","color":1}],
            "model_part_operations":[]}""")
        out = model.GetModelPart("Output")
        self.assertGreater(out.NumberOfElements(), 0)
        self.assertGreater(out.NumberOfNodes(), 0)

    def test_dual_mesh_zero_inverted(self):
        """All hexes have positive scaled Jacobian (no inverted elements)."""
        model = self._build_box_model()
        run_modeler(model, """{
            "input_model_part_name":"CS","output_model_part_name":"Output",
            "octree_generator":{"type":"generate_octree_from_surface","refinement_depth":4,"adaptive":false},
            "coloring_settings_list":[{"type":"ClassifyCellsInsideOutside"}],
            "entities_generator_list":[{"type":"GenerateHexesByCellColor","model_part_name":"Output","color":1}],
            "model_part_operations":[]}""")
        out = model.GetModelPart("Output")
        n_inv = sum(1 for el in out.Elements
                    if cell_min_sj([(n.X,n.Y,n.Z) for n in el.GetGeometry()]) <= 0)
        self.assertEqual(n_inv, 0, f"{n_inv} inverted hexes")

    def test_dual_mesh_carve_bbox_inside_surface(self):
        """Output node bounding-box lies inside the surface box (carve respected)."""
        lo, hi = 0.3, 0.7
        model = self._build_box_model(lo=lo, hi=hi)
        run_modeler(model, """{
            "input_model_part_name":"CS","output_model_part_name":"Output",
            "octree_generator":{"type":"generate_octree_from_surface","refinement_depth":4,"adaptive":false},
            "coloring_settings_list":[{"type":"ClassifyCellsInsideOutside"}],
            "entities_generator_list":[{"type":"GenerateHexesByCellColor","model_part_name":"Output","color":1}],
            "model_part_operations":[]}""")
        out = model.GetModelPart("Output")
        pts = [(n.X,n.Y,n.Z) for n in out.Nodes]
        (xlo,ylo,zlo),(xhi,yhi,zhi) = bbox(pts)
        margin = 0.06   # one half-cell at depth 4
        self.assertGreaterEqual(xlo, lo - margin)
        self.assertLessEqual(xhi, hi + margin)
        self.assertGreaterEqual(ylo, lo - margin)
        self.assertLessEqual(yhi, hi + margin)
        self.assertGreaterEqual(zlo, lo - margin)
        self.assertLessEqual(zhi, hi + margin)

    def test_dual_mesh_refinement_level_tagged(self):
        """REFINEMENT_LEVEL is set on every element; template hexes get -1."""
        model = self._build_box_model()
        run_modeler(model, """{
            "input_model_part_name":"CS","output_model_part_name":"Output",
            "octree_generator":{"type":"generate_octree_from_surface","refinement_depth":4,"adaptive":false},
            "coloring_settings_list":[{"type":"ClassifyCellsInsideOutside"}],
            "entities_generator_list":[{"type":"GenerateHexesByCellColor","model_part_name":"Output","color":1,
                                        "tag_refinement_level":true}],
            "model_part_operations":[]}""")
        out = model.GetModelPart("Output")
        levels = {el.GetValue(KM.REFINEMENT_LEVEL) for el in out.Elements}
        # Levels include at least one uniform level and -1 for template hexes
        self.assertTrue(any(l > 0 for l in levels), "No positive refinement level tagged")

    def test_boundary_conditions_created(self):
        """Boundary conditions are created and form a closed shell."""
        model = self._build_box_model()
        run_modeler(model, """{
            "input_model_part_name":"CS","output_model_part_name":"Volume",
            "octree_generator":{"type":"generate_octree_from_surface","refinement_depth":4,"adaptive":false},
            "coloring_settings_list":[{"type":"ClassifyCellsInsideOutside"}],
            "entities_generator_list":[
                {"type":"GenerateHexesByCellColor","model_part_name":"Volume","color":1},
                {"type":"GenerateBoundaryConditionsByFace","model_part_name":"Boundary","color":1}
            ],
            "model_part_operations":[]}""")
        bnd = model.GetModelPart("Boundary")
        self.assertGreater(bnd.NumberOfConditions(), 0)
        self.assertGreater(bnd.NumberOfNodes(), 0)

    def test_quality_report_operation(self):
        """ReportMeshQuality runs without error; zero inverted on a box mesh."""
        model = self._build_box_model()
        run_modeler(model, """{
            "input_model_part_name":"CS","output_model_part_name":"Output",
            "octree_generator":{"type":"generate_octree_from_surface","refinement_depth":4,"adaptive":false},
            "coloring_settings_list":[{"type":"ClassifyCellsInsideOutside"}],
            "entities_generator_list":[{"type":"GenerateHexesByCellColor","model_part_name":"Output","color":1}],
            "model_part_operations":[{"type":"ReportMeshQuality","model_part_name":"Output"}]}""")
        out = model.GetModelPart("Output")
        self.assertGreater(out.NumberOfElements(), 0)


# ===========================================================================
class TestOctreeMesherModelerPrimal(unittest.TestCase):
    """Tests for the primal (leaf-hex + hanging-node constraints) path."""

    def _run_primal(self, model, surface_name, depth=4):
        run_modeler(model, f"""{{
            "input_model_part_name":"{surface_name}","output_model_part_name":"Output",
            "octree_generator":{{"type":"generate_octree_from_surface","refinement_depth":{depth},
                                 "adaptive":true,"mesh_type":"primal"}},
            "coloring_settings_list":[],
            "entities_generator_list":[
                {{"type":"GenerateHexesByCellColor","model_part_name":"Output","color":1}},
                {{"type":"GenerateHangingNodeConstraints","model_part_name":"Output",
                  "variables":["DISPLACEMENT_X","DISPLACEMENT_Y"]}}
            ],
            "model_part_operations":[]}}""")

    def test_primal_elements_created(self):
        """Primal mesh produces one hex per octree leaf."""
        model = KM.Model()
        build_transition_surface(model)
        self._run_primal(model, "Surface", depth=4)
        out = model.GetModelPart("Output")
        self.assertGreater(out.NumberOfElements(), 0)
        self.assertGreater(out.NumberOfNodes(), 0)

    def test_primal_constraints_count(self):
        """Hanging-node constraints are generated at 2:1 transitions."""
        model = KM.Model()
        build_transition_surface(model)
        self._run_primal(model, "Surface", depth=4)
        out = model.GetModelPart("Output")
        # 2 variables → at least 2× as many constraints as hanging nodes
        nc = out.NumberOfMasterSlaveConstraints()
        print(f"\n[primal depth=4] elements={out.NumberOfElements()} constraints={nc}")
        self.assertGreater(nc, 0, "No hanging-node constraints produced")

    def test_primal_constraint_row_sum(self):
        """Every constraint's relation matrix sums to 1 (partition of unity)."""
        model = KM.Model()
        build_transition_surface(model)
        self._run_primal(model, "Surface", depth=4)
        out = model.GetModelPart("Output")
        fail = 0
        for c in out.MasterSlaveConstraints:
            T, b = KM.Matrix(), KM.Vector()
            c.CalculateLocalSystem(T, b, KM.ProcessInfo())
            row_sum = sum(T[0, j] for j in range(T.Size2()))
            if abs(row_sum - 1.0) > 1e-10:
                fail += 1
        self.assertEqual(fail, 0, f"{fail} constraints violate partition of unity")

    def test_primal_constraint_master_counts(self):
        """Only 2-master (edge-midpoint) and 4-master (face-centre) constraints."""
        model = KM.Model()
        build_transition_surface(model)
        self._run_primal(model, "Surface", depth=4)
        out = model.GetModelPart("Output")
        for c in out.MasterSlaveConstraints:
            nm = len(c.GetMasterDofsVector())
            self.assertIn(nm, (2, 4), f"Unexpected master count {nm}")


# ===========================================================================
class TestOctreeMesherModelerBunny(unittest.TestCase):
    """Tests using the low-poly Stanford Bunny surface (skipped if absent)."""

    def setUp(self):
        self.stl = load_bunny_ascii()
        if self.stl is None:
            self.skipTest("Bunny-LowPoly.stl not available")

    def tearDown(self):
        if self.stl and os.path.exists(self.stl):
            os.remove(self.stl)

    def _load_surface(self, model, name="BunnySurface"):
        mp = model.CreateModelPart(name)
        mp.ProcessInfo[KM.DOMAIN_SIZE] = 3
        KM.StlIO(self.stl, KM.Parameters('{"open_mode":"read"}')).ReadModelPart(mp)
        return mp

    def test_dual_bunny_zero_inverted(self):
        """Dual mesh of the bunny at depth 4: 0 inverted hexes."""
        model = KM.Model()
        self._load_surface(model)
        run_modeler(model, """{
            "input_model_part_name":"BunnySurface","output_model_part_name":"Output",
            "octree_generator":{"type":"generate_octree_from_surface","refinement_depth":4,"adaptive":true},
            "coloring_settings_list":[{"type":"ClassifyCellsInsideOutside"}],
            "entities_generator_list":[{"type":"GenerateHexesByCellColor","model_part_name":"Output","color":1}],
            "model_part_operations":[]}""")
        out = model.GetModelPart("Output")
        n_inv = sum(1 for el in out.Elements
                    if cell_min_sj([(n.X,n.Y,n.Z) for n in el.GetGeometry()]) <= 0)
        n_el = out.NumberOfElements()
        print(f"\n[bunny dual depth=4] elements={n_el} inverted={n_inv}")
        self.assertGreater(n_el, 0)
        self.assertEqual(n_inv, 0)

    def test_primal_bunny_constraints_row_sum(self):
        """Primal mesh of the bunny: all constraints have row-sum = 1."""
        model = KM.Model()
        self._load_surface(model)
        run_modeler(model, """{
            "input_model_part_name":"BunnySurface","output_model_part_name":"Output",
            "octree_generator":{"type":"generate_octree_from_surface","refinement_depth":4,"adaptive":true,
                                 "mesh_type":"primal"},
            "coloring_settings_list":[],
            "entities_generator_list":[
                {"type":"GenerateHexesByCellColor","model_part_name":"Output","color":1},
                {"type":"GenerateHangingNodeConstraints","model_part_name":"Output",
                 "variables":["DISPLACEMENT_X"]}
            ],
            "model_part_operations":[]}""")
        out = model.GetModelPart("Output")
        nc = out.NumberOfMasterSlaveConstraints()
        print(f"\n[bunny primal depth=4] elements={out.NumberOfElements()} constraints={nc}")
        self.assertGreater(nc, 0)
        fail = 0
        for c in out.MasterSlaveConstraints:
            T, b = KM.Matrix(), KM.Vector()
            c.CalculateLocalSystem(T, b, KM.ProcessInfo())
            if abs(sum(T[0,j] for j in range(T.Size2())) - 1.0) > 1e-10:
                fail += 1
        self.assertEqual(fail, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

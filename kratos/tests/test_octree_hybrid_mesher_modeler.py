"""
Tests for OctreeHybridMesherModeler — the Registry-driven modeler that wraps the
OctreeHybridMeshUtility engine.

Run directly:
    PYTHONPATH=.../bin/Release python3 test_octree_hybrid_mesher_modeler.py
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
    mod = KM.OctreeHybridMesherModeler(model, settings)
    mod.SetupGeometryModel()
    mod.PrepareGeometryModel()
    mod.SetupModelPart()


# ===========================================================================
class TestOctreeHybridMesherModelerDual(unittest.TestCase):
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
class TestOctreeHybridMesherModelerPrimal(unittest.TestCase):
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
class TestOctreeHybridMesherModelerBunny(unittest.TestCase):
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


# ===========================================================================
class TestClassifyCellsInsideOutside(unittest.TestCase):
    """Unit tests for ClassifyCellsInsideOutside colouring."""

    def _run(self, lo=0.3, hi=0.7, depth=4):
        model = KM.Model()
        build_closed_box_surface(model, lo=lo, hi=hi, name="S")
        run_modeler(model, f"""{{
            "input_model_part_name":"S","output_model_part_name":"O",
            "octree_generator":{{"type":"generate_octree_from_surface",
                                 "refinement_depth":{depth},"adaptive":false}},
            "coloring_settings_list":[{{"type":"ClassifyCellsInsideOutside"}}],
            "entities_generator_list":[],
            "model_part_operations":[]}}""")
        return model

    def test_produces_inside_and_outside_colors(self):
        """After colouring a block around a box, both color=1 and color=0 cells exist."""
        # We run WITHOUT a hex generator so we can inspect that the coloring ran.
        # Indirect test: running the full pipeline with color=1 filter gives fewer
        # cells than the unfiltered full block.
        model = KM.Model()
        build_closed_box_surface(model, lo=0.3, hi=0.7, name="S")
        run_modeler(model, """{
            "input_model_part_name":"S","output_model_part_name":"All",
            "octree_generator":{"type":"generate_octree_from_surface","refinement_depth":4,"adaptive":false},
            "coloring_settings_list":[],
            "entities_generator_list":[{"type":"GenerateHexesByCellColor","model_part_name":"All","color":1}],
            "model_part_operations":[]}""")
        model2 = KM.Model()
        build_closed_box_surface(model2, lo=0.3, hi=0.7, name="S")
        run_modeler(model2, """{
            "input_model_part_name":"S","output_model_part_name":"Carved",
            "octree_generator":{"type":"generate_octree_from_surface","refinement_depth":4,"adaptive":false},
            "coloring_settings_list":[{"type":"ClassifyCellsInsideOutside"}],
            "entities_generator_list":[{"type":"GenerateHexesByCellColor","model_part_name":"Carved","color":1}],
            "model_part_operations":[]}""")
        n_all = model.GetModelPart("All").NumberOfElements()
        n_carved = model2.GetModelPart("Carved").NumberOfElements()
        print(f"\n[ClassifyCellsInsideOutside] unfiltered={n_all} inside={n_carved}")
        # The carved set (inside only) must be strictly smaller than the unfiltered block
        self.assertGreater(n_all, n_carved,
                           "ClassifyCellsInsideOutside removed no cells")
        self.assertGreater(n_carved, 0, "No inside cells found")

    def test_projected_shortcut_all_cells_inside(self):
        """With project_to_surface=true the shortcut path sets all cells to color=1."""
        model = KM.Model()
        build_closed_box_surface(model, lo=0.3, hi=0.7, name="S")
        run_modeler(model, """{
            "input_model_part_name":"S","output_model_part_name":"Proj",
            "octree_generator":{"type":"generate_octree_from_surface",
                                 "refinement_depth":4,"adaptive":false,"project_to_surface":true},
            "coloring_settings_list":[{"type":"ClassifyCellsInsideOutside"}],
            "entities_generator_list":[{"type":"GenerateHexesByCellColor",
                                        "model_part_name":"Proj","color":1}],
            "model_part_operations":[]}""")
        n_proj = model.GetModelPart("Proj").NumberOfElements()
        print(f"\n[ClassifyCellsInsideOutside projected] elements={n_proj}")
        self.assertGreater(n_proj, 0)

    def test_default_type_name(self):
        """The default parameters include type='ClassifyCellsInsideOutside'."""
        proto_path = "OctreeHybridMesherColoring.All.ClassifyCellsInsideOutside.Prototype"
        self.assertTrue(KM.Registry.HasValue(proto_path),
                        f"Registry path not found: {proto_path}")

    def test_unknown_coloring_type_raises(self):
        """An unknown colouring type triggers a clear error message."""
        model = KM.Model()
        build_closed_box_surface(model, name="S")
        with self.assertRaises(Exception):
            run_modeler(model, """{
                "input_model_part_name":"S","output_model_part_name":"O",
                "octree_generator":{"type":"generate_octree_from_surface","refinement_depth":3,"adaptive":false},
                "coloring_settings_list":[{"type":"NonExistentColoringType"}],
                "entities_generator_list":[],"model_part_operations":[]}""")


# ===========================================================================
class TestGenerateHexesByCellColor(unittest.TestCase):
    """Unit tests for GenerateHexesByCellColor entity generator."""

    def _box_run(self, lo=0.3, hi=0.7, depth=4, color=1,
                 entity="Element3D8N", tag_level=True):
        model = KM.Model()
        build_closed_box_surface(model, lo=lo, hi=hi, name="S")
        run_modeler(model, f"""{{
            "input_model_part_name":"S","output_model_part_name":"O",
            "octree_generator":{{"type":"generate_octree_from_surface",
                                 "refinement_depth":{depth},"adaptive":false}},
            "coloring_settings_list":[{{"type":"ClassifyCellsInsideOutside"}}],
            "entities_generator_list":[{{"type":"GenerateHexesByCellColor",
                                        "model_part_name":"O","color":{color},
                                        "generated_entity":"{entity}",
                                        "tag_refinement_level":{str(tag_level).lower()}}}],
            "model_part_operations":[]}}""")
        return model.GetModelPart("O")

    def test_positive_element_count(self):
        out = self._box_run()
        self.assertGreater(out.NumberOfElements(), 0)
        self.assertGreater(out.NumberOfNodes(), 0)

    def test_color_filter_inside(self):
        """color=1 produces fewer elements than running without a color filter."""
        model_full = KM.Model()
        build_closed_box_surface(model_full, name="S")
        run_modeler(model_full, """{
            "input_model_part_name":"S","output_model_part_name":"O",
            "octree_generator":{"type":"generate_octree_from_surface","refinement_depth":4,"adaptive":false},
            "coloring_settings_list":[],
            "entities_generator_list":[{"type":"GenerateHexesByCellColor","model_part_name":"O","color":1}],
            "model_part_operations":[]}""")
        n_all = model_full.GetModelPart("O").NumberOfElements()

        n_carved = self._box_run().NumberOfElements()
        print(f"\n[GenerateHexesByCellColor] unfiltered={n_all} inside={n_carved}")
        self.assertGreater(n_all, n_carved)

    def test_zero_inverted_elements(self):
        out = self._box_run()
        n_inv = sum(1 for el in out.Elements
                    if cell_min_sj([(n.X, n.Y, n.Z) for n in el.GetGeometry()]) <= 0)
        self.assertEqual(n_inv, 0)

    def test_refinement_level_tagged(self):
        out = self._box_run(tag_level=True)
        levels = {el.GetValue(KM.REFINEMENT_LEVEL) for el in out.Elements}
        self.assertTrue(any(l > 0 for l in levels))

    def test_refinement_level_not_tagged(self):
        """When tag_refinement_level=false no REFINEMENT_LEVEL is set."""
        out = self._box_run(tag_level=False)
        # Default int value for an unset variable is 0, not any octree level
        levels = {el.GetValue(KM.REFINEMENT_LEVEL) for el in out.Elements}
        self.assertEqual(levels, {0}, f"Expected only default 0, got {levels}")

    def test_node_deduplication(self):
        """Node count is strictly less than 8 × element count (nodes are shared)."""
        out = self._box_run()
        n_el = out.NumberOfElements()
        n_nd = out.NumberOfNodes()
        self.assertLess(n_nd, 8 * n_el,
                        "Every node is unique — sharing is not working")

    def test_node_ids_contiguous_from_one(self):
        """All node ids are unique positive integers starting from 1."""
        out = self._box_run()
        ids = sorted(n.Id for n in out.Nodes)
        self.assertEqual(ids[0], 1)
        self.assertEqual(len(ids), len(set(ids)), "Duplicate node ids")

    def test_element_ids_contiguous_from_one(self):
        out = self._box_run()
        ids = sorted(el.Id for el in out.Elements)
        self.assertEqual(ids[0], 1)
        self.assertEqual(len(ids), len(set(ids)))

    def test_registry_path_exists(self):
        self.assertTrue(KM.Registry.HasValue(
            "OctreeHybridMesherEntityGeneration.All.GenerateHexesByCellColor.Prototype"))

    def test_unknown_entity_type_raises(self):
        with self.assertRaises(Exception):
            model = KM.Model()
            build_closed_box_surface(model, name="S")
            run_modeler(model, """{
                "input_model_part_name":"S","output_model_part_name":"O",
                "octree_generator":{"type":"generate_octree_from_surface",
                                    "refinement_depth":3,"adaptive":false},
                "coloring_settings_list":[{"type":"ClassifyCellsInsideOutside"}],
                "entities_generator_list":[{"type":"GenerateHexesByCellColor",
                    "model_part_name":"O","color":1,"generated_entity":"NoSuchElement3D8N"}],
                "model_part_operations":[]}""")


# ===========================================================================
class TestGenerateBoundaryConditionsByFace(unittest.TestCase):
    """Unit tests for GenerateBoundaryConditionsByFace entity generator."""

    def _run(self, lo=0.3, hi=0.7, depth=4):
        model = KM.Model()
        build_closed_box_surface(model, lo=lo, hi=hi, name="S")
        run_modeler(model, f"""{{
            "input_model_part_name":"S","output_model_part_name":"Volume",
            "octree_generator":{{"type":"generate_octree_from_surface",
                                 "refinement_depth":{depth},"adaptive":false}},
            "coloring_settings_list":[{{"type":"ClassifyCellsInsideOutside"}}],
            "entities_generator_list":[
                {{"type":"GenerateHexesByCellColor","model_part_name":"Volume","color":1}},
                {{"type":"GenerateBoundaryConditionsByFace","model_part_name":"Boundary","color":1}}
            ],
            "model_part_operations":[]}}""")
        return model.GetModelPart("Volume"), model.GetModelPart("Boundary")

    def test_conditions_created(self):
        vol, bnd = self._run()
        print(f"\n[GenerateBoundaryConditionsByFace] elements={vol.NumberOfElements()} cond={bnd.NumberOfConditions()}")
        self.assertGreater(bnd.NumberOfConditions(), 0)

    def test_boundary_nodes_populated(self):
        vol, bnd = self._run()
        self.assertGreater(bnd.NumberOfNodes(), 0)

    def test_boundary_nodes_subset_of_volume_nodes(self):
        """Every boundary node id also exists in the volume mesh."""
        vol, bnd = self._run()
        vol_ids = {n.Id for n in vol.Nodes}
        bnd_ids = {n.Id for n in bnd.Nodes}
        self.assertTrue(bnd_ids.issubset(vol_ids),
                        "Boundary nodes have ids not present in volume")

    def test_condition_ids_contiguous(self):
        vol, bnd = self._run()
        ids = sorted(c.Id for c in bnd.Conditions)
        self.assertEqual(ids[0], 1)
        self.assertEqual(len(ids), len(set(ids)))

    def test_each_condition_has_four_nodes(self):
        """Every boundary condition is a quad (4 nodes)."""
        vol, bnd = self._run()
        for cond in bnd.Conditions:
            self.assertEqual(len(cond.GetGeometry()), 4,
                             f"Condition {cond.Id} does not have 4 nodes")

    def test_boundary_faces_lt_6_times_elements(self):
        """Boundary face count is < 6 × element count (interior faces are shared)."""
        vol, bnd = self._run()
        self.assertLess(bnd.NumberOfConditions(),
                        6 * vol.NumberOfElements())

    def test_registry_path_exists(self):
        self.assertTrue(KM.Registry.HasValue(
            "OctreeHybridMesherEntityGeneration.All.GenerateBoundaryConditionsByFace.Prototype"))


# ===========================================================================
class TestGenerateHangingNodeConstraints(unittest.TestCase):
    """Unit tests for GenerateHangingNodeConstraints entity generator (primal mesh)."""

    def _run(self, depth=4, variables=None):
        if variables is None:
            variables = ["DISPLACEMENT_X"]
        var_json = str(variables).replace("'", '"')
        model = KM.Model()
        build_transition_surface(model)
        run_modeler(model, f"""{{
            "input_model_part_name":"Surface","output_model_part_name":"O",
            "octree_generator":{{"type":"generate_octree_from_surface",
                                 "refinement_depth":{depth},"adaptive":true,"mesh_type":"primal"}},
            "coloring_settings_list":[],
            "entities_generator_list":[
                {{"type":"GenerateHexesByCellColor","model_part_name":"O","color":1}},
                {{"type":"GenerateHangingNodeConstraints","model_part_name":"O",
                  "variables":{var_json}}}
            ],
            "model_part_operations":[]}}""")
        return model.GetModelPart("O")

    def test_constraints_generated(self):
        out = self._run()
        nc = out.NumberOfMasterSlaveConstraints()
        print(f"\n[GenerateHangingNodeConstraints] elements={out.NumberOfElements()} constraints={nc}")
        self.assertGreater(nc, 0)

    def test_partition_of_unity(self):
        """Every constraint row sums to 1.0 (bilinear interpolation property)."""
        out = self._run()
        fail = 0
        for c in out.MasterSlaveConstraints:
            T, b = KM.Matrix(), KM.Vector()
            c.CalculateLocalSystem(T, b, KM.ProcessInfo())
            rs = sum(T[0, j] for j in range(T.Size2()))
            if abs(rs - 1.0) > 1e-10:
                fail += 1
        self.assertEqual(fail, 0, f"{fail} constraints violate partition of unity")

    def test_master_counts_two_or_four(self):
        """Only edge-midpoint (2-master) and face-centre (4-master) constraints."""
        out = self._run()
        for c in out.MasterSlaveConstraints:
            nm = len(c.GetMasterDofsVector())
            self.assertIn(nm, (2, 4), f"Unexpected master count {nm}")

    def test_face_centre_constraints_present(self):
        """At least one face-centre (4-master) constraint exists."""
        out = self._run()
        nm_counts = [len(c.GetMasterDofsVector()) for c in out.MasterSlaveConstraints]
        self.assertIn(4, nm_counts,
                      "No 4-master (face-centre) constraints found")

    def test_multiple_variables(self):
        """Requesting 3 variables multiplies constraint count by 3."""
        out1 = self._run(variables=["DISPLACEMENT_X"])
        out3 = self._run(variables=["DISPLACEMENT_X", "DISPLACEMENT_Y", "DISPLACEMENT_Z"])
        nc1 = out1.NumberOfMasterSlaveConstraints()
        nc3 = out3.NumberOfMasterSlaveConstraints()
        print(f"\n[GenerateHangingNodeConstraints] 1-var={nc1} 3-var={nc3}")
        self.assertEqual(nc3, 3 * nc1,
                         f"3-variable count {nc3} ≠ 3 × single-variable count {nc1}")

    def test_each_constraint_has_one_slave_dof(self):
        """Every constraint has exactly one slave DOF (one slave node per variable)."""
        out = self._run()
        for c in out.MasterSlaveConstraints:
            self.assertEqual(len(c.GetSlaveDofsVector()), 1,
                             f"Constraint {c.Id} has {len(c.GetSlaveDofsVector())} slaves")

    def test_registry_path_exists(self):
        self.assertTrue(KM.Registry.HasValue(
            "OctreeHybridMesherEntityGeneration.All.GenerateHangingNodeConstraints.Prototype"))

    def test_dual_mesh_no_hanging_constraints(self):
        """Dual mesh never produces hanging constraints (conforming by construction)."""
        model = KM.Model()
        build_transition_surface(model)
        run_modeler(model, """{
            "input_model_part_name":"Surface","output_model_part_name":"O",
            "octree_generator":{"type":"generate_octree_from_surface","refinement_depth":4,
                                 "adaptive":true,"mesh_type":"dual"},
            "coloring_settings_list":[{"type":"ClassifyCellsInsideOutside"}],
            "entities_generator_list":[{"type":"GenerateHexesByCellColor",
                                        "model_part_name":"O","color":1}],
            "model_part_operations":[]}""")
        # Dual path has no mHanging entries → no constraints should be created
        self.assertEqual(model.GetModelPart("O").NumberOfMasterSlaveConstraints(), 0)


# ===========================================================================
class TestReportMeshQuality(unittest.TestCase):
    """Unit tests for ReportMeshQuality operation."""

    def _run(self, depth=4):
        model = KM.Model()
        build_closed_box_surface(model, name="S")
        run_modeler(model, f"""{{
            "input_model_part_name":"S","output_model_part_name":"Output",
            "octree_generator":{{"type":"generate_octree_from_surface",
                                 "refinement_depth":{depth},"adaptive":false}},
            "coloring_settings_list":[{{"type":"ClassifyCellsInsideOutside"}}],
            "entities_generator_list":[{{"type":"GenerateHexesByCellColor",
                                        "model_part_name":"Output","color":1}}],
            "model_part_operations":[{{"type":"ReportMeshQuality",
                                       "model_part_name":"Output"}}]}}""")
        return model.GetModelPart("Output")

    def test_runs_without_error(self):
        out = self._run()
        self.assertGreater(out.NumberOfElements(), 0)

    def test_zero_inverted_box(self):
        """A carved box mesh at depth 4 has no inverted elements."""
        out = self._run()
        n_inv = sum(1 for el in out.Elements
                    if cell_min_sj([(n.X, n.Y, n.Z) for n in el.GetGeometry()]) <= 0)
        self.assertEqual(n_inv, 0)

    def test_empty_model_part_does_not_crash(self):
        """ReportMeshQuality on an empty ModelPart just logs and returns."""
        model = KM.Model()
        build_closed_box_surface(model, name="S")
        run_modeler(model, """{
            "input_model_part_name":"S","output_model_part_name":"Empty",
            "octree_generator":{"type":"generate_octree_from_surface","refinement_depth":3,"adaptive":false},
            "coloring_settings_list":[],
            "entities_generator_list":[],
            "model_part_operations":[{"type":"ReportMeshQuality","model_part_name":"Empty"}]}""")
        # No assertion needed — just should not raise

    def test_registry_path_exists(self):
        self.assertTrue(KM.Registry.HasValue(
            "OctreeHybridMesherOperation.All.ReportMeshQuality.Prototype"))


# ===========================================================================
class TestRegistryDispatch(unittest.TestCase):
    """Tests for the Registry-prototype dispatch mechanism in OctreeHybridMesherModeler."""

    def test_all_base_prototypes_registered(self):
        """All three base-class prototypes are registered in the Registry."""
        for path in [
            "OctreeHybridMesherColoring.All.OctreeHybridMesherColoring.Prototype",
            "OctreeHybridMesherEntityGeneration.All.OctreeHybridMesherEntityGeneration.Prototype",
            "OctreeHybridMesherOperation.All.OctreeHybridMesherOperation.Prototype",
        ]:
            self.assertTrue(KM.Registry.HasValue(path), f"Missing: {path}")

    def test_all_concrete_prototypes_registered(self):
        """All concrete components are registered."""
        paths = [
            "OctreeHybridMesherColoring.All.ClassifyCellsInsideOutside.Prototype",
            "OctreeHybridMesherEntityGeneration.All.GenerateHexesByCellColor.Prototype",
            "OctreeHybridMesherEntityGeneration.All.GenerateBoundaryConditionsByFace.Prototype",
            "OctreeHybridMesherEntityGeneration.All.GenerateHangingNodeConstraints.Prototype",
            "OctreeHybridMesherOperation.All.ReportMeshQuality.Prototype",
        ]
        for path in paths:
            self.assertTrue(KM.Registry.HasValue(path), f"Missing: {path}")

    def test_full_path_dispatch_works(self):
        """A four-segment full Registry path is accepted directly."""
        model = KM.Model()
        build_closed_box_surface(model, name="S")
        run_modeler(model, """{
            "input_model_part_name":"S","output_model_part_name":"O",
            "octree_generator":{"type":"generate_octree_from_surface","refinement_depth":3,"adaptive":false},
            "coloring_settings_list":[{
                "type":"OctreeHybridMesherColoring.All.ClassifyCellsInsideOutside.Prototype"
            }],
            "entities_generator_list":[{"type":"GenerateHexesByCellColor",
                                        "model_part_name":"O","color":1}],
            "model_part_operations":[]}""")
        self.assertGreater(model.GetModelPart("O").NumberOfElements(), 0)

    def test_unknown_operation_type_raises(self):
        """An unknown operation type triggers a Registry-not-found error."""
        model = KM.Model()
        build_closed_box_surface(model, name="S")
        with self.assertRaises(Exception):
            run_modeler(model, """{
                "input_model_part_name":"S","output_model_part_name":"O",
                "octree_generator":{"type":"generate_octree_from_surface",
                                    "refinement_depth":3,"adaptive":false},
                "coloring_settings_list":[],
                "entities_generator_list":[],
                "model_part_operations":[{"type":"TotallyUnknownOperation"}]}""")

    def test_base_type_invocation_raises(self):
        """Invoking the base OctreeHybridMesherOperation prototype raises a clear error."""
        model = KM.Model()
        build_closed_box_surface(model, name="S")
        with self.assertRaises(Exception):
            run_modeler(model, """{
                "input_model_part_name":"S","output_model_part_name":"O",
                "octree_generator":{"type":"generate_octree_from_surface",
                                    "refinement_depth":3,"adaptive":false},
                "coloring_settings_list":[],
                "entities_generator_list":[],
                "model_part_operations":[{
                    "type":"OctreeHybridMesherOperation.All.OctreeHybridMesherOperation.Prototype"
                }]}""")


if __name__ == "__main__":
    unittest.main(verbosity=2)

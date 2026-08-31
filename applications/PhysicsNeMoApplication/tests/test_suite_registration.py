"""Guards three things that rot while the suite stays green.

The first two are static: they parse sources with `ast` and never import
the tests or the application, so they run unchanged in the torch-free CI
and cannot be disabled by a missing optional dependency. The third runs the
benchmark script, which is pure Kratos + numpy for the same reason.

**Registration.** Suites are assembled by hand, and CLAUDE.md lists the
consequence as a known gotcha - adding a `test_*.py` does not add it to the
suite. An unregistered class is invisible: the run passes, the count looks
healthy, and the tests never execute. Nothing else in the suite notices.

**Documented identifiers.** The README and the documentation pages name
modules and functions. A rename leaves the prose pointing at something that
no longer exists, and no test would fail.

**The benchmark.** `benchmarks/benchmark_bridges.py` is executed by nothing
- not CMake, not this suite, not CI - while the README cites its result to
justify a roadmap decision. See the class docstring.

What the identifier check deliberately does *not* do: catch a stale claim about
project state. The roadmap once said the DoMINO predictor-corrector recipe
was still open after it had shipped; every identifier in that sentence
existed, so this check would have passed it. Prose about status has to be
read against the tree. This catches renames, which is a different failure.
"""

import ast
import glob
import os
import re
import subprocess
import sys
from pathlib import Path

import KratosMultiphysics.KratosUnittest as KratosUnittest

_TESTS_DIR = Path(__file__).resolve().parent
_APP_DIR = _TESTS_DIR.parent
_SCRIPTS_DIR = _APP_DIR / "python_scripts"
_BENCHMARK = _APP_DIR / "benchmarks" / "benchmark_bridges.py"
_DOCS_DIR = (_APP_DIR.parent.parent / "docs" / "pages" / "Applications"
             / "PhysicsNeMo_Application")

_RUNNERS = ("test_PhysicsNeMoApplication.py", "test_PhysicsNeMoApplication_mpi.py")

# tokens ending in one of these are file names, not attribute references:
# `graph_bridge.py` must not be read as module graph_bridge, attribute py
_FILE_SUFFIXES = (".py", ".md", ".json", ".mdpa", ".npz", ".pt", ".mdlus",
                  ".h5", ".so", ".txt", ".sh", ".bat", ".onnx", ".ipynb",
                  ".vtu", ".zarr", ".stl", ".csv", ".yaml", ".yml")


def _ParseFile(path):
    try:
        return ast.parse(Path(path).read_text(errors="ignore"))
    except (SyntaxError, OSError):
        return None


def _TestCaseClassesByFile():
    """{class name: file name} for every TestCase subclass outside the runners."""
    found = {}
    # rglob, not glob: the suite puts subdirectories such as mesh_bridge/ on
    # sys.path, so their test files are as much part of the suite as the top
    # level ones - and were the blind spot that let one slip through
    for path in sorted(_TESTS_DIR.rglob("test_*.py")):
        if path.name in _RUNNERS or "__pycache__" in path.parts:
            continue
        tree = _ParseFile(path)
        if tree is None:
            continue
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            bases = [getattr(b, "attr", None) or getattr(b, "id", "")
                     for b in node.bases]
            if any("TestCase" in base for base in bases):
                found[node.name] = path.name
    return found


def _RegisteredClasses():
    """Class names actually added to a suite.

    Only names inside a `loadTestsFromTestCases([...])` call count. An import
    alone does not: importing a class and forgetting to add it is precisely
    the half-finished registration this is here to catch.
    """
    registered = set()
    for name in _RUNNERS:
        path = _TESTS_DIR / name
        if not path.is_file():
            continue
        for group in re.findall(r"loadTestsFromTestCases\(\[([\w,\s]+)\]\)",
                                path.read_text(errors="ignore")):
            registered |= {token.strip() for token in group.split(",") if token.strip()}
    return registered


def _PublicNamesByModule():
    """{module stem: {public top-level def/class names}} for the application.

    Read with `ast` rather than by importing, so a module whose imports need
    torch is still covered here.
    """
    public = {}
    for path in glob.glob(str(_SCRIPTS_DIR / "**" / "*.py"), recursive=True):
        stem = os.path.basename(path)[:-3]
        if stem == "__init__":
            continue
        tree = _ParseFile(path)
        if tree is None:
            continue
        public[stem] = {node.name for node in tree.body
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                             ast.ClassDef))
                        and not node.name.startswith("_")}
    return public


def _DocumentationFiles():
    files = [_APP_DIR / "README.md"]
    files += [Path(p) for p in
              glob.glob(str(_DOCS_DIR / "**" / "*.md"), recursive=True)]
    return [f for f in files if f.is_file()]


class TestSuiteRegistration(KratosUnittest.TestCase):

    def test_EveryTestCaseClassIsRegisteredInASuite(self):
        defined = _TestCaseClassesByFile()
        self.assertTrue(
            defined,
            f"no TestCase classes found under {_TESTS_DIR}; this guard is not "
            "actually checking anything")

        unregistered = sorted(set(defined) - _RegisteredClasses())
        self.assertFalse(
            unregistered,
            "these TestCase classes are never added to a suite, so their tests "
            "do not run - add them to AssembleTestSuites() in "
            f"{_RUNNERS[0]} or {_RUNNERS[1]}: "
            + ", ".join(f"{defined[name]}::{name}" for name in unregistered))


class TestDocumentedIdentifiersExist(KratosUnittest.TestCase):

    def test_BacktickedModulesAndAttributesResolve(self):
        public = _PublicNamesByModule()
        self.assertTrue(public, f"no modules found under {_SCRIPTS_DIR}")
        documents = _DocumentationFiles()
        self.assertTrue(documents, "no documentation files found to check")

        broken = []
        module_hits = attribute_hits = 0
        for document in documents:
            tokens = set(re.findall(r"`([^`\n]+)`",
                                    document.read_text(errors="ignore")))
            for token in tokens:
                name = token.strip().replace("()", "")
                if name.endswith(_FILE_SUFFIXES):
                    continue
                if name in public:
                    module_hits += 1
                    continue
                dotted = re.fullmatch(r"([a-z_][a-z0-9_]*)\.([A-Za-z_]\w*)", name)
                # only our own modules; external APIs are not ours to verify
                if dotted and dotted.group(1) in public:
                    attribute_hits += 1
                    if dotted.group(2) not in public[dotted.group(1)]:
                        broken.append(f"{document.name}: `{token}`")

        self.assertFalse(
            broken,
            "documentation names application code that does not exist "
            "(renamed or removed?): " + ", ".join(sorted(set(broken))))
        self.assertTrue(
            module_hits or attribute_hits,
            "no documented references were resolved at all, so this guard "
            "passed without checking anything")


class TestBenchmarkStillRuns(KratosUnittest.TestCase):
    """The benchmark is a script nothing else executes.

    Not CMake, not this suite, not CI - and the README cites its result to
    justify a roadmap decision. That is how its headline number drifted from
    what it actually measures without anyone noticing, so at minimum the
    script itself must keep running against the bridges it times.

    Deliberately asserts nothing about *timings*. A wall-clock assertion in
    a unit suite is flaky by construction; this checks the API, and the
    numbers stay a thing a human runs and reads.
    """

    def test_TheBridgeBenchmarkExecutes(self):
        self.assertTrue(_BENCHMARK.is_file(), f"{_BENCHMARK} is missing")

        completed = subprocess.run(
            [sys.executable, str(_BENCHMARK),
             "--divisions", "4", "--grid", "8", "--repeat", "1"],
            capture_output=True, text=True, timeout=600)

        self.assertEqual(
            completed.returncode, 0,
            f"{_BENCHMARK.name} failed:\n{completed.stdout[-2000:]}\n"
            f"{completed.stderr[-2000:]}")
        self.assertIn("us/entity", completed.stdout,
                      "the benchmark ran but printed no results table")


if __name__ == "__main__":
    KratosUnittest.main()

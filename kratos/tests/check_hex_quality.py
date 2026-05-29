"""
Diagnostic: read a legacy VTK hex mesh and report degenerate / inverted hexes.

Splits the report by the "level" cell-scalar so we can tell whether the bad
elements come from the plain dual hexes (level >= 0) or the 13-element
transition template (level == -1).

Usage:
    python3 check_hex_quality.py [mesh.vtk]
"""
import sys

# VTK_HEXAHEDRON node order: 0-3 bottom face, 4-7 top face (matching)
HEX_FACES = [  # outward for a right-handed cube
    (0, 3, 2, 1),  # bottom (-z)
    (4, 5, 6, 7),  # top    (+z)
    (0, 1, 5, 4),  # -y
    (1, 2, 6, 5),  # +x
    (2, 3, 7, 6),  # +y
    (3, 0, 4, 7),  # -x
]
# The 8 corner tetra of a hex, for scaled-Jacobian sign at each corner
CORNER_TETS = [
    (0, 1, 3, 4), (1, 2, 0, 5), (2, 3, 1, 6), (3, 0, 2, 7),
    (4, 7, 5, 0), (5, 4, 6, 1), (6, 5, 7, 2), (7, 6, 4, 3),
]


def read_vtk(path):
    pts, cells, levels = [], [], []
    with open(path) as f:
        lines = f.read().split("\n")
    i = 0
    n = len(lines)
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
                c = list(map(int, lines[i].split()))
                cells.append(c[1:])
                i += 1
            continue
        if t[0] == "SCALARS" and "level" in lines[i]:
            i += 2  # skip SCALARS + LOOKUP_TABLE
            while len(levels) < len(cells):
                levels.extend(map(int, lines[i].split()))
                i += 1
            continue
        i += 1
    return pts, cells, levels


def sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def norm(a):
    return dot(a, a) ** 0.5


def analyze(pts, cells, levels):
    if not levels:
        levels = [0] * len(cells)
    stats = {}  # level -> dict of counters
    for h, lv in zip(cells, levels):
        s = stats.setdefault(lv, dict(total=0, degenerate=0, inverted=0, nonplanar=0))
        s["total"] += 1
        coords = [pts[v] for v in h]
        # 1. degenerate: any two coincident corners
        deg = len(set(h)) != 8
        if not deg:
            for a in range(8):
                for b in range(a + 1, 8):
                    if norm(sub(coords[a], coords[b])) < 1e-12:
                        deg = True
                        break
                if deg:
                    break
        if deg:
            s["degenerate"] += 1
            continue
        # 2. inverted: scaled Jacobian negative at any corner
        worst = 1e30
        for (o, x, y, z) in CORNER_TETS:
            e1, e2, e3 = sub(coords[x], coords[o]), sub(coords[y], coords[o]), sub(coords[z], coords[o])
            n1, n2, n3 = norm(e1), norm(e2), norm(e3)
            if n1 < 1e-14 or n2 < 1e-14 or n3 < 1e-14:
                worst = -1
                break
            j = dot(e1, cross(e2, e3)) / (n1 * n2 * n3)
            worst = min(worst, j)
        if worst <= 0:
            s["inverted"] += 1
    return stats


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "octree_hex_mesh.vtk"
    pts, cells, levels = read_vtk(path)
    print(f"File   : {path}")
    print(f"Points : {len(pts)}")
    print(f"Hexes  : {len(cells)}")
    stats = analyze(pts, cells, levels)
    print(f"\n{'level':>7} {'total':>10} {'degenerate':>11} {'inverted':>9}  {'%bad':>6}")
    tot = dict(total=0, degenerate=0, inverted=0)
    for lv in sorted(stats):
        s = stats[lv]
        bad = s["degenerate"] + s["inverted"]
        pct = 100.0 * bad / s["total"] if s["total"] else 0.0
        label = "TEMPLATE" if lv == -1 else f"L{lv}"
        print(f"{label:>7} {s['total']:>10} {s['degenerate']:>11} {s['inverted']:>9}  {pct:>5.1f}%")
        for k in tot:
            tot[k] += s[k]
    bad = tot["degenerate"] + tot["inverted"]
    print(f"{'ALL':>7} {tot['total']:>10} {tot['degenerate']:>11} {tot['inverted']:>9}  "
          f"{100.0*bad/tot['total'] if tot['total'] else 0:>5.1f}%")


if __name__ == "__main__":
    main()

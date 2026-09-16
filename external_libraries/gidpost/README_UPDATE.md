# UPDATE WARNING

**Current vendored version: gidpost 2.14** (`GP_VERSION_MAJOR`/`GP_VERSION_MINOR`
in `source/gidpost.h`). Re-vendored from meshio++'s copy at
`meshioplusplus/src/cpp/third_party/gidpost` (see that project's
`README.meshioplusplus.md` for what it excluded from upstream and why).

On any future version bump, re-apply every item below to the new upstream
sources before replacing the vendored tree:

- `CMakeLists.txt` -> delete examples (`ENABLE_EXAMPLES`/`ENABLE_PARALLEL_EXAMPLE`
  subdirectories stay commented out).
- `source/CMakeLists.txt` -> add cluster files to the `gidpost_source` list
  (`gidpost_cluster.c`).
- `gidpost_functions.h` -> `#include "gidpost_cluster_functions.h"`.
- `gidpostFILES.c` -> `case GiD_Cluster:` in `ValidateConnectivity` (alongside
  `GiD_Point`/`GiD_Sphere`/`GiD_Circle`) && a trailing `"Point"` entry in
  `strElementType[]` (the cluster's GiD spelling).
- `gidpostHDF5.c` -> `case GiD_Cluster: num_int=3; break;` in the mesh-attribute
  switch, alongside the `GiD_Sphere`/`GiD_Circle` cases.
- `gidpost_types.h` -> append `GiD_Cluster` to the `GiD_ElementType` enum
  (append only, so the enum stays ABI-compatible with any file written under a
  previous version).
- `gidpost.c` -> keep the `printf("Debug version: ...")` in
  `GiD_PostInit`/`GiD_HashInit` commented out (Kratos-local silencing; upstream
  re-enables it on some releases).
- `gidpostInt.c` -> keep the bounds-checked `GiD_PostSetFormatReal` /
  `GiD_PostSetFormatStep` setters (Kratos fix: upstream 2.14 introduced an
  unchecked `strcpy` into both `G_format_real[100]` and `format_step[]` --
  the latter sized to fit only its own 6-byte default -- so a caller-supplied
  format string of unbounded length was an overflow. Both are now bounds
  checked and `format_step` is sized to match `G_format_real`. Report this
  upstream to CIMNE and to meshio++ if not already fixed there.)
- Not vendored, and not needed by Kratos: the Fortran binding (`cfortran/`,
  `fortran_module/`, `source/gidpostfor.c`, `source/gidpostfor.h`,
  `source/gidpostforAPI.c`) and `set_nv_compilers.sh`. Nothing in this repo
  references them (verified against `source/CMakeLists.txt`'s `gidpost_source`
  list and a repo-wide grep). If a future upstream version is needed only
  through its Fortran API, these will need to be re-added.

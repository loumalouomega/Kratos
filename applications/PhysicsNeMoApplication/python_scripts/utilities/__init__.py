"""Small helpers with no better home.

``tensor_adaptor_dataset_utils``
    Shared gather/scatter helpers over ``Kratos.TensorAdaptors``, used by the
    bridges and by the MPI export paths.
``array_backend_utils``
    Opt-in CuPy acceleration for the array-heavy bridge paths, with numpy as
    the default and the fallback. Also the zero-copy CuPy-to-torch handoff.
``nvtx_utils``
    Opt-in NVTX ranges around deployment hot paths, for Nsight Systems.
``shallow_water_reference``
    A numpy-only shallow-water integrator generating reference trajectories for
    the GraphCast recipe. No torch, no physicsnemo.

If something here grows a theme, give it a package.
"""

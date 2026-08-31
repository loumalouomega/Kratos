"""Small helpers with no better home.

``tensor_adaptor_dataset_utils``
    Shared gather/scatter helpers over ``Kratos.TensorAdaptors``, used by the
    bridges and by the MPI export paths.
``nvtx_utils``
    Opt-in NVTX ranges around deployment hot paths, for Nsight Systems.
``shallow_water_reference``
    A numpy-only shallow-water integrator generating reference trajectories for
    the GraphCast recipe. No torch, no physicsnemo.

If something here grows a theme, give it a package.
"""

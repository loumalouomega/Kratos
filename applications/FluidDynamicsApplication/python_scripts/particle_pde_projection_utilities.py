"""Helpers to use the particle-based level set transport (LEoPart PDE-constrained projection) in the two-fluid solvers."""

# Importing the Kratos Library
import KratosMultiphysics
from KratosMultiphysics import python_linear_solver_factory

# Import applications
import KratosMultiphysics.FluidDynamicsApplication as KratosCFD

EULERIAN_TRANSPORT = "eulerian"
PARTICLE_PDE_PROJECTION_TRANSPORT = "particle_pde_projection"
AVAILABLE_TRANSPORT_TYPES = (EULERIAN_TRANSPORT, PARTICLE_PDE_PROJECTION_TRANSPORT)

def GetDefaultLinearSolverSettings():
    """Default linear solver of the facet system of the PDE-constrained projection.

    The condensed facet system is symmetric positive definite but not elliptic, so algebraic multigrid does not help:
    conjugate gradients with AMGCL's single-level relaxation preconditioner is used instead. With the relaxation preconditioner
    the Kratos AMGCL wrapper ignores "smoother_type", so AMGCL's default (SPAI0) is used and it is not set here. The facet
    system has no Kratos DOFs, so the AMGCL block size must be given explicitly.
    """
    return KratosMultiphysics.Parameters("""{
        "solver_type"         : "amgcl",
        "krylov_type"         : "cg",
        "preconditioner_type" : "relaxation",
        "tolerance"           : 1e-8,
        "max_iteration"       : 1000,
        "block_size"          : 1,
        "verbosity"           : 0
    }""")

def GetLevelSetTransportType(settings, solver_name):
    """Returns the validated "levelset_transport_type" of the given solver settings."""
    transport_type = settings["levelset_transport_type"].GetString()
    if transport_type not in AVAILABLE_TRANSPORT_TYPES:
        raise ValueError(f"{solver_name}: unknown 'levelset_transport_type' '{transport_type}'. Available options are {AVAILABLE_TRANSPORT_TYPES}.")
    return transport_type

def CreateParticlesOutputModelPart(model, settings):
    """Creates the model part of the particles output, if requested in the "particle_pde_projection_settings".

    The solvers call it in their constructor: the output processes are created before the solver is initialized and they
    need the model part to exist. The process fills it later.
    """
    if not settings.Has("particles_output") or not settings["particles_output"].Has("model_part_name"):
        return
    particles_model_part_name = settings["particles_output"]["model_part_name"].GetString()
    if particles_model_part_name and not model.HasModelPart(particles_model_part_name):
        model.CreateModelPart(particles_model_part_name)

def CreateParticlePdeProjectionProcess(model_part, domain_size, settings, epetra_communicator=None):
    """Creates the particle-based level set transport process.

    Keyword arguments:
    model_part -- the fluid computing model part (linear simplices)
    domain_size -- 2 or 3
    settings -- the "particle_pde_projection_settings" (an optional "linear_solver_settings" configures the facet system solver)
    epetra_communicator -- Epetra communicator of MPI runs (the facet system is then distributed with Trilinos)
    """
    if domain_size not in (2, 3):
        raise ValueError(f"Wrong domain size {domain_size}. Expected 2 or 3.")

    process_settings = settings.Clone()
    if process_settings.Has("linear_solver_settings"):
        linear_solver_settings = process_settings["linear_solver_settings"].Clone()
        process_settings.RemoveValue("linear_solver_settings")
    else:
        linear_solver_settings = GetDefaultLinearSolverSettings()
    if linear_solver_settings.Has("solver_type") and linear_solver_settings["solver_type"].GetString().lower() == "amgcl" and not linear_solver_settings.Has("block_size"):
        linear_solver_settings.AddInt("block_size", 1)

    if model_part.IsDistributed():
        if epetra_communicator is None:
            raise ValueError("An Epetra communicator is required to use the particle level set transport in MPI runs.")
        _PrepareDistributedParticlesModelPart(model_part, process_settings)
        from KratosMultiphysics.TrilinosApplication import trilinos_linear_solver_factory
        import KratosMultiphysics.FluidDynamicsApplication.TrilinosExtension as TrilinosExtension
        linear_solver = trilinos_linear_solver_factory.ConstructSolver(linear_solver_settings)
        process_factory = TrilinosExtension.TrilinosParticleMeshPdeProjectionProcess2D if domain_size == 2 else TrilinosExtension.TrilinosParticleMeshPdeProjectionProcess3D
        return process_factory(epetra_communicator, model_part, linear_solver, process_settings)

    linear_solver = python_linear_solver_factory.ConstructSolver(linear_solver_settings)
    if domain_size == 2:
        return KratosCFD.ParticleMeshPdeProjectionProcess2D(model_part, linear_solver, process_settings)
    return KratosCFD.ParticleMeshPdeProjectionProcess3D(model_part, linear_solver, process_settings)

def _PrepareDistributedParticlesModelPart(model_part, process_settings):
    """In MPI runs the particles output model part needs an MPI communicator, so that every rank writes its own output file."""
    if not process_settings.Has("particles_output") or not process_settings["particles_output"].Has("model_part_name"):
        return
    particles_model_part_name = process_settings["particles_output"]["model_part_name"].GetString()
    if not particles_model_part_name:
        return
    model = model_part.GetModel()
    if model.HasModelPart(particles_model_part_name):
        particles_model_part = model.GetModelPart(particles_model_part_name)
    else:
        particles_model_part = model.CreateModelPart(particles_model_part_name)
    if not particles_model_part.IsDistributed():
        import KratosMultiphysics.mpi as KratosMPI
        KratosMPI.ModelPartCommunicatorUtilities.SetMPICommunicator(particles_model_part, model_part.GetCommunicator().GetDataCommunicator())

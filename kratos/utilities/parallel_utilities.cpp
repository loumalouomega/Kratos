//    |  /           |
//    ' /   __| _` | __|  _ \   __|
//    . \  |   (   | |   (   |\__ `
//   _|\_\_|  \__,_|\__|\___/ ____/
//                   Multi-Physics
//
//  License:		 BSD License
//					 Kratos default license: kratos/license.txt
//
//  Main authors:    Riccardo Rossi
//                   Denis Demidov
//                   Philipp Bucher
//

// System includes
#include <algorithm>
#include <cstdlib> // for std::getenv
#include <memory>  // For std::unique_ptr

// External includes
// Note: Headers are now included in parallel_utilities.h based on framework defines
#if defined(KRATOS_PARALLEL_FRAMEWORK_OPENMP)
#include <omp.h>
#elif defined(KRATOS_PARALLEL_FRAMEWORK_TBB)
#include <tbb/global_control.h>
#include <tbb/task_arena.h> // For tbb::this_task_arena::max_concurrency()
#endif


// Project includes
#include "parallel_utilities.h" // Should have the framework defines from workaround
#include "input_output/logger.h"
#include "includes/lock_object.h"


namespace Kratos {

namespace {
    std::once_flag global_lock_flag_once;
    // This flag was for the old mspNumThreads initialization, keep it for that path.
#if !defined(KRATOS_PARALLEL_FRAMEWORK_TBB)
    std::once_flag num_threads_init_flag_once;
#endif
}

// Initialize static members
LockObject* ParallelUtilities::mspGlobalLock = nullptr;

#if defined(KRATOS_PARALLEL_FRAMEWORK_TBB)
std::unique_ptr<tbb::global_control> ParallelUtilities::mspTbbGlobalControl = nullptr;
#else
int* ParallelUtilities::mspNumThreads = nullptr; // Only define if not TBB
#endif


int ParallelUtilities::GetNumThreads()
{
#if defined(KRATOS_PARALLEL_FRAMEWORK_TBB)
    if (mspTbbGlobalControl) {
        return tbb::global_control::active_value(tbb::global_control::max_allowed_parallelism);
    } else {
        // Fallback if SetNumThreads (and thus mspTbbGlobalControl initialization) hasn't been called.
        // TBB initializes to hardware_concurrency by default.
        return tbb::this_task_arena::max_concurrency();
    }
#elif defined(KRATOS_PARALLEL_FRAMEWORK_OPENMP)
    int nthreads = omp_get_max_threads();
    KRATOS_DEBUG_ERROR_IF(nthreads <= 0) << "omp_get_max_threads() returned " << nthreads << ", which is not possible." << std::endl;
    return nthreads;
#elif defined(KRATOS_PARALLEL_FRAMEWORK_CXX11)
    // This path uses the manually managed static int
    return GetNumberOfThreads(); // Calls private static int& GetNumberOfThreads()
#else // KRATOS_PARALLEL_FRAMEWORK_NONE
    return 1;
#endif
}

void ParallelUtilities::SetNumThreads(const int NumThreads)
{
    KRATOS_ERROR_IF(NumThreads <= 0) << "Attempting to set NumThreads to <= 0. This is not allowed" << std::endl;

#if defined(KRATOS_PARALLEL_FRAMEWORK_TBB)
    const int num_procs = GetNumProcs(); // System hardware concurrency
    KRATOS_WARNING_IF("ParallelUtilities", NumThreads > num_procs) << "TBB: Requested threads (" << NumThreads << ") exceeds available hardware threads (" << num_procs << ")!" << std::endl;
    if (mspTbbGlobalControl) {
        mspTbbGlobalControl.reset(); // Destroy previous global_control object
    }
    mspTbbGlobalControl = std::make_unique<tbb::global_control>(tbb::global_control::max_allowed_parallelism, NumThreads);
#elif defined(KRATOS_PARALLEL_FRAMEWORK_OPENMP)
    const int num_procs = GetNumProcs();
    KRATOS_WARNING_IF("ParallelUtilities", NumThreads > num_procs) << "OpenMP: Requested threads (" << NumThreads << ") exceeds available hardware threads (" << num_procs << ")!" << std::endl;
    GetNumberOfThreads() = NumThreads; // Update the static int for consistency
    omp_set_num_threads(NumThreads);
#elif defined(KRATOS_PARALLEL_FRAMEWORK_CXX11)
    const int num_procs = GetNumProcs();
    KRATOS_WARNING_IF("ParallelUtilities", NumThreads > num_procs) << "C++11: Requested threads (" << NumThreads << ") exceeds available hardware threads (" << num_procs << ")!" << std::endl;
    GetNumberOfThreads() = NumThreads; // Update the static int
#else // KRATOS_PARALLEL_FRAMEWORK_NONE
    if (NumThreads > 1) {
        KRATOS_WARNING("ParallelUtilities") << "Attempting to set " << NumThreads << " threads, but KRATOS_PARALLEL_FRAMEWORK_NONE is active. Only 1 thread will be used." << std::endl;
    }
    // If there was a static int for thread count, ensure it's 1.
    // GetNumberOfThreads() = 1; // This would require GetNumberOfThreads to exist for NONE too.
    // For TBB path under NONE, mspTbbGlobalControl would be set to 1 if InitializeNumberOfThreads calls SetNumThreads(1)
#endif
}

int ParallelUtilities::GetNumProcs()
{
// GetNumProcs is independent of the chosen KRATOS_PARALLEL_FRAMEWORK_XXX for loops,
// it describes the machine. OpenMP provides a direct way, otherwise use C++11.
#ifdef KRATOS_SMP_OPENMP // Use the original broader check for omp_get_num_procs
    return omp_get_num_procs();
#else
    int num_procs = std::thread::hardware_concurrency();
    KRATOS_WARNING_IF("ParallelUtilities", num_procs == 0) << "std::thread::hardware_concurrency() returned 0. Check system configuration." << std::endl;
    return std::max(1, num_procs); // Ensure at least 1 is returned
#endif
}

void ParallelUtilities::InitializeNumberOfThreads()
{
    // This function determines the number of threads from environment or hardware
    // and then calls SetNumThreads, which will dispatch to the correct mechanism.
    int num_threads_determined;

#if defined(KRATOS_PARALLEL_FRAMEWORK_NONE)
    num_threads_determined = 1;
#else
    const char* env_kratos = std::getenv("KRATOS_NUM_THREADS");
    const char* env_omp    = std::getenv("OMP_NUM_THREADS");

    if (env_kratos) {
        num_threads_determined = std::atoi(env_kratos);
    } else if (env_omp) {
        num_threads_determined = std::atoi(env_omp);
    } else {
        num_threads_determined = std::thread::hardware_concurrency();
        KRATOS_INFO_IF("ParallelUtilities", num_threads_determined == 0) << "std::thread::hardware_concurrency() returned 0. Defaulting to 1 thread." << std::endl;
    }
    num_threads_determined = std::max(1, num_threads_determined);
#endif

    SetNumThreads(num_threads_determined);
}

LockObject& ParallelUtilities::GetGlobalLock()
{
    if (!mspGlobalLock) {
        std::call_once(global_lock_flag_once, [](){
            static LockObject global_lock;
            mspGlobalLock = &global_lock;
        });
    }
    return *mspGlobalLock;
}

#if !defined(KRATOS_PARALLEL_FRAMEWORK_TBB)
// This private static method is only needed for OpenMP and CXX11 pure std::thread based options
// to manage mspNumThreads.
int& ParallelUtilities::GetNumberOfThreads()
{
    // Initialize mspNumThreads once using the public InitializeNumberOfThreads logic
    // This ensures that the environment variables are read correctly and SetNumThreads is called.
    // However, InitializeNumberOfThreads is void now.
    // The static int here should be the single source of truth for OMP/CXX11 thread count.
    if (!mspNumThreads) {
        std::call_once(num_threads_init_flag_once, [](){
            static int static_num_threads; // This will store the number of threads for OMP/CXX11

            // Determine initial value based on environment or hardware, similar to InitializeNumberOfThreads
            // but without calling SetNumThreads to avoid recursion here.
            // This is a bit tricky because InitializeNumberOfThreads now calls SetNumThreads.
            // Let's simplify: InitializeNumberOfThreads should be called once globally.
            // This function just provides access to the static variable.
            // The actual setting of this variable should happen in SetNumThreads for OMP/CXX11.

            // Simplified initialization for the static variable, actual control via SetNumThreads
            const char* env_kratos = std::getenv("KRATOS_NUM_THREADS");
            const char* env_omp    = std::getenv("OMP_NUM_THREADS");
            if (env_kratos) static_num_threads = std::atoi(env_kratos);
            else if (env_omp) static_num_threads = std::atoi(env_omp);
            else static_num_threads = std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
            static_num_threads = std::max(1, static_num_threads);

#if defined(KRATOS_PARALLEL_FRAMEWORK_OPENMP)
            // For OpenMP, also ensure omp_set_num_threads is called with this initial value
            // if SetNumThreads hasn't been called yet from InitializeNumberOfThreads.
            // This is tricky due to initialization order.
            // Simplest is to ensure InitializeNumberOfThreads() is called from Kernel constructor.
            // For now, assume this static int is the master for OMP if TBB is not used.
            omp_set_num_threads(static_num_threads);
#endif
            mspNumThreads = &static_num_threads;
        });
    }
    return *mspNumThreads;
}
#endif

}  // namespace Kratos.

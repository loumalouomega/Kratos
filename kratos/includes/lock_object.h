//    |  /           |
//    ' /   __| _` | __|  _ \   __|
//    . \  |   (   | |   (   |\__ `
//   _|\_\_|  \__,_|\__|\___/ ____/
//                   Multi-Physics
//
//  License:		 BSD License
//					 Kratos default license: kratos/license.txt
//
//  Main authors:    Pooyan Dadvand
//                   Philipp Bucher (https://github.com/philbucher)
//

#pragma once

// -- BEGIN WORKAROUND for define.h access issues --
#if defined(KRATOS_SMP_TBB)
  #if defined(KRATOS_SMP_OPENMP)
    #error "KRATOS_SMP_TBB and KRATOS_SMP_OPENMP cannot be defined simultaneously. Please choose only one."
  #endif
  #define KRATOS_PARALLEL_FRAMEWORK_TBB
#elif defined(KRATOS_SMP_OPENMP)
  #define KRATOS_PARALLEL_FRAMEWORK_OPENMP
#elif defined(KRATOS_SMP_CXX11)
  #define KRATOS_PARALLEL_FRAMEWORK_CXX11
#else
  #define KRATOS_PARALLEL_FRAMEWORK_NONE
#endif
// -- END WORKAROUND --

// System includes
#include <mutex> // For KRATOS_PARALLEL_FRAMEWORK_CXX11 and std::lock_guard

// External includes
#if defined(KRATOS_PARALLEL_FRAMEWORK_OPENMP)
#include <omp.h>
#elif defined(KRATOS_PARALLEL_FRAMEWORK_TBB)
#include <tbb/mutex.h>
#endif

// Project includes
#include "includes/define.h"


namespace Kratos
{
///@addtogroup KratosCore
///@{

///@name Kratos Classes
///@{

/// This class defines and stores a lock and gives an interface to it.
/** The class makes a tiny wrapper over shared memory locking mechanisms
 * it is compliant with C++ Lockable
 * see https://en.cppreference.com/w/cpp/named_req/Lockable
 */
class LockObject
{
public:
    ///@name Life Cycle
    ///@{

    /// Default constructor.
    LockObject() noexcept
    {
#if defined(KRATOS_PARALLEL_FRAMEWORK_OPENMP)
        omp_init_lock(&mLock);
#endif
        // For tbb::mutex and std::mutex, default constructor is sufficient.
    }

    /// Copy constructor.
    LockObject(LockObject const& rOther) = delete;

    /// Destructor.
    ~LockObject() noexcept
    {
#if defined(KRATOS_PARALLEL_FRAMEWORK_OPENMP)
        omp_destroy_lock(&mLock);
#endif
        // For tbb::mutex and std::mutex, destructor is sufficient.
    }

    ///@}
    ///@name Operators
    ///@{

    /// Assignment operator.
    LockObject& operator=(LockObject const& rOther) = delete;

    ///@}
    ///@name Access
    ///@{

    inline void lock() const
    {
#if defined(KRATOS_PARALLEL_FRAMEWORK_TBB)
        mLock.lock();
#elif defined(KRATOS_PARALLEL_FRAMEWORK_OPENMP)
        omp_set_lock(&mLock);
#elif defined(KRATOS_PARALLEL_FRAMEWORK_CXX11)
        mLock.lock();
#else // KRATOS_PARALLEL_FRAMEWORK_NONE
        // No operation for non-parallel version
#endif
    }

    KRATOS_DEPRECATED_MESSAGE("Please use lock instead")
    inline void SetLock() const
    {
        this->lock();
    }

    inline void unlock() const
    {
#if defined(KRATOS_PARALLEL_FRAMEWORK_TBB)
        mLock.unlock();
#elif defined(KRATOS_PARALLEL_FRAMEWORK_OPENMP)
        omp_unset_lock(&mLock);
#elif defined(KRATOS_PARALLEL_FRAMEWORK_CXX11)
        mLock.unlock();
#else // KRATOS_PARALLEL_FRAMEWORK_NONE
        // No operation for non-parallel version
#endif
    }

    KRATOS_DEPRECATED_MESSAGE("Please use unlock instead")
    inline void UnSetLock() const
    {
        this->unlock();
    }

    inline bool try_lock() const
    {
#if defined(KRATOS_PARALLEL_FRAMEWORK_TBB)
        return mLock.try_lock();
#elif defined(KRATOS_PARALLEL_FRAMEWORK_OPENMP)
        return omp_test_lock(&mLock);
#elif defined(KRATOS_PARALLEL_FRAMEWORK_CXX11)
        return mLock.try_lock();
#else // KRATOS_PARALLEL_FRAMEWORK_NONE
        return true; // Non-parallel version always succeeds
#endif
    }

    ///@}

private:
    ///@name Member Variables
    ///@{

#if defined(KRATOS_PARALLEL_FRAMEWORK_TBB)
        mutable tbb::mutex mLock;
#elif defined(KRATOS_PARALLEL_FRAMEWORK_OPENMP)
	    mutable omp_lock_t mLock;
#elif defined(KRATOS_PARALLEL_FRAMEWORK_CXX11)
        mutable std::mutex mLock;
#else // KRATOS_PARALLEL_FRAMEWORK_NONE
        // For a truly non-parallel version, no lock member is needed.
        // However, to allow seamless compilation, a dummy or std::mutex might be used
        // if LockObject is instantiated in non-parallel contexts but expected to compile.
        // For simplicity, if no framework is active, no lock member is declared.
        // Operations will be no-ops or error if called unexpectedly.
#endif

    ///@}

}; // Class LockObject

///@}

///@} addtogroup block

}  // namespace Kratos.

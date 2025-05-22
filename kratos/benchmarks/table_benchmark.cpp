//    |  /           |
//    ' /   __| _` | __|  _ \   __|
//    . \  |   (   | |   (   |\__ \.
//   _|\_\_|  \__,_|\__|\___/ ____/
//                   Multi-Physics
//
//  License:         BSD License
//                   Kratos default license: kratos/license.txt
//
//  Main authors:    [Your Agent Name]
//                   (based on example_benchmark.cpp by Riccardo Rossi)
//

// System includes
#include <string>
#include <iostream>
#include <vector>
#include <chrono>    // For std::chrono, used by Kratos::Timer potentially or for seeding
#include <algorithm> // For std::sort (though not directly used in these benchmarks)
#include <random>    // For std::mt19937 and std::uniform_real_distribution
#include <cmath>     // For std::sin, std::abs

// External includes

// --- START OF PLACEHOLDER DEFINITIONS FOR KRATOS UTILITIES ---
// These are simplified stubs based on their usage in the provided table.h and benchmark structure.
// In a real Kratos build, these would come from actual Kratos headers.
// It's important these are defined before including Kratos headers that might depend on them,
// or are guarded appropriately.

// Placeholder for Kratos macros from define.h (KRATOS_CLASS_POINTER_DEFINITION, KRATOS_ERROR_IF)
// These are expected by table.h
#ifndef KRATOS_DEFINE_H_INCLUDED_BENCHMARK_STUB // Custom guard for this benchmark file's stubs
#define KRATOS_DEFINE_H_INCLUDED_BENCHMARK_STUB

#ifndef KRATOS_CLASS_POINTER_DEFINITION
#define KRATOS_CLASS_POINTER_DEFINITION(class_name) \
    class class_name; \
    typedef class_name* Pointer; \
    typedef const class_name* ConstPointer;
#endif

#ifndef KRATOS_ERROR_IF
#define KRATOS_ERROR_IF(condition) \
    if (condition) { \
        std::cerr << "KRATOS BENCHMARK ERROR (placeholder): " #condition << std::endl; \
        /* In a real app, this might throw or abort. */ \
    }
#endif

// Forward declare Kratos namespace and some classes to resolve dependencies
// These are forward declared here as their full stub definitions are below,
// but Kratos headers might expect them to be known within the Kratos namespace.
namespace Kratos {
    class Serializer; 
    template<class TDataType> class Variable;
    // Logger is defined in its own stub section below
}

#endif // KRATOS_DEFINE_H_INCLUDED_BENCHMARK_STUB

// Placeholder for Kratos::Logger and KRATOS_WARNING (as used in table.h and benchmark example)
// Assuming logger.h would provide these
#ifndef KRATOS_LOGGER_H_INCLUDED_BENCHMARK_STUB // Guard against multiple inclusions
#define KRATOS_LOGGER_H_INCLUDED_BENCHMARK_STUB
namespace Kratos {
    class Logger {
    public:
        Logger(const std::string& label) : mLabel(label) {}

        template<typename T>
        Logger& operator<<(const T& value) {
            // Minimal output for benchmark purposes to keep console clean during timing
            // std::cout << "[" << mLabel << "] " << value; 
            return *this;
        }

        Logger& operator<<(std::ostream& (*manip)(std::ostream&)) {
            // std::cout << manip; // For std::endl etc.
            return *this;
        }
    private:
        std::string mLabel;
    };
} // namespace Kratos

#ifndef KRATOS_WARNING
#define KRATOS_WARNING(label_str) Kratos::Logger(label_str)
#endif

#endif // KRATOS_LOGGER_H_INCLUDED_BENCHMARK_STUB


// Placeholder for Kratos::Serializer (as used in table.h)
#ifndef KRATOS_SERIALIZER_H_INCLUDED_BENCHMARK_STUB
#define KRATOS_SERIALIZER_H_INCLUDED_BENCHMARK_STUB
namespace Kratos {
    class Serializer {
    public:
        enum class Mode { Save, Load }; 
        Serializer(Mode mode = Mode::Save) : mMode(mode) {}

        template<typename T>
        void save(const std::string& rName, const T& rValue) { /* no-op for benchmark */ }
        
        template<typename T>
        void load(const std::string& rName, T& rValue) { /* no-op for benchmark */ }
    private:
        Mode mMode;
    };
}
#endif // KRATOS_SERIALIZER_H_INCLUDED_BENCHMARK_STUB


// Placeholder for Kratos::Variable (as used in table.h for Table<double,double> specialization)
#ifndef KRATOS_VARIABLE_H_INCLUDED_BENCHMARK_STUB
#define KRATOS_VARIABLE_H_INCLUDED_BENCHMARK_STUB
namespace Kratos {
    template<class TDataType>
    class Variable {
    public:
        Variable(const std::string& NewName = "UnnamedVariable") : mName(NewName) {}
        const std::string& Name() const { return mName; }
    private:
        std::string mName;
    };
}
#endif // KRATOS_VARIABLE_H_INCLUDED_BENCHMARK_STUB


// Project includes (now that stubs are defined)
#include "../includes/table.h"     // Path to table.h as specified
#include "../utilities/timer.h"    // Kratos timer for benchmarking


// Placeholder for Kratos benchmark macros (from example_benchmark.cpp)
#ifndef KRATOS_BENCHMARK
#define KRATOS_BENCHMARK(name) static void Test##name() // Using static to keep them in namespace
#endif
#ifndef KRATOS_BENCHMARKING_SESSION
#define KRATOS_BENCHMARKING_SESSION(name) namespace name {
#define KRATOS_BENCHMARKING_SESSION_END }
#endif
// --- END OF PLACEHOLDER DEFINITIONS ---


KRATOS_BENCHMARKING_SESSION(TableBenchmark)

// Helper function to create and populate a Kratos::Table<double, double>
Kratos::Table<double, double> CreateAndPopulateTable(std::size_t num_elements) {
    Kratos::Table<double, double> table;
    for (std::size_t i = 0; i < num_elements; ++i) {
        // Insert sorted data. Using PushBack as it's efficient for sorted additions.
        // X values are i*0.1, Y values are i*0.15 + a small sinusoidal variation.
        table.PushBack(static_cast<double>(i) * 0.1, 
                       static_cast<double>(i) * 0.15 + std::sin(static_cast<double>(i) * 0.05));
    }
    return table;
}

// Benchmark for Table<double,double>::GetValue
// "TableGetValuePerformance" is the "name" part for KRATOS_BENCHMARK macro,
// corresponding to "TableNameGetValue" from prompt.
KRATOS_BENCHMARK(TableGetValuePerformance) // Results in static void TestTableGetValuePerformance()
{
    const std::size_t num_elements_in_table = 20000;
    const int num_lookups_to_perform = 100000;

    auto table = CreateAndPopulateTable(num_elements_in_table);

    std::vector<double> lookup_x_values(num_lookups_to_perform);
    std::mt19937 random_engine(12345); // Mersenne Twister with a fixed seed for reproducibility
    
    double table_min_x_val = 0.0;
    double table_max_x_val = (num_elements_in_table > 0) ? (static_cast<double>(num_elements_in_table - 1) * 0.1) : 10.0;
    // Generate X values for lookup: 20% outside on lower end, 20% outside on upper end of table's X range
    std::uniform_real_distribution<> distribution(table_min_x_val - (table_max_x_val - table_min_x_val) * 0.2, 
                                                  table_max_x_val + (table_max_x_val - table_min_x_val) * 0.2);

    for (int i = 0; i < num_lookups_to_perform; ++i) {
        lookup_x_values[i] = distribution(random_engine);
    }

    volatile double sum_of_values = 0.0; // Use volatile to prevent compiler from optimizing away the loop

    Kratos::Timer::Start("TableGetValuePerformance::Lookup"); // Timer section name as per example
    for (int i = 0; i < num_lookups_to_perform; ++i) {
        sum_of_values += table.GetValue(lookup_x_values[i]);
    }
    Kratos::Timer::Stop("TableGetValuePerformance::Lookup");

    // Optional: print sum to verify it's non-zero, but do it outside timing.
    // std::cout << "Sum of GetValue (for verification, not timed): " << sum_of_values << std::endl;
}

// Benchmark for Table<double,double>::GetDerivative
// "TableGetDerivativePerformance" is the "name" part for KRATOS_BENCHMARK macro,
// corresponding to "TableNameGetDerivative" from prompt.
KRATOS_BENCHMARK(TableGetDerivativePerformance) // Results in static void TestTableGetDerivativePerformance()
{
    const std::size_t num_elements_in_table = 20000;
    const int num_lookups_to_perform = 100000;

    auto table = CreateAndPopulateTable(num_elements_in_table);

    std::vector<double> lookup_x_values(num_lookups_to_perform);
    std::mt19937 random_engine(54321); // Different seed for variety, still reproducible
    
    double table_min_x_val = 0.0;
    double table_max_x_val = (num_elements_in_table > 0) ? (static_cast<double>(num_elements_in_table - 1) * 0.1) : 10.0;
    std::uniform_real_distribution<> distribution(table_min_x_val - (table_max_x_val - table_min_x_val) * 0.2, 
                                                  table_max_x_val + (table_max_x_val - table_min_x_val) * 0.2);

    for (int i = 0; i < num_lookups_to_perform; ++i) {
        lookup_x_values[i] = distribution(random_engine);
    }

    volatile double sum_of_derivatives = 0.0; // Use volatile

    Kratos::Timer::Start("TableGetDerivativePerformance::Lookup"); // Timer section name
    for (int i = 0; i < num_lookups_to_perform; ++i) {
        sum_of_derivatives += table.GetDerivative(lookup_x_values[i]);
    }
    Kratos::Timer::Stop("TableGetDerivativePerformance::Lookup");

    // Optional: print sum to verify it's non-zero, but do it outside timing.
    // std::cout << "Sum of GetDerivative (for verification, not timed): " << sum_of_derivatives << std::endl;
}

KRATOS_BENCHMARKING_SESSION_END

// Main function to run benchmarks (following example_benchmark.cpp pattern)
#ifdef KRATOS_EXECUTE_BENCHMARKS_VIA_MAIN 
int main()
{
    std::cout << "Executing Kratos Table Benchmarks..." << std::endl;

    // Manually call benchmark functions as per example_benchmark.cpp.
    // The KRATOS_BENCHMARK macro defines them as Test<Name> within the namespace.
    TableBenchmark::TestTableGetValuePerformance();
    TableBenchmark::TestTableGetDerivativePerformance();

    std::cout << "Kratos Table Benchmarks finished." << std::endl;

    // Print timer results using Kratos::Timer
    // Assuming Kratos::Timer accumulates results globally and has a static PrintAll method.
    std::cout << "\nKratos Timer Report:" << std::endl;
    std::cout << "--------------------" << std::endl;
    Kratos::Timer::PrintAll(std::cout); 

    return 0;
}
#endif // KRATOS_EXECUTE_BENCHMARKS_VIA_MAIN

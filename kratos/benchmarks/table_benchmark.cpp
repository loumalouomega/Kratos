//    |  /           |
//    ' /   __| _` | __|  _ \   __|
//    . \  |   (   | |   (   |\__ `
//   _|\_\_|  \__,_|\__|\___/ ____/
//                   Multi-Physics
//
//  License:         BSD License
//                   Kratos default license: kratos/license.txt
//
//  Main authors:    Vicente Mataix Ferrandiz
//

// System includes
#include <algorithm> // For std::sort (though not directly used in these benchmarks)
#include <random>    // For std::mt19937 and std::uniform_real_distribution
#include <cmath>     // For std::sin, std::abs

// External includes
#include <benchmark/benchmark.h>

// Project includes
#include "includes/table.h"     // Path to table.h as specified

namespace Kratos
{

// Helper function to create and populate a Table<double, double>
Table<double, double> CreateAndPopulateTable(const std::size_t NumElements) {
    Table<double, double> table;
    for (std::size_t i = 0; i < NumElements; ++i) {
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
template <std::size_t Size>
static void TestTableGetValuePerformance(benchmark::State& state)
{
    const int num_lookups_to_perform = 5 * Size;

    auto table = CreateAndPopulateTable(Size);

    std::vector<double> lookup_x_values(num_lookups_to_perform);
    std::mt19937 random_engine(12345); // Mersenne Twister with a fixed seed for reproducibility
    
    double table_min_x_val = 0.0;
    double table_max_x_val = (Size > 0) ? (static_cast<double>(Size - 1) * 0.1) : 10.0;
    // Generate X values for lookup: 20% outside on lower end, 20% outside on upper end of table's X range
    std::uniform_real_distribution<> distribution(table_min_x_val - (table_max_x_val - table_min_x_val) * 0.2, 
                                                  table_max_x_val + (table_max_x_val - table_min_x_val) * 0.2);

    for (int i = 0; i < num_lookups_to_perform; ++i) {
        lookup_x_values[i] = distribution(random_engine);
    }

    for (auto _ : state) {
        volatile double sum_of_values = 0.0; // Use volatile to prevent compiler from optimizing away the loop
        for (int i = 0; i < num_lookups_to_perform; ++i) {
            sum_of_values += table.GetValue(lookup_x_values[i]);
        }
    }
}

// Register the function as benchmarks for different sizes
BENCHMARK_TEMPLATE(TestTableGetValuePerformance, 5);
BENCHMARK_TEMPLATE(TestTableGetValuePerformance, 10);
BENCHMARK_TEMPLATE(TestTableGetValuePerformance, 20);
BENCHMARK_TEMPLATE(TestTableGetValuePerformance, 200);
BENCHMARK_TEMPLATE(TestTableGetValuePerformance, 2000);
BENCHMARK_TEMPLATE(TestTableGetValuePerformance, 20000);

// Benchmark for Table<double,double>::GetDerivative
// "TableGetDerivativePerformance" is the "name" part for KRATOS_BENCHMARK macro,
// corresponding to "TableNameGetDerivative" from prompt.
template <std::size_t Size>
static void TestTableGetDerivativePerformance(benchmark::State& state)
{
    const int num_lookups_to_perform = 5 * Size;

    auto table = CreateAndPopulateTable(Size);

    std::vector<double> lookup_x_values(num_lookups_to_perform);
    std::mt19937 random_engine(54321); // Different seed for variety, still reproducible
    
    double table_min_x_val = 0.0;
    double table_max_x_val = (Size > 0) ? (static_cast<double>(Size - 1) * 0.1) : 10.0;
    std::uniform_real_distribution<> distribution(table_min_x_val - (table_max_x_val - table_min_x_val) * 0.2, 
                                                  table_max_x_val + (table_max_x_val - table_min_x_val) * 0.2);

    for (int i = 0; i < num_lookups_to_perform; ++i) {
        lookup_x_values[i] = distribution(random_engine);
    }

    for (auto _ : state) {
        volatile double sum_of_derivatives = 0.0; // Use volatile
        for (int i = 0; i < num_lookups_to_perform; ++i) {
            sum_of_derivatives += table.GetDerivative(lookup_x_values[i]);
        }
    }
}

// Register the function as benchmarks for different sizes
BENCHMARK_TEMPLATE(TestTableGetDerivativePerformance, 5);
BENCHMARK_TEMPLATE(TestTableGetDerivativePerformance, 10);
BENCHMARK_TEMPLATE(TestTableGetDerivativePerformance, 20);
BENCHMARK_TEMPLATE(TestTableGetDerivativePerformance, 200);
BENCHMARK_TEMPLATE(TestTableGetDerivativePerformance, 2000);
BENCHMARK_TEMPLATE(TestTableGetDerivativePerformance, 20000);

// Benchmark for Table<double,double>::GetValue with cached access pattern
template <std::size_t Size>
static void TestTableGetValuePerformanceCached(benchmark::State& state)
{
    const int num_lookups_per_base = 5; // Number of close lookups for each base point
    const int num_base_points = Size;   // Number of distinct points to test cache around
    const int num_lookups_to_perform = num_lookups_per_base * num_base_points;
    auto table = CreateAndPopulateTable(Size);
    std::vector<double> lookup_x_values(num_lookups_to_perform);
    std::mt19937 random_engine(67890); // Different seed
    double table_min_x_val = 0.0;
    double table_max_x_val = (Size > 0) ? (static_cast<double>(Size - 1) * 0.1) : 10.0;
    std::uniform_real_distribution<> base_dist(table_min_x_val, table_max_x_val);
    double step = (Size > 1) ? (0.1 / (num_lookups_per_base + 1)) : 0.001; // Small step for close lookups
    // Ensure step is not zero if num_lookups_per_base is large or 0.1 is too small relative to it.
    if (step == 0.0 && num_lookups_per_base > 0) step = 0.001;


    int k = 0;
    for (int i = 0; i < num_base_points; ++i) {
        double base_x = base_dist(random_engine);
        // Ensure base_x allows for meaningful steps if table/step is small
        if (Size <=1 && num_lookups_per_base > 1) base_x = table_min_x_val + (table_max_x_val - table_min_x_val)/2.0 ;

        for (int j = 0; j < num_lookups_per_base; ++j) {
            double val = base_x + static_cast<double>(j - num_lookups_per_base/2) * step;
            // Clamp values to be within a reasonable range of the table if necessary,
            // especially if step calculation leads to far-off values for small tables.
            // This benchmark is for cached/close access, not extreme extrapolation.
            if (Size > 0) { // only clamp if table has data
                val = std::max(table_min_x_val - (table_max_x_val - table_min_x_val) * 0.1, val); // allow slight extrapolation
                val = std::min(table_max_x_val + (table_max_x_val - table_min_x_val) * 0.1, val);
            }
            lookup_x_values[k++] = val;
            if (k >= num_lookups_to_perform) break;
        }
        if (k >= num_lookups_to_perform) break;
    }
    // If k < num_lookups_to_perform due to break (e.g. Size=0), resize lookup_x_values
    if (k < num_lookups_to_perform) {
        lookup_x_values.resize(k);
    }


    for (auto _ : state) {
        volatile double sum_of_values = 0.0;
        for (std::size_t i = 0; i < lookup_x_values.size(); ++i) {
            sum_of_values += table.GetValue(lookup_x_values[i]);
        }
    }
}

BENCHMARK_TEMPLATE(TestTableGetValuePerformanceCached, 5);
BENCHMARK_TEMPLATE(TestTableGetValuePerformanceCached, 10);
BENCHMARK_TEMPLATE(TestTableGetValuePerformanceCached, 20);
BENCHMARK_TEMPLATE(TestTableGetValuePerformanceCached, 200);
BENCHMARK_TEMPLATE(TestTableGetValuePerformanceCached, 2000);
BENCHMARK_TEMPLATE(TestTableGetValuePerformanceCached, 20000);

}  // namespace Kratos

BENCHMARK_MAIN();

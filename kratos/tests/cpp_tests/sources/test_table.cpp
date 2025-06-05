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

// Project includes
#include "testing/testing.h"
#include "includes/table.h"
#include "includes/expect.h"
#include "includes/variables.h"

namespace Kratos::Testing {

KRATOS_TEST_CASE_IN_SUITE(BaseTable, KratosCoreFastSuite)
{
    Table<double> table;
    for (std::size_t i = 0; i < 6; ++i)
        table.PushBack(static_cast<double>(i), 2.0 * static_cast<double>(i));
            
    double nearest = (table.GetNearestRow(2.1))[0];
    KRATOS_EXPECT_DOUBLE_EQ(nearest, 4.0);
    KRATOS_EXPECT_DOUBLE_EQ(table.GetValue(2.1), 4.2);
    KRATOS_EXPECT_DOUBLE_EQ(table(2.1), 4.2);
    KRATOS_EXPECT_DOUBLE_EQ(table.GetDerivative(2.1), 2.0);

    auto& r_data = table.Data();
    KRATOS_EXPECT_EQ(r_data.size(), 6);
            
    // Clear database
    table.Clear();
    KRATOS_EXPECT_EQ(r_data.size(), 0);

    // Inverse filling with insert
    for (std::size_t i = 6; i > 0; --i)
        table.insert(static_cast<double>(i), 2.0 * static_cast<double>(i));

    KRATOS_EXPECT_EQ(r_data.size(), 6);

    nearest = (table.GetNearestRow(2.1))[0];
    KRATOS_EXPECT_DOUBLE_EQ(nearest, 4.0);
    KRATOS_EXPECT_DOUBLE_EQ(table.GetValue(2.1), 4.2);
    KRATOS_EXPECT_DOUBLE_EQ(table(2.1), 4.2);
    KRATOS_EXPECT_DOUBLE_EQ(table.GetDerivative(2.1), 2.0);
}

KRATOS_TEST_CASE_IN_SUITE(NamesOfXAndYInTable, KratosCoreFastSuite)
{
    Table<double> table; // uses the class template

    // New tables shouldn't have any names set
    KRATOS_EXPECT_TRUE(table.NameOfX().empty());
    KRATOS_EXPECT_TRUE(table.NameOfY().empty());

    table.SetNameOfX("Foo");
    KRATOS_EXPECT_EQ(table.NameOfX(), "Foo");

    table.SetNameOfY("Bar");
    KRATOS_EXPECT_EQ(table.NameOfY(), "Bar");
}

KRATOS_TEST_CASE_IN_SUITE(NamesOfXAndYInTableSpecialization, KratosCoreFastSuite)
{
    Table<double, double> table; // uses the template specialization

    // New tables shouldn't have any names set
    KRATOS_EXPECT_TRUE(table.NameOfX().empty());
    KRATOS_EXPECT_TRUE(table.NameOfY().empty());

    table.SetNameOfX("Foo");
    KRATOS_EXPECT_EQ(table.NameOfX(), "Foo");

    table.SetNameOfY("Bar");
    KRATOS_EXPECT_EQ(table.NameOfY(), "Bar");
}

KRATOS_TEST_CASE_IN_SUITE(TableDerivativeUsesExtrapolationWhenOutsideOfDomain, KratosCoreFastSuite)
{
    Table<double, double> table;
    table.PushBack(0.0, 0.0);
    table.PushBack(1.0, 2.0);
    table.PushBack(3.0, 3.0);

    constexpr auto abs_tolerance = 1.0e-08;
    KRATOS_EXPECT_NEAR(table.GetDerivative(-2.0), 2.0, abs_tolerance); // before first point in table
    KRATOS_EXPECT_NEAR(table.GetDerivative(0.0), 2.0, abs_tolerance); // at first point in table
    KRATOS_EXPECT_NEAR(table.GetDerivative(3.0), 0.5, abs_tolerance); // at last point in table
    KRATOS_EXPECT_NEAR(table.GetDerivative(5.0), 0.5, abs_tolerance); // beyond last point in table
}

KRATOS_TEST_CASE_IN_SUITE(TableCacheCorrectness, KratosCoreFastSuite)
{
    Table<double, double> table;

    // Setup Table
    table.PushBack(0.0, 0.0);
    table.PushBack(1.0, 10.0);
    table.PushBack(2.0, 15.0);
    table.PushBack(3.0, 18.0);
    table.PushBack(4.0, 22.0);

    // Test Cache Hits (Sequential/Close Access)
    KRATOS_EXPECT_DOUBLE_EQ(table.GetValue(1.0), 10.0); // Initial access
    KRATOS_EXPECT_DOUBLE_EQ(table.GetValue(1.1), 10.5); // Should use cache for neighborhood: (1.1-1.0)/(2.0-1.0) * (15.0-10.0) + 10.0 = 0.1 * 5.0 + 10.0 = 10.5
    KRATOS_EXPECT_DOUBLE_EQ(table.GetValue(0.9), 9.0);   // Neighborhood: (0.9-0.0)/(1.0-0.0) * (10.0-0.0) + 0.0 = 0.9 * 10.0 = 9.0.
    KRATOS_EXPECT_DOUBLE_EQ(table.GetValue(1.0), 10.0); // Exact cache hit

    // Test Cache Miss then Hit
    KRATOS_EXPECT_DOUBLE_EQ(table.GetValue(3.5), 20.0); // (3.5-3.0)/(4.0-3.0) * (22.0-18.0) + 18.0 = 0.5 * 4.0 + 18.0 = 20.0
    KRATOS_EXPECT_DOUBLE_EQ(table.GetValue(3.6), 20.4); // Cache hit: (3.6-3.0)/(4.0-3.0) * (22.0-18.0) + 18.0 = 0.6 * 4.0 + 18.0 = 20.4

    // Test Extrapolation with Cache
    KRATOS_EXPECT_DOUBLE_EQ(table.GetValue(-1.0), -10.0); // Extrapolate from (0,0) and (1,10)
    KRATOS_EXPECT_DOUBLE_EQ(table.GetValue(-0.5), -5.0);  // Should use cache based on previous extrapolation
    KRATOS_EXPECT_DOUBLE_EQ(table.GetValue(5.0), 26.0);   // Extrapolate from (3,18) and (4,22)
    KRATOS_EXPECT_DOUBLE_EQ(table.GetValue(4.5), 24.0);   // Should use cache

    // Test Cache Invalidation on `insert`
    table.Clear();
    table.PushBack(0.0, 0.0);
    table.PushBack(2.0, 20.0);
    KRATOS_EXPECT_DOUBLE_EQ(table.GetValue(1.0), 10.0); // Populate cache
    table.insert(1.5, 15.0); // Insert, should invalidate cache
    KRATOS_EXPECT_DOUBLE_EQ(table.Data()[1].first, 1.5); // Check insertion worked. Data should be (0,0), (1.5,15), (2,20)
    KRATOS_EXPECT_DOUBLE_EQ(table.GetValue(1.0), (1.0-0.0)/(1.5-0.0) * (15.0-0.0) + 0.0); // Re-evaluate: (1.0/1.5)*15 = 10.0
    KRATOS_EXPECT_DOUBLE_EQ(table.GetValue(1.6), (1.6-1.5)/(2.0-1.5) * (20.0-15.0) + 15.0); // (0.1/0.5)*5 + 15 = 1+15=16

    // Test Cache Invalidation on `PushBack`
    table.Clear();
    table.PushBack(0.0, 0.0);
    table.PushBack(1.0, 10.0);
    KRATOS_EXPECT_DOUBLE_EQ(table.GetValue(0.5), 5.0); // Populate cache
    table.PushBack(2.0, 20.0); // PushBack, should invalidate cache. Data: (0,0), (1,10), (2,20)
    KRATOS_EXPECT_DOUBLE_EQ(table.GetValue(1.5), (1.5-1.0)/(2.0-1.0) * (20.0-10.0) + 10.0); // Re-evaluate: 0.5*10+10=15

    // Test Cache Invalidation on `Clear`
    table.Clear();
    table.PushBack(0.0, 0.0);
    table.PushBack(1.0, 10.0);
    KRATOS_EXPECT_DOUBLE_EQ(table.GetValue(0.5), 5.0); // Populate cache
    table.Clear(); // Clear, should invalidate cache
    table.PushBack(0.0, 0.0);
    table.PushBack(2.0, 20.0); // New data
    KRATOS_EXPECT_DOUBLE_EQ(table.GetValue(1.0), 10.0); // Should use new data, not stale cache

    // Test Single Element Table (Cache logic should effectively be bypassed or handle gracefully)
    table.Clear();
    table.PushBack(1.0, 100.0);
    KRATOS_EXPECT_DOUBLE_EQ(table.GetValue(0.5), 100.0);
    KRATOS_EXPECT_DOUBLE_EQ(table.GetValue(1.0), 100.0);
    KRATOS_EXPECT_DOUBLE_EQ(table.GetValue(1.5), 100.0);

    // Test Two Element Table (Cache)
    table.Clear();
    table.PushBack(0.0,0.0);
    table.PushBack(1.0,10.0);
    KRATOS_EXPECT_DOUBLE_EQ(table.GetValue(-1.0), -10.0); // Extrapolation
    KRATOS_EXPECT_DOUBLE_EQ(table.GetValue(0.1), 1.0);   // Interpolation, cache
    KRATOS_EXPECT_DOUBLE_EQ(table.GetValue(0.5), 5.0);   // Interpolation, cache
    KRATOS_EXPECT_DOUBLE_EQ(table.GetValue(0.9), 9.0);   // Interpolation, cache
    KRATOS_EXPECT_DOUBLE_EQ(table.GetValue(2.0), 20.0);   // Extrapolation
}

}  // namespace Kratos::Testing.

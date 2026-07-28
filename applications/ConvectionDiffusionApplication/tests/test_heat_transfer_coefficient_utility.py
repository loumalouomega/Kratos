import math

import KratosMultiphysics
import KratosMultiphysics.ConvectionDiffusionApplication as KratosConvDiff
import KratosMultiphysics.KratosUnittest as UnitTest

# Stefan Boltzmann constant in SI units, matching ThermalFace::StefanBoltzmann
STEFAN_BOLTZMANN = 5.67e-8


class HeatTransferCoefficientUtilityTest(UnitTest.TestCase):
    """Tests for the HeatTransferCoefficientUtility.

    The tests are self-contained: the utility builds its own mesh, so neither an mdpa file nor a
    materials file is required.
    """

    def _CreateUtility(self, model, settings_string):
        settings = KratosMultiphysics.Parameters(settings_string)
        return KratosConvDiff.HeatTransferCoefficientUtility(model, settings)

    def _CreateRadiationOnlyUtility(self, model, name, emissivity_1=1.0, emissivity_2=1.0):
        """Builds a utility in which only the linearised radiative term survives."""
        return self._CreateUtility(model, r'''{
            "model_part_name"        : "''' + name + r'''",
            "max_octree_level"       : 1,
            "f_contact"              : 0.0,
            "include_conduction"     : false,
            "include_gap_conduction" : false,
            "include_radiation"      : true,
            "material_1"             : {"emissivity" : ''' + str(emissivity_1) + r'''},
            "material_2"             : {"emissivity" : ''' + str(emissivity_2) + r'''},
            "sample_temperatures"    : [1000.0]
        }''')

    def testAxisAlignedInterfaceArea(self):
        """The tributary areas must tile the cut exactly, at every refinement level.

        The plane z = 0.5 coincides with a lattice plane at every level greater than zero, which
        is the degenerate case the half-open corner classification is designed to handle.
        """
        for level in range(0, 4):
            model = KratosMultiphysics.Model()
            utility = self._CreateUtility(model, r'''{
                "model_part_name"     : "AxisAligned''' + str(level) + r'''",
                "box_min"             : [0.0, 0.0, 0.0],
                "box_max"             : [1.0, 1.0, 1.0],
                "plane_point"         : [0.0, 0.0, 0.5],
                "plane_normal"        : [0.0, 0.0, 1.0],
                "max_octree_level"    : ''' + str(level) + r''',
                "sample_temperatures" : [500.0]
            }''')
            utility.GenerateMesh()

            # The polygons tile the cut regardless of the refinement level, so the area is exact
            self.assertAlmostEqual(utility.GetInterfaceArea(), 1.0, delta=1e-10)

    def testInclinedInterfaceArea(self):
        """A plane through the centre of the cube along (1,1,1) cuts a regular hexagon."""
        model = KratosMultiphysics.Model()
        utility = self._CreateUtility(model, r'''{
            "model_part_name"     : "Inclined",
            "box_min"             : [0.0, 0.0, 0.0],
            "box_max"             : [1.0, 1.0, 1.0],
            "plane_point"         : [0.5, 0.5, 0.5],
            "plane_normal"        : [1.0, 1.0, 1.0],
            "max_octree_level"    : 3,
            "sample_temperatures" : [500.0]
        }''')
        utility.GenerateMesh()

        exact_area = 3.0 * math.sqrt(3.0) / 4.0
        self.assertAlmostEqual(utility.GetInterfaceArea(), exact_area, delta=1e-6)

    def testInterfaceNodalArea(self):
        """The interface nodes must carry their own tributary area in NODAL_AREA."""
        model = KratosMultiphysics.Model()
        utility = self._CreateUtility(model, r'''{
            "model_part_name"     : "NodalArea",
            "max_octree_level"    : 2,
            "sample_temperatures" : [500.0]
        }''')
        utility.GenerateMesh()

        model_part = model.GetModelPart("NodalArea")
        interface_model_part = model_part.GetSubModelPart("Interface")

        # Level two splits the cut into a four by four pattern of equal squares
        self.assertEqual(interface_model_part.NumberOfNodes(), 16)

        accumulated_area = 0.0
        for node in interface_model_part.Nodes:
            nodal_area = node.GetSolutionStepValue(KratosMultiphysics.NODAL_AREA)
            self.assertAlmostEqual(nodal_area, 1.0 / 16.0, delta=1e-12)
            accumulated_area += nodal_area

        self.assertAlmostEqual(accumulated_area, utility.GetInterfaceArea(), delta=1e-12)

        # Every leaf must end up on one of the two sides
        self.assertEqual(
            model_part.NumberOfElements(),
            model_part.GetSubModelPart("Material1").NumberOfElements()
            + model_part.GetSubModelPart("Material2").NumberOfElements())

    def testRadiationBlackBodies(self):
        """Two black bodies give the plain linearised Stefan Boltzmann coefficient."""
        model = KratosMultiphysics.Model()
        utility = self._CreateRadiationOnlyUtility(model, "BlackBodies")

        temperature_1 = 1000.0
        temperature_2 = 300.0
        expected = STEFAN_BOLTZMANN \
            * (temperature_1**2 + temperature_2**2) * (temperature_1 + temperature_2)

        obtained = utility.ComputeHeatTransferCoefficient(temperature_1, temperature_2)
        self.assertAlmostEqual(obtained, expected, delta=1e-14 * expected)

    def testRadiationRecoversStefanBoltzmann(self):
        """The linearised form must reproduce Q = sigma*F12*(T1^4 - T2^4)."""
        model = KratosMultiphysics.Model()
        utility = self._CreateRadiationOnlyUtility(model, "StefanBoltzmann")

        for temperature_1, temperature_2 in [(1000.0, 300.0), (1500.0, 500.0), (400.0, 350.0)]:
            coefficient = utility.ComputeHeatTransferCoefficient(temperature_1, temperature_2)
            flux = coefficient * (temperature_1 - temperature_2)
            exact_flux = STEFAN_BOLTZMANN * (temperature_1**4 - temperature_2**4)
            self.assertAlmostEqual(flux, exact_flux, delta=1e-12 * abs(exact_flux))

    def testRadiationHasNoSingularity(self):
        """As T2 approaches T1 the coefficient tends to 4*sigma*T^3 rather than diverging.

        Note that the reference expression T1^4 - T2^4 loses all its significant digits in this
        regime, whereas the factored form used by the utility does not. This is the reason for
        linearising algebraically instead of dividing the quartic difference by the gap.
        """
        model = KratosMultiphysics.Model()
        utility = self._CreateRadiationOnlyUtility(model, "NoSingularity")

        temperature = 800.0
        limit = 4.0 * STEFAN_BOLTZMANN * temperature**3

        # The deviation from the limit is of the order of 3*gap/(2*T), so the gaps below are the
        # ones that actually probe it. A vanishing gap must stay finite rather than divide by zero.
        for gap in [1e-6, 1e-10, 1e-14, 0.0]:
            coefficient = utility.ComputeHeatTransferCoefficient(temperature, temperature - gap)
            self.assertAlmostEqual(coefficient, limit, delta=1e-6 * limit)

        # Approaching the limit from a coarser gap must be smooth and bounded
        previous_deviation = None
        for gap in [1.0, 1e-1, 1e-2, 1e-3]:
            coefficient = utility.ComputeHeatTransferCoefficient(temperature, temperature - gap)
            deviation = abs(coefficient - limit)
            self.assertLess(deviation, 0.01 * limit)
            if previous_deviation is not None:
                self.assertLess(deviation, previous_deviation)
            previous_deviation = deviation

    def testGrayBodyExchangeFactor(self):
        """Non unit emissivities must enter through F12 = 1/(1/e1 + 1/e2 - 1)."""
        model = KratosMultiphysics.Model()
        utility = self._CreateRadiationOnlyUtility(model, "GrayBody", 0.8, 0.5)

        temperature_1 = 900.0
        temperature_2 = 400.0
        exchange_factor = 1.0 / (1.0 / 0.8 + 1.0 / 0.5 - 1.0)
        expected = STEFAN_BOLTZMANN * exchange_factor \
            * (temperature_1**2 + temperature_2**2) * (temperature_1 + temperature_2)

        obtained = utility.ComputeHeatTransferCoefficient(temperature_1, temperature_2)
        self.assertAlmostEqual(obtained, expected, delta=1e-14 * expected)

    def testFreeSurfaceExchangeFactor(self):
        """For a free surface the exchange factor degenerates to the solid emissivity."""
        model = KratosMultiphysics.Model()
        utility = self._CreateUtility(model, r'''{
            "model_part_name"        : "FreeSurface",
            "max_octree_level"       : 1,
            "f_contact"              : 0.0,
            "is_air"                 : true,
            "ambient_temperature"    : 300.0,
            "include_conduction"     : false,
            "include_gap_conduction" : false,
            "material_1"             : {"emissivity" : 0.7},
            "sample_temperatures"    : [1000.0]
        }''')

        temperature = 1000.0
        expected = STEFAN_BOLTZMANN * 0.7 * (temperature**2 + 300.0**2) * (temperature + 300.0)

        obtained = utility.ComputeHeatTransferCoefficient(temperature, 300.0)
        self.assertAlmostEqual(obtained, expected, delta=1e-14 * expected)

    def testSeriesConduction(self):
        """With full contact the coefficient is the series solid conduction one."""
        model = KratosMultiphysics.Model()
        utility = self._CreateUtility(model, r'''{
            "model_part_name"        : "Conduction",
            "max_octree_level"       : 1,
            "f_contact"              : 1.0,
            "include_gap_conduction" : false,
            "include_radiation"      : false,
            "material_1"             : {"conductivity_table" : [[0.0, 40.0]], "thickness" : 2.0e-3},
            "material_2"             : {"conductivity_table" : [[0.0, 10.0]], "thickness" : 5.0e-3},
            "sample_temperatures"    : [500.0]
        }''')

        expected = 1.0 / (2.0e-3 / 40.0 + 5.0e-3 / 10.0)
        obtained = utility.ComputeHeatTransferCoefficient(500.0, 400.0)
        self.assertAlmostEqual(obtained, expected, delta=1e-12 * expected)

    def testTemperatureDependentConductivity(self):
        """The conductivity tables must be interpolated linearly and clamped outside their range."""
        model = KratosMultiphysics.Model()
        utility = self._CreateUtility(model, r'''{
            "model_part_name"        : "TemperatureDependent",
            "max_octree_level"       : 1,
            "f_contact"              : 1.0,
            "include_gap_conduction" : false,
            "include_radiation"      : false,
            "material_1"             : {"conductivity_table" : [[300.0, 50.0], [700.0, 30.0]], "thickness" : 1.0e-3},
            "material_2"             : {"conductivity_table" : [[300.0, 20.0], [700.0, 20.0]], "thickness" : 1.0e-3},
            "sample_temperatures"    : [500.0]
        }''')

        # Halfway through the tabulated range of the first material
        expected = 1.0 / (1.0e-3 / 40.0 + 1.0e-3 / 20.0)
        self.assertAlmostEqual(
            utility.ComputeHeatTransferCoefficient(500.0, 500.0), expected, delta=1e-12 * expected)

        # Below the tabulated range the first value is held
        expected_clamped = 1.0 / (1.0e-3 / 50.0 + 1.0e-3 / 20.0)
        self.assertAlmostEqual(
            utility.ComputeHeatTransferCoefficient(100.0, 500.0),
            expected_clamped, delta=1e-12 * expected_clamped)

    def testGapConduction(self):
        """The gas conduction term is the gap conductivity over the gap thickness."""
        model = KratosMultiphysics.Model()
        utility = self._CreateUtility(model, r'''{
            "model_part_name"        : "GapConduction",
            "max_octree_level"       : 1,
            "f_contact"              : 0.0,
            "gap_thickness"          : 2.0e-5,
            "air_conductivity_table" : [[0.0, 0.04]],
            "include_conduction"     : false,
            "include_radiation"      : false,
            "sample_temperatures"    : [500.0]
        }''')

        expected = 0.04 / 2.0e-5
        obtained = utility.ComputeHeatTransferCoefficient(500.0, 400.0)
        self.assertAlmostEqual(obtained, expected, delta=1e-12 * expected)

    def testContactFractionBlending(self):
        """The contact fraction must blend the solid branch against the gap branch."""
        settings_template = r'''{
            "model_part_name"        : "%s",
            "max_octree_level"       : 1,
            "f_contact"              : %s,
            "gap_thickness"          : 1.0e-5,
            "air_conductivity_table" : [[0.0, 0.03]],
            "material_1"             : {"conductivity_table" : [[0.0, 20.0]], "thickness" : 1.0e-3, "emissivity" : 0.9},
            "material_2"             : {"conductivity_table" : [[0.0, 20.0]], "thickness" : 1.0e-3, "emissivity" : 0.9},
            "sample_temperatures"    : [800.0]
        }'''

        model = KratosMultiphysics.Model()
        full_contact = self._CreateUtility(model, settings_template % ("FullContact", "1.0"))
        no_contact = self._CreateUtility(model, settings_template % ("NoContact", "0.0"))
        half_contact = self._CreateUtility(model, settings_template % ("HalfContact", "0.5"))

        solid_branch = full_contact.ComputeHeatTransferCoefficient(800.0, 700.0)
        gap_branch = no_contact.ComputeHeatTransferCoefficient(800.0, 700.0)
        blended = half_contact.ComputeHeatTransferCoefficient(800.0, 700.0)

        expected = 0.5 * solid_branch + 0.5 * gap_branch
        self.assertAlmostEqual(blended, expected, delta=1e-12 * expected)

    def testTableRoundTrip(self):
        """The emitted table must round trip through Parameters and drive a PiecewiseLinearTable."""
        model = KratosMultiphysics.Model()
        utility = self._CreateUtility(model, r'''{
            "model_part_name"        : "Table",
            "max_octree_level"       : 2,
            "f_contact"              : 0.0,
            "include_conduction"     : false,
            "include_gap_conduction" : false,
            "sample_temperatures"    : [728.15, 944.15],
            "plateau_epsilon"        : 0.01
        }''')

        table = utility.ComputeTable()
        round_tripped = KratosMultiphysics.Parameters(table.WriteJsonString())

        self.assertEqual(round_tripped["input_variable"].GetString(), "TEMPERATURE")
        self.assertEqual(round_tripped["output_variable"].GetString(), "CONVECTION_COEFFICIENT")

        data = round_tripped["data"]
        self.assertEqual(data.size(), 4)

        # The input column must be strictly increasing for the interpolation to be well defined
        abscissas = [data[i][0].GetDouble() for i in range(data.size())]
        for i in range(1, len(abscissas)):
            self.assertTrue(abscissas[i] > abscissas[i - 1])

        self.assertAlmostEqual(abscissas[0], 728.14, delta=1e-12)
        self.assertAlmostEqual(abscissas[1], 728.15, delta=1e-12)
        self.assertAlmostEqual(abscissas[2], 944.14, delta=1e-12)
        self.assertAlmostEqual(abscissas[3], 944.15, delta=1e-12)

        # Each sample is written twice, so the pairs share their value
        self.assertAlmostEqual(data[0][1].GetDouble(), data[1][1].GetDouble(), delta=1e-12)
        self.assertAlmostEqual(data[2][1].GetDouble(), data[3][1].GetDouble(), delta=1e-12)

        # Feeding the rows to a Kratos table must reproduce the plateaus
        piecewise_table = KratosMultiphysics.PiecewiseLinearTable()
        for i in range(data.size()):
            piecewise_table.AddRow(data[i][0].GetDouble(), data[i][1].GetDouble())

        for sample in [728.15, 944.15]:
            self.assertAlmostEqual(
                piecewise_table.GetValue(sample),
                piecewise_table.GetValue(sample - 0.01),
                delta=1e-12)

    def testTableAreaWeighting(self):
        """The tabulated value must match the pointwise one for uniform material data."""
        model = KratosMultiphysics.Model()
        utility = self._CreateUtility(model, r'''{
            "model_part_name"          : "Weighting",
            "max_octree_level"         : 2,
            "f_contact"                : 0.0,
            "include_conduction"       : false,
            "include_gap_conduction"   : false,
            "partner_temperature_mode" : "fixed",
            "sink_temperature"         : 300.0,
            "sample_temperatures"      : [1000.0]
        }''')

        table = utility.ComputeTable()
        tabulated = table["data"][1][1].GetDouble()
        pointwise = utility.ComputeHeatTransferCoefficient(1000.0, 300.0)

        self.assertAlmostEqual(tabulated, pointwise, delta=1e-12 * pointwise)

    def testPartnerTemperatureModes(self):
        """The delta and fixed modes must select the documented partner temperature."""
        settings_template = r'''{
            "model_part_name"          : "%s",
            "max_octree_level"         : 1,
            "f_contact"                : 0.0,
            "include_conduction"       : false,
            "include_gap_conduction"   : false,
            "partner_temperature_mode" : "%s",
            "delta_temperature"        : 150.0,
            "sink_temperature"         : 350.0,
            "sample_temperatures"      : [900.0]
        }'''

        model = KratosMultiphysics.Model()

        delta_utility = self._CreateUtility(model, settings_template % ("Delta", "delta"))
        delta_value = delta_utility.ComputeTable()["data"][1][1].GetDouble()
        self.assertAlmostEqual(
            delta_value,
            delta_utility.ComputeHeatTransferCoefficient(900.0, 750.0),
            delta=1e-12 * delta_value)

        fixed_utility = self._CreateUtility(model, settings_template % ("Fixed", "fixed"))
        fixed_value = fixed_utility.ComputeTable()["data"][1][1].GetDouble()
        self.assertAlmostEqual(
            fixed_value,
            fixed_utility.ComputeHeatTransferCoefficient(900.0, 350.0),
            delta=1e-12 * fixed_value)

    def testValidation(self):
        """Invalid input must be rejected rather than silently producing wrong numbers."""
        # A degenerate plane normal
        with self.assertRaises(RuntimeError):
            self._CreateUtility(KratosMultiphysics.Model(), r'''{
                "model_part_name"     : "BadNormal",
                "plane_normal"        : [0.0, 0.0, 0.0],
                "sample_temperatures" : [500.0]
            }''')

        # A plateau wider than the gap between consecutive samples would break the interpolation
        with self.assertRaises(RuntimeError):
            self._CreateUtility(KratosMultiphysics.Model(), r'''{
                "model_part_name"     : "BadPlateau",
                "sample_temperatures" : [500.0, 500.5],
                "plateau_epsilon"     : 1.0
            }''')

        # Temperatures are absolute
        with self.assertRaises(RuntimeError):
            self._CreateUtility(KratosMultiphysics.Model(), r'''{
                "model_part_name"     : "BadTemperature",
                "sample_temperatures" : [-10.0]
            }''')

        # Non increasing samples
        with self.assertRaises(RuntimeError):
            self._CreateUtility(KratosMultiphysics.Model(), r'''{
                "model_part_name"     : "BadOrder",
                "sample_temperatures" : [700.0, 500.0]
            }''')

        # A degenerate box
        with self.assertRaises(RuntimeError):
            self._CreateUtility(KratosMultiphysics.Model(), r'''{
                "model_part_name"     : "BadBox",
                "box_min"             : [0.0, 0.0, 0.0],
                "box_max"             : [1.0, 0.0, 1.0],
                "sample_temperatures" : [500.0]
            }''')

        # A vanishing gap thickness with the gas conduction term active
        with self.assertRaises(RuntimeError):
            self._CreateUtility(KratosMultiphysics.Model(), r'''{
                "model_part_name"        : "BadGap",
                "gap_thickness"          : 0.0,
                "include_gap_conduction" : true,
                "sample_temperatures"    : [500.0]
            }''')

        # An unknown partner temperature mode
        with self.assertRaises(RuntimeError):
            self._CreateUtility(KratosMultiphysics.Model(), r'''{
                "model_part_name"          : "BadMode",
                "partner_temperature_mode" : "quadratic",
                "sample_temperatures"      : [500.0]
            }''')

    def testPlaneOutsideBox(self):
        """A plane that misses the box must be reported instead of yielding an empty table."""
        model = KratosMultiphysics.Model()
        utility = self._CreateUtility(model, r'''{
            "model_part_name"     : "Outside",
            "box_min"             : [0.0, 0.0, 0.0],
            "box_max"             : [1.0, 1.0, 1.0],
            "plane_point"         : [0.0, 0.0, 5.0],
            "plane_normal"        : [0.0, 0.0, 1.0],
            "sample_temperatures" : [500.0]
        }''')

        with self.assertRaises(RuntimeError):
            utility.ComputeTable()


if __name__ == '__main__':
    KratosMultiphysics.Logger.GetDefaultOutput().SetSeverity(KratosMultiphysics.Logger.Severity.WARNING)
    UnitTest.main()

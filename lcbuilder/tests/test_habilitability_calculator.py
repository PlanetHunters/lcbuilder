import unittest

from lcbuilder.star.HabitabilityCalculator import HabitabilityCalculator

from lcbuilder.HarmonicSelector import HarmonicSelector
from lcbuilder.tests.test_lcbuilder_abstract import TestsLcBuilderAbstract


class TestsHabitabilityCalculator(TestsLcBuilderAbstract):
    habitability_calculator: HabitabilityCalculator = HabitabilityCalculator()

    def test_au_to_period(self):
        self.assertAlmostEqual(self.habitability_calculator.au_to_period(1, 0.1), 11.51742564487683, 3)

    def test_calculate_semi_major_axis(self):
        au = 0.100
        self.assertAlmostEqual(
            self.habitability_calculator.calculate_semi_major_axis(
                self.habitability_calculator.au_to_period(1, au),
                0, 0, 1, 0, 0
            )[0],
            au,
            3
        )

    def test_calculate_teq(self):
        teq_default_albedo = (
            self.habitability_calculator.calculate_teq(1, 0.1, 0.1, 1,
                                                       0.1, 0.1,20, 0.1,
                                                       0.1, 6000, 100, 100,
                                                       albedo=0.3)
        )
        teq_albedo_0 = (
            self.habitability_calculator.calculate_teq(1, 0.1, 0.1, 1,
                                                       0.1, 0.1,20, 0.1,
                                                       0.1, 6000, 100, 100,
                                                       albedo=0)
        )
        self.assertAlmostEqual(teq_default_albedo[0], 696.2537, 3)
        self.assertAlmostEqual(teq_default_albedo[1], 38.5044, 3)
        self.assertAlmostEqual(teq_albedo_0[0], 761.1899, 3)
        self.assertAlmostEqual(teq_albedo_0[1], 42.0955, 3)

    def test_calculate_teff(self):
        teff = self.habitability_calculator.calculate_teff(6000, 100, 100,
                                                    0.01, 0.001, 0.001,
                                                    0.05, 0.001, 0.001)
        self.assertAlmostEqual(teff[0], 4012.4418, 3)

    def test_calculate_temperature_stat(self):
        teff = self.habitability_calculator.calculate_teff(6000, 100, 100,
                                                    0.01, 0.001, 0.001,
                                                    0.05, 0.001, 0.001)
        teq_default_albedo = (
            self.habitability_calculator.calculate_teq(1, 0.1, 0.1, 1,
                                                       0.1, 0.1, 20, 0.1,
                                                       0.1, 6000, 100, 100,
                                                       albedo=0.3)
        )
        stat = (
            self.habitability_calculator.calculate_planet_temperature_stat(teq_default_albedo[0], teq_default_albedo[1],
                                                                       teq_default_albedo[2], teff[0], teff[1], teff[2])
        )
        self.assertAlmostEqual(stat, 25.8796815452458, 3)

    def test_calculate_albedo(self):
        albedo = self.habitability_calculator.calculate_albedo(0.01, 0.001, 0.001,
                                                               20, 0.1, 0.1, 1, 0.1, 0.1,
                                                               10, 0.1, 0.1)
        self.assertAlmostEqual(albedo[0], 0.096, 2)

    def test_calculate_albedo_stat(self):
        albedo = self.habitability_calculator.calculate_albedo(0.01, 0.001, 0.001,
                         20, 0.1, 0.1, 1, 0.1, 0.1,
                         10, 0.1, 0.1)
        stat = self.habitability_calculator.calculate_albedo_stat(albedo[0], albedo[1], albedo[2])
        self.assertAlmostEqual(stat, 76.72, 2)


if __name__ == '__main__':
    unittest.main()

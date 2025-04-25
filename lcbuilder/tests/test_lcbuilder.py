import os
import unittest
import numpy as np
from lcbuilder.helper import LcbuilderHelper

from lcbuilder.lcbuilder_class import LcBuilder
from lcbuilder.objectinfo.InputObjectInfo import InputObjectInfo
from lcbuilder.objectinfo.MissionInputObjectInfo import MissionInputObjectInfo
from lcbuilder.objectinfo.MissionObjectInfo import MissionObjectInfo
from lcbuilder import constants
from lcbuilder.tests.test_lcbuilder_abstract import TestsLcBuilderAbstract


class TestsLcBuilder(TestsLcBuilderAbstract):
    def test_build_object_info(self):
        file = None
        cadence = [120]
        sectors = "all"
        lcbuilder = LcBuilder()
        target_name = "TIC 1234"
        object_info = lcbuilder.build_object_info(target_name, [constants.SPOC_AUTHOR], sectors, file, cadence, None, None, None, None,
                                                  None)
        assert object_info.mission_id() == target_name
        self.assertIsNone(object_info.lower_outliers_sigma)
        target_name = "TIC 1234"
        object_info = lcbuilder.build_object_info(target_name, [constants.SPOC_AUTHOR], sectors, file, cadence, None, None, None, None,
                                                  None, lower_outliers_sigma=3)
        assert object_info.mission_id() == target_name
        self.assertEqual(3, object_info.lower_outliers_sigma)
        target_name = "KIC 1234"
        object_info = lcbuilder.build_object_info(target_name, [constants.KEPLER_AUTHOR], sectors, file, cadence, None, None, None, None,
                                                  None)
        assert object_info.mission_id() == target_name
        target_name = "EPIC 1234"
        object_info = lcbuilder.build_object_info(target_name, [constants.SPOC_AUTHOR], sectors, file, cadence, None, None, None, None,
                                                  None)
        assert object_info.mission_id() == target_name
        target_name = "25.9_-19.3"
        cadence = [1800]
        object_info = lcbuilder.build_object_info(target_name, [constants.TESS_SPOC_AUTHOR], sectors, file, cadence, None, None, None, None,
                                                  None)
        assert object_info.mission_id() is None and object_info.sherlock_id() == "25.9_-19.3_FFI_all"
        target_name = "25.9_-19.3"
        cadence = [120]
        object_info = lcbuilder.build_object_info(target_name, [constants.SPOC_AUTHOR], sectors, file, cadence, None, None, None, None,
                                                  None)
        assert object_info.mission_id() is None and object_info.sherlock_id() == "25.9_-19.3_FFI_all"
        target_name = "WHATEVER"
        file = "fake_lc.csv"
        object_info = lcbuilder.build_object_info(target_name, None, sectors, file, cadence, None, None, None, None,
                                                  None)
        assert object_info.mission_id() is None and object_info.sherlock_id() == "INP_" + os.path.splitext(file)[
            0].replace("/", "_")

    def test_lcbuilder_get_author(self):
        self.assertEqual(constants.TESS_SPOC_AUTHOR, LcBuilder().get_default_author("TIC 12345", 'long'))
        self.assertEqual(constants.TESS_SPOC_AUTHOR, LcBuilder().get_default_author("TIC 12345", 'long'))
        self.assertEqual(constants.TESS_SPOC_AUTHOR, LcBuilder().get_default_author("TIC 12345", 'long'))
        self.assertEqual(constants.TESS_SPOC_AUTHOR, LcBuilder().get_default_author("TIC 12345", 'long'))
        self.assertEqual(constants.TESS_SPOC_AUTHOR, LcBuilder().get_default_author("TIC 12345", 'long'))
        self.assertEqual(constants.TESS_SPOC_AUTHOR, LcBuilder().get_default_author("TIC 12345", 'long'))

    def test_short_cadence_kic(self):
        lc_build = LcBuilder().build(MissionObjectInfo('all', "KIC 12557548",
                                                       author=[constants.KEPLER_AUTHOR], cadence=[60]), "./")
        self.assertEqual(lc_build.cadence, 59)
        self.assertGreater(len(lc_build.lc), 0)
        self._test_kepler_star_params(lc_build.star_info)

    def test_long_cadence_missing_star_kic(self):
        lc_build = LcBuilder().build(MissionObjectInfo('all', "KIC 12106934",
                                                       author=[constants.KEPLER_AUTHOR], cadence=[1800]), "./")
        self.assertEqual(lc_build.cadence, 1765)
        self.assertGreater(len(lc_build.lc), 0)

    def test_short_cadence_epic(self):
        lc_build = LcBuilder().build(MissionObjectInfo('all', "EPIC 211945201",
                                                       author=[constants.K2_AUTHOR], cadence=[60]), "./")
        self.assertEqual(lc_build.cadence, 59)
        self.assertGreater(len(lc_build.lc), 0)
        self._test_k2_star_params(lc_build.star_info)
        lc_build = LcBuilder().build(MissionObjectInfo('all', "EPIC 211945201", cadence=[1800],
                                                       author=[constants.EVEREST_AUTHOR]), "./")
        self.assertEqual(lc_build.cadence, 1766)
        self.assertGreater(len(lc_build.lc), 0)
        self._test_k2_star_params(lc_build.star_info)

    def test_long_cadence(self):
        lc_build = LcBuilder().build(MissionObjectInfo('all', "TIC 352315023", cadence=[600],
                                                       author=[constants.ELEANOR_AUTHOR]), "./")
        self.assertEqual(lc_build.cadence, 200)
        self.assertGreater(len(lc_build.lc), 0)
        self._test_tess_star_params(lc_build.star_info)

    def test_long_cadence_other_author(self):
        lc_build = LcBuilder().build(MissionObjectInfo('all', "TIC 192833836", cadence=[600],
                                                       author=[constants.TESS_SPOC_AUTHOR]), "./")
        self.assertEqual(lc_build.cadence, 200)
        self.assertGreater(len(lc_build.lc), 0)

    def test_long_cadence_coords(self):
        lc_build = LcBuilder().build(MissionObjectInfo('all', ra=300.47, dec=-71.96, cadence=[600],
                                                       author=[constants.ELEANOR_AUTHOR]),
                                     "./")
        self.assertEqual(lc_build.cadence, 200)
        self.assertGreater(len(lc_build.lc), 0)
        self._test_tess_star_params(lc_build.star_info)

    def test_input_with_id(self):
        directory = self._get_test_resource_file("input.csv")
        lc_build = LcBuilder().build(MissionInputObjectInfo("TIC 352315023", directory, initial_transit_mask=[{'P': 3, 'D': 60, 'T0': 50}]), "./")
        self.assertGreater(len(lc_build.lc), 0)
        self._test_tess_star_params(lc_build.star_info)

    def test_input_without_id(self):
        directory = self._get_test_resource_file("input.csv")
        lc_build = LcBuilder().build(InputObjectInfo(directory), "./")
        self.assertGreater(len(lc_build.lc), 0)
        self.assertTrue(lc_build.star_info.mass_assumed)
        self.assertTrue(lc_build.star_info.radius_assumed)

    def test_build(self):
        lc_build = LcBuilder().build(MissionObjectInfo([13], "TIC 352315023", cadence=[1800],
                                                       author=[constants.ELEANOR_AUTHOR]), "./")
        self.assertEqual(1222, len(lc_build.lc))
        self.assertEqual(1320, len(lc_build.lc_data))
        lc_build = LcBuilder().build(MissionObjectInfo([13], "TIC 352315023", cadence=[120, 1800],
                                                       author=[constants.ELEANOR_AUTHOR, constants.SPOC_AUTHOR]), "./")
        self.assertEqual(19327, len(lc_build.lc))
        self.assertEqual(21131, len(lc_build.lc_data))
        lc_build = LcBuilder().build(MissionObjectInfo([13], "TIC 352315023",
                                                       author=[constants.SPOC_AUTHOR], cadence=[120]), "./")
        self.assertEqual(18107, len(lc_build.lc))
        self.assertEqual(19811, len(lc_build.lc_data))
        lc_build = LcBuilder().build(MissionObjectInfo([13], "KIC 12557548",
                                                       author=[constants.KEPLER_AUTHOR], cadence=[60]), "./")
        self.assertEqual(127850, len(lc_build.lc))
        self.assertEqual(130290, len(lc_build.lc_data))
        lc_build = LcBuilder().build(MissionObjectInfo('all', "EPIC 211945201",
                                                       author=[constants.K2_AUTHOR], cadence=[60]), "./")
        self.assertEqual(107670, len(lc_build.lc))
        self.assertEqual(116640, len(lc_build.lc_data))
        lc_build = LcBuilder().build(MissionObjectInfo([1], "KIC 12557548",
                                                       author=[constants.KEPLER_AUTHOR], cadence=[1800]), "./")
        self.assertEqual(1543, len(lc_build.lc))
        self.assertEqual(1639, len(lc_build.lc_data))
        lc_build = LcBuilder().build(MissionObjectInfo([5], "EPIC 211945201",
                                                       author=[constants.K2_AUTHOR], cadence=[1800]), "./")
        self.assertEqual(3324, len(lc_build.lc))
        self.assertEqual(3657, len(lc_build.lc_data))

    def test_binning(self):
        directory = os.path.dirname(__file__) + "/resources/input.csv"
        lc_build = LcBuilder().build(MissionInputObjectInfo("TIC 352315023", directory), "./")
        self.assertEqual(4551, len(lc_build.lc))
        lc_build = LcBuilder().build(MissionInputObjectInfo("TIC 352315023", directory, binning=0.05), "./")
        self.assertEqual(994, len(lc_build.lc))
        lc_build = LcBuilder().build(MissionInputObjectInfo("TIC 352315023", directory, binning=0.1), "./")
        self.assertEqual(472, len(lc_build.lc))


if __name__ == '__main__':
    unittest.main()

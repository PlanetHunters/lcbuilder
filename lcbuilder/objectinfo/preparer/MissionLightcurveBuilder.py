import logging
import pandas
from lightkurve import LightCurve

from lcbuilder import constants
from lcbuilder.LcBuild import LcBuild
from lcbuilder.objectinfo.ObjectProcessingError import ObjectProcessingError
from lcbuilder.objectinfo.MissionObjectInfo import MissionObjectInfo
from lcbuilder.objectinfo.preparer.mission_data_preparer import StandardMissionDataPreparer, EverestMissionDataPreparer, \
    MissionDataPreparer
from lcbuilder.star import starinfo
from lcbuilder.objectinfo.preparer.LightcurveBuilder import LightcurveBuilder
import lightkurve as lk
import os
import numpy as np


class MissionLightcurveBuilder(LightcurveBuilder):
    """
    Prepares the data from the mission official repositories of a given target
    """
    def __init__(self):
        super().__init__()

    def log_available_data(self, object_info: MissionObjectInfo) -> None:
        mission_id = object_info.mission_id()
        mission, mission_prefix, id = MissionDataPreparer.parse_object_id(mission_id)
        if mission_prefix not in self.star_catalogs:
            raise ValueError("Wrong object id " + mission_id)
        logging.info("Downloading lightcurve files...")
        tpf_search_results = lk.search_targetpixelfile(str(mission_id), author=None)
        lc_search_results = lk.search_lightcurve(str(mission_id), author=None)
        tpf_searchcut_results = lk.search_tesscut(str(mission_id))
        if len(tpf_search_results) == 0:
            logging.warning("No TPF data %s")
        for tpf_search_result in tpf_search_results:
            # in this logging line i get " TypeError: must be real number, not numpy.str_"
            logging.info(f"There are TPF data: {tpf_search_result.mission[0]}, Year "
                         f"{tpf_search_result.year[0]}, Author: {tpf_search_result.mission[0]}, ExpTime: "
                         f"{tpf_search_result.exptime[0].value}")
        if len(tpf_searchcut_results) == 0:
            logging.warning("No TessCut data %s")
        for tpf_searchcut_result in tpf_searchcut_results:
            logging.info(f"There are TessCut data: {tpf_searchcut_result.mission[0]}, Year "
                         f"{tpf_searchcut_result.year[0]}, Author: {tpf_searchcut_result.author[0]}, ExpTime: "
                         f"{tpf_searchcut_result.exptime[0].value}")
        if len(lc_search_results) == 0:
            logging.warning("No LightCurve data with author")
        for lc_search_result in lc_search_results:
            logging.info(f"There are LightCurve data: {lc_search_result.mission[0]}, Year "
                         f"{lc_search_result.year[0]}, Author: {lc_search_result.author[0]}, ExpTime: "
                         f"{lc_search_result.exptime[0].value}")

    def get_star_info(self, object_info: MissionObjectInfo) -> starinfo.StarInfo:
        logging.info("Retrieving star catalog info")
        mission_id = object_info.mission_id()
        sherlock_id = object_info.sherlock_id()
        mission, mission_prefix, id = MissionDataPreparer.parse_object_id(mission_id)
        if mission_prefix not in self.star_catalogs:
            raise ValueError("Wrong object id " + mission_id)
        if object_info.ra is not None and object_info.dec is not None:
            star_info = starinfo.StarInfo(sherlock_id,
                                          *self.star_catalogs[constants.MISSION_ID_TESS].coords_catalog_info(
                                              object_info.ra, object_info.dec)
                                          )
        else:
            star_info = starinfo.StarInfo(sherlock_id, *self.star_catalogs[mission_prefix].catalog_info(id))
        return star_info

    def _append_lc_data(self, author: str, cadence: int, lc: LightCurve | None, lc_data: pandas.DataFrame | None,
                        sources: list[str] | None, apertures: dict | None, sectors: list[int] | set[int] | None,
                        sectors_to_start_end_times: dict | None,
                        new_lc: LightCurve, new_lc_data: pandas.DataFrame | None, new_source: str | None,
                        new_apertures: dict | None, new_sectors: list[int] | None,
                        new_sectors_to_start_end_times: dict | None):
        if new_lc:
            lc = lc.append(new_lc) if lc else new_lc
            lc.sort("time")
        if new_lc_data is not None:
            lc_data = pandas.concat([lc_data, new_lc_data]) if lc_data is not None else new_lc_data
            lc_data.sort_values(by=['time'], inplace=True)
        if new_source:
            if sources:
                sources.append(new_source)
            else:
                sources = [new_source]
        if new_apertures:
            for key in new_apertures.keys():
                if key not in apertures:
                    apertures[key] = {}
                if author not in apertures[key]:
                    apertures[key][author] = {}
                apertures[key][author][cadence] = new_apertures[key].tolist() \
                    if isinstance(new_apertures[key], np.ndarray) else new_apertures[key]
        sectors = set(sectors)
        if new_sectors:
            if not sectors:
                sectors = set([])
            for sector in new_sectors:
                sectors.add(sector)
        if new_sectors_to_start_end_times:
            for key in new_sectors_to_start_end_times.keys():
                if key not in sectors_to_start_end_times:
                    sectors_to_start_end_times[key] = {}
                if author not in sectors_to_start_end_times[key]:
                    sectors_to_start_end_times[key][author] = {}
                sectors_to_start_end_times[key][author][cadence] = new_sectors_to_start_end_times[key]
        return lc, lc_data, sources, apertures, sectors, sectors_to_start_end_times

    @staticmethod
    def get_mission_data_preparer(mission_prefix: str, author: str, cadence: int):
        preparer = StandardMissionDataPreparer()
        if mission_prefix == constants.MISSION_ID_KEPLER_2 and author == constants.EVEREST_AUTHOR:
            preparer = EverestMissionDataPreparer()
        return preparer

    def get_data(self, author: str, cadence: int, object_info: MissionObjectInfo, sherlock_dir: str,
                 caches_root_dir: str, keep_tpfs: bool = True):
        mission_id = object_info.mission_id()
        mission, mission_prefix, id = MissionDataPreparer.parse_object_id(mission_id)
        lc, lc_data, source, apertures, sectors, sectors_to_start_end_times = (
            MissionLightcurveBuilder.get_mission_data_preparer(mission_prefix, author, cadence)
            .prepare(object_info, author, cadence, sherlock_dir, caches_root_dir, keep_tpfs))
        return lc, lc_data, source, apertures, sectors, sectors_to_start_end_times

    def build(self, object_info: MissionObjectInfo, sherlock_dir: str, caches_root_dir: str, keep_tpfs: bool = True):
        star_info = self.get_star_info(object_info)
        self.log_available_data(object_info)
        tpfs_dir = sherlock_dir + "/tpfs/"
        if not os.path.exists(tpfs_dir):
            os.mkdir(tpfs_dir)
        lc = None
        lc_data = None
        sectors = []
        sources = None
        apertures = {}
        sectors_to_start_end_times = {}
        for author in object_info.author:
            for cadence in object_info.cadence:
                try:
                    new_lc, new_lc_data, new_source, new_apertures, new_sectors, new_sectors_to_start_end_times = (
                        self.get_data(author, cadence, object_info, sherlock_dir, caches_root_dir, keep_tpfs))
                    lc, lc_data, sources, apertures, sectors, sectors_to_start_end_times = (
                        self._append_lc_data(author, cadence, lc, lc_data, sources, apertures, sectors,
                                             sectors_to_start_end_times, new_lc, new_lc_data, new_source, new_apertures,
                                             new_sectors, new_sectors_to_start_end_times))
                except ObjectProcessingError:
                    logging.warning(f"No data for author {author} and cadence {cadence}")
        return LcBuild(lc, lc_data, star_info, 1, None, object_info.author,
                       object_info.cadence, sectors, sources, apertures,
                       sectors_to_start_end_times=sectors_to_start_end_times)

import logging
import os
import re
import shutil
import sys
import types

try:
    from mpl_toolkits.axes_grid1.inset_locator import InsetPosition
except ImportError:
    dummy = types.ModuleType('inset_locator')
    dummy.InsetPosition = object
    sys.modules['mpl_toolkits.axes_grid1.inset_locator'] = dummy

import everest
import lightkurve as lk

import pandas as pd
from abc import ABC, abstractmethod
import astropy.io.fits as astropy_fits
import astropy.units as u
from everest.missions.k2 import Season
from lightkurve import LightCurveCollection, KeplerLightCurve

from lcbuilder.photometry.aperture_extractor import ApertureExtractor

from lcbuilder.objectinfo.ObjectProcessingError import ObjectProcessingError

from lcbuilder.constants import LIGHTKURVE_CACHE_DIR, CUTOUT_SIZE, ELEANOR_CACHE_DIR

from lcbuilder import constants
from lcbuilder.objectinfo import MissionObjectInfo
from lcbuilder.objectinfo.preparer.LightcurveBuilder import LightcurveBuilder

import lcbuilder.eleanor
sys.modules['eleanor'] = sys.modules['lcbuilder.eleanor']
import eleanor
from lcbuilder.eleanor.targetdata import TargetData

import numpy as np
from astropy.coordinates import SkyCoord

class MissionDataPreparer(ABC):
    NUMBERS_REGEX = "[0-9]+$"
    OBJECT_ID_REGEX = "^(KIC|TIC|EPIC)[-_ ]([0-9]+)$"

    def __init__(self):
        super().__init__()

    @abstractmethod
    def prepare_mission_data(self, object_info: MissionObjectInfo, author: str, cadence: int, sherlock_dir, caches_root_dir,
                     keep_tpfs: bool = True):
        pass

    def prepare(self, object_info: MissionObjectInfo, author: str, cadence: int, sherlock_dir, caches_root_dir,
                     keep_tpfs: bool = True):
        self.mission_id = object_info.mission_id()
        self.mission, self.mission_prefix, self.id = MissionDataPreparer.parse_object_id(self.mission_id)
        self.sectors = None if object_info.sectors == 'all' or self.mission != constants.MISSION_TESS else object_info.sectors
        self.campaigns = None if object_info.sectors == 'all' or self.mission != constants.MISSION_K2 else object_info.sectors
        self.quarters = None if object_info.sectors == 'all' or self.mission != constants.MISSION_KEPLER else object_info.sectors
        self.tokens = self.sectors if self.sectors is not None else self.campaigns if self.campaigns is not None else self.quarters
        self.apertures = {}
        self.tpfs_dir = sherlock_dir + "/tpfs/"
        self.lc = None
        self.lc_data = None
        self.sectors_to_start_end_times = {}
        return self.prepare_mission_data(object_info, author, cadence, sherlock_dir, caches_root_dir, keep_tpfs)

    @staticmethod
    def parse_object_id(object_id):
        if object_id is None:
            return constants.MISSION_TESS, constants.MISSION_ID_TESS, None
        object_id_parsed = re.search(MissionDataPreparer.OBJECT_ID_REGEX, object_id)
        if object_id_parsed is None:
            return None, None, None
        mission_prefix = object_id[object_id_parsed.regs[1][0]:object_id_parsed.regs[1][1]]
        id = object_id[object_id_parsed.regs[2][0]:object_id_parsed.regs[2][1]]
        if mission_prefix == constants.MISSION_ID_KEPLER:
            mission = constants.MISSION_KEPLER
        elif mission_prefix == constants.MISSION_ID_KEPLER_2:
            mission = constants.MISSION_K2
        elif mission_prefix == constants.MISSION_ID_TESS:
            mission = constants.MISSION_TESS
        else:
            mission = None
        return mission, mission_prefix, int(id)

class StandardMissionDataPreparer(MissionDataPreparer):
    def __init__(self):
        super().__init__()

    def prepare_mission_data(self, object_info: MissionObjectInfo, author: str, cadence: int, sherlock_dir, caches_root_dir,
                     keep_tpfs: bool = True):
        target_name = str(self.mission_id)
        lcf = LightcurveBuilder.search_lightcurve(target_name, self.mission_prefix, self.mission, cadence, self.sectors,
                                                  self.quarters, self.campaigns, author,
                                                  caches_root_dir + LIGHTKURVE_CACHE_DIR,
                                                  object_info.quality_flag, object_info.initial_trim_sectors)
        if lcf is None:
            raise ObjectProcessingError("The target " + str(self.mission_id) + " is not available for the author " +
                                        author + ", cadence " + str(cadence) + "s and sectors " + str(self.tokens))
        tpfs = LightcurveBuilder.search_tpf(target_name, self.mission_prefix, self.mission, cadence,
                                            self.sectors, self.quarters,
                                            self.campaigns, author, caches_root_dir + LIGHTKURVE_CACHE_DIR,
                                            object_info.quality_flag, (CUTOUT_SIZE, CUTOUT_SIZE),
                                            object_info.initial_trim_sectors)
        self.lc_data = self.extract_lc_data(lcf)
        self.lc = None
        matching_objects = []
        for tpf in tpfs:
            if keep_tpfs:
                shutil.copy(tpf.path, self.tpfs_dir + f'/{author}_{cadence}_' + os.path.basename(tpf.path))
            if self.mission_prefix == constants.MISSION_ID_KEPLER:
                sector = tpf.quarter
            elif self.mission_prefix == constants.MISSION_ID_TESS:
                sector = tpf.sector
            if self.mission_prefix == constants.MISSION_ID_KEPLER_2:
                sector = tpf.campaign
            self.sectors_to_start_end_times[int(sector)] = (tpf.time[0].value, tpf.time[-1].value)
            self.apertures[int(sector)] = ApertureExtractor.from_boolean_mask(tpf.pipeline_mask, tpf.column, tpf.row)
            try:
                if 'DATE-OBS' in tpf.meta:
                    logging.info("Sector %s dates: Start (%s) End(%s)", sector, tpf.meta['DATE-OBS'],
                                 tpf.meta['DATE-END'])
                elif 'DATE' in tpf.meta:
                    logging.info("Sector %s date (%s)", sector, tpf.meta['DATE'])
            except:
                logging.exception("Problem extracting sector dates from TPF")
        for i in range(0, len(lcf)):
            if lcf.data[i].label == self.mission_id:
                if self.lc is None:
                    self.lc = lcf.data[i].normalize()
                else:
                    self.lc = self.lc.append(lcf.data[i].normalize())
            else:
                matching_objects.append(lcf.data[i].label)
        matching_objects = set(matching_objects)
        if len(matching_objects) > 0:
            logging.warning("================================================")
            logging.warning("TICS IN THE SAME PIXEL: " + str(matching_objects))
            logging.warning("================================================")
        if self.lc is None:
            self.tokens = self.sectors if self.sectors is not None else self.campaigns if self.campaigns is not None else self.quarters
            self.tokens = self.tokens if self.tokens is not None else "all"
            raise ObjectProcessingError("The target " + target_name + " is not available for the author " + author +
                                        ", cadence " + str(cadence) + "s and sectors " + str(self.tokens))
        self.lc = self.lc.remove_nans()
        if self.mission_prefix == constants.MISSION_ID_KEPLER:
            self.sectors = [lcfile.quarter for lcfile in lcf]
        elif self.mission_prefix == constants.MISSION_ID_TESS:
            self.sectors = [file.sector for file in lcf]
        elif self.mission_prefix == constants.MISSION_ID_KEPLER_2:
            logging.info("Correcting K2 motion in light curve...")
            self.sectors = [lcfile.campaign for lcfile in lcf]
            self.lc = self.lc.to_corrector("sff").correct(windows=20)
        source = "tpf"
        return self.lc, self.lc_data, source, self.apertures, self.sectors, self.sectors_to_start_end_times

    def extract_lc_data(self, lcf: LightCurveCollection):
        fit_files = [astropy_fits.open(lcf.filename) for lcf in lcf]
        time = []
        flux = []
        flux_err = []
        background_flux = []
        quality = []
        centroids_x = []
        centroids_y = []
        motion_x = []
        motion_y = []
        for fit_file in fit_files:
            time.append(fit_file[1].data['TIME'])
            try:
                flux.append(fit_file[1].data['PDCSAP_FLUX'])
                flux_err.append(fit_file[1].data['PDCSAP_FLUX_ERR'])
            except:
                # QLP curves that can contain KSPSAP_FLUX or DET_FLUX: https://tess.mit.edu/qlp/
                try:
                    flux.append(fit_file[1].data['KSPSAP_FLUX'])
                    flux_err.append(fit_file[1].data['KSPSAP_FLUX_ERR'])
                except:
                    flux.append(fit_file[1].data['DET_FLUX'])
                    flux_err.append(fit_file[1].data['DET_FLUX_ERR'])
            background_flux.append(fit_file[1].data['SAP_BKG'])
            try:
                quality.append(fit_file[1].data['QUALITY'])
            except KeyError:
                logging.info("QUALITY info is not available.")
                quality.append(np.full(len(fit_file[1].data['TIME']), np.nan))
            try:
                centroids_x.append(fit_file[1].data['MOM_CENTR1'])
                centroids_y.append(fit_file[1].data['MOM_CENTR2'])
                motion_x.append(fit_file[1].data['POS_CORR1'])
                motion_y.append(fit_file[1].data['POS_CORR2'])
            except:
                logging.warning("No centroid and position data in light curve")
        time = np.concatenate(time)
        flux = np.concatenate(flux)
        flux_err = np.concatenate(flux_err)
        background_flux = np.concatenate(background_flux)
        quality = np.concatenate(quality)
        if len(centroids_x) > 0:
            centroids_x = np.concatenate(centroids_x)
            centroids_y = np.concatenate(centroids_y)
            motion_x = np.concatenate(motion_x)
            motion_y = np.concatenate(motion_y)
        lc_data = pd.DataFrame(columns=['time', 'flux', 'flux_err', 'background_flux', 'quality', 'centroids_x',
                                        'centroids_y', 'motion_x', 'motion_y'])
        lc_data['time'] = time
        lc_data['flux'] = flux
        lc_data['flux_err'] = flux_err
        lc_data['background_flux'] = background_flux
        lc_data['quality'] = quality
        if len(centroids_x) > 0:
            lc_data['centroids_x'] = centroids_x
            lc_data['centroids_y'] = centroids_y
            lc_data['motion_x'] = motion_x
            lc_data['motion_y'] = motion_y
        lc_data.dropna(subset=['time'], inplace=True)
        for fit_file in fit_files:
            fit_file.close()
        return lc_data

class EleanorMissionDataPreparer(MissionDataPreparer):
    def __init__(self):
        super().__init__()

    def extract_eleanor_lc_data(self, eleanor_data):
        time = []
        flux = []
        flux_err = []
        background_flux = []
        quality = []
        centroids_x = []
        centroids_y = []
        motion_x = []
        motion_y = []
        [time.append(data.time) for data in eleanor_data]
        [flux.append(data.pca_flux) for data in eleanor_data]
        [flux_err.append(data.flux_err) for data in eleanor_data]
        [background_flux.append(data.flux_bkg) for data in eleanor_data]
        try:
            [quality.append(data.quality) for data in eleanor_data]
        except KeyError:
            logging.info("QUALITY info is not available.")
            [quality.append(np.full(len(data.time), np.nan)) for data in eleanor_data]
        [centroids_x.append(data.centroid_xs - data.cen_x) for data in eleanor_data]
        [centroids_y.append(data.centroid_ys - data.cen_y) for data in eleanor_data]
        [motion_x.append(data.x_com) for data in eleanor_data]
        [motion_y.append(data.y_com) for data in eleanor_data]
        time = np.concatenate(time)
        flux = np.concatenate(flux)
        flux_err = np.concatenate(flux_err)
        background_flux = np.concatenate(background_flux)
        quality = np.concatenate(quality)
        centroids_x = np.concatenate(centroids_x)
        centroids_y = np.concatenate(centroids_y)
        motion_x = np.concatenate(motion_x)
        motion_y = np.concatenate(motion_y)
        lc_data = pd.DataFrame(columns=['time', 'flux', 'flux_err', 'background_flux', 'quality', 'centroids_x',
                                            'centroids_y', 'motion_x', 'motion_y'])
        lc_data['time'] = time
        lc_data['flux'] = flux
        lc_data['flux_err'] = flux_err
        lc_data['background_flux'] = background_flux
        lc_data['quality'] = quality
        lc_data['centroids_x'] = centroids_x
        lc_data['centroids_y'] = centroids_y
        lc_data['motion_x'] = motion_x
        lc_data['motion_y'] = motion_y
        return lc_data

    def prepare_mission_data(self, object_info: MissionObjectInfo, author: str, cadence: int, sherlock_dir, caches_root_dir,
                     keep_tpfs: bool = True):
        logging.info(f"Retrieving ELEANOR data with author {author} and cadence {cadence}s")
        source = "eleanor"
        if object_info.ra is not None and object_info.dec is not None:
            coords = SkyCoord(ra=object_info.ra, dec=object_info.dec, unit=(u.deg, u.deg))
            star = eleanor.source.multi_sectors(coords=coords, sectors=object_info.sectors,
                                                post_dir=caches_root_dir + ELEANOR_CACHE_DIR,
                                                metadata_path=caches_root_dir + ELEANOR_CACHE_DIR)
        else:
            object_id_parsed = re.search(MissionDataPreparer.NUMBERS_REGEX, object_info.id)
            object_id_parsed = object_info.id[object_id_parsed.regs[0][0]:object_id_parsed.regs[0][1]]
            star = eleanor.multi_sectors(tic=object_id_parsed, sectors=object_info.sectors,
                                         post_dir=caches_root_dir + ELEANOR_CACHE_DIR,
                                         metadata_path=caches_root_dir + ELEANOR_CACHE_DIR)
        if star is None:
            raise ObjectProcessingError("No data for this object")
        if star[0].tic:
            # TODO FIX star info objectid
            logging.info("Assotiated TIC is " + str(star[0].tic))
            tpfs = lk.search_tesscut("TIC " + str(star[0].tic), sector=self.sectors) \
                .download_all(download_dir=caches_root_dir + LIGHTKURVE_CACHE_DIR,
                              cutout_size=(CUTOUT_SIZE, CUTOUT_SIZE))
        data = []
        for s in star:
            datum = TargetData(s, height=CUTOUT_SIZE, width=CUTOUT_SIZE, do_pca=True)
            data.append(datum)
            for tpf in tpfs:
                if tpf.sector == s.sector:
                    self.sectors_to_start_end_times[s.sector] = (tpf.time[0].value, tpf.time[-1].value)
                    self.apertures[s.sector] = ApertureExtractor.from_boolean_mask(datum.aperture.astype(bool),
                                                                              tpf.column, tpf.row)
                    if keep_tpfs:
                        shutil.copy(tpf.path, self.tpfs_dir + f'/{author}_{cadence}_' + os.path.basename(tpf.path))
        quality_bitmask = np.bitwise_and(data[0].quality.astype(int),
                                         object_info.quality_flag if object_info.quality_flag != 'default' else 175)
        self.lc_data = self.extract_eleanor_lc_data(data)
        self.lc = data[0].to_lightkurve(data[0].__dict__[object_info.eleanor_corr_flux],
                                   quality_mask=quality_bitmask).remove_nans().flatten()
        self.sectors = [datum.source_info.sector for datum in data]
        if len(data) > 1:
            for datum in data[1:]:
                quality_bitmask = np.bitwise_and(datum.quality,
                                                 object_info.quality_flag if object_info.quality_flag != 'default' else 175)
                self.lc = self.lc.append(datum.to_lightkurve(datum.pca_flux, quality_mask=quality_bitmask).remove_nans()
                               .flatten())
        return self.lc, self.lc_data, source, self.apertures, self.sectors, self.sectors_to_start_end_times

class EverestMissionDataPreparer(MissionDataPreparer):
    def __init__(self):
        super().__init__()

    def prepare_mission_data(self, object_info: MissionObjectInfo, author: str, cadence: int, sherlock_dir, caches_root_dir,
                     keep_tpfs: bool = True):
        logging.info(f"Retrieving K2 data with author {author} and cadence {cadence}s")
        source = 'everest'
        everest_cadence = 'sc' if isinstance(cadence, str) and (cadence == 'short' or cadence == 'fast') or (
                isinstance(cadence, int) and cadence < 600) else 'lc'
        if self.campaigns is None:
            self.campaigns = Season(self.id)
        if not isinstance(self.campaigns, (list, np.ndarray)):
            self.campaigns = [self.campaigns]
        for campaign in self.campaigns:
            try:
                everest_star = everest.user.Everest(self.id, campaign, quiet=True, cadence=everest_cadence)
            except:
                raise ObjectProcessingError("Can't find object " + str(id) + " with " + str(cadence) + " cadence and " +
                                 str(campaign) + " campaign in Everest")
            quality_mask = ((everest_star.quality != 0) & (everest_star.quality != 27)) \
                if object_info.quality_flag == 'default' \
                else np.where(object_info.quality_flag & everest_star.quality)
            time = np.delete(everest_star.time, quality_mask)
            flux = np.delete(everest_star.flux, quality_mask)
            if self.lc is None:
                self.lc = KeplerLightCurve(time, flux).normalize()
            else:
                self.lc = self.lc.append(KeplerLightCurve(time, flux).normalize())
        self.lc = self.lc.remove_nans()
        return self.lc, self.lc_data, source, self.apertures, self.sectors, self.sectors_to_start_end_times
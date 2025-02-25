from math import floor, ceil

import foldedleastsquares
import logging
import numpy as np
import wotan
from foldedleastsquares import DefaultTransitTemplateGenerator
from lcbuilder import constants
from scipy import stats
from lightkurve import TessLightCurve
from astropy.stats.sigma_clipping import sigma_clip


class LcbuilderHelper:
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def convert_from_to(value, unit_from, unit_to):
        value_with_units = value * unit_from
        value_with_units = value_with_units.to(unit_to)
        return value_with_units.value

    @staticmethod
    def compute_t0s(time, period, t0, duration):
        last_time = time[len(time) - 1]
        first_time = time[0]
        num_of_transits_back = int(floor(((t0 - first_time) / period)))
        transits_lists_back = t0 - period * np.arange(num_of_transits_back, 0,
                                                      -1) if num_of_transits_back > 0 else np.array([])
        num_of_transits = int(ceil(((last_time - t0) / period)))
        transit_lists = t0 + period * np.arange(0, num_of_transits)
        transit_lists = np.append(transits_lists_back, transit_lists)
        plot_range = duration * 2
        transits_in_data = [
            time[(transit > time - plot_range) & (transit < time + plot_range)] for
            transit in transit_lists]
        transit_t0s_list = transit_lists[[len(transits_in_data_set) > 0 for transits_in_data_set in transits_in_data]]
        return transit_t0s_list

    @staticmethod
    def mask_transits(time, flux, period, duration, epoch, flux_err=None):
        mask = foldedleastsquares.transit_mask(time, period, duration, epoch)
        time = time[~mask]
        flux = flux[~mask]
        if flux_err is not None:
            flux_err = flux_err[~mask]
        return time, flux, flux_err

    @staticmethod
    def mask_transits_dict(lc, transits_mask):
        time, flux, flux_err = lc.time.value, lc.flux.value, lc.flux_err.value
        for transit_mask in transits_mask:
            logging.info('* Transit mask with P=%.2f d, T0=%.2f d, Dur=%.2f min *', transit_mask["P"],
                         transit_mask["T0"], transit_mask["D"])
            time, flux, flux_err = LcbuilderHelper.mask_transits(time, flux, transit_mask["P"],
                                                                 transit_mask["D"] / 60 / 24, transit_mask["T0"])
        lc = TessLightCurve(time=time, flux=flux, flux_err=flux_err, quality=np.zeros(len(time)))
        return lc

    @staticmethod
    def clip_outliers(array, sigma, sigma_lower=None, sigma_upper=None):
        mask = outlier_mask = sigma_clip(
            data=array,
            sigma=sigma,
            sigma_lower=sigma_lower,
            sigma_upper=sigma_upper,
        ).mask
        return array[~outlier_mask], outlier_mask

    @staticmethod
    def correct_epoch(mission, epoch):
        result = epoch
        if mission == constants.MISSION_TESS and epoch - constants.TBJD > 0:
            result = epoch - constants.TBJD
        elif (mission == constants.MISSION_K2 or mission == constants.MISSION_KEPLER) and epoch - constants.KBJD > 0:
            result = epoch - constants.TBJD
        return result

    @staticmethod
    def normalize_mission_epoch(mission, epoch):
        corrected_epoch = epoch
        if mission == constants.MISSION_TESS:
            corrected_epoch = epoch + constants.TBJD
        elif mission == constants.MISSION_KEPLER:
            corrected_epoch = epoch + constants.KBJD
        elif mission == constants.MISSION_K2:
            corrected_epoch = epoch + constants.KBJD
        return corrected_epoch

    @staticmethod
    def bin(time, values, bins, values_err=None, bin_err_mode='values_std'):
        if len(time) <= bins:
            value_err = values_err if values_err is not None else np.nanstd(values)
            time_err = (time[1] - time[0]) if len(time) > 1 else np.nan
            return time, values, time_err, value_err
        bin_means, bin_edges, binnumber = stats.binned_statistic(time, values, statistic='mean', bins=bins)
        if bin_err_mode == 'flux_err':
            bin_stds, _, _ = stats.binned_statistic(time, values_err, statistic='mean', bins=bins)
        elif bin_err_mode == 'values_std':
            bin_stds, _, _ = stats.binned_statistic(time, values, statistic='std', bins=bins)
        elif bin_err_mode == 'values_snr':
            bin_stds, _, _ = stats.binned_statistic(time, values, statistic='std', bins=bins)
            bin_counts, _, _ = stats.binned_statistic(time, values, statistic='count', bins=bins)
            bin_counts[bin_counts == 0] = 1
            bin_stds = bin_stds / np.sqrt(bin_counts)
        bin_width = (bin_edges[1] - bin_edges[0])
        bin_centers = bin_edges[1:] - bin_width / 2
        bin_means_data_mask = np.isnan(bin_means)
        bin_centers = bin_centers[~bin_means_data_mask]
        bin_means = bin_means[~bin_means_data_mask]
        bin_stds = bin_stds[~bin_means_data_mask]
        return bin_centers, bin_means, bin_width, bin_stds

    @staticmethod
    def calculate_period_grid(time, min_period, max_period, oversampling, star_info, transits_min_count,
                              max_oversampling=15):
        time_span_curve = time[-1] - time[0]
        dif = time[1:] - time[:-1]
        jumps = np.where(dif > 1)[0]
        jumps = np.append(jumps, len(time) - 1)
        previous_jump_index = 0
        time_span_all_sectors = 0
        empty_days = 0
        for jumpIndex in jumps[0:-1]:
            empty_days = empty_days + time[jumpIndex + 1] - time[jumpIndex - 1]
        if oversampling is None:
            oversampling = int(1 / ((time_span_curve - empty_days) / time_span_curve))
            oversampling = oversampling if oversampling < max_oversampling else max_oversampling
            oversampling = oversampling if oversampling > 3 else 3
        for jumpIndex in jumps:
            time_chunk = time[
                         previous_jump_index + 1:jumpIndex]  # ignoring first measurement as could be the last from the previous chunk
            if len(time_chunk) > 0:
                time_span_all_sectors = time_span_all_sectors + (time_chunk[-1] - time_chunk[0])
            previous_jump_index = jumpIndex
        return DefaultTransitTemplateGenerator() \
                   .period_grid(star_info.radius, star_info.mass, time_span_curve, min_period,
                                max_period, oversampling, transits_min_count, time_span_curve), oversampling

    @staticmethod
    def truncate_borders(time, flux, flux_err, min_gap_size=0.5, truncate_border=0):
        if truncate_border <= 0:
            return time, flux, flux_err
        dif = time[1:] - time[:-1]
        jumps = np.where(dif > min_gap_size)[0]
        jumps = np.append(jumps, len(time) - 1)
        previous_jump_index = 0
        truncated_flux = []
        truncated_flux_err = []
        truncated_time = []
        for jumpIndex in jumps:
            time_chunk = time[previous_jump_index + 1:jumpIndex]  # ignoring first measurement as could be the last from the previous chunk
            flux_chunk = flux[previous_jump_index + 1:jumpIndex]  # ignoring first measurement as could be the last from the previous chunk
            flux_err_chunk = flux_err[previous_jump_index + 1:jumpIndex]  # ignoring first measurement as could be the last from the previous chunk
            if len(time_chunk) > 0:
                valid_indexes = np.argwhere((time_chunk > time_chunk[0] + truncate_border) & (time_chunk < time_chunk[-1] - truncate_border)).flatten()
                truncated_time = np.append(truncated_time, time_chunk[valid_indexes])
                truncated_flux = np.append(truncated_flux, flux_chunk[valid_indexes])
                truncated_flux_err = np.append(truncated_flux_err, flux_err_chunk[valid_indexes])
            previous_jump_index = jumpIndex
        return np.array(truncated_time), np.array(truncated_flux), np.array(truncated_flux_err)

    @staticmethod
    def compute_cadence(time):
        cadence_array = np.diff(time) * 24 * 60 * 60
        cadence_array = cadence_array[~np.isnan(cadence_array)]
        cadence_array = cadence_array[cadence_array > 0]
        return int(np.round(np.nanmedian(cadence_array)))

    @staticmethod
    def estimate_transit_cadences(cadence_s, duration_d):
        cadence = cadence_s / 3600 / 24
        return duration_d // cadence
    
    @staticmethod
    def detrend(time, flux, window_size, check_cadence=False, method='biweight'):
        if np.all(flux < 0):
            flux = flux - np.min(flux)
        if check_cadence:
            cadence = LcbuilderHelper.compute_cadence(time) / 24 / 3600
            if window_size > cadence * 3:
                flatten_lc, trend = wotan.flatten(time, flux, window_length=window_size, return_trend=True,
                                                  method=method)
            else:
                flatten_lc, trend = wotan.flatten(time, flux, window_length=cadence * 4, return_trend=True,
                                                  method=method)
        else:
            flatten_lc, trend = wotan.flatten(time, flux, window_length=window_size, return_trend=True,
                                                  method=method)
        return flatten_lc, trend

    @staticmethod
    def mission_lightkurve_sector_extraction(mission: str, lightkurve_item: object) -> np.ndarray:
        """
        Returns the sectors, quarters or campaigns from the given lightkurve object
        :param mission: the mission of the lightkurve item
        :param lightkurve_item: the lightkurve object containing the data
        :return name: the name of the time series groups (quarter, campaign, sector)
        :return sector: the list containing the identifiers
        """
        sector_name = None
        sector = None
        if mission == constants.MISSION_TESS:
            sector = lightkurve_item.sector
            sector_name = 'sector'
        elif mission == constants.MISSION_KEPLER:
            sector = lightkurve_item.quarter
            sector_name = 'quarter'
        elif mission == constants.MISSION_K2:
            sector = lightkurve_item.campaign
            sector_name = 'campaign'
        if isinstance(sector, int):
            sector = np.array([sector])
        elif isinstance(sector, list):
            sector = np.array(sector)
        return sector_name, sector

    @staticmethod
    def mission_pixel_size(mission):
        px_size_arcs = None
        if mission == constants.MISSION_TESS:
            px_size_arcs = 20.25
        elif mission == constants.MISSION_KEPLER:
            px_size_arcs = 4
        elif mission == constants.MISSION_K2:
            px_size_arcs = 4
        return px_size_arcs


import logging
import math
import os
import re

import lightkurve
import numpy
import pandas
import yaml
from scipy import stats
from scipy.signal import savgol_filter
from wotan import flatten

from lcbuilder.star.starinfo import StarInfo

from lcbuilder.objectinfo.InputObjectInfo import InputObjectInfo
from lcbuilder.objectinfo.MissionFfiCoordsObjectInfo import MissionFfiCoordsObjectInfo
from lcbuilder.objectinfo.MissionFfiIdObjectInfo import MissionFfiIdObjectInfo
from lcbuilder.objectinfo.MissionInputObjectInfo import MissionInputObjectInfo
from lcbuilder.objectinfo.MissionObjectInfo import MissionObjectInfo
from lcbuilder.objectinfo.ObjectInfo import ObjectInfo
from lcbuilder.objectinfo.preparer.MissionFfiLightcurveBuilder import MissionFfiLightcurveBuilder
from lcbuilder.objectinfo.preparer.MissionInputLightcurveBuilder import MissionInputLightcurveBuilder
from lcbuilder.objectinfo.preparer.MissionLightcurveBuilder import MissionLightcurveBuilder
import matplotlib.pyplot as plt
import foldedleastsquares as tls


class LcBuilder:
    COORDS_REGEX = "^(-{0,1}[0-9.]+)_(-{0,1}[0-9.]+)$"
    DEFAULT_CADENCES_FOR_MISSION = {"Kepler": 60, "K2": 60, "TESS": 120}

    def __init__(self) -> None:
        self.lightcurve_builders = {InputObjectInfo: MissionInputLightcurveBuilder(),
                                    MissionInputObjectInfo: MissionInputLightcurveBuilder(),
                                    MissionObjectInfo: MissionLightcurveBuilder(),
                                    MissionFfiIdObjectInfo: MissionFfiLightcurveBuilder(),
                                    MissionFfiCoordsObjectInfo: MissionFfiLightcurveBuilder()}

    def build(self, object_info: ObjectInfo, object_dir: str, caches_root_dir=os.path.expanduser('~') + "/"):
        lc_build = self.lightcurve_builders[type(object_info)].build(object_info, object_dir, caches_root_dir)
        if lc_build.tpf_apertures is not None:
            with open(object_dir + "/apertures.yaml", 'w') as f:
                apertures = {sector: [aperture.tolist() for aperture in apertures]
                             for sector, apertures in lc_build.tpf_apertures.items()}
                apertures = {"sectors": apertures}
                f.write(yaml.dump(apertures, default_flow_style=True))
        sherlock_id = object_info.sherlock_id()
        star_info = self.__complete_star_info(object_info.mission_id(), object_info.star_info, lc_build.star_info,
                                              object_dir)
        time_float = lc_build.lc.time.value
        flux_float = lc_build.lc.flux.value
        flux_err_float = lc_build.lc.flux_err.value
        lc = lightkurve.LightCurve(time=time_float, flux=flux_float, flux_err=flux_err_float)
        lc_df = pandas.DataFrame(columns=['#time', 'flux', 'flux_err'])
        lc_df['#time'] = time_float
        lc_df['flux'] = flux_float
        lc_df['flux_err'] = flux_err_float
        lc_df.to_csv(object_dir + "lc.csv", index=False)
        lc = lc.remove_outliers(sigma_lower=float('inf'), sigma_upper=object_info.outliers_sigma)
        time_float = lc.time.value
        flux_float = lc.flux.value
        flux_err_float = lc.flux_err.value
        cadence_array = numpy.diff(time_float) * 24 * 60 * 60
        cadence_array = cadence_array[~numpy.isnan(cadence_array)]
        cadence_array = cadence_array[cadence_array > 0]
        lc_build.cadence = int(numpy.round(numpy.nanmedian(cadence_array)))
        clean_time, flatten_flux, clean_flux_err = self.__clean_initial_flux(object_info, time_float, flux_float,
                                                                             flux_err_float, star_info,
                                                                             lc_build.cadence, object_dir)
        lc = lightkurve.LightCurve(time=clean_time, flux=flatten_flux, flux_err=clean_flux_err)
        periodogram = lc.to_periodogram(minimum_period=0.05, maximum_period=15, oversample_factor=10)
        # power_norm = self.running_median(periodogram.power.value, 20)
        periodogram.plot(view='period', scale='log')
        plt.title(str(sherlock_id) + " Lightcurve periodogram")
        plt.savefig(object_dir + "/Periodogram_" + str(sherlock_id) + ".png", bbox_inches='tight')
        plt.clf()
        # power_mod = periodogram.power.value - power_norm
        # power_mod = power_mod / np.mean(power_mod)
        # periodogram.power = power_mod * u.d / u.d
        # periodogram.plot(view='period', scale='log')
        # plt.title(str(sherlock_id) + " Lightcurve normalized periodogram")
        # plt.savefig(object_dir + "PeriodogramNorm_" + str(sherlock_id) + ".png", bbox_inches='tight')
        # plt.clf()
        if object_info.auto_detrend_period is not None:
            lc_build.detrend_period = object_info.auto_detrend_period
        elif object_info.auto_detrend_enabled:
            lc_build.detrend_period = self.__calculate_max_significant_period(lc, periodogram)
        if lc_build.detrend_period is not None:
            logging.info('================================================')
            logging.info('AUTO-DETREND EXECUTION')
            logging.info('================================================')
            logging.info("Period = %.3f", lc_build.detrend_period)
            lc.fold(lc_build.detrend_period).scatter()
            plt.title("Phase-folded period: " + format(lc_build.detrend_period, ".2f") + " days")
            plt.savefig(object_dir + "/Phase_detrend_period_" + str(sherlock_id) + "_" +
                        format(lc_build.detrend_period, ".2f") + "_days.png", bbox_inches='tight')
            plt.clf()
            flatten_flux, lc_trend = self.__detrend_by_period(object_info.auto_detrend_method, clean_time, flatten_flux,
                                                              lc_build.detrend_period * object_info.auto_detrend_ratio)
        if object_info.initial_mask is not None:
            logging.info('================================================')
            logging.info('INITIAL MASKING')
            logging.info('================================================')
            initial_mask = object_info.initial_mask
            logging.info('** Applying ordered masks to the lightcurve **')
            for mask_range in initial_mask:
                logging.info('* Initial mask since day %.2f to day %.2f. *', mask_range[0], mask_range[1])
                mask = [(clean_time < mask_range[0] if not math.isnan(mask_range[1]) else False) |
                        (clean_time > mask_range[1] if not math.isnan(mask_range[1]) else False)]
                clean_time = clean_time[mask]
                flatten_flux = flatten_flux[mask]
                clean_flux_err = clean_flux_err[mask]
        if object_info.initial_transit_mask is not None:
            logging.info('================================================')
            logging.info('INITIAL TRANSIT MASKING')
            logging.info('================================================')
            initial_transit_mask = object_info.initial_transit_mask
            logging.info('** Applying ordered transit masks to the lightcurve **')
            for transit_mask in initial_transit_mask:
                logging.info('* Transit mask with P=%.2f d, T0=%.2f d, Dur=%.2f min *', transit_mask["P"],
                             transit_mask["T0"], transit_mask["D"])
                mask = tls.transit_mask(clean_time, transit_mask["P"], transit_mask["D"] / 60 / 24, transit_mask["T0"])
                clean_time = clean_time[~mask]
                flatten_flux = flatten_flux[~mask]
                clean_flux_err = clean_flux_err[~mask]
        lc = lightkurve.LightCurve(time=clean_time, flux=flatten_flux, flux_err=clean_flux_err)
        lc_build.lc = lc
        return lc_build

    def smooth(self, flux, window_len=11, window='blackman'):
        clean_flux = savgol_filter(flux, window_len, 3)
        # if window_len < 3:
        #     return flux
        # if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        #     raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
        # s = np.r_[np.ones(window_len//2), clean_flux, np.ones(window_len//2)]
        # # print(len(s))
        # if window == 'flat':  # moving average
        #     w = np.ones(window_len, 'd')
        # else:
        #     w = eval('np.' + window + '(window_len)')
        # # TODO probably problematic with big gaps in the middle of the curve
        # clean_flux = np.convolve(w / w.sum(), s, mode='valid')
        return clean_flux

    def __calculate_max_significant_period(self, lc, periodogram):
        #max_accepted_period = (lc.time[len(lc.time) - 1] - lc.time[0]) / 4
        max_accepted_period = numpy.float64(10)
        # TODO related to https://github.com/franpoz/SHERLOCK/issues/29 check whether this fits better
        max_power_index = numpy.argmax(periodogram.power)
        period = periodogram.period[max_power_index]
        if max_power_index > 0.0008:
            period = period.value
            logging.info("Auto-Detrend found the strong period: " + str(period) + ".")
        else:
            logging.info("Auto-Detrend did not find relevant periods.")
            period = None
        return period

    def __detrend_by_period(self, method, time, flux, period_window):
        if method == 'gp':
            flatten_lc, lc_trend = flatten(time, flux, method=method, kernel='matern',
                                   kernel_size=period_window, return_trend=True, break_tolerance=0.5)
        else:
            flatten_lc, lc_trend = flatten(time, flux, window_length=period_window, return_trend=True,
                                           method=method, break_tolerance=0.5)
        return flatten_lc, lc_trend

    def __complete_star_info(self, object_id, input_star_info, catalogue_star_info, object_dir):
        if catalogue_star_info is None:
            catalogue_star_info = StarInfo()
        result_star_info = catalogue_star_info
        if input_star_info is None:
            input_star_info = StarInfo()
        if input_star_info.radius is not None:
            result_star_info.radius = input_star_info.radius
            result_star_info.radius_assumed = False
        if input_star_info.radius_min is not None:
            result_star_info.radius_min = input_star_info.radius_min
        if input_star_info.radius_max is not None:
            result_star_info.radius_max = input_star_info.radius_max
        if input_star_info.mass is not None:
            result_star_info.mass = input_star_info.mass
            result_star_info.mass_assumed = False
        if input_star_info.mass_min is not None:
            result_star_info.mass_min = input_star_info.mass_min
        if input_star_info.mass_max is not None:
            result_star_info.mass_max = input_star_info.mass_max
        if input_star_info.ra is not None:
            result_star_info.ra = input_star_info.ra
        if input_star_info.dec is not None:
            result_star_info.dec = input_star_info.dec
        if input_star_info.teff is not None:
            result_star_info.teff = input_star_info.teff
        if input_star_info.lum is not None:
            result_star_info.lum = input_star_info.lum
        if input_star_info.logg is not None:
            result_star_info.logg = input_star_info.logg
        if input_star_info.ld_coefficients is not None:
            result_star_info.ld_coefficients = input_star_info.ld_coefficients
        if result_star_info is not None:
            logging.info("Star info prepared.")
            if result_star_info.mass is None or numpy.isnan(result_star_info.mass):
                logging.info("Star catalog doesn't provide mass. Assuming M=0.1Msun")
                result_star_info.assume_model_mass()
            if result_star_info.radius is None or numpy.isnan(result_star_info.radius):
                logging.info("Star catalog doesn't provide radius. Assuming R=0.1Rsun")
                result_star_info.assume_model_radius()
            if result_star_info.mass_min is None or numpy.isnan(result_star_info.mass_min):
                result_star_info.mass_min = result_star_info.mass - (0.5 if result_star_info.mass > 0.5 else result_star_info.mass / 2)
                logging.info("Star catalog doesn't provide M_low_err. Assuming M_low_err=%.3fMsun", result_star_info.mass_min)
            if result_star_info.mass_max is None or numpy.isnan(result_star_info.mass_max):
                result_star_info.mass_max = result_star_info.mass + 0.5
                logging.info("Star catalog doesn't provide M_up_err. Assuming M_low_err=%.3fMsun", result_star_info.mass_max)
            if result_star_info.radius_min is None or numpy.isnan(result_star_info.radius_min):
                result_star_info.radius_min = result_star_info.radius - (0.5 if result_star_info.radius > 0.5 else result_star_info.radius / 2)
                logging.info("Star catalog doesn't provide R_low_err. Assuming R_low_err=%.3fRsun", result_star_info.radius_min)
            if result_star_info.radius_max is None or numpy.isnan(result_star_info.radius_max):
                result_star_info.radius_max = result_star_info.radius + 0.5
                logging.info("Star catalog doesn't provide R_up_err. Assuming R_up_err=%.3fRsun", result_star_info.radius_max)
        logging.info('================================================')
        logging.info('STELLAR PROPERTIES')
        logging.info('================================================')
        logging.info('limb-darkening estimates using quadratic LD (a,b)= %s', result_star_info.ld_coefficients)
        logging.info('mass = %.6f', result_star_info.mass)
        logging.info('mass_min = %.6f', result_star_info.mass_min)
        logging.info('mass_max = %.6f', result_star_info.mass_max)
        logging.info('radius = %.6f', result_star_info.radius)
        logging.info('radius_min = %.6f', result_star_info.radius_min)
        logging.info('radius_max = %.6f', result_star_info.radius_max)
        logging.info('teff = %.6f', result_star_info.teff)
        logging.info('lum = %.6f', result_star_info.lum)
        logging.info('logg = %.6f', result_star_info.logg)
        star_df = pandas.DataFrame(columns=['obj_id', 'ra', 'dec', 'R_star', 'R_star_lerr', 'R_star_uerr', 'M_star',
                                            'M_star_lerr', 'M_star_uerr', 'Teff_star', 'Teff_star_lerr',
                                            'Teff_star_uerr', 'ld_a', 'ld_b'])
        ld_a = result_star_info.ld_coefficients[0] if result_star_info.ld_coefficients is not None else None
        ld_b = result_star_info.ld_coefficients[1] if result_star_info.ld_coefficients is not None else None
        star_df = star_df.append(
            {'obj_id': object_id, 'ra': result_star_info.ra, 'dec': result_star_info.dec,
             'R_star': result_star_info.radius,
             'R_star_lerr': result_star_info.radius - result_star_info.radius_min,
             'R_star_uerr': result_star_info.radius_max - result_star_info.radius,
             'M_star': result_star_info.mass, 'M_star_lerr': result_star_info.mass - result_star_info.mass_min,
             'M_star_uerr': result_star_info.mass_max - result_star_info.mass,
             'Teff_star': result_star_info.teff, 'Teff_star_lerr': 200, 'Teff_star_uerr': 200,
             'logg': result_star_info.logg, 'logg_err': result_star_info.logg_err,
             'ld_a': ld_a, 'ld_b': ld_b,
             'feh': result_star_info.feh,
             'feh_err': result_star_info.feh_err, 'v': result_star_info.v, 'v_err': result_star_info.v_err,
             'j': result_star_info.j, 'j_err': result_star_info.j_err,
             'k': result_star_info.k, 'k_err': result_star_info.k_err,
             'h': result_star_info.h, 'h_err': result_star_info.h_err,
             'kp': result_star_info.kp},
            ignore_index=True)
        star_df.to_csv(object_dir + "params_star.csv", index=False)
        return result_star_info

    def __clean_initial_flux(self, object_info, time, flux, flux_err, star_info, cadence, object_dir):
        clean_time = time
        clean_flux = flux
        clean_flux_err = flux_err
        is_short_cadence = cadence <= 300
        if object_info.prepare_algorithm is not None:
            clean_time, clean_flux, clean_flux_err = object_info.prepare_algorithm.prepare(object_info, clean_time,
                                                                                           clean_flux, clean_flux_err)
        if (is_short_cadence and object_info.smooth_enabled) or (object_info.high_rms_enabled and object_info.initial_mask is None):
            logging.info('================================================')
            logging.info('INITIAL FLUX CLEANING')
            logging.info('================================================')
        if object_info.high_rms_enabled and object_info.initial_mask is None:
            logging.info('Masking high RMS areas by a factor of %.2f with %.1f hours binning',
                         object_info.high_rms_threshold, object_info.high_rms_bin_hours)
            bins_per_day = 24 / object_info.high_rms_bin_hours
            dif = time[1:] - time[:-1]
            jumps = numpy.where(dif > 3)[0]
            jumps = numpy.append(jumps, len(clean_time))
            before_flux = clean_flux
            previous_jump_index = 0
            entire_high_rms_mask = numpy.array([], dtype=bool)
            entire_bin_centers = numpy.array([])
            entire_bin_stds = numpy.array([])
            entire_rms_threshold_array = numpy.array([])
            for jumpIndex in jumps:
                time_partial = clean_time[previous_jump_index:jumpIndex]
                flux_partial = clean_flux[previous_jump_index:jumpIndex]
                before_flux_partial = before_flux[previous_jump_index:jumpIndex]
                bins = (time_partial[len(time_partial) - 1] - time_partial[0]) * bins_per_day
                bin_stds, bin_edges, binnumber = stats.binned_statistic(time_partial, flux_partial, statistic='std', bins=bins)
                stds_median = numpy.nanmedian(bin_stds[bin_stds > 0])
                stds_median_array = numpy.full(len(bin_stds), stds_median)
                rms_threshold_array = stds_median_array * object_info.high_rms_threshold
                too_high_bin_stds_indexes = numpy.argwhere(bin_stds > rms_threshold_array)
                high_std_mask = numpy.array([bin_id - 1 in too_high_bin_stds_indexes for bin_id in binnumber])
                entire_high_rms_mask = numpy.append(entire_high_rms_mask, high_std_mask)
                bin_width = (bin_edges[1] - bin_edges[0])
                bin_centers = bin_edges[1:] - bin_width / 2
                entire_bin_centers = numpy.append(entire_bin_centers, bin_centers)
                entire_bin_stds = numpy.append(entire_bin_stds, bin_stds)
                entire_rms_threshold_array = numpy.append(entire_rms_threshold_array, rms_threshold_array)
                previous_jump_index = jumpIndex
                self.__plot_rms_mask(star_info.object_id, object_info.high_rms_bin_hours, bin_centers,
                                     bin_stds, rms_threshold_array, high_std_mask, time_partial, flux_partial,
                                     'High_RMS_Mask_' + str(star_info.object_id) + '_time_' +
                                     str(time_partial[1]) + '_' + str(time_partial[-1]), object_dir)
            self.__plot_rms_mask(star_info.object_id, object_info.high_rms_bin_hours, entire_bin_centers,
                                 entire_bin_stds, entire_rms_threshold_array, entire_high_rms_mask, clean_time,
                                 clean_flux, 'High_RMS_Mask_' + str(star_info.object_id), object_dir)
        if is_short_cadence and object_info.smooth_enabled:
            #logging.info('Applying Smooth phase (savgol + weighted average)')
            logging.info('Applying Smooth phase (savgol)')
            clean_flux = self.smooth(clean_flux)
            # TODO to use convolve we need to remove the borders effect
            # clean_flux = np.convolve(clean_flux, [0.025, 0.05, 0.1, 0.155, 0.34, 0.155, 0.1, 0.05, 0.025], "same")
            # clean_flux = np.convolve(clean_flux, [0.025, 0.05, 0.1, 0.155, 0.34, 0.155, 0.1, 0.05, 0.025], "same")
            # clean_flux[0:5] = 1
            # clean_flux[len(clean_flux) - 6: len(clean_flux) - 1] = 1
            #clean_flux = uniform_filter1d(clean_flux, 11)
            #clean_flux = self.flatten_bw(self.FlattenInput(clean_time, clean_flux, 0.02))[0]
        return clean_time, clean_flux, clean_flux_err

    def __plot_rms_mask(self, object_id, rms_bin_hours, bin_centers, bin_stds, rms_threshold_array, rms_mask,
                        time, flux, filename, object_dir):
        plot_dir = object_dir + "/rms_mask/"
        if not os.path.exists(plot_dir):
            os.mkdir(plot_dir)
        fig, axs = plt.subplots(2, 1, figsize=(8, 8), constrained_layout=True)
        axs[0].set_title(str(rms_bin_hours) + " hours binned RMS")
        axs[1].set_title("Total and masked high RMS flux")
        fig.suptitle(str(object_id) + " High RMS Mask")
        axs[0].set_xlabel('Time (d)')
        axs[0].set_ylabel('Flux RMS')
        axs[1].set_xlabel('Time (d)')
        axs[1].set_ylabel('Flux norm.')
        axs[0].plot(bin_centers, bin_stds, color='black', alpha=0.75, rasterized=True, label="RMS")
        axs[0].plot(bin_centers, rms_threshold_array, color='red', rasterized=True,
                    label='Mask Threshold')
        axs[1].scatter(time[1:], flux[1:], color='gray', alpha=0.5, rasterized=True, label="Flux norm.")
        axs[1].scatter(time[rms_mask][1:], flux[rms_mask][1:], linewidth=1, color='red',
                       alpha=1.0,
                       label="High RMS")
        axs[0].legend(loc="upper right")
        axs[1].legend(loc="upper right")
        fig.savefig(plot_dir + filename + '.png')
        fig.clf()

    def __running_median(self, data, kernel):
        """Returns sliding median of width 'kernel' and same length as data """
        idx = numpy.arange(kernel) + numpy.arange(len(data) - kernel + 1)[:, None]
        med = numpy.percentile(data[idx], 90, axis=1)

        # Append the first/last value at the beginning/end to match the length of
        # data and returned median
        first_values = med[0]
        last_values = med[-1]
        missing_values = len(data) - len(med)
        values_front = int(missing_values * 0.5)
        values_end = missing_values - values_front
        med = numpy.append(numpy.full(values_front, first_values), med)
        med = numpy.append(med, numpy.full(values_end, last_values))
        return med

    def build_object_info(self, target_name, author, sectors, file, cadence, initial_mask, initial_transit_mask,
                          star_info, aperture, eleanor_corr_flux='pca_flux',
                          outliers_sigma=None, high_rms_enabled=True, high_rms_threshold=2.5,
                          high_rms_bin_hours=4, smooth_enabled=False,
                          auto_detrend_enabled=False, auto_detrend_method="cosine", auto_detrend_ratio=0.25,
                          auto_detrend_period=None, prepare_algorithm=None):
        mission, mission_prefix, id = MissionLightcurveBuilder().parse_object_id(target_name)
        coords = None if mission is not None else self.parse_coords(target_name)
        cadence = cadence if cadence is not None else self.DEFAULT_CADENCES_FOR_MISSION[mission]
        if mission is not None and file is None and cadence <= 300:
            return MissionObjectInfo(target_name, sectors, author, cadence, initial_mask, initial_transit_mask,
                                     star_info, aperture, outliers_sigma, high_rms_enabled,
                                     high_rms_threshold, high_rms_bin_hours, smooth_enabled, auto_detrend_enabled,
                                     auto_detrend_method, auto_detrend_ratio, auto_detrend_period, prepare_algorithm)
        elif mission is not None and file is None and cadence > 300:
            return MissionFfiIdObjectInfo(target_name, sectors, author, cadence, initial_mask, initial_transit_mask,
                                          star_info, aperture, eleanor_corr_flux,
                                          outliers_sigma, high_rms_enabled, high_rms_threshold, high_rms_bin_hours,
                                          smooth_enabled, auto_detrend_enabled, auto_detrend_method, auto_detrend_ratio,
                                          auto_detrend_period, prepare_algorithm)
        elif mission is not None and file is not None:
            return MissionInputObjectInfo(target_name, file, initial_mask, initial_transit_mask,
                                          star_info, outliers_sigma, high_rms_enabled, high_rms_threshold,
                                          high_rms_bin_hours, smooth_enabled, auto_detrend_enabled, auto_detrend_method,
                                          auto_detrend_ratio, auto_detrend_period, prepare_algorithm)
        elif mission is None and coords is not None and cadence > 300:
            return MissionFfiCoordsObjectInfo(coords[0], coords[1], sectors, author, cadence, initial_mask,
                                              initial_transit_mask, star_info, aperture,
                                              eleanor_corr_flux, outliers_sigma, high_rms_enabled, high_rms_threshold,
                                              high_rms_bin_hours, smooth_enabled, auto_detrend_enabled,
                                              auto_detrend_method, auto_detrend_ratio, auto_detrend_period,
                                              prepare_algorithm)
        elif mission is None and file is not None:
            return InputObjectInfo(file, initial_mask, initial_transit_mask, star_info,
                                   outliers_sigma, high_rms_enabled, high_rms_threshold, high_rms_bin_hours,
                                   smooth_enabled, auto_detrend_enabled, auto_detrend_method, auto_detrend_ratio,
                                   auto_detrend_period, prepare_algorithm)
        else:
            raise ValueError(
                "Invalid target definition with target_name={}, mission={}, id={}, coords={}, sectors={}, file={}, "
                "cadence={}".format(target_name, mission, id, coords, sectors, file, cadence))

    def parse_object_info(self, target: str):
        return MissionLightcurveBuilder().parse_object_id(target)

    def parse_coords(self, target: str):
        coords_parsed = re.search(self.COORDS_REGEX, target)
        coords = [coords_parsed.group(1), coords_parsed.group(2)] if coords_parsed is not None else None
        return coords

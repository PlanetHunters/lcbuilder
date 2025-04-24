import logging
import math
import os
import re
import scipy
import lightkurve
import numpy
import pandas
import yaml
import multiprocessing

from lcbuilder.constants import ELEANOR_CACHE_DIR, LIGHTKURVE_CACHE_DIR
from scipy import stats
from scipy.signal import savgol_filter
from wotan import flatten

from lcbuilder import constants
from lcbuilder.helper import LcbuilderHelper
from lcbuilder.objectinfo.preparer.mission_data_preparer import MissionDataPreparer
from lcbuilder.star.starinfo import StarInfo

from lcbuilder.objectinfo.InputObjectInfo import InputObjectInfo
from lcbuilder.objectinfo.MissionInputObjectInfo import MissionInputObjectInfo
from lcbuilder.objectinfo.MissionObjectInfo import MissionObjectInfo
from lcbuilder.objectinfo.ObjectInfo import ObjectInfo
from lcbuilder.objectinfo.preparer.MissionInputLightcurveBuilder import MissionInputLightcurveBuilder
from lcbuilder.objectinfo.preparer.MissionLightcurveBuilder import MissionLightcurveBuilder
import matplotlib.pyplot as plt
import foldedleastsquares as tls


class LcBuilder:
    COORDS_REGEX = "^(-{0,1}[0-9.]+)_(-{0,1}[0-9.]+)$"
    DEFAULT_CADENCES_FOR_MISSION = {constants.MISSION_KEPLER: 60, constants.MISSION_K2: 60,
                                    constants.MISSION_TESS: 120}

    def __init__(self) -> None:
        self.lightcurve_builders = {InputObjectInfo: MissionInputLightcurveBuilder(),
                                    MissionInputObjectInfo: MissionInputLightcurveBuilder(),
                                    MissionObjectInfo: MissionLightcurveBuilder()}

    def build(self, object_info: ObjectInfo, object_dir: str, caches_root_dir=os.path.expanduser('~') + "/",
              cpus=multiprocessing.cpu_count() - 1):
        if not os.path.exists(caches_root_dir):
            os.mkdir(caches_root_dir)
        lightkurve_cache_dir = caches_root_dir + LIGHTKURVE_CACHE_DIR
        if not os.path.exists(lightkurve_cache_dir):
            os.mkdir(lightkurve_cache_dir)
        eleanor_cache_dir = caches_root_dir + ELEANOR_CACHE_DIR
        if not os.path.exists(eleanor_cache_dir):
            os.mkdir(eleanor_cache_dir)
        lc_build = self.lightcurve_builders[type(object_info)].build(object_info, object_dir, caches_root_dir)
        if lc_build.tpf_apertures is not None:
            with open(object_dir + "/apertures.yaml", 'w') as f:
                apertures = {"sectors": lc_build.tpf_apertures}
                f.write(yaml.dump(apertures, default_flow_style=True))
        sherlock_id = object_info.sherlock_id()
        star_info = self.__complete_star_info(object_info.mission_id(), object_info.star_info, lc_build.star_info,
                                              object_dir)
        time_float = lc_build.lc.time.value
        flux_float = lc_build.lc.flux.value
        flux_err_float = lc_build.lc.flux_err.value
        if object_info.initial_trim is not None:
            logging.info(f"Trimming data keeping first {object_info.initial_trim} days")
            initial_trim_args = numpy.argwhere(time_float < time_float[0] + object_info.initial_trim).flatten()
            time_float = time_float[initial_trim_args]
            flux_float = flux_float[initial_trim_args]
            flux_err_float = flux_err_float[initial_trim_args]
        lc = lightkurve.LightCurve(time=time_float, flux=flux_float, flux_err=flux_err_float)
        lc_df = pandas.DataFrame(columns=['time', 'flux', 'flux_err', 'sector'])
        time_float = numpy.array(time_float)
        flux_float = numpy.array(flux_float)
        flux_err_float = numpy.array(flux_err_float)
        lc_df['time'] = time_float
        lc_df['flux'] = flux_float
        lc_df['flux_err'] = flux_err_float
        if lc_build.sectors_to_start_end_times is not None and len(lc_build.sectors_to_start_end_times.keys()) > 0:
            for sector, author_cadence_data in lc_build.sectors_to_start_end_times.items():
                for author, cadence_data in author_cadence_data.items():
                    for cadence, sectors_to_start_end_times in cadence_data.items():
                        lc_df.loc[(lc_df['time'] >= sectors_to_start_end_times[0]) &
                                  (lc_df['time'] <= sectors_to_start_end_times[1]), 'sector'] = sector
        lc_df.to_csv(object_dir + "lc.csv", index=False)
        lc = lc.remove_outliers(sigma_lower=float('inf') if object_info.lower_outliers_sigma is None else object_info.lower_outliers_sigma,
                                sigma_upper=object_info.outliers_sigma)
        time_float = lc.time.value
        flux_float = lc.flux.value
        flux_err_float = lc.flux_err.value
        cadence_array = numpy.diff(time_float) * 24 * 60 * 60
        cadence_array = cadence_array[~numpy.isnan(cadence_array)]
        cadence_array = cadence_array[cadence_array > 0]
        lc_build.cadence = int(numpy.round(numpy.nanmedian(cadence_array)))
        lc = self.__clean_initial_flux(object_info, time_float, flux_float, flux_err_float, star_info,
                                       lc_build.cadence, object_dir)
        clean_time = lc.time.value
        flatten_flux = lc.flux.value
        clean_flux_err = lc.flux_err.value
        periodogram = self.__plot_periodogram(lc, 0.05, 15, 10, sherlock_id,
                                              object_dir + "/periodicity/Periodogram_Initial_" + str(sherlock_id) + ".png")
        self.__plot_autocorrelation(lc, lc_build.cadence, sherlock_id,
                                    object_dir + "/periodicity/Autocorrelation_Initial_" + str(sherlock_id) + ".png")
        original_time = clean_time
        original_flux = flatten_flux
        if object_info.auto_detrend_period is not None:
            lc_build.detrend_period = object_info.auto_detrend_period
        elif object_info.auto_detrend_enabled:
            lc_build.detrend_period = self.__calculate_max_significant_period(lc, periodogram)
        if object_info.initial_mask is not None:
            logging.info('================================================')
            logging.info('INITIAL MASKING')
            logging.info('================================================')
            initial_mask = object_info.initial_mask
            logging.info('** Applying ordered masks to the lightcurve **')
            for mask_range in initial_mask:
                logging.info('* Initial mask since day %.2f to day %.2f. *', mask_range[0], mask_range[1])
                mask = ((clean_time < mask_range[0] if not math.isnan(mask_range[1]) else False) |
                        (clean_time > mask_range[1] if not math.isnan(mask_range[1]) else False))
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
        if object_info.reduce_simple_oscillations:
            logging.info('================================================')
            logging.info('STELLAR OSCILLATIONS REDUCTION')
            logging.info('================================================')
            flatten_flux = self.__reduce_simple_oscillations(object_dir, object_info.mission_id(), clean_time,
                                                             flatten_flux, star_info,
                                                             object_info.oscillation_snr_threshold,
                                                             object_info.oscillation_amplitude_threshold,
                                                             object_info.oscillation_ws_scale,
                                                             object_info.oscillation_min_period,
                                                             object_info.oscillation_max_period,
                                                             cpus=cpus, search_engine=object_info.search_engine)
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
        lc = lightkurve.LightCurve(time=clean_time, flux=flatten_flux, flux_err=clean_flux_err)
        lc_build.lc = lc
        self.__plot_periodogram(lc, 0.05, 15, 10, sherlock_id,
                                object_dir + "/periodicity/Periodogram_Final_" + str(sherlock_id) + ".png")
        self.__plot_autocorrelation(lc, lc_build.cadence, sherlock_id,
                                    object_dir + "/periodicity/Autocorrelation_Final_" + str(sherlock_id) + ".png")
        self.__plot_flux_diff(sherlock_id, object_dir, original_time, original_flux, lc_build.lc.time.value, lc_build.lc.flux.value)
        return lc_build

    def __plot_flux_diff(self, object_id, object_dir, original_time, original_flux, time, flux):
        # Plot tokenized curve detrends
        dir = object_dir + '/flux_diff/'
        dif = original_time[1:] - original_time[:-1]
        jumps = numpy.where(dif > 3)[0]
        jumps = numpy.append(jumps, len(original_time))
        previous_jump_index = 0
        for jumpIndex in jumps:
            original_time_partial = original_time[previous_jump_index:jumpIndex]
            original_flux_partial = original_flux[previous_jump_index:jumpIndex]
            time_partial = time[(time > original_time_partial[0]) & (time <= original_time_partial[-1])]
            flux_partial = flux[(time > original_time_partial[0]) & (time <= original_time_partial[-1])]
            if len(flux_partial) != len(original_flux_partial) or numpy.any(flux_partial[1:-1] != original_flux_partial[1:-1]):
                if not os.path.exists(dir):
                    os.mkdir(dir)
                fig, axs = plt.subplots(1, 1, figsize=(16, 6), constrained_layout=True)
                axs.scatter(original_time_partial[1:-1], original_flux_partial[1:-1], color="firebrick")
                axs.scatter(time_partial[1:-1], flux_partial[1:-1], color="orange", alpha=0.5)
                axs.set_xlabel("Time (TBJD)", fontsize=35)
                axs.set_ylabel("Flux norm.", fontsize=35)
                axs.tick_params(axis='both', which='major', labelsize=35)
                axs.tick_params(axis='both', which='minor', labelsize=35)
                plt.savefig(dir + str(object_id) + '_time_' + str(time_partial[1]) + '_' + str(time_partial[-1]) + '.png', bbox_inches='tight')
                plt.clf()
            previous_jump_index = jumpIndex

    def __plot_autocorrelation(self, lc, cadence_s, object_id, filename, max_days=5):
        if not os.path.exists(os.path.dirname(filename)):
           os.mkdir(os.path.dirname(filename))
        nlags = int(max_days / (cadence_s / 60 / 60 / 24))
        x = numpy.asarray(lc.flux.value)
        x = x - numpy.mean(x)
        result = numpy.correlate(x, x, mode='full')
        mid = len(result) // 2
        result = result[mid:mid + nlags + 1]
        correlation = result / result[0]
        lags = numpy.linspace(0, max_days, len(correlation))
        fig, axs = plt.subplots(1, 1, figsize=(8, 4), constrained_layout=True)
        axs.set_ylabel("Autocorrelation")
        axs.set_xlabel("Time lags (d)")
        axs.plot(lags, correlation, color="blue")
        plt.savefig(filename, bbox_inches='tight')
        plt.clf()
        plt.close()

    def __plot_periodogram(self, lc, min_period, max_period, oversample, object_id, filename):
        if not os.path.exists(os.path.dirname(filename)):
           os.mkdir(os.path.dirname(filename))
        periodogram = lc.to_periodogram(minimum_period=min_period, maximum_period=max_period,
                                        oversample_factor=oversample)
        # power_norm = self.running_median(periodogram.power.value, 20)
        periodogram.plot(view='period', scale='log')
        plt.savefig(filename, bbox_inches='tight')
        plt.clf()
        plt.close()
        # power_mod = periodogram.power.value - power_norm
        # power_mod = power_mod / np.mean(power_mod)
        # periodogram.power = power_mod * u.d / u.d
        # periodogram.plot(view='period', scale='log')
        # plt.title(str(sherlock_id) + " Lightcurve normalized periodogram")
        # plt.savefig(object_dir + "PeriodogramNorm_" + str(sherlock_id) + ".png", bbox_inches='tight')
        # plt.clf()
        return periodogram

    def smooth(self, flux, sg_window_len=11, convolve_window_len=7, window='blackman'):
        # if convolve_window_len < 3 or sg_window_len < 3:
        #     return flux
        # if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        #     raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
        # s = numpy.r_[numpy.ones(convolve_window_len // 2), flux, numpy.ones(convolve_window_len // 2)]
        # # print(len(s))
        # if window == 'flat':  # moving average
        #     w = numpy.ones(convolve_window_len, 'd')
        # else:
        #     w = eval('numpy .' + window + '(window_len)')
        # # TODO probably problematic with big gaps in the middle of the curve
        # clean_flux = numpy.convolve(w / w.sum(), s, mode='valid')
        if sg_window_len % 2 == 0:
            sg_window_len = sg_window_len + 1
        clean_flux = savgol_filter(flux, sg_window_len, 3)
        return clean_flux

    def __calculate_max_significant_period(self, lc, periodogram):
        # max_accepted_period = (lc.time[len(lc.time) - 1] - lc.time[0]) / 4
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

    def __reduce_simple_oscillations(self, object_dir, object_id, time, flux, star_info, snr_threshold=4,
                                     amplitude_threshold=0.1, window_size_scale=60, oscillation_min_period=0.002,
                                     oscillation_max_period=0.2, cpus=multiprocessing.cpu_count() - 1,
                                     search_engine='cpu'):
        no_transits_time, no_transits_flux = self.__reduce_visible_transits(time, flux, star_info, cpus, search_engine)
        snr = 10
        number = 0
        pulsations_df = pandas.DataFrame(columns=['period_s', 'frequency_microHz', 'amplitude', 'phase', 'snr',
                                                  'number'])
        lc = lightkurve.LightCurve(time=no_transits_time, flux=no_transits_flux)
        periodogram = lc.to_periodogram(minimum_period=oscillation_min_period, maximum_period=oscillation_max_period,
                                        oversample_factor=1)
        remove_signal = snr > snr_threshold
        sa_dir = object_dir + "sa/"
        while remove_signal:
            window_size = 1 / (window_size_scale * 10e-6) / 3600 / 24
            max_power_index = numpy.nanargmax(periodogram.power.value)
            period = periodogram.period[max_power_index].value
            frequency = 1 / period
            omega = frequency * 2. * numpy.pi
            indexes_around = numpy.argwhere((periodogram.period.value > period - window_size / 2) &
                                            (periodogram.period.value < period + window_size / 2))
            median_power_around = numpy.nanmedian(periodogram.power[indexes_around].value)
            snr = periodogram.power[max_power_index].value / median_power_around
            remove_signal = snr > snr_threshold
            if remove_signal:
                if not os.path.exists(sa_dir):
                    os.mkdir(sa_dir)
                guess_amp = numpy.std(no_transits_flux) * 2. ** 0.5
                guess = numpy.array([guess_amp, 0.])

                def sinfunc(t, a, p):
                    return a * numpy.sin(omega * t + p) + 1

                popt, pcov = scipy.optimize.curve_fit(sinfunc, no_transits_time, no_transits_flux, p0=guess)
                perr = numpy.sqrt(numpy.diag(pcov))
                A_err, p_err = perr
                A, p = popt
                fitfunc = lambda t: A * numpy.sin(omega * t + p) + 1
                fit_flux = fitfunc(time)
                fit_no_transit_flux = fitfunc(no_transits_time)
                flux_corr = flux - fit_flux + 1
                A = numpy.sqrt(A ** 2)
                remove_signal = self.__is_simple_oscillation_good_enough(snr, snr_threshold, A, A_err,
                                                                         p, p_err, numpy.std(flux),
                                                                         numpy.std(flux_corr), amplitude_threshold)
                if remove_signal:
                    logging.info(
                        "Reducing pulsation with period %sd, flux amplitude of %s, phase at %s and snr %s",
                        period, A, p, snr)
                    pulsations_df = pandas.concat([pulsations_df,
                                                      pandas.DataFrame([{'period_s': period * 24 * 3600,
                                                                         'frequency_microHz': frequency / 24 / 3600 * 1000000,
                                                                         'amplitude': A, 'phase': p,
                                                                         'snr': snr, 'number': number}])],
                                                     ignore_index=True)
                    self.__plot_pulsation_fit(sa_dir, object_id, time, flux, fit_flux, period, number)
                    flux = flux_corr
                    no_transits_flux = no_transits_flux - fit_no_transit_flux + 1
                    lc = lightkurve.LightCurve(time=no_transits_time, flux=no_transits_flux)
                    periodogram = lc.to_periodogram(minimum_period=oscillation_min_period, maximum_period=2,
                                                    oversample_factor=1)
                    self.__plot_pulsation_periodogram(sa_dir, object_id, periodogram, period, number)
                    number = number + 1
        if len(pulsations_df) > 0:
            pulsations_df = pulsations_df.sort_values(['number'], ascending=[True])
            pulsations_df.to_csv(sa_dir + "signals.csv", index=False)
        return flux

    def __reduce_visible_transits(self, time, flux, star_info, cpus, search_engine):
        min_sde = 13
        sde = min_sde + 1
        logging.info("Searching for obvious transits to mask them before reducing the stellar pulsations")
        while sde > min_sde:
            model = tls.transitleastsquares(time, flux)
            transit_period_min = 0.3
            transit_period_max = 2
            period_grid, oversampling = LcbuilderHelper.calculate_period_grid(time, transit_period_min,
                                                                              transit_period_max, 1, star_info, 2)
            power_args = {"period_min": 0.3,
                          "period_max": 2, "n_transits_min": 2, "show_progress_bar": False,
                          "duration_grid_step": 1.15,
                          "use_threads": cpus, "oversampling_factor": oversampling,
                          "period_grid": period_grid,
                          'use_gpu': search_engine == 'gpu' or search_engine == 'gpu_approximate',
                          'gpu_approximate': search_engine == 'gpu_approximate'}
            if star_info.ld_coefficients is not None:
                power_args["u"] = star_info.ld_coefficients
            power_args["R_star"] = star_info.radius
            power_args["R_star_min"] = star_info.radius_min
            power_args["R_star_max"] = star_info.radius_max
            power_args["M_star"] = star_info.mass
            power_args["M_star_min"] = star_info.mass_min
            power_args["M_star_max"] = star_info.mass_max
            results = model.power(**power_args)
            sde = results.SDE
            if sde > min_sde:
                logging.info("Masking transit at period %.2f d, T0 %.2f and duration %.2f m.", results.period,
                             results.T0,
                             results.duration * 60 * 24)
                in_transit = tls.transit_mask(time, results.period,
                                              results.duration if results.duration > 0.01 else 0.01, results.T0)
                time = time[~in_transit]
                flux = flux[~in_transit]
        return time, flux

    def __is_simple_oscillation_good_enough(self, snr, snr_threshold, A, A_err, p, p_err, flux_std, flux_corr_std,
                                            amplitude_threshold):
        return snr > snr_threshold and (A_err == numpy.inf or (A_err / A < 0.2)) and \
               (p_err == numpy.inf or (p_err < 0.2)) and A / flux_std > amplitude_threshold and \
               flux_corr_std < flux_std

    def __plot_pulsation_fit(self, sa_dir, object_id, time, flux, fit_flux, period, number):
        folded_time = tls.core.fold(time, period, time[0])
        fig, axs = plt.subplots(1, 1, figsize=(8, 4), constrained_layout=True)
        axs.set_ylabel("Flux norm.")
        axs.set_xlabel("Time (d)")
        title = "stellar activity P=" + str(float(f'{period:.2g}')) + "d"
        if object_id is not None:
            title = f'{object_id} {title}'
        axs.set_xlabel("Phase", fontsize=25)
        axs.set_ylabel("Flux norm.", fontsize=25)
        axs.tick_params(axis='both', which='major', labelsize=25)
        axs.tick_params(axis='both', which='minor', labelsize=25)
        axs.scatter(folded_time, flux, 2, color="blue", alpha=0.3)
        axs.scatter(folded_time, fit_flux, 2, color="orange", alpha=1)
        signal_dir = sa_dir + "/" + str(number)
        if not os.path.exists(signal_dir):
            os.mkdir(signal_dir)
        plt.savefig(signal_dir + "/folded_curve.png", bbox_inches='tight')
        plt.clf()
        plt.close()

    def __plot_pulsation_periodogram(self, sa_dir, object_id, periodogram, period, number):
        signal_dir = sa_dir + "/" + str(number)
        if not os.path.exists(signal_dir):
            os.mkdir(signal_dir)
        periodogram.plot(view='period', scale='log')
        title = "Lightcurve periodogram without signal at P=" + str(round(period, 6)) + "d"
        if object_id is not None:
            title = f'{object_id} {title}'
        plt.title(title)
        plt.savefig(signal_dir + "/Periodogram.png", bbox_inches='tight')
        plt.clf()
        plt.close()

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
                result_star_info.mass_min = result_star_info.mass - (
                    0.5 if result_star_info.mass > 0.5 else result_star_info.mass / 2)
                logging.info("Star catalog doesn't provide M_low_err. Assuming M_low_err=%.3fMsun",
                             result_star_info.mass_min)
            if result_star_info.mass_max is None or numpy.isnan(result_star_info.mass_max):
                result_star_info.mass_max = result_star_info.mass + 0.5
                logging.info("Star catalog doesn't provide M_up_err. Assuming M_low_err=%.3fMsun",
                             result_star_info.mass_max)
            if result_star_info.radius_min is None or numpy.isnan(result_star_info.radius_min):
                result_star_info.radius_min = result_star_info.radius - (
                    0.5 if result_star_info.radius > 0.5 else result_star_info.radius / 2)
                logging.info("Star catalog doesn't provide R_low_err. Assuming R_low_err=%.3fRsun",
                             result_star_info.radius_min)
            if result_star_info.radius_max is None or numpy.isnan(result_star_info.radius_max):
                result_star_info.radius_max = result_star_info.radius + 0.5
                logging.info("Star catalog doesn't provide R_up_err. Assuming R_up_err=%.3fRsun",
                             result_star_info.radius_max)
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
        if result_star_info.teff:
            logging.info('teff = %.6f', result_star_info.teff)
        if result_star_info.lum:
            logging.info('lum = %.6f', result_star_info.lum)
        if result_star_info.logg:
            logging.info('logg = %.6f', result_star_info.logg)
        star_df = pandas.DataFrame(columns=['obj_id', 'ra', 'dec', 'R_star', 'R_star_lerr', 'R_star_uerr', 'M_star',
                                            'M_star_lerr', 'M_star_uerr', 'Teff_star', 'Teff_star_lerr',
                                            'Teff_star_uerr', 'ld_a', 'ld_b'])
        ld_a = result_star_info.ld_coefficients[0] if result_star_info.ld_coefficients is not None else None
        ld_b = result_star_info.ld_coefficients[1] if result_star_info.ld_coefficients is not None else None
        star_df = pandas.concat([star_df,
                       pandas.DataFrame([{'obj_id': object_id, 'ra': result_star_info.ra, 'dec': result_star_info.dec,
                         'R_star': result_star_info.radius, 'radius': result_star_info.radius,
                         'R_star_lerr': result_star_info.radius - result_star_info.radius_min,
                         'R_star_uerr': result_star_info.radius_max - result_star_info.radius,
                         'M_star': result_star_info.mass, 'mass': result_star_info.mass,
                         'M_star_lerr': result_star_info.mass - result_star_info.mass_min,
                         'M_star_uerr': result_star_info.mass_max - result_star_info.mass,
                         'Teff_star': result_star_info.teff, 'Teff': result_star_info.teff,
                         'Teff_star_lerr': 200, 'Teff_star_uerr': 200,
                         'logg': result_star_info.logg, 'logg_err': result_star_info.logg_err,
                         'ld_a': ld_a, 'ld_b': ld_b,
                         'feh': result_star_info.feh,
                         'feh_err': result_star_info.feh_err, 'v': result_star_info.v, 'v_err': result_star_info.v_err,
                         'j': result_star_info.j, 'j_err': result_star_info.j_err,
                         'k': result_star_info.k, 'k_err': result_star_info.k_err,
                         'h': result_star_info.h, 'h_err': result_star_info.h_err,
                         'kp': result_star_info.kp, 'lum': result_star_info.lum}])],
                      ignore_index=True)
        star_df.to_csv(object_dir + "params_star.csv", index=False)
        return result_star_info

    def __clean_initial_flux(self, object_info, time, flux, flux_err, star_info, cadence, object_dir):
        clean_time = time
        clean_flux = flux
        clean_flux_err = flux_err
        lc = lightkurve.LightCurve(time=clean_time, flux=clean_flux, flux_err=clean_flux_err)
        if (object_info.binning > 0 or object_info.prepare_algorithm or
                object_info.smooth_enabled or
                (object_info.high_rms_enabled and object_info.initial_mask is None) or object_info.truncate_border > 0):
            logging.info('================================================')
            logging.info('INITIAL FLUX CLEANING')
            logging.info('================================================')
        if object_info.truncate_border > 0:
            clean_time, clean_flux, clean_flux_err = \
                LcbuilderHelper.truncate_borders(clean_time, clean_flux, clean_flux_err, truncate_border=object_info.truncate_border)
            lc = lightkurve.LightCurve(time=clean_time, flux=clean_flux, flux_err=clean_flux_err)
            lc = lc.remove_nans()
        if object_info.binning > 0:
            bin_edges_for_time = numpy.arange(clean_time[0], clean_time[-1], object_info.binning)
            bin_means, bin_edges, binnumber = stats.binned_statistic(clean_time, clean_flux, statistic='mean',
                                                                     bins=bin_edges_for_time)
            bin_stds, _, _ = stats.binned_statistic(clean_time, clean_flux, statistic='std', bins=bin_edges_for_time)
            bin_width = (bin_edges[1] - bin_edges[0])
            bin_centers = bin_edges[1:] - bin_width / 2
            clean_time = bin_centers
            clean_flux = bin_means
            clean_flux_err = bin_stds
            lc = lightkurve.LightCurve(time=clean_time, flux=clean_flux, flux_err=clean_flux_err)
            lc = lc.remove_nans()
        if object_info.prepare_algorithm is not None:
            clean_time, clean_flux, clean_flux_err = object_info.prepare_algorithm.prepare(
                object_info, lc.time.value, lc.flux.value, lc.flux_err.value, star_info=star_info)
            lc = lightkurve.LightCurve(time=clean_time, flux=clean_flux, flux_err=clean_flux_err)
            lc = lc.remove_nans()
        clean_time = lc.time.value
        clean_flux = lc.flux.value
        clean_flux_err = lc.flux_err.value
        if object_info.high_rms_enabled and object_info.initial_mask is None:
            logging.info('Masking high RMS areas by a factor of %.2f with %.1f hours binning',
                         object_info.high_rms_threshold, object_info.high_rms_bin_hours)
            bins_per_day = 24 / object_info.high_rms_bin_hours
            dif = clean_time[1:] - clean_time[:-1]
            jumps = numpy.where(dif > 3)[0]
            jumps = numpy.append(jumps, len(clean_time))
            previous_jump_index = 0
            rms_mask_values = [object_info.high_rms_threshold, 1.75, 1.5, 1.25]
            entire_high_rms_mask = []
            entire_bin_centers = []
            entire_bin_stds = []
            entire_rms_threshold_array = []
            for rms_mask_value in rms_mask_values:
                entire_high_rms_mask = entire_high_rms_mask + [numpy.array([], dtype=bool)]
                entire_bin_centers = entire_bin_centers + [numpy.array([])]
                entire_bin_stds = entire_bin_stds + [numpy.array([])]
                entire_rms_threshold_array = entire_rms_threshold_array + [numpy.array([])]
            for jumpIndex in jumps:
                time_partial = clean_time[previous_jump_index:jumpIndex]
                flux_partial = clean_flux[previous_jump_index:jumpIndex]
                high_std_masks = []
                bin_centers_array = []
                bin_stds_array = []
                rms_threshold_array_array = []
                for rms_mask_value_index, rms_mask_value in enumerate(rms_mask_values):
                    high_std_mask, bin_centers, bin_stds, rms_threshold_array = \
                        self.__compute_rms_mask(time_partial, flux_partial, bins_per_day, rms_mask_value)
                    high_std_masks = high_std_masks + [high_std_mask]
                    bin_centers_array = bin_centers_array + [bin_centers]
                    bin_stds_array = bin_stds_array + [bin_stds]
                    rms_threshold_array_array = rms_threshold_array_array + [rms_threshold_array]
                    entire_high_rms_mask[rms_mask_value_index] = numpy.append(entire_high_rms_mask[rms_mask_value_index], high_std_mask)
                    entire_bin_centers[rms_mask_value_index] = numpy.append(entire_bin_centers[rms_mask_value_index], bin_centers)
                    entire_bin_stds[rms_mask_value_index] = numpy.append(entire_bin_stds[rms_mask_value_index], bin_stds)
                    entire_rms_threshold_array[rms_mask_value_index] = numpy.append(entire_rms_threshold_array[rms_mask_value_index], rms_threshold_array)
                    previous_jump_index = jumpIndex
                self.__plot_rms_mask(star_info.object_id, object_info.high_rms_bin_hours, rms_mask_values,
                                     bin_centers_array, bin_stds_array, rms_threshold_array_array, high_std_masks,
                                     time_partial, flux_partial,
                                     'High_RMS_Mask_' + str(star_info.object_id) + '_time_' +
                                     str(time_partial[1]) + '_' + str(time_partial[-1]), object_dir)
            self.__plot_rms_mask(star_info.object_id, object_info.high_rms_bin_hours, rms_mask_values,
                                 entire_bin_centers, entire_bin_stds, entire_rms_threshold_array,
                                 entire_high_rms_mask, clean_time, clean_flux,
                                 'High_RMS_Mask_' + str(star_info.object_id), object_dir)
            clean_time = clean_time[~entire_high_rms_mask[0]]
            clean_flux = clean_flux[~entire_high_rms_mask[0]]
            clean_flux_err = clean_flux_err[~entire_high_rms_mask[0]]
            lc = lightkurve.LightCurve(time=clean_time, flux=clean_flux, flux_err=clean_flux_err)
            lc = lc.remove_nans()
        smooth_cadences = 20 * 60 // cadence
        if smooth_cadences > 3 and object_info.smooth_enabled:
            # logging.info('Applying Smooth phase (savgol + weighted average)')
            logging.info('Applying Smooth phase (savgol)')
            # Using SavGol filter with 20 minutes window
            clean_flux = self.smooth(lc.flux.value, sg_window_len=smooth_cadences)
            # TODO to use convolve we need to remove the borders effect
            # clean_flux = np.convolve(clean_flux, [0.025, 0.05, 0.1, 0.155, 0.34, 0.155, 0.1, 0.05, 0.025], "same")
            # clean_flux = np.convolve(clean_flux, [0.025, 0.05, 0.1, 0.155, 0.34, 0.155, 0.1, 0.05, 0.025], "same")
            # clean_flux[0:5] = 1
            # clean_flux[len(clean_flux) - 6: len(clean_flux) - 1] = 1
            # clean_flux = uniform_filter1d(clean_flux, 11)
            # clean_flux = self.flatten_bw(self.FlattenInput(clean_time, clean_flux, 0.02))[0]
            lc = lightkurve.LightCurve(time=clean_time, flux=clean_flux, flux_err=clean_flux_err)
            lc = lc.remove_nans()
        return lc

    def __compute_rms_mask(self, time, flux, bins_per_day, rms_threshold):
        bins = (time[len(time) - 1] - time[1]) * bins_per_day
        bin_stds, bin_edges, binnumber = stats.binned_statistic(time[1:], flux[1:], statistic='std', bins=bins)
        stds_median = numpy.nanmedian(bin_stds[bin_stds > 0])
        stds_median_array = numpy.full(len(bin_stds), stds_median)
        rms_threshold_array = stds_median_array * rms_threshold
        too_high_bin_stds_indexes = numpy.argwhere(bin_stds > rms_threshold_array)
        high_std_mask = numpy.array([bin_id - 1 in too_high_bin_stds_indexes for bin_id in binnumber])
        high_std_mask = numpy.append(high_std_mask[0], high_std_mask)
        bin_width = (bin_edges[1] - bin_edges[0])
        bin_centers = bin_edges[1:] - bin_width / 2
        return high_std_mask, bin_centers, bin_stds, rms_threshold_array

    def __plot_rms_mask(self, object_id, rms_bin_hours, rms_values, bin_centers, bin_stds, rms_threshold_array,
                        rms_mask, time, flux, filename, object_dir):
        rms_values_len = len(rms_values)
        plot_dir = object_dir + "/rms_mask/"
        colors = ['red', 'orange', 'pink', 'purple']
        if not os.path.exists(plot_dir):
            os.mkdir(plot_dir)
        fig, axs = plt.subplots(1 + rms_values_len, 1, figsize=(12, 4 + 4 * rms_values_len), constrained_layout=True,
                                gridspec_kw={'height_ratios': [1.5, 1, 1, 1, 1]})
        for index, rms_value in enumerate(rms_values):
            axs[0].plot(bin_centers[index], rms_threshold_array[index], color=colors[index], rasterized=True,
                        label=f'{rms_values[index]} * RMS Threshold')
            axs[index + 1].set_ylabel('Relative flux', fontsize=18)
            if index == len(rms_values) - 1:
                axs[index + 1].set_xlabel('Time (d)', fontsize=18)
            axs[index + 1].scatter(time[1:], flux[1:], color='gray', alpha=0.5, rasterized=True, label="Flux norm.")
            axs[index + 1].scatter(time[rms_mask[index]], flux[rms_mask[index]], linewidth=1, color=colors[index],
                                   alpha=1.0, label=f"Masked Flux > {rms_values[index]} * RMS")
            axs[index + 1].legend(loc="upper right", fontsize=16)
            axs[index + 1].tick_params(axis='both', which='major', labelsize=18)
        fig.suptitle(object_id + " High RMS Mask", fontsize=24)
        axs[0].set_ylabel('Flux RMS', fontsize=18)
        axs[0].plot(bin_centers[0], bin_stds[0], color='black', alpha=0.75, rasterized=True, label=str(rms_bin_hours) + "h binned RMS")
        axs[0].legend(loc="upper right", fontsize=16)
        axs[0].tick_params(axis='both', which='major', labelsize=18)
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
                          auto_detrend_period=None, prepare_algorithm=None, reduce_simple_oscillations=False,
                          oscillation_snr_threshold=4, oscillation_amplitude_threshold=0.1, oscillation_ws_scale=60,
                          oscillation_min_period=0.002, oscillation_max_period=0.2, binning=1, truncate_border=0,
                          lower_outliers_sigma: float = None, initial_trim: float = None,
                          initial_trim_sectors: int = None, search_engine='cpu'):
        mission, mission_prefix, id = MissionDataPreparer.parse_object_id(target_name)
        coords = None if mission is not None else self.parse_coords(target_name)
        cadence = cadence if cadence is not None else self.DEFAULT_CADENCES_FOR_MISSION[mission]
        if (mission is not None or coords is not None) and file is None:
            return MissionObjectInfo(sectors, target_name,
                                     coords[0] if coords is not None else None,
                                     coords[1] if coords is not None else None, author, cadence, initial_mask,
                                     initial_transit_mask,
                                     star_info, aperture, outliers_sigma, high_rms_enabled,
                                     high_rms_threshold, high_rms_bin_hours, smooth_enabled, auto_detrend_enabled,
                                     auto_detrend_method, auto_detrend_ratio, auto_detrend_period, prepare_algorithm,
                                     reduce_simple_oscillations, oscillation_snr_threshold,
                                     oscillation_amplitude_threshold, oscillation_ws_scale, oscillation_min_period,
                                     oscillation_max_period, binning, eleanor_corr_flux, truncate_border,
                                     lower_outliers_sigma=lower_outliers_sigma, initial_trim=initial_trim,
                                     initial_trim_sectors=initial_trim_sectors, search_engine=search_engine)
        elif mission is not None and file is not None:
            return MissionInputObjectInfo(target_name, file, initial_mask, initial_transit_mask,
                                          star_info, outliers_sigma, high_rms_enabled, high_rms_threshold,
                                          high_rms_bin_hours, smooth_enabled, auto_detrend_enabled, auto_detrend_method,
                                          auto_detrend_ratio, auto_detrend_period, prepare_algorithm,
                                          reduce_simple_oscillations, oscillation_snr_threshold,
                                          oscillation_amplitude_threshold, oscillation_ws_scale,
                                          oscillation_min_period, oscillation_max_period, binning, truncate_border,
                                          lower_outliers_sigma=lower_outliers_sigma, initial_trim=initial_trim,
                                          search_engine=search_engine)
        elif mission is None and file is not None:
            return InputObjectInfo(file, initial_mask, initial_transit_mask, star_info,
                                   outliers_sigma, high_rms_enabled, high_rms_threshold, high_rms_bin_hours,
                                   smooth_enabled, auto_detrend_enabled, auto_detrend_method, auto_detrend_ratio,
                                   auto_detrend_period, prepare_algorithm,
                                   reduce_simple_oscillations, oscillation_snr_threshold,
                                   oscillation_amplitude_threshold, oscillation_ws_scale, oscillation_min_period,
                                   oscillation_max_period, binning, truncate_border,
                                   lower_outliers_sigma=lower_outliers_sigma, initial_trim=initial_trim,
                                   search_engine=search_engine)
        else:
            raise ValueError(
                "Invalid target definition with target_name={}, mission={}, id={}, coords={}, sectors={}, file={}, "
                "cadence={}".format(target_name, mission, id, coords, sectors, file, cadence))

    def parse_object_info(self, target: str):
        return MissionDataPreparer.parse_object_id(target)

    def get_default_author(self, target, cadence):
        mission, mission_prefix, id = MissionDataPreparer.parse_object_id(target)
        if mission == constants.MISSION_KEPLER:
            author = constants.KEPLER_AUTHOR
        elif mission == constants.MISSION_K2:
            author = constants.K2_AUTHOR
        elif mission == constants.MISSION_TESS and ((cadence == 'short' or cadence == 'fast') or
                                                    (not isinstance(cadence, (str)) and cadence < 600)):
            author = constants.SPOC_AUTHOR
        elif mission == constants.MISSION_TESS and (cadence == 'long' or cadence >= 600):
            author = constants.TESS_SPOC_AUTHOR
        return author

    def parse_coords(self, target: str):
        coords_parsed = re.search(self.COORDS_REGEX, target)
        coords = [coords_parsed.group(1), coords_parsed.group(2)] if coords_parsed is not None else None
        return coords

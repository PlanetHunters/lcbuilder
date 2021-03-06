from lcbuilder.objectinfo.ObjectInfo import ObjectInfo


class MissionInputObjectInfo(ObjectInfo):
    """
    Implementation of ObjectInfo to be used to characterize objects which are to be loaded from a csv file and knowing
    the source mission id of the input light curve.
    """
    def __init__(self, mission_id, input_file, initial_mask=None, initial_transit_mask=None,
                 star_info=None, outliers_sigma=3,
                 high_rms_enabled=True, high_rms_threshold=2.5,
                 high_rms_bin_hours=4, smooth_enabled=False,
                 auto_detrend_enabled=False, auto_detrend_method="cosine", auto_detrend_ratio=0.25,
                 auto_detrend_period=None, prepare_algorithm=None
                 ):
        """
        @param mission_id: the mission identifier. TIC ##### for TESS, KIC ##### for Kepler and EPIC ##### for K2.
        @param input_file: the file to be used for loading the light curve
        @param initial_mask: an array of time ranges provided to mask them into the initial object light curve.
        from the initial light curve before processing.
        @param star_info: input star information
        @param outliers_sigma: sigma used to cut upper outliers.
        @param high_rms_enabled: whether RMS masking is enabled
        @param high_rms_threshold: RMS masking threshold
        @param high_rms_bin_hours: RMS masking binning
        @param smooth_enabled: whether short-window smooth is enabled
        @param auto_detrend_enabled: whether automatic high-amplitude periodicities detrending is enabled
        @param auto_detrend_method: biweight or cosine
        @param auto_detrend_ratio: the ratio to be used as window size in relationship to the strongest found period
        @param auto_detrend_period: the fixed detrend period (disables auto_detrend)
        @param prepare_algorithm: custom curve preparation logic
        """
        super().__init__(initial_mask, initial_transit_mask, star_info, None,
                         outliers_sigma, high_rms_enabled, high_rms_threshold, high_rms_bin_hours, smooth_enabled,
                         auto_detrend_enabled, auto_detrend_method, auto_detrend_ratio, auto_detrend_period,
                         prepare_algorithm)
        self.id = mission_id
        self.input_file = input_file

    def sherlock_id(self):
        return self.id.replace(" ", "") + "_INP"

    def mission_id(self):
        return self.id

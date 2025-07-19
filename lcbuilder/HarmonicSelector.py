import numpy as np


class HarmonicSelector:
    @staticmethod
    def is_harmonic(a_t0, b_t0, a_period, b_period, t0_tolerance=0.075, period_tolerance=0.025):
        multiplicity = HarmonicSelector.multiple_of(a_period, b_period, period_tolerance)
        return multiplicity != 0 and HarmonicSelector.matches_t0(a_t0, b_t0, a_period, multiplicity, t0_tolerance)

    @staticmethod
    def multiple_of(a, b, tolerance=0.05):
        a = float(a)
        b = float(b)
        if a <= 0 or b <= 0:
            return 0
        mod_ab = a % b
        mod_ba = b % a
        is_a_multiple_of_b = a >= b and a < b * 3 + tolerance * 3 and (
            (mod_ab < 1 and abs(mod_ab % 1) <= tolerance) or ((b - mod_ab) < 1 and abs((b - mod_ab) % 1) <= tolerance))
        if is_a_multiple_of_b:
            return round(a / b)
        is_b_multiple_of_a = b >= a and a > b / 3 - tolerance / 3 and (
            (mod_ba < 1 and abs(mod_ba % 1) <= tolerance) or ((a - mod_ba) < 1 and abs((a - mod_ba) % 1) <= tolerance))
        if is_b_multiple_of_a:
            return - round(b / a)
        return 0

    @staticmethod
    def matches_t0(a_t0, b_t0, a_period, multiplicity, tolerance=0.02):
        if multiplicity == 1:
            return abs(b_t0 - a_t0) < tolerance
        elif multiplicity < 0:
            allowed_t0s_centers = np.linspace(a_t0 - a_period * (-multiplicity), a_t0 + a_period * (-multiplicity), -2 * multiplicity + 1)
        elif multiplicity > 0:
            allowed_t0s_centers = np.linspace(a_t0 - a_period, a_t0 + a_period, 2 * multiplicity + 1)
        else:
            return False
        matching_t0s = [abs(b_t0 - allowed_t0_center) <= tolerance for allowed_t0_center in allowed_t0s_centers]
        return True in matching_t0s
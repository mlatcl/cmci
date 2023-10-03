import numpy as np
from scipy.interpolate import splrep, BSpline
from scipy.signal import find_peaks

import logging

logging.basicConfig(level=logging.ERROR)

class CallFinder:
    def __init__(self):
        self.band_to_consider = (1.5e3, 7.2e3)

    def threshold_spectrum(self, S, f, smoothing=1100, # more smoothing = less peaks
                           freq_to_ignore=5):
        spec_with_peaks = np.zeros_like(S)
        for i in range(len(S.T)):
            smooth_spec = BSpline(*splrep(f, S[:, i], s=smoothing))(f)
            peaks, _ = find_peaks(smooth_spec)
            spec_with_peaks[peaks, i] = 1.0
        spec_with_peaks[:freq_to_ignore, :] = 0

        # set bands to zero
        band_min = np.argmin(abs(f - self.band_to_consider[0]))
        band_max = np.argmin(abs(f - self.band_to_consider[1]))

        spec_with_peaks[:band_min, :] = 0.0
        spec_with_peaks[band_max:, :] = 0.0
        return spec_with_peaks

    @staticmethod
    def _compute_one_feature(S, condition):
        S = S.copy()
        S[~condition] = 0.0
        return S.sum(axis=0).reshape(-1, 1)

    def compute_features(self, S, f):
        computed_features = [self._compute_one_feature(S,c[0](f)) for c in self.conditions]
        return np.concatenate(computed_features, axis=1)

    @staticmethod
    def _validate_starts_and_ends(starts, ends):
        if (len(starts) != len(ends)) or (ends < starts).any():
            raise ValueError("starts and ends are inconsistent")

    @staticmethod
    def get_starts_and_ends(labels):
        labels = np.diff(labels, prepend=0)

        starts = np.where(labels == 1)[0]
        ends = np.where(labels == -1)[0]

        if len(starts) != len(ends):
            if labels[np.where(labels != 0)[0][0]] == 1:
                # call identified at the end and doesn't finish
                ends = np.hstack([ends, len(labels) - 1])
            else:
                # call ends in the beginning and the start is unseen
                starts = np.hstack([0, starts])
        CallFinder._validate_starts_and_ends(starts, ends)
        return np.vstack((starts, ends)).T

    @staticmethod
    def clean_labels(t, start_end_indices):
        if len(start_end_indices) > 0:
            extended_sei = np.array([list((max(s-2, 0), e)) for (s, e) in start_end_indices])
            range_timepoints = t[extended_sei]
            range_timepoints[np.diff(range_timepoints, axis=1)[:, 0] > 0.2/60, :] # small windows are ignored
            return range_timepoints
        else:
            return np.zeros((0,2))

    def find_calls(self, S, f, t, threshold=1.0, mininum_call_duration=0.05, smoothing=1100):
        thresholded_spectrum = self.threshold_spectrum(S, f, smoothing=smoothing)
        feature = thresholded_spectrum.sum(axis=0)
        final_feature = (feature >= threshold).astype(float)

        start_end_indices = self.get_starts_and_ends(final_feature)
        segments = self.clean_labels(t, start_end_indices)
        
        segments = segments[np.diff(segments, axis=1)[:, 0] > mininum_call_duration, :] # filter out short duration calls
        return segments

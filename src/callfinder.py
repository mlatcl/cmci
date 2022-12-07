
import numpy as np
from hmmlearn.hmm import GaussianHMM

class CallFinder:
    NOT_CALL_STATE, CALL_STATE = 0, 1
    n_states = 2

    def __init__(self, conditions=None):
        # states are (no call, call)
        self.hmm = GaussianHMM(n_components=self.n_states, params="st", init_params="st")
        # self.conditions = [(condition, call_is_zeroes), ...]
        self.conditions = [
            (lambda x: ((x > 4.5e3) & (x < 6.5e3)), False, 1.),
            # (lambda x: ((x > 10e3) & (x < 12e3)), False),
            # (lambda x: ((x < 4e3)), True, 1.),
            (lambda x: ((x > 7e3) & (x < 10e3)), True, 1.),
            ] if conditions is None else conditions
        self.hmm.n_features = 1

        self.hmm.means_ = np.zeros((self.n_states, self.hmm.n_features)) + 20 # these don't change, the observation parameters.
        self.hmm.covars_ = np.zeros((self.n_states, self.hmm.n_features)) + 100 # these don't change, the observation parameters.

        self._init_hmm_params(0, call_is_zeroes=False)
        
    def _init_hmm_params(self, i, call_is_zeroes=False):
        """
        :param:
        If the feature *i* has non-zero numbers when it is indicating a call, set in_call to True.
        If the feature has zeroes when it is indicating a call, set in_call to False.
        """
        state = self.NOT_CALL_STATE if not call_is_zeroes else self.CALL_STATE
        self.hmm.means_[state, i] = 0.0

        covars_ = self.hmm.covars_.copy()
        covars_[state, i, i] = 9 # sqrt of this * 2 controls roughly how many theshold hits will trigger a "call" signal.
        self.hmm.covars_ = covars_[:, range(self.hmm.n_features), range(self.hmm.n_features)]

    @staticmethod
    def threshold_spectrum(S, threshold=0.85, quantile=None):
        freq_to_ignore = 40
        if quantile is not None:
            threshold = np.quantile(S, quantile)
        print("Quantile {} Threshold {}".format(quantile, threshold))
        S = S > threshold # maybe change this to quantile
        S[:freq_to_ignore, :] = 0
        return S

    @staticmethod
    def whale_threshold_spectrum(S, c1=3, c2=2.5):
        # from pdb import set_trace; set_trace();
        t1 = c1*np.tile(np.median(S, axis=1)[:, None], (1, S.shape[1]))
        t2 = c1*np.tile(np.median(S, axis=0)[None, :], (S.shape[0], 1))
        print("T1: {} T2: {}".format(t1.shape, t2.shape))
        threshold = np.maximum(t1, t2) / c2
        return S > threshold

    @staticmethod
    def normalize_spectrum(S):
        s_min, s_max = S.min(), S.max()
        S = (S - s_min) / (s_max - s_min)
        return S

    @staticmethod
    def _compute_one_feature(S, condition):
        S = S.copy()
        S[~condition] = 0.0
        return S.sum(axis=0).reshape(-1, 1)

    def compute_features(self, S, f):
        computed_features = [self._compute_one_feature(S,c[0](f)) for c in self.conditions]
        return np.concatenate(computed_features, axis=1)

    def fit_hmm(self, features):
        self.hmm.fit(features)

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
        print("SEI: {}".format(start_end_indices))
        extended_sei = np.array([list((max(s-2, 0), e)) for (s, e) in start_end_indices])
        range_timepoints = t[extended_sei]
        range_timepoints[np.diff(range_timepoints, axis=1)[:, 0] > 0.2/60, :] # small windows are ignored
        print(range_timepoints)
        return range_timepoints

    def find_calls(self, S, f, t):
        nS = self.normalize_spectrum(S)
        tnS = self.threshold_spectrum(nS, quantile=0.99)
        wnS = self.whale_threshold_spectrum(nS, c1=3, c2=2.4)
        thresholded_spectrum = tnS
        print("Whale theshold {}".format(wnS))
        print("Normie theshold {}".format(tnS))
        # threshold
        features = self.compute_features(thresholded_spectrum, f)
        final_feature = np.zeros((features.shape[0], 1))
        for i, (c, f) in enumerate(zip(self.conditions, list(features.T))):
            if c[1] is True: # column negative
                tf = f * -1 * c[2]
            else:
                tf = f * c[2]
            print("TF {}".format(tf))
            final_feature = final_feature + tf[:, None]
            
            print("Final Feature {}".format(final_feature))
            
            
        print("Features: {}".format(features))
        print("Final Feature: {}".format(final_feature))
        print("Final Feature diff: {}".format(final_feature-features[:,[0]]))

        self.fit_hmm(final_feature)
        labels = self.hmm.predict(final_feature)

        start_end_indices = self.get_starts_and_ends(labels)
        segments = self.clean_labels(t, start_end_indices)
        mininum_call_duration = 0.004
        segments = segments[np.diff(segments, axis=1)[:, 0] > mininum_call_duration, :] # filter out short duration calls
        return segments, thresholded_spectrum, features, final_feature

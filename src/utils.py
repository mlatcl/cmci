
import numpy as np
from hmmlearn.hmm import GaussianHMM


class CallFinder:
    not_call_state, call_state = 0, 1
    n_states = 2
    n_features = 1

    def __init__(self):
        # states are (no call, call)
        self.hmm = GaussianHMM(n_components=2, params="st", init_params="st")
        self.hmm.means_ = np.zeros((self.n_states, self.n_features)) + 20
        self.hmm.covars_ = np.zeros((self.n_states, self.n_features)) + 100
        self.hmm.n_features = self.n_features

    def set_hmm_params(self, i, in_call=True):
        state = self.not_call_state if in_call else self.call_state
        self.hmm.means_[state, i] = 0.0

        covars_ = self.hmm.covars_.copy()
        covars_[state, i, i] = 1e-10
        self.hmm.covars_ = covars_[:, range(self.n_features), range(self.n_features)]

    @staticmethod
    def normalize_spectrum(S):
        s_min, s_max = S.min(), S.max()
        S = (S - s_min) / (s_max - s_min)
        S = S > 0.85 # maybe change this to quantile
        return S

    @staticmethod
    def _one_feature(S, condition):
        S = S.copy()
        S[~condition] = 0.0
        return S.sum(axis=0).reshape(-1, 1)

    def features(self, S, f):
        features = [
            self._one_feature(S, (4e3 < f) <= 7e3),
        ]
        self.set_hmm_params(0, in_call=True)
        return np.concatenate(features, axis=1)

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
        range_timepoints = t[start_end_indices]
        range_timepoints[np.diff(range_timepoints, axis=1)[:, 0] > 0.2/60, :] # small windows are ignored
        return range_timepoints

    def find_calls(self, S, f, t):
        S = self.normalize_spectrum(S)
        features = self.features(S, f)

        self.fit_hmm(features)
        labels = self.hmm.predict(features)

        start_end_indices = self.get_starts_and_ends(labels)
        range_timepoints = self.clean_labels(t, start_end_indices)
        return range_timepoints

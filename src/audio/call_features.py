import scipy
import numpy as np

class CallFeatures:

    def __init__(self, S, f, t):
        self.S = S
        self.f = f
        self.t = t

    def maximal_power_frequencies(self, positions=None):
        """
        :positions: 0 - 1 based on the percentage of time into the spectrogram
        """
        if positions is None:
            positions = np.linspace(0, 1, 5)
        freqs = np.array([self._freq_at_time(p) for p in positions])
        return freqs

    def _freq_at_time(self, position):
        return self.f[self.S[:, int((self.S.shape[1]-1)*position)].argmax()]

    def min_freq(self):
        """
        # TODO check with Jen?
        """
        return self._freq_at_time(0.0)


    def max_freq(self):
        return self.f[np.unravel_index(self.S.argmax(), self.S.shape)[0]]

    def maximal_power(self):
        return self.S.max()

    def duration(self):
        return self.t.max() - self.t.min()


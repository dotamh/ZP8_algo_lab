"""
background.py — composite background noise model

Noise chain (applied in order to a clean signal frame):
    1. Speckle      : multiplicative Gamma noise (coherent illumination)
    2. Poisson      : shot noise from photon/electron statistics
    3. Readout      : additive Gaussian camera noise N(bias, rn²)

Each layer can be enabled/disabled independently.
All values in DN units; GAIN [e⁻/DN] converts DN → electrons for Poisson.
"""

import numpy as np
import matplotlib.pyplot as plt

from config import H, W, RN, BIAS, GAIN


class Background:
    """
    Composite noise background: speckle × Poisson + Gaussian readout.

    Parameters
    ----------
    H, W       : frame dimensions [pixels]
    rn         : readout noise sigma [DN]
    bias       : DC bias level [DN]
    gain       : ADC gain [e⁻ / DN]  — used for Poisson scaling
    n_looks    : speckle number-of-looks (L); larger L = less speckle.
                 None or 0 disables speckle.
    use_poisson: enable shot noise (default True)
    rng        : np.random.RandomState for reproducibility
    """

    def __init__(self, H=H, W=W, rn=RN, bias=BIAS, gain=GAIN,
                 n_looks=None, use_poisson=True, rng=None):
        self.H           = H
        self.W           = W
        self.rn          = rn
        self.bias        = bias
        self.gain        = gain
        self.n_looks     = n_looks      # None → no speckle
        self.use_poisson = use_poisson
        self.rng         = rng if rng is not None else np.random.RandomState()

    # ── noise layers ──────────────────────────────────────────────────────

    def apply_speckle(self, frame):
        """
        Multiplicative Gamma speckle: I_out = I_in * G,  G ~ Gamma(L, 1/L).
        Mean(G)=1, Var(G)=1/L.  Single-look (L=1) is exponential.
        """
        L = self.n_looks
        G = self.rng.gamma(shape=L, scale=1.0 / L, size=frame.shape)
        return frame * G

    def apply_poisson(self, frame):
        """
        Shot noise: convert DN→electrons, draw Poisson, convert back.
        Negative values are clamped to 0 before Poisson sampling.
        """
        frame_e = np.clip(frame, 0, None) * self.gain
        noisy_e = self.rng.poisson(frame_e).astype(float)
        return noisy_e / self.gain

    def readout_noise(self):
        """Gaussian readout noise ~ N(0, rn²), added to every pixel."""
        return self.rng.normal(0.0, self.rn, size=(self.H, self.W))

    # ── full sample ───────────────────────────────────────────────────────

    def sample(self, signal=None):
        """
        Generate one observed frame.

        Parameters
        ----------
        signal : ndarray (H, W) or None
            Clean signal in DN.  None → pure noise frame (signal = 0).

        Returns
        -------
        frame : ndarray (H, W)  [DN]
        """
        if signal is None:
            signal = np.zeros((self.H, self.W))

        frame = signal.copy().astype(float)

        if self.n_looks is not None and self.n_looks > 0:
            frame = self.apply_speckle(frame)

        if self.use_poisson:
            frame = self.apply_poisson(frame)

        frame += self.bias + self.readout_noise()
        return frame

    # ── diagnostics ───────────────────────────────────────────────────────

    def noise_std_theory(self, signal_level=0.0):
        """
        Theoretical total noise std at a given signal level [DN].
        Combines readout, shot noise (and speckle if enabled).
        """
        sigma2 = self.rn ** 2                                     # readout
        sigma2 += signal_level / self.gain                        # shot noise (DN²)
        if self.n_looks is not None and self.n_looks > 0:
            sigma2 += (signal_level ** 2) / self.n_looks          # speckle
        return np.sqrt(sigma2)

    # ── plotting ──────────────────────────────────────────────────────────

    def plot(self, frame, title='', defects=None, layout='horizontal', figsize=(8, 6)):
        if layout == 'horizontal':
            fig, axes = plt.subplots(1, 2, figsize=figsize,
                                     gridspec_kw={'width_ratios': [4, 1]})
        else:
            fig, axes = plt.subplots(2, 1, figsize=figsize,
                                     gridspec_kw={'height_ratios': [4, 1]})

        ax_img, ax_hist = axes[0], axes[1]
        vmax = max(np.percentile(frame, 99.9), self.bias + 3 * self.rn)
        im = ax_img.imshow(frame, cmap='gray', interpolation='nearest',
                           vmin=self.bias - 3 * self.rn, vmax=vmax, aspect='auto')
        if defects:
            for (xd, yd, _) in defects:
                ax_img.plot(xd, yd, 'c+', ms=8, mew=1.2)
        ax_img.set_title(title)
        plt.colorbar(im, ax=ax_img, label='DN')

        ax_hist.hist(frame.ravel(), bins=100, density=True,
                     color='steelblue', alpha=0.7, orientation='horizontal')
        ax_hist.set_xlabel('density')
        ax_hist.grid(True)
        plt.tight_layout()
        return fig, axes

    def __repr__(self):
        spk = f'Gamma(L={self.n_looks})' if self.n_looks else 'off'
        return (f"Background(rn={self.rn}, bias={self.bias}, gain={self.gain})\n"
                f"  speckle={spk}  poisson={self.use_poisson}")

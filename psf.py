import numpy as np
from scipy.special import j1
from scipy.ndimage import shift as ndshift
import matplotlib.pyplot as plt

from config import R_AIRY_PX, PSF_FINE, OVERSAMPLE


def annotated_heatmap(ax, arr, title, fmt='{:.2f}', cmap='coolwarm', fontsize=8,
                      vmin=None, vmax=None):
    im = ax.imshow(arr, cmap=cmap, interpolation='nearest', vmin=vmin, vmax=vmax)
    H, W = arr.shape
    threshold = arr.max() * 0.6 if arr.max() > 0 else 0.5
    for i in range(H):
        for j in range(W):
            ax.text(j, i, fmt.format(arr[i, j]), ha='center', va='center',
                    fontsize=fontsize,
                    color='black' if arr[i, j] > threshold else 'white')
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, shrink=0.75)
    return im


class PSF:
    def __init__(self, r_airy_px=R_AIRY_PX, psf_fine=PSF_FINE, oversample=OVERSAMPLE,
                 jitter_spx=0.0, jitter_spy=0.0, A=1.0):
        self.r_airy_px  = r_airy_px
        self.psf_fine   = psf_fine
        self.oversample = oversample
        self.jitter_spx = jitter_spx
        self.jitter_spy = jitter_spy
        self.A          = A

    @staticmethod
    def _bin2d(img, factor):
        H, W = img.shape
        H2, W2 = H // factor, W // factor
        return (img[:H2*factor, :W2*factor]
                .reshape(H2, factor, W2, factor)
                .sum(axis=(1, 3)))

    def make_fine(self, jitter_spx=None, jitter_spy=None):
        jx = self.jitter_spx if jitter_spx is None else jitter_spx
        jy = self.jitter_spy if jitter_spy is None else jitter_spy
        half = self.psf_fine // 2
        y, x = np.mgrid[-half:half, -half:half].astype(float)
        r = np.sqrt(x**2 + y**2)
        u_safe = np.where(r == 0, 1.0, 3.8317 * r / (self.r_airy_px * self.oversample))
        k = np.where(r == 0, 1.0, (2.0 * j1(u_safe) / u_safe)**2)
        k /= k.sum()
        if jx != 0.0 or jy != 0.0:
            k = ndshift(k, (jy, jx), order=3, mode='constant', cval=0.0)
        return k

    def make_template(self, A=None, jitter_spx=None, jitter_spy=None):
        """Camera-pixel PSF array, sum = A (default: self.A)."""
        a = self.A if A is None else A
        return self._bin2d(self.make_fine(jitter_spx, jitter_spy), self.oversample) * a

    def stamp(self, frame, x_px, y_px, A=None, jitter_spx=None, jitter_spy=None):
        """Add defect PSF centred at (x_px, y_px) into frame (in-place)."""
        tmpl = self.make_template(A, jitter_spx, jitter_spy)
        th, tw = tmpl.shape
        r0, c0 = y_px - th // 2, x_px - tw // 2
        r1, c1 = r0 + th, c0 + tw
        if r0 >= 0 and c0 >= 0 and r1 <= frame.shape[0] and c1 <= frame.shape[1]:
            frame[r0:r1, c0:c1] += tmpl

    def plot(self, A=None, jitter_spx=None, jitter_spy=None, figsize=(13, 4)):
        a      = self.A          if A          is None else A
        jx     = self.jitter_spx if jitter_spx is None else jitter_spx
        jy     = self.jitter_spy if jitter_spy is None else jitter_spy
        k_fine = self.make_fine(jx, jy)
        tmpl   = self.make_template(a, jx, jy)
        n_cam  = self.psf_fine // self.oversample

        fig, axes = plt.subplots(1, 3, figsize=figsize)
        axes[0].imshow(k_fine, cmap='hot', interpolation='nearest')
        axes[0].set_title(f'Airy PSF  {self.psf_fine}x{self.psf_fine} fine px\n'
                          f'r_airy={self.r_airy_px} cam-px = '
                          f'{self.r_airy_px*self.oversample:.0f} fine px')
        axes[1].imshow(k_fine, cmap='hot', interpolation='nearest')
        for i in range(0, self.psf_fine + 1, self.oversample):
            axes[1].axhline(i - 0.5, color='cyan', lw=0.8)
            axes[1].axvline(i - 0.5, color='cyan', lw=0.8)
        axes[1].set_title(f'Bin grid: {self.oversample}x{self.oversample} -> 1 cam-px')
        annotated_heatmap(axes[2], tmpl,
                          f'Camera PSF {tmpl.shape[0]}x{tmpl.shape[1]} px\nsum={tmpl.sum():.4f}',
                          fmt='{:.4f}', fontsize=12)
        fig.suptitle(f'PSF  jitter=({jx},{jy}) fine-px  A={a}', fontsize=11)
        plt.tight_layout()
        plt.show()
        return k_fine, tmpl

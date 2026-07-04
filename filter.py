"""
filter.py — detection filters + SNR estimators.

Aligned to imagelag_filter.ipynb (Module 4).

Filters:
    box_map / box   : n×m uniform-kernel local-sum response (no-filter baseline)
    matched         : normalized matched-filter response, matched to a template

Noise/SNR:
    sigma_theory    : analytical white-noise std for an n×m box sum
    sigma_empirical : far-field sampled std from a response map
    snr_theory      : theoretical MF SNR bound (white noise)
"""

import numpy as np
from scipy.signal import fftconvolve

from config import RN, BIAS


class Filter:
    def __init__(self, rn=RN, bias=BIAS):
        self.rn   = rn
        self.bias = bias

    # ── Box map: n×m 均匀核滑动卷积响应图 ───────────────────────────
    def box_map(self, frame, n=2, m=2):
        """
        n×m 均匀核滑动卷积，返回每个位置的局部和响应图。
        减去 n*m*bias 扣除直流。
        """
        kernel = np.ones((n, m))
        resp   = fftconvolve(frame, kernel[::-1, ::-1], mode='same')
        return resp - n * m * self.bias

    # ── Box: 指定位置的 box_map 值 ────────────────────────────────────
    def box(self, frame, n=2, m=2):
        """box_map 全图最大值作为信号。"""
        return self.box_map(frame, n, m).max()

    # ── 匹配滤波器 ────────────────────────────────────────────────────
    def matched(self, frame, template):
        """归一化 MF 响应图。缺陷峰值 ≈ A·||template||，噪声 std ≈ RN。"""
        h    = template / np.linalg.norm(template)
        resp = fftconvolve(frame, h[::-1, ::-1], mode='same')
        return resp

    # ── Noise 1: 理论值 ───────────────────────────────────────────────
    def sigma_theory(self, n=2, m=2):
        """n×m box sum 理论噪声：sqrt(n*m)·RN。"""
        return np.sqrt(n * m) * self.rn

    # ── Noise 2: 经验值（远场响应图采样）─────────────────────────────
    def sigma_empirical(self, resp_map, cx, cy, n_samples=2000, excl_r=40, seed=0):
        """从响应图远场背景区采样取 std。excl_r: 排除半径 [pixels]。"""
        H, W = resp_map.shape
        rng  = np.random.RandomState(seed)
        vals = []
        while len(vals) < n_samples:
            ys = rng.randint(2, H - 2, n_samples)
            xs = rng.randint(2, W - 2, n_samples)
            for y, x in zip(ys, xs):
                if abs(y - cy) > excl_r or abs(x - cx) > excl_r:
                    vals.append(resp_map[y, x])
                if len(vals) >= n_samples:
                    break
        return np.std(vals[:n_samples])

    # ── 理论 SNR 上界（白噪声，MF）───────────────────────────────────
    def snr_theory(self, template, A):
        """MF SNR = A · ||template|| / RN"""
        return A * np.linalg.norm(template) / self.rn

    def __repr__(self):
        return (f"Filter(rn={self.rn}, bias={self.bias})\n"
                f"  sigma_theory(2×2) = {self.sigma_theory():.4f}")

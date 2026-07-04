"""
imagelag.py — Hammerstein image-lag forward model, Q-dependent biexponential kernel.

Aligned to imagelag_filter.ipynb (Module 3). `apply()` uses the per-pixel
F-dependent kernel: every trapped pixel is released with its own kernel(F),
tail spread along axis=1 (trigger/readout direction).

    r(t, Q) = [Q - f(Q)]·δ(t=0) + f(Q)·h(t; f(Q))
    f(Q)    = Nt·(1-exp(-Q/Qc))                      static nonlinearity
    h(t;F)  = A1(F)·g_fast(t) + (1-A1(F))·g_slow(t)  Q-dependent release kernel
    A1(F)   = A1max·(1-exp(-F/Fc1))                  fast-trap fraction

Calibration source:
    Nt, Qc          : Mitscherlich fit, R²=0.996
    α1,α2,A1max,Fc1 : joint biexponential global fit (lag_model_fit_test.ipynb)
"""

import numpy as np


class ImageLag:
    """
    Hammerstein image-lag forward model — Q-dependent biexponential kernel.

        r(t, Q) = [Q − f(Q)]·δ(t=0) + f(Q)·h(t; f(Q))

        f(Q)   = Nt·(1−exp(−Q/Qc))                      静态非线性
        h(t;F) = A1(F)·g_fast(t) + (1−A1(F))·g_slow(t)  Q 依赖释放核
        A1(F)  = A1max·(1−exp(−F/Fc1))                   快陷阱占比
        g_slow = (1−α1)·α1^t / Σ,  g_fast = (1−α2)·α2^t / Σ  (窗口归一化)

    标定来源:
        Nt, Qc        : Mitscherlich 拟合  R²=0.996
        α1,α2,A1max,Fc1: 双指数联合全局优化 (lag_model_fit_test.ipynb)
    """

    # ── 默认标定参数 ──────────────────────────────────────────────────
    NT_DEFAULT     = 62.63
    QC_DEFAULT     = 85.4
    ALPHA1_DEFAULT = 0.7779   # 慢陷阱  τ1 ≈ 3.98 帧
    ALPHA2_DEFAULT = 0.2344   # 快陷阱  τ2 ≈ 0.69 帧
    A1MAX_DEFAULT  = 1.0
    FC1_DEFAULT    = 78.5     # 特征填充电荷 [DN]
    LEN_DEFAULT    = 20       # 拖尾窗口（帧）

    def __init__(self,
                 Nt=NT_DEFAULT, Qc=QC_DEFAULT,
                 alpha1=ALPHA1_DEFAULT, alpha2=ALPHA2_DEFAULT,
                 A1max=A1MAX_DEFAULT, Fc1=FC1_DEFAULT,
                 T=LEN_DEFAULT):
        self.Nt     = Nt
        self.Qc     = Qc
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.A1max  = A1max
        self.Fc1    = Fc1
        self.T      = T

    # ── 静态非线性 ────────────────────────────────────────────────────
    def f_trap(self, Q):
        """Per-pixel 陷阱填充量"""
        return self.Nt * (1.0 - np.exp(-np.clip(Q, 0, None) / self.Qc))

    def remaining(self, Q):
        return Q - self.f_trap(Q)

    # ── 释放核 ────────────────────────────────────────────────────────
    def kernel(self, f=None):
        """
        Q 依赖双指数释放核，shape (1, T+1)，t=0 为占位 0，其余 Σ=1。
        f : f_trap 标量，None 时取 f→0（纯慢陷阱）。
        """
        f  = max(float(f), 1e-6) if f is not None else 1e-6
        t  = np.arange(1, self.T + 1)
        A1 = self.A1max * (1 - np.exp(-f / self.Fc1))
        g_sl = (1-self.alpha1)*self.alpha1**t;  g_sl /= g_sl.sum()
        g_fa = (1-self.alpha2)*self.alpha2**t;  g_fa /= g_fa.sum()
        h    = A1*g_fa + (1-A1)*g_sl
        return np.concatenate([[0.0], h]).reshape(1, -1)

    # ── 2D forward model ─────────────────────────────────────────────
    def apply(self, frame, return_parts=False):
        """
        线性叠加 forward model（逐像素 F 依赖核）。
        对每个有捕获的像素直接调用 self.kernel()。
        拖尾沿 axis=1（触发/读出方向）。
        """
        H, W    = frame.shape
        trapped = self.f_trap(frame)
        remain  = frame - trapped
        lag     = np.zeros((H, W))

        ys, xs = np.where(trapped > 0.01)
        for i, j in zip(ys, xs):
            k = self.kernel(trapped[i, j]).flatten()   # 该像素自己的 F → 你的 kernel
            n = min(self.T, W - 1 - j)                  # 不越右边界
            for dt in range(1, n + 1):
                lag[i, j + dt] += trapped[i, j] * k[dt]

        out = remain + lag
        if return_parts:
            return out, trapped, remain, lag
        return out

    # ── 标量冲激响应 ──────────────────────────────────────────────────
    def predict(self, Q, t_max=None):
        """单像素接受 Q DN 时的理论响应，返回 dict。"""
        T   = self.T if t_max is None else t_max
        fQ  = self.f_trap(Q)
        t   = np.arange(1, T + 1)
        A1  = self.A1max * (1 - np.exp(-max(float(fQ), 1e-6) / self.Fc1))
        g_sl = (1-self.alpha1)*self.alpha1**t;  g_sl /= g_sl.sum()
        g_fa = (1-self.alpha2)*self.alpha2**t;  g_fa /= g_fa.sum()
        lag  = fQ * (A1*g_fa + (1-A1)*g_sl)
        return {'peak': Q - fQ, 'lag': lag, 'fQ': fQ, 'A1': A1,
                'sum':  Q - fQ + lag.sum()}

    # ── 参数热更新 ────────────────────────────────────────────────────
    def update(self, **kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise KeyError(f"未知参数: {k}")
            setattr(self, k, v)
        return self

    def __repr__(self):
        tau1 = -1 / np.log(self.alpha1)
        tau2 = -1 / np.log(self.alpha2)
        return (f"ImageLag(Nt={self.Nt}, Qc={self.Qc})\n"
                f"  slow: α1={self.alpha1:.4f}  τ1={tau1:.2f} 帧\n"
                f"  fast: α2={self.alpha2:.4f}  τ2={tau2:.2f} 帧\n"
                f"  A1(F) = {self.A1max} · (1−exp(−F/{self.Fc1}))")

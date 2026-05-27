"""
Image Lag Model Fitting
=======================
Model: single-species Hammerstein
  r(t; Q) = [Q - f(Q)] * delta(t=peak) + f(Q) * h(t)

  f(Q) = Nt * (1 - exp(-Q / Qc))          Mitscherlich trap filling
  h(t) = A1 * alpha1^(t-1) + A2 * alpha2^(t-1)   double-exponential release kernel

Inputs:
  - (Q, lag1) pairs for f(Q) fit
  - normalized h(t) sequence for h(t) fit
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ============================================================
# 1. DATA — replace with your measurements
# ============================================================

# f(Q) data: DC signal level Q, and first post-peak lag value
# Exclude saturated points (e.g. 946 DN)
fQ_data = [
    # Q (DN)   lag1 (DN)
    ( 33.32,    5.89),
    ( 35.28,    6.14),
    ( 48.71,    9.00),
    (101.98,   15.86),
    (218.43,   24.08),
    (223.47,   24.84),
    (236.24,   24.91),
    (305.86,   27.03),
    # (946.25, 53.00),  # excluded: saturated
]

# h(t) data: mean normalized post-peak sequence (t=1 is first lag point)
# Replace with your per-frame measurements when available
ht_t    = np.array([1, 2, 3, 4, 5, 6, 7])
ht_meas = np.array([1.000, 0.420, 0.234, 0.147, 0.094, 0.059, 0.044])

# ============================================================
# 2. MODEL FUNCTIONS
# ============================================================

def f_mitscherlich(Q, Nt, Qc):
    """Trap filling curve: f(Q) = Nt * (1 - exp(-Q/Qc))"""
    return Nt * (1 - np.exp(-Q / Qc))

def h_single(t, A, alpha):
    """Single-exponential release kernel"""
    return A * alpha ** (t - 1)

def h_double(t, A1, alpha1, A2, alpha2):
    """Double-exponential release kernel (two trap species)"""
    return A1 * alpha1 ** (t - 1) + A2 * alpha2 ** (t - 1)

def full_model(t_and_Q, Nt, Qc, A1, alpha1, A2, alpha2):
    """Full Hammerstein model: lag(t, Q) = f(Q) * h(t)"""
    t, Q = t_and_Q
    return f_mitscherlich(Q, Nt, Qc) * h_double(t, A1, alpha1, A2, alpha2)

# ============================================================
# 3. FIT f(Q)
# ============================================================

Q_arr    = np.array([d[0] for d in fQ_data])
lag1_arr = np.array([d[1] for d in fQ_data])

popt_fQ, pcov_fQ = curve_fit(
    f_mitscherlich, Q_arr, lag1_arr,
    p0=[40, 200], bounds=(0, [500, 2000]), maxfev=10000
)
Nt, Qc = popt_fQ
perr_fQ = np.sqrt(np.diag(pcov_fQ))

lag1_fit  = f_mitscherlich(Q_arr, Nt, Qc)
resid_fQ  = lag1_arr - lag1_fit
rms_fQ    = np.sqrt(np.mean(resid_fQ ** 2))
ss_res    = np.sum(resid_fQ ** 2)
ss_tot    = np.sum((lag1_arr - lag1_arr.mean()) ** 2)
r2_fQ     = 1 - ss_res / ss_tot

print("=" * 55)
print("f(Q) fit:  Nt * (1 - exp(-Q/Qc))")
print("=" * 55)
print(f"  Nt  = {Nt:.3f} +/- {perr_fQ[0]:.3f} DN")
print(f"  Qc  = {Qc:.2f}  +/- {perr_fQ[1]:.2f} DN")
print(f"  R2  = {r2_fQ:.5f}")
print(f"  RMS = {rms_fQ:.4f} DN")
print()
print(f"  {'Q':>8}  {'lag1_meas':>10}  {'lag1_fit':>10}  {'resid':>8}  {'rel%':>8}")
for i in range(len(Q_arr)):
    rel = 100 * resid_fQ[i] / lag1_arr[i]
    print(f"  {Q_arr[i]:8.2f}  {lag1_arr[i]:10.3f}  {lag1_fit[i]:10.3f}  "
          f"{resid_fQ[i]:8.3f}  {rel:7.2f}%")

# ============================================================
# 4. FIT h(t) — single and double exponential
# ============================================================

# Single exponential
p1, pc1 = curve_fit(
    h_single, ht_t, ht_meas,
    p0=[1.0, 0.5], bounds=([0, 0], [2, 1]), maxfev=10000
)
fit_h1  = h_single(ht_t, *p1)
rms_h1  = np.sqrt(np.mean((ht_meas - fit_h1) ** 2))

# Double exponential
p2, pc2 = curve_fit(
    h_double, ht_t, ht_meas,
    p0=[0.5, 0.2, 0.5, 0.7],
    bounds=([0, 0, 0, 0], [2, 1, 2, 1]), maxfev=10000
)
A1, alpha1, A2, alpha2 = p2
fit_h2  = h_double(ht_t, *p2)
rms_h2  = np.sqrt(np.mean((ht_meas - fit_h2) ** 2))
perr_h2 = np.sqrt(np.diag(pc2))

tau1 = -1 / np.log(alpha1)
tau2 = -1 / np.log(alpha2)

print()
print("=" * 55)
print("h(t) fit — single exponential: A * alpha^(t-1)")
print("=" * 55)
print(f"  A     = {p1[0]:.4f}")
print(f"  alpha = {p1[1]:.4f}  (tau = {-1/np.log(p1[1]):.2f} frames)")
print(f"  RMS   = {rms_h1:.5f}")

print()
print("=" * 55)
print("h(t) fit — double exponential: A1*alpha1^(t-1) + A2*alpha2^(t-1)")
print("=" * 55)
print(f"  A1     = {A1:.4f} +/- {perr_h2[0]:.4f}   (fast trap, tau={tau1:.2f} frames)")
print(f"  alpha1 = {alpha1:.4f} +/- {perr_h2[1]:.4f}")
print(f"  A2     = {A2:.4f} +/- {perr_h2[2]:.4f}   (slow trap, tau={tau2:.2f} frames)")
print(f"  alpha2 = {alpha2:.4f} +/- {perr_h2[3]:.4f}")
print(f"  RMS    = {rms_h2:.5f}  ({rms_h1/rms_h2:.0f}x better than single)")
print()
print(f"  {'t':>4}  {'h_meas':>8}  {'single':>8}  {'double':>8}  {'resid_s':>8}  {'resid_d':>8}")
for i in range(len(ht_t)):
    print(f"  {ht_t[i]:>4}  {ht_meas[i]:>8.4f}  {fit_h1[i]:>8.4f}  "
          f"{fit_h2[i]:>8.4f}  {ht_meas[i]-fit_h1[i]:>8.4f}  {ht_meas[i]-fit_h2[i]:>8.4f}")

# ============================================================
# 5. SNR GAIN
# ============================================================

t_long = np.arange(1, 20)
ht_long = h_double(t_long, A1, alpha1, A2, alpha2)
snr_temporal = np.sum(ht_long ** 2)
snr_spatial  = 4.0   # 2x2 PSF, uniform (conservative lower bound)
snr_total    = snr_temporal * snr_spatial

print()
print("=" * 55)
print("SNR gain (GLRT matched filter)")
print("=" * 55)
print(f"  Spatial  2x2 PSF : {10*np.log10(snr_spatial):.2f} dB  (uniform, lower bound)")
print(f"  Temporal lag h(t): {10*np.log10(snr_temporal):.2f} dB")
print(f"  Joint total      : {10*np.log10(snr_total):.2f} dB")

# ============================================================
# 6. PLOT
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# --- Panel 1: f(Q) fit ---
ax = axes[0]
q_line   = np.linspace(10, 400, 500)
q_extrap = np.linspace(300, 1000, 300)
ax.scatter(Q_arr, lag1_arr, color='#378ADD', s=60, zorder=5, label='measured lag1')
ax.plot(q_line,   f_mitscherlich(q_line,   Nt, Qc), color='#D85A30', lw=2,
        label=f'fit: Nt={Nt:.1f}, Qc={Qc:.0f}')
ax.plot(q_extrap, f_mitscherlich(q_extrap, Nt, Qc), color='#888', lw=1.5,
        ls='--', label='extrapolation')
ax.set_xscale('log')
ax.set_xlabel('Q (DN)')
ax.set_ylabel('lag1 (DN)')
ax.set_title(f'f(Q) trap filling\nR2={r2_fQ:.4f}, RMS={rms_fQ:.3f} DN')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# --- Panel 2: h(t) fit ---
ax = axes[1]
t_dense = np.linspace(1, 7, 200)
ax.plot(ht_t, ht_meas, 'o', color='#378ADD', ms=7, zorder=5, label='measured h(t)')
ax.plot(t_dense, h_single(t_dense, *p1), color='#888', lw=1.5,
        ls='--', label=f'single exp (RMS={rms_h1:.4f})')
ax.plot(t_dense, h_double(t_dense, *p2), color='#D85A30', lw=2,
        label=f'double exp (RMS={rms_h2:.4f})')

# log scale inset check
ax.set_xlabel('t (frames after peak)')
ax.set_ylabel('h(t) normalized')
ax.set_title(f'h(t) release kernel\ndouble exp {rms_h1/rms_h2:.0f}x better than single')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# --- Panel 3: log(h) linearity check ---
ax = axes[2]
ax.plot(ht_t, np.log(ht_meas), 'o-', color='#378ADD', ms=6, lw=1.5,
        label='log(h_meas)')
ax.plot(t_dense, np.log(np.maximum(h_single(t_dense, *p1), 1e-9)),
        color='#888', lw=1.5, ls='--', label='single exp (straight line = single)')
ax.plot(t_dense, np.log(np.maximum(h_double(t_dense, *p2), 1e-9)),
        color='#D85A30', lw=2, label='double exp')
ax.set_xlabel('t (frames after peak)')
ax.set_ylabel('log h(t)')
ax.set_title('Semi-log plot\nstraight line = single exp, curve = multi')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/lag_model_fit.png', dpi=150, bbox_inches='tight')
print("\nSaved: lag_model_fit.png")

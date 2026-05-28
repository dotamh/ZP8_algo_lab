# CLAUDE.md

Project context for the **PSF + image-lag filter comparison** work.
Read this before editing `imagelag_filter.ipynb` or `lag_model_fit_test.ipynb`.

## Key files

| File | Role |
|------|------|
| `imagelag_filter.ipynb` | **Main simulation** — PSF / Background / ImageLag / Filter classes + demos |
| `lag_model_fit_test.ipynb` | **Parameter fitting** — f(Q) Mitscherlich + Q-dependent biexponential kernel |
| `lag_model_fit.py` | Early single-species fit (archived; superseded by the notebook) |
| `imagelag_data.xlsx` | Raw measurement data: Sheet2 = h(t) curves, Sheet3 = f_trap areas |

## Goal

Simulate a forward imaging chain (Airy PSF → jitter → Q-dependent image lag) and
compare detection filters for finding point defects. Evaluation metric: **output SNR**
(highest-SNR filter wins).

## Forward model (authoritative — `imagelag_filter.ipynb` Modules 1–3)

Chain, in order:

1. **PSF** (`class PSF`): Airy disk on a `128×128` fine grid (OVERSAMPLE = 32).
   Binned to a `4×4` camera-pixel footprint; central `2×2` contains ~91.5% of energy.
   Sub-pixel jitter applied via `ndshift` on the fine grid before binning.

2. **Image lag** (`class ImageLag`): Q-dependent Hammerstein model.
   - Static nonlinearity: `f_trap(Q) = Nt·(1 − exp(−Q / Qc))`
   - Q-dependent biexponential kernel:
     `h(t; F) = A1(F)·g_fast(t) + (1−A1(F))·g_slow(t)`
     where `A1(F) = A1max·(1 − exp(−F / Fc1))`
   - `apply(frame)`: `trapped = f_trap(Q)` → `lag = trapped ⊛ kernel(mean_f)` → `out = remain + lag`

3. **Observation** = PSF output + image lag + readout noise `RN`.

Frame size: `H×W = 256×256`. Default `RN = 2.0 ADC`, `FWC = 1e4 e-`.

### Calibrated parameters (from `lag_model_fit_test.ipynb`)

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `Nt` | 62.63 DN | Trap capacity |
| `Qc` | 85.4 DN | Half-saturation signal |
| `α1` | 0.7779 | Slow trap decay (τ₁ ≈ 3.98 frames) |
| `α2` | 0.2344 | Fast trap decay (τ₂ ≈ 0.69 frames) |
| `A1max` | 1.0 | Max fast-trap weight |
| `Fc1` | 78.5 DN | Characteristic fill charge |

## Filters (`imagelag_filter.ipynb` Module 4 — `class Filter`)

| Filter | Status | Idea |
|--------|--------|------|
| `box` | **done** | Sum of central 2×2 pixels — no-filter baseline |
| `matched` | **done** | `Filter.matched()`: MF matched to PSF shape; optimal for white noise |
| `lag-kernel MF` | **done** | MF matched to full PSF+lag response template; tested in filter-test cell |
| Wiener | **TODO** | Clutter-matched `g = C⁻¹h`; treats lag tail as colored clutter |
| GLRT-matched | **TODO** | GLRT with MF on PSF+lag; handles unknown amplitude |

## What has been done

### Forward model (`imagelag_filter.ipynb`)
- `class PSF`: Airy disk, OVERSAMPLE=32, stores `jitter_spx/spy` and `A`; `make_template()`, `stamp()`, `plot()`
- `class Background`: Gaussian readout noise; `sample()`, `plot()`
- `class ImageLag`: Q-dependent Hammerstein model; calibrated from `lag_model_fit_test.ipynb`
- `class Filter`: `box()`, `matched()`, `sigma_theory()`, `sigma_empirical()`, `snr_theory()`

### Parameter fitting (`lag_model_fit_test.ipynb`)
- f(Q) Mitscherlich fit: R²=0.996, Nt=62.63 DN, Qc=85.4 DN
- Global biexponential kernel fit: α1=0.7779 (slow, τ≈3.98f), α2=0.2344 (fast, τ≈0.69f), A1max=1.0, Fc1=78.5 DN
- Validated: normalized lag tails from 3 signal levels collapse → single-species Hammerstein confirmed

### Filter SNR comparison (filter-test cell in `imagelag_filter.ipynb`)
- Box (no lag): SNR_emp ≈ 23.8
- Box (with lag): SNR_emp ≈ 11.0  (signal halved by trapping, σ unchanged)
- Lag-kernel MF (with lag frame): SNR_emp ≈ 16.5  (+1.5× vs box-with-lag, −0.7× vs box-no-lag)
- Two noise methods per filter: `sigma_theory` (analytical) and `sigma_empirical` (far-field sampling)

## Output-SNR metric

- **signal** `s` = filter peak response to the defect's PSF (or PSF+lag) alone (no noise).
- **noise** `σ` = rms of filter output sampled from background pixels far from the defect (|Δx|>40 or |Δy|>40).
- `SNR = s / σ`.

Two estimators: `sigma_theory` (analytical, white-noise only) and `sigma_empirical` (far-field sampling, captures lag clutter).

## TODO

- **Monte Carlo SNR sweep**: repeat filter comparison over many noise realizations; plot mean SNR ± std vs amplitude A for all filters.
- **Matched filter (PSF-only MF)**: `Filter.matched()` with normalized PSF template; compare to box and lag-kernel MF.
- **Wiener filter**: `g = S_signal / (S_signal + S_noise)` in frequency domain, where `S_noise` includes lag PSD as colored clutter.
- **GLRT matched filter**: GLRT statistic for unknown amplitude A; reduces to `hᵀC⁻¹x` (matched filter with noise-whitening); compare to Wiener.
- **f(Q) biexponential fit**: try `f(Q) = N1·(1−exp(−Q/Qc1)) + N2·(1−exp(−Q/Qc2))` and compare to Mitscherlich.

## Conventions

- Reproducible Monte-Carlo via `np.random.RandomState(seed)`; never global seed.
- Plot labels in English (container fonts lack CJK glyphs).
- `fftconvolve(..., mode='full')[:H, :W]` for all 2-D convolutions — keep consistent.
- Output plots saved to the working directory (e.g. `psf_lag_*.png`, `cmp_*.png`).

# CLAUDE.md

Project context for the **PSF + saturated image-lag filter comparison** work.
Read this before editing `step5.py` or `filter_compare.py`.

## Goal

Simulate a forward imaging chain (2×2 PSF → jitter → saturated image lag) and
compare detection filters for finding point defects. Current evaluation metric:
**output SNR** (highest-SNR filter wins).

## Forward model (authoritative — see `step5.py`)

Chain, in order:

1. **PSF**: each defect is a `2×2` box of ones, scaled by amplitude `A`.
2. **Jitter**: convolve with `(1×3)/3` box kernel along x.
   PSF ⊛ jitter = effective **2×4 target PSF** `h`; `||h|| = 1.4907`.
3. **Saturated lag** (the key nonlinearity):
   - `mask = (jittered > 0.01)` — binary, marks illuminated pixels.
   - `lag = mask ⊛ L_fixed`, with `L_fixed = [3, 2, 1, 0.5, 0.2]` (1D x-tail).
   - **Lag tail is a FIXED absolute charge, independent of `A`.** Bright and
     faint defects get the *same* lag pattern. Any filter that treats lag as
     signal-proportional blur is wrong by construction.
4. **Observation** = `jittered + lag + RN`, with `RN = 1.0` (white Gaussian).

Image is `H×W = 30×160`; defects sit on row `Y = 15`.

Invariant: any new code must keep the forward model bit-identical to `step5.py`
(kernels, threshold `0.01`, `fftconvolve(..., 'full')[:H,:W]`).

## Filters under comparison (`filter_compare.py`)

| Filter      | Idea                                                            |
|-------------|-----------------------------------------------------------------|
| MF          | Matched filter to `h`. Optimal for white noise, ignores lag.    |
| Wiener      | Clutter-matched `g = C⁻¹h`; lag treated as colored clutter.     |
| CLRT-ideal  | Constrained LRT with the *true* lag subtracted (upper bound).   |
| CLRT-real   | CLRT with lag estimated blindly from MF peaks, then subtracted. |

## Output-SNR metric

- **signal** `s_i` = filter peak response to the defect's PSF *alone*, measured
  on an **isolated single-defect scene** (linear in `A`, no neighbor cross-talk).
- **noise** `σ_nc` = rms of `T_filter(obs) − signal_part` on the realistic
  8-defect scene → captures lag clutter + RN leaking through the filter.
- `SNR_i = s_i / σ_nc`.

## Findings so far

- Plain **MF** is crippled by lag clutter: `σ_nc ≈ 5.3` vs RN-only `≈ 1.49`.
- **CLRT** (real and ideal) recovers ≈ the full white-noise SNR `1.49·A`,
  ~3.5× better than plain MF across all amplitudes.
- CLRT-real ≈ CLRT-ideal once the MF-peak→corner offset `(1, 2)` is calibrated
  so the estimated lag template registers correctly.

## Known issues / TODO

- **Wiener filter has a PSD unit bug** (in progress). The white-noise term
  `RN²` and the lag periodogram `|FFT(lag)|²` are on different scales — the
  periodogram is larger by a factor `H·W = 4800`, so the filter massively
  over-whitens. Fix: put both on the same convention, e.g. use
  `RN²·H·W + P_lag`, or normalize the periodogram by `H·W`. Re-verify the
  Wiener row of the table after the fix.
- CLRT-real depends on blind peak detection; faint defects near the detection
  threshold may get mis-registered lag templates.

## Conventions

- Reproducible Monte-Carlo via `np.random.RandomState(seed)`; never global seed.
- Plot labels in English (container fonts lack CJK glyphs).
- Outputs written to the working dir as `cmp_*.png`.

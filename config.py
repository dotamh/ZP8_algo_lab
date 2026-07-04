
# ── camera sensor ──────────────────────────────────────────────────────────
H, W       = 256, 256   # camera frame [sensor pixels]
FWC        = 1e4        # full-well capacity [electrons]
GAIN       = 1.0        # ADC gain [e⁻ / DN]
RN         = 2.0        # readout noise sigma [DN]
BIAS       = 0.0        # bias level [DN]

# ── PSF sub-pixel model ────────────────────────────────────────────────────
R_AIRY_PX  = 1.0        # Airy first-dark-ring radius [camera pixels]
PSF_FINE   = 128        # fine-grid size for PSF [fine pixels, square]
OVERSAMPLE = 32         # fine pixels per camera pixel

# ── jitter ────────────────────────────────────────────────────────────────
JITTER_MAX_SPX = OVERSAMPLE   # ±1 camera pixel max

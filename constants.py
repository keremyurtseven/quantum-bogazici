import numpy as np

# ---------- Problem constants ----------
T = 8.22e-3                 # seconds (one period)
FS_REF = 1.4e9              # Hz (sampling for reference signal)
UV_PER_V_THRESHOLD = 150.0  # µV/V spec

# ---------- Reference signal coefficients (sum of Gaussians) ----------
A_S = np.array([288.5, 3477, 1700, 1.036, 3058, 2533, 2923, 357.1])
B_S = np.array([0.00645, 0.006347, 0.004174, 0.001731, -0.001027, 0.008404, 0.002118, 0.007327])
C_S = np.array([0.000343, 0.001874, 0.001678, 1.079e-5, 0.002381, 0.0009477, 0.002298, 0.0004674])

# ----------- Controls for plotting stems -----------
FULL_FFT           = True        # True = use all ~11.5M samples at 1.4 GHz (heavy); False = preview via decimation
PREVIEW_TARGET_N   = 1_000_000   # samples for fast preview FFT when FULL_FFT=False
STEM_MODE          = "db_floor"  # "all" | "topk" | "db_floor"
TOP_K              = 2000        # used if STEM_MODE=="topk"
DB_FLOOR           = -50         # keep lines within 80 dB of max if STEM_MODE=="db_floor"
FREQ_MAX_MHZ       = None        # None or a number (e.g., 100.0) to cap x-axis
MAX_PLOT_POINTS = 20000          # decimate time plots for speed

# Define coefficients for modulator
A1 = 0.581873872695153
A2 = 0.348747127467881
B1 = A1
B2 = A2
B3 = 1.0
G1 = 0
C1 = 0.091626468407932
C2 = 1.034906999953018
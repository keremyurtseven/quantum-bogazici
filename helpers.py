import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy.signal import lfilter
from tqdm import tqdm
from constants import *

def fmt_fs(fs_hz):
    """Frequency unity conversion for better visualization"""
    if fs_hz >= 1e9:  return f"{fs_hz/1e9:.3f} GHz"
    if fs_hz >= 1e6:  return f"{fs_hz/1e6:.3f} MHz"
    if fs_hz >= 1e3:  return f"{fs_hz/1e3:.3f} kHz"
    return f"{fs_hz:.3f} Hz"

def gen_one_period(fs):
    """Sample exactly one period [0, T) with N=round(fs*T)."""
    N = int(round(fs * T))
    N = max(N, 1)
    dt = T / N
    t = np.arange(N) * dt
    x = sum(A_S[i] * np.exp(-((t - B_S[i]) / C_S[i])**2) for i in range(len(A_S)))
    return x.astype(np.float64), dt

def one_period_spectrum(x, dt):
    """One-sided discrete line spectrum from one period. Returns (freqs, mag, phase)."""
    N = x.size
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(N, d=dt)
    mag = np.abs(X) / N
    if N % 2 == 0:
        if mag.size > 2: mag[1:-1] *= 2.0
    else:
        if mag.size > 1: mag[1:] *= 2.0
    phase = np.angle(X)
    return freqs, mag, phase

def cosine_sum_on_ref_grid(tones, N_ref):
    """Reconstruct on the reference grid using sum A*cos(2π f t + φ)."""
    if tones is None or tones.size == 0:
        return np.zeros(N_ref, dtype=np.float64)
    t = np.linspace(0.0, T, N_ref, endpoint=False)
    xhat = np.zeros_like(t)
    for f, A, ph in tones:
        xhat += A * np.cos(2*np.pi*f*t + ph)
    return xhat

def metric_uVrms_per_V(x_ref, x_syn):
    """µV/V = RMS(error)/RMS(ref) * 1e6."""
    err = x_ref - x_syn
    rms_err = np.sqrt(np.mean(err**2))
    rms_ref = np.sqrt(np.mean(x_ref**2)) + 1e-300
    return float(rms_err / rms_ref * 1e6)

def moving_average(x, N=128):
    """Reconstruct the modulator output"""
    return lfilter(np.ones(N)/N, 1, x)

def to_float(s):
    """Converter to read data from txt"""
    if isinstance(s, bytes):
        s = s.decode("utf-8", errors="ignore")
    return float(s.replace(",", "."))  # convert comma decimal to dot

def period_spectrum_select_tones(
    x, fs, dt=None,          # signal, sampling freq, time step
    db_floor_db=60,          # keep bins within this many dB of the strongest (in band)
    *, freq_max=None,        # limit to ≤ this frequency (Hz). None → Nyquist
    top_k=None,              # after dB floor, optionally keep only strongest K lines
    plot_time=True,          # time plot; amp vs time
    plot_spectrum=True,      # stem plot; dB vs log-frequency
    save_txt=False,          # save tones to TXT
    txt_name="tones.txt",    # file name if save_txt=True
):
    """
    Treat x as ONE exact period at sampling rate fs.
    Return tones array with columns [freq_Hz, amplitude, phase_rad].
    Amplitudes are one-sided Fourier series line amplitudes.
    """
    x = np.asarray(x, float)
    N = x.size
    if N < 2:
        raise ValueError("x must have at least 2 samples")

    # --- discrete Fourier series over one period ---
    X = np.fft.rfft(x) / N
    freqs = np.fft.rfftfreq(N, d=1.0/fs)

    # one-sided amplitude for real signals
    amps = np.abs(X)
    if N % 2 == 0:        # even N includes Nyquist bin
        if amps.size > 2: amps[1:-1] *= 2.0
    else:
        if amps.size > 1: amps[1:] *= 2.0
    phases = np.angle(X)

    # --- in-band selection ---
    if freq_max is None:
        freq_max = fs/2
    band = freqs <= float(freq_max)
    f_b, a_b, p_b = freqs[band], amps[band], phases[band]

    if a_b.size == 0:
        return np.empty((0,3)), {"freqs": freqs, "amps": amps, "phases": phases}

    # --- apply relative dB floor vs strongest in-band line ---
    amax = np.max(a_b) + 1e-300
    rel_db = 20*np.log10(a_b/amax)
    keep = rel_db >= -abs(db_floor_db)
    f_sel, a_sel, p_sel = f_b[keep], a_b[keep], p_b[keep]

    # --- optional top-K cap ---
    if top_k is not None and f_sel.size > top_k:
        idx = np.argsort(a_sel)[::-1][:top_k]
        f_sel, a_sel, p_sel = f_sel[idx], a_sel[idx], p_sel[idx]

    # sort by frequency for nice output
    order = np.argsort(f_sel)
    tones = np.column_stack([f_sel[order], a_sel[order], p_sel[order]])

    # --- plot (optional) ---
    if plot_time:
        # decimate for speed if huge
        N = x.size
        max_pts = 20000
        if N > max_pts:
            idx = np.linspace(0, N-1, max_pts, dtype=int)
            t_plot = idx * dt
            x_plot = x[idx]
        else:
            t_plot = np.arange(N) * dt
            x_plot = x

        plt.figure(figsize=(10, 3.4))
        plt.plot(t_plot*1e3, x_plot, lw=0.9)
        plt.title(f"Synthesized One-Period (fs={fmt_fs(fs)})")
        plt.xlabel("Time (ms)")
        plt.ylabel("Amplitude (V)")
        plt.grid(True, alpha=0.5)
        plt.tight_layout()
        plt.show()

    if plot_spectrum:
        a_b_db = 20*np.log10(a_b/amax)
        plt.figure(figsize=(9.5, 3.6))
        plt.plot(f_b*1e-6, a_b_db, lw=0.8, color='0.7', label="All bins (rel dB)")
        if tones.size:
            A_sel_db = 20*np.log10(tones[:,1]/amax)
            plt.stem(tones[:,0]*1e-6, A_sel_db, linefmt='C1-', markerfmt='C1o',
                     basefmt=" ", label=fr"$\geq${db_floor_db} dB from max")
        plt.xscale("log")
        plt.xlabel("Frequency [MHz]")
        plt.ylabel("Amplitude [dB rel. max]")
        plt.title("Discrete Spectrum (one-period assumption)")
        plt.grid(True, which="both", alpha=0.4)
        plt.tight_layout(); plt.legend(); plt.show()

    # --- save (optional) ---
    if save_txt:
        tosave = tones
        header = "freq_Hz amplitude phase_rad"
        np.savetxt(txt_name, tosave, fmt="%.12g", header=header)
        print(f"Saved tones to {txt_name}")

    return tones, {"freqs": freqs, "amps": amps, "phases": phases}

def sweep_optimize_cosine_sum(
    fs_grid,
    dbfloor_grid,
    x_ref, N_ref,
    threshold_uV_per_V=151.0,
    save_csv=True,
    csv_path="sweep_results.csv",
    save_best_tone_list=False,
    best_tone_fname=None,
    blank_col2=False,
):
    """
    Sweeps all (fs, db_floor_db). For each:
      - build tones via synth_period_and_tones (Cell 2)
      - reconstruct on reference grid via cosine sum
      - compute metric in µV/V vs original x_ref
      - log fs, db_floor_db, metric, num_tones

    Returns:
      results: list of dicts
      best: dict for best feasible (min #tones; tie → lower metric) or None
      failing: list of dicts with metric > threshold
    """
    results = []
    best = None
    for fs in fs_grid:
        for dbf in tqdm(dbfloor_grid):
            # Build tones (no plots, no save per-point)
            x_synth, dt_synth = gen_one_period(fs)

            tones, _ = period_spectrum_select_tones(
                x_synth, fs, dt_synth,
                db_floor_db=dbf,
                plot_time=False,
                plot_spectrum=False,
                save_txt=False
            )
            num_tones = int(tones.shape[0])

            # Cosine-sum reconstruction on reference grid
            x_syn = cosine_sum_on_ref_grid(tones, N_ref)

            # Metric µV/V
            metric = metric_uVrms_per_V(x_ref, x_syn)

            rec = {
                "fs": float(fs),
                "db_floor_db": float(dbf),
                "metric_uV_per_V": float(metric),
                "num_tones": num_tones,
                "tones": tones,  # keep for potential best save
            }
            results.append(rec)

            # Track best feasible
            if metric <= threshold_uV_per_V:
                if (best is None) or (num_tones < best["num_tones"]) or \
                   (num_tones == best["num_tones"] and metric < best["metric_uV_per_V"]):
                    best = rec
        print(f"{fmt_fs(fs)} is done")

    # Save CSV if requested
    if save_csv and results:
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["fs_Hz", "db_floor_db", "metric_uV_per_V", "num_tones"])
            for r in results:
                w.writerow([r["fs"], r["db_floor_db"], r["metric_uV_per_V"], r["num_tones"]])
        print(f"Saved sweep table: {csv_path}")

    # Save best tone list if requested
    if best is not None and save_best_tone_list:
        # Reuse Cell 2's saver indirectly by calling it once with save_txt=True
        # but we already have tones; so write here to avoid recomputation:
        def _fmt_fs(fs_hz):
            if fs_hz >= 1e9:  return f"{fs_hz/1e9:.3f}GHz"
            if fs_hz >= 1e6:  return f"{fs_hz/1e6:.3f}MHz"
            if fs_hz >= 1e3:  return f"{fs_hz/1e3:.3f}kHz"
            return f"{fs_hz:.3f}Hz"
        fs_str = _fmt_fs(best["fs"])
        sp_str = f"{abs(best['db_floor_db']):.0f}dB"
        outname = best_tone_fname or f"tones_fs{fs_str}_sp{sp_str}.txt"
        tones = best["tones"]
        if blank_col2:
            if tones.size:
                blank = np.zeros((tones.shape[0],1))
                tosave = np.column_stack([tones[:,0], tones[:,1], blank, tones[:,2]])
            else:
                tosave = np.empty((0,4))
            header = "freq_Hz amplitude _blank phase_rad"
        else:
            tosave = tones
            header = "freq_Hz amplitude phase_rad"
        np.savetxt(outname, tosave, fmt="%.12g", header=header)
        print(f"Saved best tone list: {outname}")

    # Failing points (> threshold)
    failing = [r for r in results if r["metric_uV_per_V"] > threshold_uV_per_V]
    print(f"Total points: {len(results)} | Feasible (≤ {threshold_uV_per_V:.1f} µV/V): {len(results)-len(failing)} | Failing: {len(failing)}")
    if failing:
        # Show a few worst offenders
        worst = sorted(failing, key=lambda r: r["metric_uV_per_V"], reverse=True)[:5]
        print("Worst failing points (top 5):")
        for r in worst:
            print(f"  fs={r['fs']:.3g} Hz, floor={r['db_floor_db']:.0f} dB, "
                  f"metric={r['metric_uV_per_V']:.1f} µV/V, tones={r['num_tones']}")
            
    # --- Show a few best ones ---
    # Feasible points (≤ threshold): prioritize FEWEST tones, then lowest metric
    feasible = [r for r in results if r["metric_uV_per_V"] <= threshold_uV_per_V]
    if feasible:
        best_fewtones = sorted(feasible, key=lambda r: (r["num_tones"], r["metric_uV_per_V"]))[:5]
        print("Best feasible points (top 5: fewest tones → lowest metric):")
        for r in best_fewtones:
            print(f"  fs={r['fs']:.3g} Hz, floor={r['db_floor_db']:.0f} dB, "
                f"metric={r['metric_uV_per_V']:.3f} µV/V, tones={r['num_tones']}")
    else:
        print(f"No feasible points ≤ {threshold_uV_per_V:.1f} µV/V.")

    # Overall best by METRIC only (even if above threshold)
    best_metric_overall = sorted(results, key=lambda r: r["metric_uV_per_V"])[:5]
    print("Lowest metric overall (top 5):")
    for r in best_metric_overall:
        print(f"  fs={r['fs']:.3g} Hz, floor={r['db_floor_db']:.0f} dB, "
            f"metric={r['metric_uV_per_V']:.3f} µV/V, tones={r['num_tones']}")
        


    return results, best, failing

def delta_sigma_cifb(
        u,             # Signal to modulated
        mode="pm1"     # Modulation mode. "01" for 0,1. "pm1" for -1,1. "ternary" for -1,0,1
):
    """
        Create a delta modulator using the reference in the task slide
    """
    # Normalize the input and shift between -1 to 1
    u = np.asarray(u, dtype=float)
    x_min = np.min(u)
    x_max = np.max(u)

    # avoid division by zero
    if x_max == x_min:
        u_normalized = np.zeros_like(u, dtype=float)
    else:
        u_normalized = 2.0 * (u - x_min) / (x_max - x_min) - 1.0

    N = len(u_normalized)
    y  = np.zeros(N)
    vq = np.zeros(N)

    # assume these are defined elsewhere: a1,a2,b1,b2,b3,c1,c2,g1
    s1 = 0.0
    s2 = 0.0
    y_fb_prev = 0.0  # what goes back through the DAC (±1 or ternary)

    # symmetric ternary thresholds on v (not on u)
    th1, th2 = -1.0/3.0, 1.0/3.0

    for n in range(N):
        s1 += B1*u_normalized[n] - A1*y_fb_prev - G1*s1
        s2 += B2*u_normalized[n] + C1*s1       - A2*y_fb_prev
        vq[n] = B3*u_normalized[n] + C2*s2     # quantizer input

        if mode == "pm1":
            y[n] =  1.0 if vq[n] >= 0.0 else -1.0
            y_fb  = y[n]  # feedback is ±1

        elif mode == "01":
            bit   = 1.0 if vq[n] >= 0.0 else 0.0
            y[n]  = bit               # keep {0,1} as the recorded output
            y_fb  = 2.0*bit - 1.0     # but feed back ±1 to keep loop centered

        elif mode == "ternary":
            if   vq[n] >  th2: y[n] =  1.0
            elif vq[n] <  th1: y[n] = -1.0
            else:             y[n] =  0.0
            y_fb = y[n]  # requires a 3-level DAC; otherwise map as needed

        y_fb_prev = y_fb

    return y, vq

# Helper functions for impedance calculations
def Z_R(R):
    R = np.asarray(R, dtype=float)
    return R.astype(complex)

def Z_C(C, w):
    C = np.asarray(C, dtype=float)
    w = np.asarray(w, dtype=float)
    jwC = 1j * w * C
    out = np.empty(np.broadcast(jwC,).shape, dtype=complex)
    np.divide(1.0, jwC, out=out, where=(jwC != 0))
    out[jwC == 0] = np.inf
    return out

def Z_L(L, w):
    L = np.asarray(L, dtype=float)
    w = np.asarray(w, dtype=float)
    return 1j * w * L


# Series impedance
def z_series(*Zs):
    s = 0
    for z in Zs:
        s = s + np.asarray(z, dtype=complex)
    return s

# Parallel impedance
def z_parallel(*Zs):
    inv_sum = 0
    for z in Zs:
        z = np.asarray(z, dtype=complex)
        inv = np.zeros_like(z, dtype=complex)
        np.divide(1.0, z, out=inv, where=(z != 0))
        inv_sum = inv_sum + inv
    out = np.empty_like(inv_sum, dtype=complex)
    np.divide(1.0, inv_sum, out=out, where=(inv_sum != 0))
    out[inv_sum == 0] = np.inf
    return out

# Transfer function
def H(f, p):
    """
    Vx/Vj for 4-block network:

      z1 = Cp_cable || C_N || R_N || R_s_cable_shunt
      z2 = R_s_cable_series + j*w*L_s_cable
      z3 = Cp_probe  || R_s_probe_shunt
      z4 = R_s_probe_series + j*w*L_s_probe

      zy = ((z1 + z2)^-1 + z3^-1)^-1
      H  = (zy / (zy + z4)) * (z1 / (z1 + z2))
    """
    f = np.atleast_1d(f).astype(float)
    w = 2*np.pi*f

    R_N   = p["R_N"]
    C_N   = p["C_N"]

    C_p_cable = p["C_p_cable"]
    R_s_cable = p["R_s_cable"]
    R_p_cable = p["R_p_cable"]
    L_s_cable = p.get("L_s_cable")

    C_p_probe = p.get("C_p_probe")
    R_p_probe = p.get("R_p_probe")
    R_s_probe = p.get("R_s_probe")
    L_s_probe = p.get("L_s_probe")

    # z1: shunt at Vx
    z1 = z_parallel(Z_C(C_p_cable, w), Z_C(C_N, w), Z_R(R_N),  Z_R(R_p_cable))
    
    # z2: series branch to Vx
    z2 = z_series(Z_R(R_s_cable), Z_L(L_s_cable, w))

    # z3: probe shunt at middle node
    z3 = z_parallel(Z_C(C_p_probe, w), Z_R(R_p_probe))

    # z4: probe series from source
    z4 = z_series(Z_R(R_s_probe), Z_L(L_s_probe, w))

    zy = z_parallel(z_series(z1, z2), z3)
    

    return (zy / z_series(zy, z4)) * (z1 / z_series(z1, z2))
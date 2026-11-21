# To run: python src/main.py data/processed/Samples.csv 512
# Pass --hplp if you also want to sweep high-pass / low-pass filter alphas

from itertools import product
from pathlib import Path
import sys
import numpy as np

# make sure repo root is in sys.path for both `python src/main.py` and `python -m src.main`
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.algorithms.cntrl_algrthm import (
    calculate_hr,
    find_peaks_control,
    high_pass_filter,
    low_pass_filter,
)
from src.algorithms.peak_detection import apply_dynamic_threshold, estimate_hr_peaks
from src.filters.bandpass import (
    butter_bandpass_filter,
    five_to_fiteen_hz_filter,
    moving_average,
    test_moving_average,
)
from src.utils.data_loader import read_samples

# hardware constants from PulseOximeter firmware
DEFAULT_CSV = "data/processed/Samples.csv"
DEFAULT_LEN = 512
FS = 64.0          # true MAX86141 rate: 128 sps / 2 avg = 64 Hz (not 80 or 100)
BUFFER_START = 32  # skips first 32 samples like BUFFER_SAMPLE_START in main.c
RR_MIN_SEC = 0.25
QRS_INT_SEC = 0.10


# --- Signal Processing & Peak-to-BPM ---

def peaks_to_bpm(peaks, fs=FS, agg="median", min_bpm=30.0, max_bpm=240.0):
    # whiteboard steps 1-5: r-peaks -> intervals -> sec -> avg/median -> invert to bpm
    if len(peaks) < 2 or fs <= 0:
        return 0.0
    ibis = np.diff(peaks) / fs
    valid = ibis[(ibis >= 60.0 / max_bpm) & (ibis <= 60.0 / min_bpm)]
    if len(valid) == 0:
        valid = ibis
    avg_ibi = float(np.median(valid) if agg == "median" else np.mean(valid))
    return 60.0 / avg_ibi if avg_ibi > 0 else 0.0


def stm32_find_peaks(squared, fs=FS, pth=2.7):
    # python port of findPeaks() in Ryan's signal_processing.c
    n_samples = len(squared)
    if n_samples == 0:
        return np.array([], dtype=int)
    rr_min, qrs_int = max(1, int(RR_MIN_SEC * fs)), max(1, int(QRS_INT_SEC * fs))
    decay = np.exp(-pth / fs)
    th, n, r_peaks, max_vals = 100.0, 0, [], []
    while n < n_samples:
        if squared[n] > th:
            win = squared[n : min(n + rr_min + qrs_int, n_samples)]
            pos = int(np.argmax(win))
            max_vals.append(float(win[pos]))
            r_peaks.append(n + pos + 2)
            # same refractory jump as C code: n += 2*RR_MIN + QRS_INT - (RR_MIN + QRS_INT - pos)
            n = n + pos + rr_min
            th = float(np.mean(max_vals))
        else:
            th *= decay
            n += 1
    return np.array(r_peaks, dtype=int)


def run_pipeline(raw, fs=FS, filt="none", hp=None, lp=None, slope="diff",
                 deriv_a=2, ma=1, pth=2.7, detector="stm32", agg="median"):
    x = np.asarray(raw, dtype=np.float64)

    # 1. filter stage (note: Samples.csv is already HP/LP filtered on the STM32 gBuffer)
    # TODO: test with --hplp once we dump raw unfiltered ADC data
    if filt == "first_order":
        if hp is not None:
            x = high_pass_filter(x, hp_alpha=hp)
        if lp is not None:
            x = low_pass_filter(x, lp_alpha=lp)
    elif filt == "bp2":
        x = five_to_fiteen_hz_filter(x, fs=fs)
    elif filt == "butter":
        x = butter_bandpass_filter(x, lowcut=0.5, highcut=4.0, fs=fs)

    if len(x) > BUFFER_START:
        x = x[BUFFER_START:]

    # direct scipy find_peaks branch (doesn't use derivative/squaring)
    if slope == "none" and detector == "scipy_peaks":
        sm = moving_average(x, ma) if ma > 1 else x
        prom = float(np.std(sm)) * 0.3 if np.std(sm) > 0 else None
        bpm, _ = estimate_hr_peaks(sm, fs=fs, prominence=prom)
        return float(bpm)

    # 2. derivative / slope: simple diff x[n+a]-x[n] or 5-pt slope2
    # doing 5-pt deriv here since bandpass.five_point_derivative doesn't return an array
    if slope == "diff":
        d = x[deriv_a:] - x[:-deriv_a] if 0 < deriv_a < len(x) else np.zeros(0)
    else:
        d = 10.0 * (2.0 * x[4:] + x[3:-1] - x[1:-3] - 2.0 * x[:-4]) if len(x) > 4 else np.zeros(0)

    # 3. moving average + squaring (note: signal_processing.c movingAverage lacks a loop so STM32 runs ma=1)
    if ma == 12:
        d = test_moving_average(d, window_size=12)
    elif ma > 1 and len(d) > 0:
        d = moving_average(d, window_size=ma)
    sq = d ** 2
    if len(sq) == 0:
        return 0.0

    # 4. peak detection -> bpm
    if detector == "ewma":
        if len(sq) < max(2, int(fs)):
            return 0.0
        peaks, _ = apply_dynamic_threshold(sq, fs=fs, pth=pth, alpha=0.85)
    else:
        peaks = stm32_find_peaks(sq, fs=fs, pth=pth)
    return peaks_to_bpm(peaks, fs=fs, agg=agg)


# --- Metrics, Comparisons & Parameter Testing ---

def calc_metrics(actual, pred):
    diff = np.asarray(pred, dtype=np.float64) - np.asarray(actual, dtype=np.float64)
    abs_err = np.abs(diff)
    return float(np.mean(abs_err)), float(np.sqrt(np.mean(diff ** 2))), float(np.max(abs_err)), int(np.sum(abs_err <= 3.0))


def check_duplicates(ss):
    seen = {}
    for s in ss:
        key = np.ascontiguousarray(s.data).tobytes()
        tag = f"{s.name} #{s.sample_num}"
        if key in seen:
            print(f"[!] Heads up: {tag} has duplicate waveform data with {seen[key]}")
        else:
            seen[key] = tag


def run_comparisons(ss, fs=FS):
    print("\n=== Algorithm Comparison (vs. Control HR) ===")
    print(f"{'Algorithm':<36} {'MAE':>6} {'RMSE':>6} {'Max':>6} {'<=3 BPM':>8}")
    print("-" * 66)

    def show(label, preds):
        mae, rmse, mx, w3 = calc_metrics(ss.actual_hr, preds)
        print(f"{label:<36} {mae:6.2f} {rmse:6.2f} {mx:6.1f} {w3:>5d}/{len(ss)}")

    # helper so cntrl_algrthm.calculate_hr doesn't divide by zero if <2 peaks found
    def ctrl_hr(data, pth):
        peaks = find_peaks_control(data, fs=fs, pth=pth)
        return float(calculate_hr(peaks, int(fs))) if len(peaks) >= 2 else 0.0

    show("STM32 Logged (Samples.csv)", ss.estimated_hr)
    show("cntrl_algrthm.py (pth=1.0)", [ctrl_hr(s.data, pth=1.0) for s in ss])
    show("cntrl_algrthm.py (pth=2.7)", [ctrl_hr(s.data, pth=2.7) for s in ss])
    tests = [
        ("STM32 Exact Port (a=2, ma=1)", dict(filt="none", slope="diff", deriv_a=2, ma=1, pth=2.7)),
        ("STM32 + Moving Avg (ma=3)", dict(filt="none", slope="diff", deriv_a=2, ma=3, pth=2.7)),
        ("STM32 + Moving Avg (ma=6)", dict(filt="none", slope="diff", deriv_a=2, ma=6, pth=2.7)),
        ("STM32 + Test Moving Avg (ma=12)", dict(filt="none", slope="diff", deriv_a=2, ma=12, pth=2.7)),
        ("BP2 (5-15Hz) + 5pt Slope + STM32", dict(filt="bp2", slope="5pt", ma=6, pth=2.7)),
        ("5pt Slope + STM32 (no refilter)", dict(filt="none", slope="5pt", ma=6, pth=2.7)),
        ("5pt Slope + EWMA Thresh", dict(filt="none", slope="5pt", ma=6, pth=2.7, detector="ewma")),
        ("Butterworth (0.5-4Hz) + SciPy Peaks", dict(filt="butter", slope="none", ma=5, detector="scipy_peaks")),
    ]
    for label, kw in tests:
        show(label, [run_pipeline(s.data, fs=fs, **kw) for s in ss])


def run_sensitivity(ss, fs=FS):
    print("\n=== Whiteboard Parameter Sensitivity (1-at-a-Time) ===")
    sweeps = [
        ("1. High Pass (alpha)", [dict(filt="none" if v is None else "first_order", hp=v) for v in [None, 0.50, 0.675, 0.80, 0.90, 0.95, 0.975, 0.99]], lambda kw: str(kw["hp"])),
        ("2. Low Pass (beta)", [dict(filt="none" if v is None else "first_order", lp=v) for v in [None, 0.20, 0.308, 0.50, 0.70, 0.840, 0.92]], lambda kw: str(kw["lp"])),
        ("3. Post-Deriv / Slope", [dict(slope="diff", deriv_a=a) for a in [1, 2, 3, 4, 5, 6, 8]] + [dict(slope="5pt", deriv_a=0)], lambda kw: "5pt" if kw["slope"] == "5pt" else f"diff({kw['deriv_a']})"),
        ("3b. Moving Avg Window", [dict(ma=m) for m in [1, 3, 5, 6, 9, 12, 15]], lambda kw: f"ma={kw['ma']}"),
        ("4. Threshold Decay (PTH)", [dict(pth=p) for p in [0.5, 1.0, 1.5, 2.0, 2.7, 3.5, 4.5, 5.4, 7.0]], lambda kw: f"pth={kw['pth']}"),
    ]
    for title, cfgs, fmt in sweeps:
        scores = []
        for kw in cfgs:
            mae, rmse, _, _ = calc_metrics(ss.actual_hr, [run_pipeline(s.data, fs=fs, **kw) for s in ss])
            scores.append((mae, rmse, fmt(kw)))
        best = min(scores, key=lambda t: (t[0], t[1]))
        print(f"  {title:<24} best={best[2]:<8} (MAE={best[0]:.2f}) | " + ", ".join(f"{l}:{m:.2f}" for m, _, l in scores))


def run_sweep(ss, fs=FS, include_hplp=False):
    slopes = [("diff", a) for a in (1, 2, 3, 4, 5, 6)] + [("5pt", 0)]
    grid = [
        dict(filt=f, hp=None, lp=None, slope=s, deriv_a=a, ma=ma, pth=pth, agg=agg)
        for f, (s, a), ma, pth, agg in product(("none", "bp2"), slopes, (1, 3, 5, 6, 9, 12), (1.0, 1.5, 2.0, 2.5, 2.7, 3.5, 4.5, 5.4), ("median", "mean"))
    ]
    if include_hplp:
        hplp_slopes = [("diff", a) for a in (1, 2, 3, 4)] + [("5pt", 0)]
        grid.extend(
            dict(filt="first_order", hp=hp, lp=lp, slope=s, deriv_a=a, ma=ma, pth=pth, agg="median")
            for hp, lp, (s, a), ma, pth in product((0.675, 0.90, 0.95, 0.975), (0.308, 0.60, 0.75, 0.840), hplp_slopes, (1, 3, 6, 9), (1.5, 2.7, 3.5, 4.5))
        )

    ranked = sorted(
        [(*calc_metrics(ss.actual_hr, preds), kw, preds) for kw in grid for preds in ([run_pipeline(s.data, fs=fs, **kw) for s in ss],)],
        key=lambda x: (x[0], x[1], x[2]),
    )
    print(f"\n=== Parameter Sweep Top 5 ({len(ranked)} configs tested) ===")
    print(f"{'Rank':<5} {'Filter':<16} {'Slope':<8} {'MA':>3} {'PTH':>4} {'Agg':<7} {'MAE':>6} {'RMSE':>6} {'Max':>5}")
    print("-" * 66)
    for i, (mae, rmse, mx, _, kw, _) in enumerate(ranked[:5], 1):
        flbl = f"hp={kw['hp']},lp={kw['lp']}" if kw["filt"] == "first_order" else kw["filt"]
        slbl = "5pt" if kw["slope"] == "5pt" else f"diff({kw['deriv_a']})"
        print(f"{i:<5d} {flbl:<16} {slbl:<8} {kw['ma']:3d} {kw['pth']:4.1f} {kw['agg']:<7} {mae:6.2f} {rmse:6.2f} {mx:5.1f}")

    top_mae, _, _, _, best_kw, best_preds = ranked[0]
    print("\n=== Per-Sample Results (STM32 CSV vs. Top #1 Config) ===")
    for i, s in enumerate(ss):
        print(f"  {s.name:<6} #{s.sample_num}: actual={s.actual_hr:3d}, stm32_csv={s.estimated_hr:3d} "
              f"({s.estimated_hr - s.actual_hr:+3d}), top1={best_preds[i]:5.1f} ({best_preds[i] - s.actual_hr:+5.1f})")

    hp_out = best_kw["hp"] if best_kw["hp"] is not None else (0.675 if best_kw["filt"] == "bp2" else 0.975)
    lp_out = best_kw["lp"] if best_kw["lp"] is not None else (0.308 if best_kw["filt"] == "bp2" else 0.840)
    print(f"\nRecommended STM32 #defines (MAE={top_mae:.2f} BPM):")
    print(f"  HP_FILTER_ALPHA={hp_out:.3f}f, LP_FILTER_ALPHA={lp_out:.3f}f, "
          f"slope={best_kw['slope']}({best_kw['deriv_a']}), MOV_AVG_AMT={best_kw['ma']}, PTH={best_kw['pth']:.1f}f, agg={best_kw['agg']}")


def main():
    args = [a for a in sys.argv[1:] if a != "--hplp"]
    ss = read_samples(args[0] if len(args) >= 1 else DEFAULT_CSV, int(args[1]) if len(args) >= 2 else DEFAULT_LEN)
    print(f"Loaded {len(ss)} samples (shape={ss.data.shape}, fs={FS} Hz)")
    check_duplicates(ss)
    run_comparisons(ss, fs=FS)
    run_sensitivity(ss, fs=FS)
    run_sweep(ss, fs=FS, include_hplp=("--hplp" in sys.argv[1:]))


if __name__ == "__main__":
    main()

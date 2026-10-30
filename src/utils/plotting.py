"""
Plotting and Visualization Utilities for PPG Heart Rate Algorithm Report (Sections 3 & 4).
Generates publication-ready Matplotlib figures for:
1. Raw vs. Filtered PPG Signal + Detected Peaks (Time Domain)
2. Power Spectral Density / FFT Periodogram (Frequency Domain)
3. Butterworth Bandpass Filter Frequency Response (Bode Magnitude Plot)
4. Parameter Optimization Sweep (Cutoff Frequencies / Moving Average vs. BPM Error)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from src.algorithms.peak_detection import estimate_hr_peaks
from src.algorithms.spectral_analysis import estimate_hr_spectral
from src.filters.bandpass import butter_bandpass_filter, moving_average, normalize_ac_dc
from src.utils.data_loader import generate_synthetic_ppg


def plot_signal_and_peaks(
    t: np.ndarray,
    raw_ppg: np.ndarray,
    filtered_ppg: np.ndarray,
    peaks: np.ndarray,
    estimated_bpm: float,
    true_bpm: float | None = None,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Plot 1: Raw PPG with baseline wander vs. Bandpass-Filtered PPG with detected peaks.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax1.plot(t, raw_ppg, color="#555555", linewidth=1.2, label="Raw Sensor Data (with Baseline Wander & Noise)")
    ax1.set_title("Subsystem 1: Raw PPG Sensor Input vs. Filtered & Normalized Output", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Raw Amplitude")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    title_str = f"Method 1 (Peak Detection): Est. HR = {estimated_bpm:.1f} BPM"
    if true_bpm is not None:
        err = abs(estimated_bpm - true_bpm)
        title_str += f" (Control = {true_bpm:.1f} BPM, Error = {err:.2f} BPM)"

    ax2.plot(t, filtered_ppg, color="#0072B2", linewidth=1.5, label="Filtered & Smoothed PPG (0.5–4.0 Hz)")
    if len(peaks) > 0:
        ax2.plot(
            t[peaks],
            filtered_ppg[peaks],
            "ro",
            markersize=6,
            label=f"Detected Systolic Peaks (n={len(peaks)})",
        )
    ax2.set_title(title_str, fontsize=11, fontweight="bold")
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Normalized AC/DC Amplitude")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)
    return fig


def plot_spectral_analysis(
    freqs: np.ndarray,
    psd: np.ndarray,
    estimated_bpm: float,
    true_bpm: float | None = None,
    min_bpm: float = 30.0,
    max_bpm: float = 240.0,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Plot 2: Frequency-Domain Power Spectral Density (FFT / Periodogram) in both Hz and BPM.
    """
    fig, ax = plt.subplots(figsize=(9, 4.5))

    mask = (freqs >= 0.1) & (freqs <= 5.0)
    freqs_bpm = freqs[mask] * 60.0
    psd_masked = psd[mask]

    ax.plot(freqs_bpm, psd_masked, color="#009E73", linewidth=1.8, label="Power Spectral Density (PSD)")
    ax.axvline(
        estimated_bpm,
        color="#D55E00",
        linestyle="--",
        linewidth=1.8,
        label=f"Dominant Peak: {estimated_bpm:.1f} BPM ({estimated_bpm / 60.0:.2f} Hz)",
    )
    if true_bpm is not None:
        ax.axvline(
            true_bpm,
            color="#333333",
            linestyle=":",
            linewidth=1.5,
            label=f"Control Reference: {true_bpm:.1f} BPM",
        )

    ax.axvspan(min_bpm, max_bpm, color="#009E73", alpha=0.08, label="Valid HR Passband (30–240 BPM)")
    ax.set_title("Method 2: Spectral Analysis (Fourier Transform / Periodogram)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Heart Rate Frequency (BPM)")
    ax.set_ylabel("Power Spectral Density")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)
    return fig


def plot_filter_frequency_response(
    lowcut: float = 0.5,
    highcut: float = 4.0,
    fs: float = 100.0,
    orders: list[int] | None = None,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Plot 3: Butterworth Bandpass Filter Bode Magnitude Response across filter orders.
    """
    if orders is None:
        orders = [1, 2, 4]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    nyq = 0.5 * fs

    for order in orders:
        sos = signal.butter(order, [lowcut / nyq, highcut / nyq], btype="bandpass", output="sos")
        w, h = signal.sosfreqz(sos, worN=2000, fs=fs)
        ax.plot(w, 20 * np.log10(np.maximum(np.abs(h), 1e-10)), linewidth=1.6, label=f"Order {order}")

    ax.axvline(lowcut, color="red", linestyle="--", alpha=0.7, label=f"Low Cutoff ({lowcut} Hz / {lowcut * 60:.0f} BPM)")
    ax.axvline(highcut, color="purple", linestyle="--", alpha=0.7, label=f"High Cutoff ({highcut} Hz / {highcut * 60:.0f} BPM)")
    ax.axhline(-3.0, color="gray", linestyle=":", label="-3 dB Cutoff Threshold")

    ax.set_xlim(0, 10.0)
    ax.set_ylim(-40, 5)
    ax.set_title("Butterworth Bandpass Filter Frequency Response", fontsize=12, fontweight="bold")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Gain (dB)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)
    return fig


def plot_parameter_optimization(
    t: np.ndarray,
    raw_ppg: np.ndarray,
    true_bpm: float,
    fs: float = 100.0,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Plot 4: Parameter Optimization Sweep (Moving Average Window Size & High-Pass Cutoff vs. Accuracy Error).
    Directly addresses Section 2 & Section 4 requirements.
    """
    window_sizes = [1, 3, 5, 9, 15, 25, 35]
    errors_peak = []
    errors_spectral = []

    normalized = normalize_ac_dc(raw_ppg, fs=fs)
    filtered = butter_bandpass_filter(normalized, lowcut=0.5, highcut=4.0, fs=fs)

    for w in window_sizes:
        smoothed = moving_average(filtered, window_size=w)
        bpm_p, _ = estimate_hr_peaks(smoothed, fs=fs)
        bpm_s, _, _ = estimate_hr_spectral(smoothed, fs=fs)
        errors_peak.append(abs(bpm_p - true_bpm))
        errors_spectral.append(abs(bpm_s - true_bpm))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(window_sizes, errors_peak, "o-", linewidth=1.8, color="#0072B2", label="Method 1: Peak Detection Error (|Est - True|)")
    ax.plot(window_sizes, errors_spectral, "s--", linewidth=1.8, color="#009E73", label="Method 2: Spectral Analysis Error (|Est - True|)")

    ax.set_title("Parameter Optimization: Moving Average Window Size vs. Heart Rate Error", fontsize=11, fontweight="bold")
    ax.set_xlabel("Moving Average Window Size (samples)")
    ax.set_ylabel("Absolute Error (BPM)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)
    return fig


def generate_all_report_graphs(output_dir: str | Path = "outputs") -> list[Path]:
    """
    Generate all 4 report figures into output_dir and return their paths.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    fs = 100.0
    true_bpm = 75.0
    t, raw_ppg = generate_synthetic_ppg(bpm=true_bpm, duration_sec=15.0, fs=fs, noise_std=0.12)

    normalized = normalize_ac_dc(raw_ppg, fs=fs)
    filtered = butter_bandpass_filter(normalized, lowcut=0.5, highcut=4.0, fs=fs)
    smoothed = moving_average(filtered, window_size=5)

    bpm_peaks, peaks = estimate_hr_peaks(smoothed, fs=fs)
    bpm_spectral, freqs, psd = estimate_hr_spectral(smoothed, fs=fs)

    paths = [
        out / "1_time_domain_peaks.png",
        out / "2_spectral_analysis_psd.png",
        out / "3_filter_frequency_response.png",
        out / "4_parameter_optimization.png",
    ]

    f1 = plot_signal_and_peaks(t, raw_ppg, smoothed, peaks, bpm_peaks, true_bpm, save_path=paths[0])
    f2 = plot_spectral_analysis(freqs, psd, bpm_spectral, true_bpm, save_path=paths[1])
    f3 = plot_filter_frequency_response(lowcut=0.5, highcut=4.0, fs=fs, save_path=paths[2])
    f4 = plot_parameter_optimization(t, raw_ppg, true_bpm, fs=fs, save_path=paths[3])

    plt.close(f1)
    plt.close(f2)
    plt.close(f3)
    plt.close(f4)

    return paths


if __name__ == "__main__":
    saved = generate_all_report_graphs()
    for p in saved:
        print(f"Saved: {p}")

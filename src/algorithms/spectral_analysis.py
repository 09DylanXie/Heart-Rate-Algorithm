"""
Method 2: Frequency-Domain Spectral Analysis (FFT / Welch) Algorithm for PPG Heart Rate.
"""

import numpy as np
from scipy import signal


def estimate_hr_spectral(
    filtered_ppg: np.ndarray,
    fs: float = 100.0,
    min_bpm: float = 30.0,
    max_bpm: float = 240.0,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Estimate heart rate (in BPM) using Periodogram / Spectral Analysis.

    Returns:
        tuple[float, np.ndarray, np.ndarray]: (Estimated BPM, frequencies in Hz, power spectrum)
    """
    freqs, psd = signal.periodogram(filtered_ppg, fs=fs, window="hann")

    min_freq = min_bpm / 60.0
    max_freq = max_bpm / 60.0

    valid_mask = (freqs >= min_freq) & (freqs <= max_freq)
    if not np.any(valid_mask):
        return 0.0, freqs, psd

    valid_freqs = freqs[valid_mask]
    valid_psd = psd[valid_mask]

    dominant_idx = int(np.argmax(valid_psd))
    dominant_freq_hz = float(valid_freqs[dominant_idx])
    bpm = dominant_freq_hz * 60.0

    return bpm, freqs, psd


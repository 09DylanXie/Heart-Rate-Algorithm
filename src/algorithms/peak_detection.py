"""
Method 1: Time-Domain Peak Detection Algorithm for PPG Heart Rate.
"""

import numpy as np
from scipy import signal


def estimate_hr_peaks(
    filtered_ppg: np.ndarray,
    fs: float = 100.0,
    min_bpm: float = 30.0,
    max_bpm: float = 240.0,
    prominence: float | None = None,
) -> tuple[float, np.ndarray]:
    """
    Estimate heart rate (in BPM) using time-domain peak detection.

    Returns:
        tuple[float, np.ndarray]: (Estimated BPM, array of peak indices)
    """
    # Minimum distance between peaks based on maximum possible heart rate
    min_distance_samples = max(1, int((60.0 / max_bpm) * fs))

    peaks, _ = signal.find_peaks(
        filtered_ppg,
        distance=min_distance_samples,
        prominence=prominence,
    )

    if len(peaks) < 2:
        return 0.0, peaks

    # Calculate inter-beat intervals (IBI) in seconds
    ibi_seconds = np.diff(peaks) / fs
    mean_ibi = float(np.mean(ibi_seconds))
    bpm = 60.0 / mean_ibi if mean_ibi > 0 else 0.0

    if bpm < min_bpm or bpm > max_bpm:
        return 0.0, peaks

    return bpm, peaks


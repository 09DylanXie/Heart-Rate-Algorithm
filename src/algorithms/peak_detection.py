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


def apply_dynamic_threshold(z_signal: np.ndarray, fs: float = 80.0, pth: float = 1.0, alpha: float = 0.85):
    """
    Applies dynamic threshold to peak detection by increasing Pth based on recent heart rate history
    Uses exponetially weighted moving average function for average
    Tracks signal ceiling and noise floor to ensure decay never bottoms out at 0
    Uses search back to check for missed beats

    """
    N = len(z_signal)
    decay_factor = np.exp(-pth / fs)
    rr_min = int(0.200 * fs) # 200 ms refractory window
    
    th_history = np.zeros(N)
    r_peaks = []
    
    th = np.mean(z_signal[:int(fs)]) * 2.0  # Initial estimate from first second
    n = 0
    
    while n < N:
        th_history[n] = th
        
        if z_signal[n] > th:
            # Find local maximum peak in the immediate candidate window
            window_end = min(n + rr_min, N)
            local_max_rel = np.argmax(z_signal[n:window_end])
            peak_idx = n + local_max_rel
            local_max = z_signal[peak_idx]
            
            r_peaks.append(peak_idx)
            
            # Update threshold based on detected pulse peak
            th = (alpha * th) + ((1.0 - alpha) * local_max)
            
            # Skip past the refractory period
            n = peak_idx + rr_min
        else:
            # Exponentially decay threshold
            th *= decay_factor
            n += 1
            
    return np.array(r_peaks), th_history
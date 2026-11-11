import numpy as np

def apply_dynamic_threshold(z_signal: np.ndarray, fs: float = 80.0, pth: float = 1.0, alpha: float = 0.85):
    """
    Applies exponential dynamic threshold decay to a squared PPG/ECG signal.
    """
    N = len(z_signal)
    decay_factor = np.exp(-pth / fs)
    rr_min = int(0.200 * fs) # 200 ms blanking/refractory window
    
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
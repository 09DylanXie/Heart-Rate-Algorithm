import numpy as np
from scipy.signal import lfilter

def find_peaks_control(data_buffer: np.ndarray, fs: float = 80.0) -> np.ndarray:
    """
    Control algorithm (current stm32 algorithm) implementing:
    - First-order High-Pass (alpha = 0.975)
    - First-order Low-Pass (alpha = 0.840)
    - Simple forward difference derivative (DERIV_DIST = 2)
    - Moving average (MOV_AVG_AMT = 6)
    - Squaring stage
    - Dynamic exponential threshold peak finding
    """
    # --- Step 1: Baseline High-Pass Filter ---
    # y[n] = 0.975 * (y[n-1] + x[n] - x[n-1])
    hp_alpha = 0.975
    b_hp = [hp_alpha, -hp_alpha]
    a_hp = [1.0, -hp_alpha]
    hp_out = lfilter(b_hp, a_hp, data_buffer)
    
    # --- Step 2: Baseline Low-Pass Filter ---
    # y[n] = 0.840 * y[n-1] + 0.160 * x[n]
    lp_alpha = 0.840
    b_lp = [1.0 - lp_alpha]
    a_lp = [1.0, -lp_alpha]
    filt_data = lfilter(b_lp, a_lp, hp_out)
    
    # --- Step 3: Simple Difference Derivative (DERIV_DIST = 2) ---
    deriv_dist = 2
    deriv = np.zeros_like(filt_data)
    deriv[:-deriv_dist] = filt_data[deriv_dist:] - filt_data[:-deriv_dist]
    
    # --- Step 4: Moving Average (Boxcar, Window = 6) ---
    mov_avg_amt = 6
    window = np.ones(mov_avg_amt) / mov_avg_amt
    ma_out = np.convolve(deriv, window, mode='same')
    
    # --- Step 5: Squaring Stage ---
    squared = ma_out ** 2
    
    # --- Step 6: Dynamic Thresholding (Exponential Decay) ---
    rr_min = int(0.2 * fs)      # Minimum RR interval (200 ms)
    qrs_int = int(0.15 * fs)    # QRS window width (150 ms)
    
    th = 100.0
    n = 0
    r_peaks = []
    max_val = []
    
    length = len(squared)
    while n < (length - deriv_dist):
        if squared[n] > th:
            window_end = min(n + rr_min + qrs_int, length - deriv_dist)
            search_window = squared[n:window_end]
            
            local_max_pos = np.argmax(search_window)
            local_max = search_window[local_max_pos]
            
            r_peaks.append(n + local_max_pos + 2)
            max_val.append(local_max)
            
            d = rr_min + qrs_int - local_max_pos
            n += (rr_min + qrs_int + rr_min - d)
            th = np.mean(max_val)
        else:
            # Exponential decay threshold update
            th *= np.exp(-1.0 / fs)
            n += 1
            
    return np.array(r_peaks, dtype=int)
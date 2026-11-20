import numpy as np
from scipy.signal import lfilter

def find_peaks_control(data_buffer: np.ndarray, fs: float = 64.0, pth: float = 1.0) -> np.ndarray:
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
    hp_alpha = 0.975
    hp_out = high_pass_filter(data_buffer, hp_alpha)
    
    # --- Step 2: Baseline Low-Pass Filter ---
    lp_alpha = 0.840
    filt_data = low_pass_filter(hp_out, lp_alpha)
    
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
            th *= np.exp(-pth / fs)
            n += 1
            
    return np.array(r_peaks, dtype=int)

# High pass filter
# y[n] = 0.975 * (y[n-1] + x[n] - x[n-1])
# a[0]*y[n] = b[0]*x[n] + b[1]*x[n-1] + ... + b[M]*x[n-M] --> (1.0)*y[n] = hp_alpha*x[n] - hp_alpha*x[n-1]
#                       - a[1]*y[n-1] - ... - a[N]*y[n-N] -->                            + hp_alpha*y[n-1]
def high_pass_filter(data_buffer: np.ndarray, hp_alpha: float = 0.975):
        
    b_hp = [hp_alpha, -hp_alpha]
    a_hp = [1.0, -hp_alpha]
    hp_out = lfilter(b_hp, a_hp, data_buffer)

    return hp_out

# Low pass Filter
# y[n] = 0.840 * y[n-1] + 0.160 * x[n]
# a[0]*y[n] = b[0]*x[n] + b[1]*x[n-1] + ... + b[M]*x[n-M]
#                       - a[1]*y[n-1] - ... - a[N]*y[n-N]
def low_pass_filter(data_buffer: np.ndarray, lp_alpha: float = 0.840):
        
    b_lp = [1.0 - lp_alpha]
    a_lp = [1.0, -lp_alpha]
    lp_out = lfilter(b_lp, a_lp, data_buffer)

    return lp_out

# Calulcate heart rate with an input set beat indices
# r_peaks: array of indices at which heart beats occured
# fs: sample rate used in collecting those peaks
def calculate_hr(r_peaks: np.ndarray, fs: int):

    r_length = len(r_peaks)

    intervals = []
    for i in range(0, r_length - 1):
        # Finds interval between two beats
        intervals.append(r_peaks[i+1] - r_peaks[i])

    # Find average interval
    sum = 0.0
    for i in range(0, r_length - 1):
        sum += intervals[i]

    avg = sum / ((float) (r_length - 1))

    # Calculate and return HR
    return 60 / (avg / fs)



# Testing
# In terminal: python3 src/algorithms/cntrl_algrthm.py  
if __name__ == "__main__":

    print("Hello world!")
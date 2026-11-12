"""
Signal Filtering Subsystem for PPG Heart Rate Algorithm.
Handles high-pass, low-pass, bandpass (0.5 - 4.0 Hz for 30-240 BPM),
moving average, and AC/DC pressure-variation normalization.
"""

import numpy as np
from scipy import signal
from scipy.signal import lfilter

"""
Re-implementation of cntrl-algrhtm filter's bandpass filters using updated alphas for 5-15hz
alphas calculated using e^-2pi(fc/fs) with fc = 5, 15
"""

def five_to_fiteen_hz_filter(data_buffer: np.ndarray, fs: float = 80.0) -> np.ndarray:
 # --- Step 1: Baseline High-Pass Filter ---
     # y[n] = 0.675 * (y[n-1] + x[n] - x[n-1])
     hp_alpha = 0.675
     b_hp = [hp_alpha, -hp_alpha]
     a_hp = [1.0, -hp_alpha]
     hp_out = lfilter(b_hp, a_hp, data_buffer)
     
     # --- Step 2: Baseline Low-Pass Filter ---
     # y[n] = 0.308 * y[n-1] + 0.160 * x[n]
     lp_alpha = 0.308
     b_lp = [1.0 - lp_alpha]
     a_lp = [1.0, -lp_alpha]
     filt_data = lfilter(b_lp, a_lp, hp_out)
     return filt_data

def five_point_derivative(data: np.ndarray, slope: float,)-> float:
    for n in range(len(data)):
        if n <5:
            continue
         
        else:
            slope = 10*(2*data[n]+ data[n-1] - data[n-3]-2*data[n-4])
    slope = slope*slope
    """
    Calculates slope (as a float) value of the filtered PPG wave. 
    Uses 5 point derivative to provide more accurate slope compared to old simple difference
    Squares slope value so pos and neg values contribute to peak energy
    """    
    


    

def butter_bandpass_filter(
    data: np.ndarray,
    lowcut: float = 0.5,
    highcut: float = 4.0,
    fs: float = 100.0,
    order: int = 2,
) -> np.ndarray:
    """
    Apply a Butterworth bandpass filter to isolate heart rate frequencies.
    Default 0.5 Hz - 4.0 Hz corresponds to 30 BPM - 240 BPM.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = signal.butter(order, [low, high], btype="bandpass", output="sos")
    return signal.sosfiltfilt(sos, data)


def moving_average(data: np.ndarray, window_size: int = 5) -> np.ndarray:
    """Apply a simple moving average filter to smooth the signal."""
    if window_size <= 1:
        return data.copy()
    kernel = np.ones(window_size) / window_size
    return np.convolve(data, kernel, mode="same")


def normalize_ac_dc(
    raw_data: np.ndarray,
    dc_cutoff: float = 0.3,
    fs: float = 100.0,
) -> np.ndarray:
    """
    Divide the AC component (heart beats) by the DC baseline component
    to account for finger pressure and light intensity variations.
    """
    nyq = 0.5 * fs
    sos_lp = signal.butter(2, dc_cutoff / nyq, btype="lowpass", output="sos")
    dc_component = signal.sosfiltfilt(sos_lp, raw_data)
    # Prevent division by zero
    dc_safe = np.where(np.abs(dc_component) < 1e-6, 1e-6, dc_component)
    ac_component = raw_data - dc_component
    return ac_component / dc_safe


"""
Unit tests for PPG filtering and heart rate estimation algorithms.
"""

import numpy as np

from src.algorithms.peak_detection import estimate_hr_peaks
from src.algorithms.spectral_analysis import estimate_hr_spectral
from src.filters.bandpass import butter_bandpass_filter, moving_average, normalize_ac_dc
from src.utils.data_loader import generate_synthetic_ppg
from src.utils.plotting import generate_all_report_graphs


def test_end_to_end_hr_estimation():
    target_bpm = 75.0
    fs = 100.0
    _, raw_ppg = generate_synthetic_ppg(bpm=target_bpm, duration_sec=20.0, fs=fs)

    # 1. Normalize AC/DC and bandpass filter (0.5 - 4.0 Hz)
    normalized = normalize_ac_dc(raw_ppg, fs=fs)
    filtered = butter_bandpass_filter(normalized, lowcut=0.5, highcut=4.0, fs=fs)
    smoothed = moving_average(filtered, window_size=5)

    # 2. Method 1: Peak Detection
    bpm_peaks, peaks = estimate_hr_peaks(smoothed, fs=fs)
    assert len(peaks) > 10
    assert np.isclose(bpm_peaks, target_bpm, atol=2.0)

    # 3. Method 2: Spectral Analysis
    bpm_spectral, _, _ = estimate_hr_spectral(smoothed, fs=fs)
    assert np.isclose(bpm_spectral, target_bpm, atol=3.0)


def test_report_graphs_generation(tmp_path):
    paths = generate_all_report_graphs(output_dir=tmp_path)
    assert len(paths) == 4
    for p in paths:
        assert p.exists()
        assert p.stat().st_size > 1000



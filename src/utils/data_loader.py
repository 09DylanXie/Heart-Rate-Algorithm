"""
Data loading utilities for STM32 STLink memory dumps and synthetic PPG signal generation.
"""

from pathlib import Path
import numpy as np
import pandas as pd


def generate_synthetic_ppg(
    bpm: float = 72.0,
    duration_sec: float = 10.0,
    fs: float = 100.0,
    noise_std: float = 0.05,
    baseline_drift_amp: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic PPG signal with baseline wander and high-frequency noise
    for testing filter and heart rate algorithms.
    """
    rng = np.random.default_rng(42)
    t = np.arange(0, duration_sec, 1.0 / fs)
    freq_hz = bpm / 60.0

    # Fundamental pulse wave + dicrotic notch harmonic
    pulse = np.sin(2 * np.pi * freq_hz * t) + 0.3 * np.sin(4 * np.pi * freq_hz * t + 0.5)
    # Low-frequency respiratory baseline wander (~0.15 Hz)
    baseline = 10.0 + baseline_drift_amp * np.sin(2 * np.pi * 0.15 * t)
    # High-frequency sensor noise
    noise = rng.normal(0.0, noise_std, size=len(t))

    return t, baseline + pulse + noise


def load_stm32_dump_csv(filepath: str | Path, column: str | int = 0) -> np.ndarray:
    """
    Load raw sensor values exported from an STM32 memory dump CSV/text file.
    """
    df = pd.read_csv(filepath)
    if isinstance(column, int):
        return df.iloc[:, column].to_numpy(dtype=float)
    return df[column].to_numpy(dtype=float)


"""Module for functions pertaining to reading sample data"""

__version__ = "1.0"
__date__ = "November 17th, 2026"


from dataclasses import dataclass, field

import csv
import numpy as np

# -------- Structures --------

# Single Sample
@dataclass
class Sample:
    name: str
    sample_num: int
    actual_hr: int
    estimated_hr: int
    data: np.ndarray

# Set of Samples
@dataclass
class SampleSet:
    """All rows stored column-wise as NumPy arrays.

    names:        (N,)           str
    sample_nums:  (N,)           int
    actual_hr:    (N,)           int
    estimated_hr: (N,)           int
    data:         (N, data_len)  float
    """
    names: np.ndarray
    sample_nums: np.ndarray
    actual_hr: np.ndarray
    estimated_hr: np.ndarray
    data: np.ndarray

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, i: int) -> Sample:
        return Sample(
            name=str(self.names[i]),
            sample_num=int(self.sample_nums[i]),
            actual_hr=int(self.actual_hr[i]),
            estimated_hr=int(self.estimated_hr[i]),
            data=self.data[i],
        )

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def to_list(self) -> list[Sample]:
        """Array of Sample structs, as in the original version."""
        return list(self)

    @property
    def hr_error(self) -> np.ndarray:
        """estimated_hr - actual_hr for every sample."""
        return self.estimated_hr - self.actual_hr

# -------- Functions --------

# FIXME: AI Code, don't know what it does
# def generate_synthetic_ppg(
#     bpm: float = 72.0,
#     duration_sec: float = 10.0,
#     fs: float = 100.0,
#     noise_std: float = 0.05,
#     baseline_drift_amp: float = 0.5,
# ) -> tuple[np.ndarray, np.ndarray]:
#     """
#     Generate a synthetic PPG signal with baseline wander and high-frequency noise
#     for testing filter and heart rate algorithms.
#     """
#     rng = np.random.default_rng(42)
#     t = np.arange(0, duration_sec, 1.0 / fs)
#     freq_hz = bpm / 60.0

#     # Fundamental pulse wave + dicrotic notch harmonic
#     pulse = np.sin(2 * np.pi * freq_hz * t) + 0.3 * np.sin(4 * np.pi * freq_hz * t + 0.5)
#     # Low-frequency respiratory baseline wander (~0.15 Hz)
#     baseline = 10.0 + baseline_drift_amp * np.sin(2 * np.pi * 0.15 * t)
#     # High-frequency sensor noise
#     noise = rng.normal(0.0, noise_std, size=len(t))

#     return t, baseline + pulse + noise

# returns if a list is empty/blank
def _is_blank(row: list[str]) -> bool:
    return not row or all(cell.strip() == "" for cell in row)

# counts rows in a csv file (excluding header line)
def count_rows(csv_path: str) -> int:
    count = 0
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # header
        for row in reader:
            if not _is_blank(row):
                count += 1
    return count

# Function to read sample data from CSV
# fileName <-- name of the file to read from
# data_len <-- length of the sample data (Ex: 64sps * 10s = 512)
# dtype <-- Data type of the data, float64 by default
# Returns structure containing all of the samples
def read_samples(file_name: str, data_len: int, dtype=np.float64) -> SampleSet:
    # Expected columns (in order):
    #    name, sample_num, actual_hr, estimated_hr, then `data_len` floats.

    num_meta_cols = 4 # number of columns utilized for describing the proceeding data
    n = count_rows(file_name)
    expected = num_meta_cols + data_len

    # Create empty numpy arrays for each type of data
    names: list[str] = []
    sample_nums = np.empty(n, dtype=np.int64)
    actual_hr = np.empty(n, dtype=np.int64)
    estimated_hr = np.empty(n, dtype=np.int64)
    data = np.empty((n, data_len), dtype=dtype)  # preallocated using the row count

    with open(file_name, newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        i = 0
        for line_num, row in enumerate(reader, start=2):
            if _is_blank(row):
                continue
            if len(row) != expected:
                raise ValueError(
                    f"Line {line_num}: expected {expected} columns "
                    f"({num_meta_cols} + {data_len} data), got {len(row)}"
                )
            names.append(row[0].strip())
            sample_nums[i] = int(row[1])
            actual_hr[i] = int(float(row[2]))
            estimated_hr[i] = int(float(row[3]))
            data[i] = np.asarray(row[num_meta_cols:], dtype=dtype)
            i += 1

    return SampleSet(
        names=np.array(names, dtype=str),
        sample_nums=sample_nums,
        actual_hr=actual_hr,
        estimated_hr=estimated_hr,
        data=data,
    )
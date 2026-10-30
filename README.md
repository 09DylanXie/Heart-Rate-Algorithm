# Heart Rate Algorithm for Raw Sensor Data (SciPy Project)

**Group Members:**
1. **Dylan Xie** — Project setup, SciPy implementation, algorithm testing & refinement
2. **Xander Oglesby-Hall** — Algorithm research, parameter optimization (`src/filters/`, `src/algorithms/`), control pulse oximeter testing
3. **Ryan Rau** — Algorithm research, STM32 memory dump data collection over STLink (`data/raw/`), data conversion, STM32 C port

---

## Project Structure

To prevent merge conflicts when all 3 collaborators are pushing code, each subsystem lives in its own isolated module:

```text
ppg-heart-rate-algorithm/
├── .github/workflows/ci.yml   # Automated pytest & ruff check on every push/PR
├── .gitattributes             # Cross-platform line-ending normalization (Mac/Windows/Linux)
├── .gitignore                 # Ignores venv/, __pycache__/, IDE configs, and temp files
├── LICENSE                    # MIT License
├── requirements.txt           # Pinned SciPy, NumPy, Matplotlib, Pandas, Pytest
├── data/
│   ├── raw/                   # Raw STM32 STLink memory dumps (Ryan)
│   └── processed/             # Cleaned SciPy-ready CSVs
├── src/
│   ├── filters/
│   │   └── bandpass.py        # High/low/bandpass filters (0.5-4Hz), moving avg, AC/DC norm
│   ├── algorithms/
│   │   ├── peak_detection.py  # Method 1: Time-domain peak detection
│   │   └── spectral_analysis.py # Method 2: Frequency-domain FFT/Periodogram
│   └── utils/
│       └── data_loader.py     # STM32 CSV loader & synthetic PPG signal generator
└── tests/
    └── test_pipeline.py       # Automated tests for accuracy & precision verification
```

---

## Quick Start (For All Collaborators)

### 1. Clone & Set Up Virtual Environment
```bash
git clone https://github.com/09DylanXie/Heart-Rate-Algorithm.git
cd Heart-Rate-Algorithm

# Create a local virtual environment (ignored by git)
python3 -m venv .venv

# Activate it:
# On macOS/Linux:
source .venv/bin/activate
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Tests Before Pushing
```bash
PYTHONPATH=. pytest
```

---

## Git Workflow for 3 Collaborators (Zero Push Errors)

To ensure nobody runs into push rejections or messy merge conflicts:

1. **Never push directly to `main`** (enable Branch Protection on GitHub for `main`).
2. **Work on feature branches** named after your task:
   - `git checkout -b dylan/peak-detection`
   - `git checkout -b xander/filter-optimization`
   - `git checkout -b ryan/stm32-data-import`
3. **Pull before starting work & before pushing:**
   ```bash
   git fetch origin
   git rebase origin/main
   ```
4. **Open a Pull Request (PR)** on GitHub — the included GitHub Action (`.github/workflows/ci.yml`) will automatically run `pytest` and `ruff` to verify nothing breaks before merging!


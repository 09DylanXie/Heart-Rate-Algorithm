# To run this main file, paste the following
# python3 src/main.py data/processed/Samples.csv 512

from utils.data_loader import read_samples

if __name__ == "__main__":
    import sys

    path, n = sys.argv[1], int(sys.argv[2])
    ss = read_samples(path, n)
    print(f"{len(ss)} samples, data shape {ss.data.shape}")
    print("mean HR error:", ss.hr_error.mean() if len(ss) else "n/a")
    for s in ss:
        print(f"{s.name} #{s.sample_num}: actual={s.actual_hr}, "
              f"estimated={s.estimated_hr}, n={len(s.data)}")

        
import os
import pandas as pd

if __name__ == "__main__":
    dir_ = "hpc/results/arora/benchmarks"
    files = os.listdir(dir_)
    files = [x for x in files if "batch=32" in x]

    files.sort()
    for file_ in files:

        df = os.path.join(dir_, file_)
        df = pd.read_csv(df)
        max_test_acc = df.test_accuracy.max()
        print(f"{file_}\t Max Accuracy:\t{max_test_acc}")

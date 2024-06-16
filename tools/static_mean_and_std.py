import os
import numpy as np
from tqdm import tqdm
import json

def run(data_file):
    with open(data_file, "r") as f:
        lines = f.readlines()
        paths = []
        for line in tqdm(lines, desc="processing"):
            line = line.strip()
            info = json.loads(line)
            patch_path = info["patch_path"]
            data = np.load(patch_path)
            paths.append(data)
    
    paths = np.concatenate(paths, axis=1)
    print(f"mean = {np.mean(paths, axis=1)}")
    print(f"std = {np.std(paths, axis=1)}")

if __name__ == "__main__":
    data_file = "data2/train.jsonl"
    run(data_file)
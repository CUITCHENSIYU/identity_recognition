import os
import numpy as np
from glob import glob
from tqdm import tqdm
from scipy import signal
from utils.preprocess import descale
from utils.filter import filter

def read_data(data_dir, prefix):
    data_dir = os.path.join(data_dir, prefix)
    file_paths = []
    for file_path in glob(data_dir+"*.txt"):
        file_paths.append(file_path)
    if len(file_paths)>1:
        file_paths = sorted(file_paths, key=lambda x: int(x.split("/")[-1].split(".")[0].split("_")[-1]))
    
    arrs = []
    for file_path in file_paths:
        arr = np.genfromtxt(file_path, delimiter=" ")
        arrs.append(arr)
    if len(arrs) > 1:
        data = np.stack([arrs], axis=0)
        data = data.squeeze()
        data = descale(data)
        # data = filter(data)
    else:
        data = arrs[0]
    
    return data.squeeze()

def run(data_file):
    with open(data_file, "r") as f:
        lines = f.readlines()
        paths = []
        for line in tqdm(lines, desc="processing"):
            line = line.strip() # subject
            for data_type in ["by", "ssvep"]: # type
                data_dir = os.path.join(line, data_type)
                if os.path.exists(data_dir) == False:
                    continue
                data = read_data(data_dir, "Channel")
                paths.append(data)
    
    paths = np.concatenate(paths, axis=1)
    print(f"mean = {np.mean(paths, axis=1)}")
    print(f"std = {np.std(paths, axis=1)}")

if __name__ == "__main__":
    data_file = "/mnt/gpuserver-1-disk0-nfs/chensiyu/identity_recognition/data/train.txt"
    run(data_file)
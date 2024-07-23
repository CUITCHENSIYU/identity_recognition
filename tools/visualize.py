import matplotlib.pyplot as plt
import numpy as np
import os
from glob import glob
from identity_recognition.utils.preprocess import descale
from identity_recognition.utils.filter import filter

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
    else:
        data = arrs[0]
    return data.squeeze()

def visualize(data, save_path):
    ch_num = data.shape[-1]
    plt.figure(figsize=(100, 20))
    subtitle = [str(i) for i in range(1, ch_num+1)]

    for i in range(0, data.shape[0]):
        plt.plot( data[i, :])
        plt.xlabel("time [sec]")
        plt.ylabel("amplitude")
        plt.grid(linestyle='-')
        plt.title(subtitle[i])
        plt.legend()
    plt.tight_layout()       
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    lines = [
        "test_data/real_time_data/2024-07-21-16-51-07.npy",
        "test_data/real_time_data/2024-07-21-16-51-17.npy",
        "test_data/real_time_data/2024-07-21-16-51-33.npy",
        "test_data/real_time_data/2024-07-21-16-51-59.npy",
        "test_data/real_time_data/2024-07-21-16-52-10.npy",
        "test_data/real_time_data/2024-07-21-16-52-20.npy",
        "test_data/real_time_data/2024-07-21-16-52-30.npy",
        "test_data/real_time_data/2024-07-21-16-52-44.npy",
        "test_data/real_time_data/2024-07-21-16-55-40.npy",
        "test_data/real_time_data/2024-07-21-16-55-54.npy",
        "test_data/real_time_data/2024-07-21-16-56-05.npy",
        "test_data/real_time_data/2024-07-21-16-56-17.npy",
        "test_data/real_time_data/2024-07-21-16-56-29.npy",
        "test_data/real_time_data/2024-07-21-16-56-39.npy"
    ]
    for line in lines:
        save_path = line.split("/")[-1].replace("npy", "jpg")
        data = np.load(line)
        visualize(data, save_path)

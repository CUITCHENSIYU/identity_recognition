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
    data = np.load("/mnt/gpuserver-1-disk0-nfs/chensiyu/identity_recognition/data/zqy/zqy5/ssvep/data/14.npy")
    visualize(data, "1.jpg")

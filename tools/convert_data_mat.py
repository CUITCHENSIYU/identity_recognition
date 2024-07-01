'''
convert raw data to train/val dataset
'''
import os
from tqdm import tqdm
from glob import glob
import numpy as np
import argparse
import yaml
import json
import subprocess
import shutil
import platform
import scipy.io as sio
from identity_recognition.utils.sliding import sliding_window

system_name = platform.system()

class ConvertData():
    def __init__(self, cfg) -> None:
        self.win_size = cfg["data"]["win_size"]
        self.step_size = cfg["data"]["win_size"]
        self.channel_num = cfg["general"]["input_channel"]
        self.data_file = cfg["data_file"]
        self.save_file = cfg["save_file"]
        self.low_freq = cfg["data"]["low_freq"]
        self.high_freq = cfg["data"]["high_freq"]
        self.sample_rate = cfg["data"]["sample_rate"]

        self.class_id_map = cfg["class_id_map"]

    def read_data(self, data_path):
        data = sio.loadmat(data_path)
        data = data["EEG"][0][0][1]

        data = data[:, 10000:]
        mask = data > 100
        data[mask] = 0

        return data

    def prepare_data(self, data, data_dir):
        for cls_id, cls_name in self.class_id_map.items():
            if cls_name in data_dir:
                break
        
        patchs = sliding_window(data, self.win_size, self.step_size)
        targets = [cls_id for i in range(len(patchs))]
        
        patch_dir = os.path.join(data_dir, "data")
        target_dir = os.path.join(data_dir, "label")
        if os.path.exists(patch_dir) == False:
            os.makedirs(patch_dir)
        else:
            if system_name == "Linux":
                command = f"rm -rf {patch_dir}/*"
                result = subprocess.run(command, shell=True, capture_output=True, text=True)
                print(f"remove exist patch file: {result}")
            else:
                shutil.rmtree(patch_dir)
                print(f"remove exist patch dir {patch_dir}")
                os.makedirs(patch_dir)
        if os.path.exists(target_dir) == False:
            os.makedirs(target_dir)
        else:
            if system_name == "Linux":
                command = f"rm -rf {target_dir}/*"
                result = subprocess.run(command, shell=True, capture_output=True, text=True)
                print(f"remove exist target file: {result}")
            else:
                shutil.rmtree(target_dir)
                print(f"remove exist patch dir {target_dir}")
                os.makedirs(target_dir)
                
        paths = []
        for i in range(len(patchs)):
            patch_path = os.path.join(patch_dir, str(i)+".npy")
            target_path = os.path.join(target_dir, str(i)+".txt")
            np.save(patch_path, patchs[i])
            with open(target_path, "w+") as f:
                f.write(str(targets[i]))
            paths.append({"patch_path": patch_path,
                          "target_path": target_path})
        return paths
    
    def run(self, ):
        with open(self.data_file, "r") as f:
            lines = f.readlines()
            paths = []
            for line in tqdm(lines, desc="processing"):
                data_dir = line.strip() # subject
                
                data_path = os.path.join(data_dir, "filter_data.mat")
                if os.path.exists(data_path) == False:
                    continue
                    
                data = self.read_data(data_path)

                assert data.shape[0] == self.channel_num

                paths_tmp = self.prepare_data(data, data_dir)
                paths.extend(paths_tmp)

        with open(self.save_file, "w+") as f:
            for path_ in paths:
                json.dump(path_, f)
                f.write('\n')

parser = argparse.ArgumentParser(description="Training")
parser.add_argument("--config_path", default="./config/config.yaml", type=str)
parser.add_argument("--data_file", default="", type=str)
parser.add_argument("--save_file", default="", type=str)
args = parser.parse_args()

if __name__ == "__main__":
    cfg_file = open(args.config_path, 'r')
    cfg = yaml.safe_load(cfg_file)
    cfg["data_file"] = args.data_file
    cfg["save_file"] = args.save_file
    convert_mapper = ConvertData(cfg)
    convert_mapper.run()

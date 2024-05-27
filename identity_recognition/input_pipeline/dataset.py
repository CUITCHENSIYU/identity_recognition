from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import numpy as np
import json

from identity_recognition.utils.registry import register_module
from identity_recognition.utils.preprocess import normlize
from identity_recognition.utils.filter import filter

@register_module(parent="input_pipelines")
def base_dataset(cfg, split):
    batch_size = cfg['general']['batch_size']
    if split =='train':
        shuffle = True
    else:
        shuffle = False
    
    dataset = DataHelper(cfg, split)
    data_loader = DataLoader(dataset=dataset, 
                             batch_size=batch_size, 
                             shuffle=shuffle, 
                             drop_last=True, 
                             num_workers=4)
    return data_loader

class DataHelper(Dataset):
    def __init__(self, cfg, split):
        self.cfg = cfg
        self.mean = cfg['data']['mean']
        self.std = cfg['data']['std']
        self.enable_filter = cfg["data"]["enable_filter"]
        self.sample_rate = cfg["data"]["sample_rate"]
        self.low_freq = cfg["data"]["low_freq"]
        self.high_freq = cfg["data"]["high_freq"]
        file_path = cfg[split]["dataset_path"]
        with open(file_path, "r") as f:
            self.infos = f.readlines()
    
    def to_tensor(self, data):
        return torch.tensor(data,dtype=torch.float)
    
    def prepare_input(self, data_path):
        data = np.load(data_path, allow_pickle=True)
        if self.enable_filter:
            data = np.concatenate([data, data], axis=1)
            data = filter(data, self.low_freq, self.high_freq, self.sample_rate)
            data = data[:, int(data.shape[1]/2):]
        data = normlize(data, self.mean, self.std)
        data = self.to_tensor(data)
        return data
    
    def prepare_target(self, target_path):
        with open(target_path, 'r') as f:
            target = [float(f.read())]
        target = self.to_tensor(target)
        return target.squeeze()

    def __getitem__(self, index):
        info = self.infos[index]
        info = json.loads(info)
        data_path = info["patch_path"]
        target_path = info["target_path"]
        data = self.prepare_input(data_path)
        target = self.prepare_target(target_path)

        return data, target

    def __len__(self):
        return len(self.infos)
    

if __name__ == "__main__":
    cfg = dict()
    split = "train"
    cfg['general'] = {}
    cfg['general']['batch_size'] = 16
    cfg['data'] = {}
    cfg['data']['mean'] = [1.37709322e-03, -1.25590711e-04, -9.78099246e-04, -1.54149900e-03,
                            -3.65268857e-04, -8.14466175e-04,  2.03103179e-05, -5.70506880e-04,
                            -1.06753382e-03, -6.38036276e-04,  4.43398198e-03, -4.42797880e-05,
                            4.58709032e-03, -4.00574380e-04,  1.18625627e-04,  1.28054139e-03,
                            -1.44552181e-03,  4.22923689e-03, -4.25630181e-04,  1.94065065e-03,
                            -1.03658997e-03, -2.67863576e-04, -6.58665125e-04, -1.81927911e-04,
                            -7.28778903e-04, -6.39883455e-04, -6.54817118e-03, -5.36829814e-04,
                            -1.65058066e-03,  5.92751761e-03,  1.98298852e-03,  9.49096881e-04]
    cfg['data']['std'] = [156.21795043, 169.71135015, 140.4069441,  136.95477891, 156.0404305,
                            133.2414394,   72.3127324,  148.45542915, 126.45601414, 142.92104616,
                            103.65128654, 100.07135721, 127.29811678, 123.69611457, 193.43011533,
                            155.93406002, 156.41003176, 257.58660447, 129.15109456, 107.70340885,
                            104.44757708, 101.69828631, 121.41448007, 192.81180237, 130.66332279,
                            173.7997312,  233.96898104, 166.46676249, 141.72560624, 221.56806554,
                            220.0013304,  306.79795205]
    cfg["data"]["enable_filter"] = True
    cfg["data"]["sample_rate"] = 1000
    cfg["data"]["low_freq"] = 5
    cfg["data"]["high_freq"] = 45
    cfg[split] = {}
    cfg[split]["dataset_path"] = "/home/root/workspace/identity_recognition/data/train.jsonl"

    dataloader = base_dataset(cfg, split)
    for i, batch in enumerate(dataloader, 0):
        input, target = batch
        print(input.shape, target.shape)
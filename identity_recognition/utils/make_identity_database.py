import numpy as np
import json
from tqdm import tqdm
import torch
from .preprocess import normlize
from .filter import filter

class MakeIdentityDatabase():
    def __init__(self, cfg):
        self.std = cfg["data"]["std"]
        self.mean = cfg["data"]["mean"]
        self.enable_filter = cfg["data"]["enable_filter"]
        self.low_freq = cfg["data"]["low_freq"]
        self.high_freq = cfg["data"]["high_freq"]
        self.sample_rate = cfg["data"]["sample_rate"]

        sample_size = cfg["inference"]["sample_size"]
        self.identity_file = cfg["inference"]["feature_file"]
        self.valid_identity = cfg["inference"]["valid_identity"]
        self.identity_map = cfg["identity_map"]

        with open(self.identity_file, "r") as f:
            lines = f.readlines()
        self.lines = lines[::sample_size]
    
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
        data = torch.unsqueeze(data, 0)
        return data

    def run(self, onnx_session):
        identity_database = []
        for line in tqdm(self.lines, desc="running identity database maker"):
            line = json.loads(line)
            data_path = line["patch_path"]
            try:
                identity_name = data_path.split("/")[-4]
            except:
                identity_name = data_path.split("\\")[-4]
            identity_id = self.identity_map[identity_name]

            if identity_name not in self.valid_identity:
                continue

            data = self.prepare_input(data_path)

            score, feature = onnx_session.run(None, {'input':data.cpu().numpy()})
            feature = np.squeeze(feature)
            identity_database.append({
                "identity_name": identity_name,
                "identity_id": identity_id,
                "feature": feature
            })
            
        return identity_database
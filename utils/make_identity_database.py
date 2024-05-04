import numpy as np
import json
from tqdm import tqdm
import torch
from utils.preprocess import normlize

class MakeIdentityDatabase():
    def __init__(self, cfg):
        self.std = cfg["data"]["std"]
        self.mean = cfg["data"]["mean"]
        sample_size = cfg["inference"]["sample_size"]
        self.identity_file = cfg["inference"]["feature_file"]
        self.valid_identity = cfg["inference"]["valid_identity"]
        self.identity_map = cfg["inference"]["identity_map"]

        with open(self.identity_file, "r") as f:
            lines = f.readlines()
        self.lines = lines[::sample_size]
    
    def to_tensor(self, data):
        return torch.tensor(data,dtype=torch.float)
    
    def prepare_input(self, data_path):
        data = np.load(data_path, allow_pickle=True)
        data = normlize(data, self.mean, self.std)
        data = self.to_tensor(data)
        data = torch.unsqueeze(data, 0)
        return data

    def run(self, onnx_session):
        identity_database = []
        for line in tqdm(self.lines, desc="running identity database maker"):
            line = json.loads(line)
            data_path = line["patch_path"]
            identity_name = data_path.split("/")[-5]
            identity_id = self.identity_map[identity_name]
            assert identity_name in self.valid_identity, f"identity_name {identity_name} not in {self.valid_identity}"
            data = self.prepare_input(data_path)

            score, feature = onnx_session.run(None, {'input':data.cpu().numpy()})
            feature = np.squeeze(feature)
            identity_database.append({
                "identity_name": identity_name,
                "identity_id": identity_id,
                "feature": feature
            })
            
        return identity_database
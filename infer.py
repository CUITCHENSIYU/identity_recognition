import onnxruntime
import numpy as np
import torch
from collections import Counter
from tqdm import tqdm

from utils.make_identity_database import MakeIdentityDatabase
from utils.filter import filter_multi
from utils.sliding import sliding_window
from utils.preprocess import descale, normlize

class Inference():
    def __init__(self, cfg):
        self.conf_thres = cfg["inference"]["conf_thres"]
        self.static_num = cfg["inference"]["static_num"]
        self.win_size = cfg["data"]["win_size"]
        self.step_size = cfg["data"]["win_size"]
        self.mean = cfg["data"]["mean"]
        self.std = cfg["data"]["std"]
        self.enable_filter = cfg["data"]["enable_filter"]
        self.low_freq = cfg["data"]["low_freq"]
        self.high_freq = cfg["data"]["high_freq"]
        self.sample_rate = cfg["data"]["sample_rate"]

        # load model
        self.onnx_session = onnxruntime.InferenceSession(cfg['inference']['model_path'], 
                                                         providers=['CUDAExecutionProvider',
                                                                    'CPUExecutionProvider', 
                                                                    "CUDAExecutionProvider"])

        identity_feature_map_maker = MakeIdentityDatabase(cfg)
        self.identity_database = identity_feature_map_maker.run(self.onnx_session)
    
    def compute_distance(self, feat1, feat2):
        # feat1 = feat1.squeeze()
        # euclidean_distance = np.linalg.norm(feat1 - feat2)
        # score = 1 / (1 + euclidean_distance)
        dot_product = np.dot(feat1, feat2)
        norm_vector1 = np.linalg.norm(feat1)
        norm_vector2 = np.linalg.norm(feat2)
        score = dot_product / (norm_vector1 * norm_vector2)

        # score = np.matmul(feat1, feat2.T)
        return score

    def match(self, feat):
        results = []
        for i in tqdm(range(len(self.identity_database)), desc="running match"):
            score = self.compute_distance(feat, 
                                           self.identity_database[i]["feature"])
            if score < self.conf_thres:
                continue
            results.append({
                "score": score,
                "identity_name": self.identity_database[i]["identity_name"],
                "identity_id": self.identity_database[i]["identity_id"]
            })
        
        sorted_results = sorted(results, key=lambda x: x["score"], reverse=False)
        sorted_results = sorted_results[:self.static_num]
        return sorted_results
    
    def compute_identity(self, results):
        identity_id_counts = Counter(result["identity_id"] for result in results)
        results_tmp= []
        for identity_id, count in identity_id_counts.items():
            max_score = 0.
            for result in results:
                if identity_id == result["identity_id"]:
                    if max_score < result["score"]:
                        max_score = result["score"]
            results_tmp.append({
                "identity_id": identity_id,
                "count": count,
                "max_score": max_score})
        if len(results_tmp) == 0:
            return {}
        init_result = results_tmp.pop(0)

        identity_id = init_result["identity_id"]
        score = init_result["max_score"]
        max_count = init_result["count"]
        for result in results_tmp:
            if max_count < result["count"]:
                max_count = result["count"]
                identity_id = result["identity_id"]
                score = result["max_score"]
            elif max_count == result["count"]:
                if score < result["max_score"]:
                    max_count = result["count"]
                    identity_id = result["identity_id"]
                    score = result["max_score"]

        return {"id": identity_id,
                "count": max_count,
                "score": score}

    def infer(self, data):
        data = descale(data)
        if self.enable_filter:
            data = filter_multi(data, self.low_freq, self.high_freq, self.sample_rate)
        data = normlize(data, self.mean, self.std)
        patchs = sliding_window(data, self.win_size, self.step_size)

        results = []
        for i, input in enumerate(tqdm(patchs)):
            print(np.mean(input), np.std(input))
            input = torch.tensor(input, dtype=torch.float)
            input = torch.unsqueeze(input, 0)
            score, feat = self.onnx_session.run(None, {'input':input.cpu().numpy()})
            pred = score.argmax(axis=1)
            print(f"pred={pred}")
            results.extend(self.match(feat))
        for res in results:
            print(res)
        identity_map = self.compute_identity(results)
        return identity_map
    
if __name__ =="__main__":
    import yaml
    config_path = "./config/config.yaml"
    cfg_file = open(config_path, 'r')
    cfg = yaml.safe_load(cfg_file)
    inference = Inference(cfg)

    data_files = [
        "/home/root/workspace/identity_recognition/data/yl/yl4/by/data/52.npy",
        "/home/root/workspace/identity_recognition/data/yl/yl4/by/data/53.npy",
        "/home/root/workspace/identity_recognition/data/yl/yl4/by/data/54.npy",
        "/home/root/workspace/identity_recognition/data/yl/yl4/by/data/55.npy",
        "/home/root/workspace/identity_recognition/data/yl/yl4/by/data/56.npy",
    ]
    datas = []
    for files in data_files:
        data = np.load(files)
        print(f"data={np.mean(data)}, {np.std(data)}")
        datas.append(data)
    data = np.concatenate(datas, 1)
    identity_map = inference.infer(data)
    print(identity_map)
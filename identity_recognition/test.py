import onnxruntime
import numpy as np
import statistics as st
import torch
from input_pipeline import build_dataloader
from evaluators import build_evaluator
from model.pipeline import build_pipeline
import time
from tqdm import tqdm

class Test():
    def __init__(self, cfg):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # load data
        self.test_loader = build_dataloader(cfg, 'test')
        self.suffix = cfg['test']['model_path'].split('.')[-1]
        
        if self.suffix=='pt':
            self.model = build_pipeline(cfg)
            self.model.load_state_dict(torch.load(cfg['test']['model_path']))
            self.model = self.model.to(self.device)
            self.model.eval()

        elif self.suffix=='onnx':
            self.onnx_session = onnxruntime.InferenceSession(cfg['test']['model_path'],
                                                              providers=['CUDAExecutionProvider',
                                                                         'CPUExecutionProvider'])
        # init evaluator
        self.evaluator = build_evaluator(cfg['evaluator']['type'], cfg)
    
    def test_onnx(self,):
        times = []
        preds = []
        targets = []
        for i, (data, target) in enumerate(tqdm(self.test_loader)):
            data = data.to(self.device)
            start_time = time.time()
            output, _ = self.onnx_session.run(None, {'input':data.cpu().numpy()})
            end_time = time.time()
            times.append(end_time-start_time)
            output = np.array(output)
            pred = output.argmax(axis=1)

            preds.append(pred)
            targets.append(target)
            
        preds = np.concatenate(preds, axis=0)
        targets = torch.cat(targets, dim=0)
        targets = targets.numpy()
                
        acc = self.evaluator.accuracy(targets,preds)
        print('accuracy: ', acc)
        precision = self.evaluator.precision(targets,preds)
        print('precision: ', precision)
        recall = self.evaluator.recall(targets,preds)
        print('recall: ', recall)
        print("infer time: ", sum(times), np.mean(times))
    
    def test_pth(self,):
        times = []
        preds = []
        targets = []

        with torch.no_grad():
            for i, (data, target) in enumerate(tqdm(self.test_loader)):
                data, target = data.to(self.device), target.to(self.device)
                start_time = time.time()
                output, _ = self.model(data)
                end_time = time.time()
                times.append(end_time-start_time)
                pred = output.argmax(dim=1, keepdim=True)

                preds.append(pred)
                targets.append(target)
            preds = torch.cat(preds, dim=0)
            targets = torch.cat(targets, dim=0)
                    
            preds = preds.cpu().numpy()
            targets = targets.cpu().numpy()
            
            acc = self.evaluator.accuracy(targets,preds)
            print('accuracy: ', acc)
            precision = self.evaluator.precision(targets,preds)
            print('precision: ',precision)
            recall = self.evaluator.recall(targets,preds)
            print('recall: ',recall)
            
            print("infer time: ", sum(times), np.mean(times))
        
    def run(self,):
        if self.suffix=='pt':
            self.test_pth()
        elif self.suffix=='onnx':
            self.test_onnx()
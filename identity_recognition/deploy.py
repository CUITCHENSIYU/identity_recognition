import torch
import onnx
from model.pipeline import build_pipeline

cuda = True if torch.cuda.is_available() else False

class Deploy():
    def __init__(self, cfg):
        self.cfg = cfg
        self.model_path = cfg['deploy']['model_path']
        self.input_channle = cfg['general']['input_channel']
        self.win_size = cfg['data']['win_size']
        self.input_tensor = torch.randn(1, self.input_channle, self.win_size).cuda()
        
    def deploy(self, ):
        model = build_pipeline(self.cfg)
        model.load_state_dict(torch.load(self.model_path))
        model = model.cuda()
        model.eval()
        onnx_path = self.model_path.replace(".pt", ".onnx")
        with torch.no_grad():
            torch.onnx.export(model,
                            self.input_tensor, 
                            onnx_path, 
                            training=torch.onnx.TrainingMode.EVAL,
                            verbose=True,
                            opset_version=11,
                            input_names=['input'],
                            output_names=['output'])
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("[ONNX_EXPORT] ONNX model has been exported to: {}".format(onnx_path))

    def run(self, ):
        self.deploy()
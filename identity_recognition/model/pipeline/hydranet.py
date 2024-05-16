import os
import sys
sys.path.append(os.path.abspath('.'))

import torch
import torch.nn as nn

from model.backbone import build_backbone
from model.head import build_head
from utils.registry import register_module

@register_module(parent="pipeline")
def hydranet(cfg):
    return HydraNet(cfg)

class HydraNet(nn.Module):
    def __init__(self, cfg):
        super(HydraNet, self).__init__()
        self.backbone = build_backbone(cfg)
        self.head = build_head(cfg, self.backbone.feature_dim)
    def forward(self, x):
        feat = self.backbone(x)
        out = self.head(feat)
        return out
    
if __name__ == "__main__":
    input = torch.rand((2, 32, 1000))
    cfg = dict()
    cfg["general"] = {}
    cfg["general"]["n_class"] = 8
    cfg["model"] = {}
    cfg["model"]["pipeline_name"] = "hydranet"
    cfg["model"]["backbone_name"] = "resnet_50"
    cfg["model"]["head_name"] = "classify_head"
    cfg["model"]["feature_dim"] = 512
    model = HydraNet(cfg)
    print(model)
    print(model(input).shape)

import torch
import torch.nn as nn
from torch.nn import functional as F

from identity_recognition.utils.registry import register_module

class ArcNet(nn.Module):
    def __init__(self, feature_num, cls_num):
        super(ArcNet, self).__init__()
        self.w = nn.Parameter(torch.randn((feature_num, cls_num)), requires_grad=True)
        self.func = nn.Softmax()

    def forward(self, x, s=64, m=0.2):
        x_norm = F.normalize(x, dim=1)
        w_norm = F.normalize(self.w, dim=0)
        cosa = torch.matmul(x_norm, w_norm) / s
        a = torch.acos(cosa)
        arcsoftmax = torch.exp(
            s * torch.cos(a + m)) / (torch.sum(torch.exp(s * cosa), dim=1, keepdim=True) - torch.exp(
            s * cosa) + torch.exp(s * torch.cos(a + m)))

        return arcsoftmax

@register_module(parent="head")
def classify_head(cfg, input_dim):
    n_class = cfg['general']['n_class']
    feature_dim = cfg["model"]["feature_dim"]
    head = ClassifyHead(n_class=n_class, 
                        input_dim=input_dim, 
                        feature_dim=feature_dim)
    return head

class ClassifyHead(nn.Module):
    def __init__(self, n_class, input_dim, feature_dim):
        super(ClassifyHead, self).__init__()

        self.feature_net = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.PReLU(),
            nn.Linear(input_dim, feature_dim, bias=False),
        )

        self.arc_net = ArcNet(feature_dim, n_class)

    def forward(self, x):
        feature = self.feature_net(x)
        score = self.arc_net(feature)
        return score, feature

if __name__ == "__main__":
    input = torch.rand((1, 2048))
    model = ClassifyHead(8, 2048, 512)
    output = model(input)
    print(output[0].shape, output[1].shape)
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import torch

from tkinter.messagebox import NO
from turtle import forward
import numpy as np

from identity_recognition.utils.registry import register_module

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

@register_module(parent="criterions")
def CELoss(config):
    return Cross_EntropyLoss(config)

class Cross_EntropyLoss(torch.nn.Module):
    def __init__(self, config):
        super(Cross_EntropyLoss, self).__init__()

        self.loss_weight = None
    def forward(self, output,labels):
        return F.cross_entropy(output,labels.long(), weight=self.loss_weight)


import torch
import torch.nn as nn

from utils.registry import register_module

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

@register_module(parent="criterions")
def NLLLoss(config):
    return NLL_Loss(config)

class NLL_Loss(torch.nn.Module):
    def __init__(self, config):
        super(NLL_Loss, self).__init__()

        self.loss_weight = None
        self.loss_func = nn.NLLLoss()
    def forward(self, output, labels):
        return self.loss_func(torch.log(output), labels)


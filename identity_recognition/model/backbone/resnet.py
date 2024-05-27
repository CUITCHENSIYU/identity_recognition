import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

from identity_recognition.model.backbone.layers import Extractor_log_spec
from identity_recognition.utils.registry import register_module


@register_module(parent="backbone")
def resnet_50(cfg):
    backbone = ResNet50()
    return backbone

class ResNet50(nn.Module):
    def __init__(self, ):
        super(ResNet50, self).__init__()
        self.extractor_log_spec = Extractor_log_spec(n_mels=12,
                                                     sample_rate=1000,
                                                     n_fft=100,
                                                     hop_length=10,
                                                     window_size=100)
        
        resnet = models.resnet50(pretrained=True)
        self.pretrained = nn.Module()
        self.pretrained.layer1 = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1
        )
        self.pretrained.layer2 = resnet.layer2
        self.pretrained.layer3 = resnet.layer3
        self.pretrained.layer4 = resnet.layer4
        self.pretrained.pool = resnet.avgpool

        self.feature_dim = 2048

    def forward(self, x):
        x = self.extractor_log_spec(x)
        for i in range(x.shape[1]):
            if i == 0:
                x_ = x[:,i,:,:]
            else:
                x_ = torch.cat((x_, x[:,i,:,:]), dim=2)
        x = torch.unsqueeze(x_, dim=1)
        x = F.interpolate(x, size=(224, 224))
        x = x.repeat(1,3,1,1)

        layer_1 = self.pretrained.layer1(x)
        layer_2 = self.pretrained.layer2(layer_1)
        layer_3 = self.pretrained.layer3(layer_2)
        layer_4 = self.pretrained.layer4(layer_3)
        feat = self.pretrained.pool(layer_4)
        feat = feat.squeeze(-1)
        return feat.squeeze(-1)


if __name__ == "__main__":
    input = torch.rand((1, 32, 1000))
    model = ResNet50()
    print(model)
    print(model(input).shape)
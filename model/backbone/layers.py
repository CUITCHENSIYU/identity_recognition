import torch.nn as nn
import torch.nn.functional as F
import torch
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
        
class Extractor_log_spec(nn.Module):
    def __init__(self, 
                n_mels=12, 
                sample_rate=1000,
                n_fft=100,
                hop_length=10,
                window="hann",
                window_size = 100,
                center = True,
                pad_mode = 'reflect',
                freeze_parameters = True):
        super(Extractor_log_spec, self).__init__()
        
        self.spectrogram_extractor = Spectrogram(n_fft=n_fft, hop_length=hop_length,
                                                win_length=window_size, window=window, center=center,
                                                pad_mode=pad_mode, freeze_parameters=freeze_parameters)

        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_mels=n_mels, n_fft=n_fft,
                                                freeze_parameters=freeze_parameters)
            
    def forward(self, input):
        channel_num = input.shape[1]
        feats = []  
        for ch_id in range(channel_num):
            x = self.spectrogram_extractor(input[:, ch_id, :])
            x = self.logmel_extractor(x)
            feats.append(x)
            
        feat = torch.concat(feats, dim=1)
        return feat

if __name__ == "__main__":
    x = torch.rand((2, 32, 1000))
    extractor_log_spec = Extractor_log_spec()
    print(extractor_log_spec(x).shape)

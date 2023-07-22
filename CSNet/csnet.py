import torch.nn as nn
import torch
import torchvision.models

class CSNet(nn.Module):
    def __init__(self):
        super(CSNet, self).__init__()
        self.self = self
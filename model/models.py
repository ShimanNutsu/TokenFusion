import torch
import torch.nn as nn

class MixingClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool1 = nn.AdaptiveAvgPool1d(1)
        self.pool2 = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.LazyLinear(100)
    def forward(self, x):
        x1 = self.pool2(x)
        x = torch.transpose(x, 1, 2)
        x2 = self.pool1(x)
        x1 = torch.squeeze(x1, dim=-1)
        x2 = torch.squeeze(x2, dim=-1)
        x = torch.concat([x1, x2], axis=1)
        x = self.classifier(x)
        return x

class TokenWiseClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.LazyLinear(100)
    def forward(self, x):
        x = self.pool(x)
        x = torch.squeeze(x, dim=-1)
        x = self.classifier(x)
        return x


class ChannelWiseClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.LazyLinear(100)
    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        x = self.pool(x)
        x = torch.squeeze(x, dim=-1)
        x = self.classifier(x)
        return x

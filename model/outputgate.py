import torch
from torch import nn

class OutputGateModule(nn.Module):
  def __init__(self, input_channel):
    super().__init__()
    self.conv = nn.Conv2d(input_channel, 1, 3, stride=2, padding=1)
  def forward(self, X1, X2):
    return torch.mul(X2, nn.functional.sigmoid(self.conv(X1)))
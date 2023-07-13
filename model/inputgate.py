import torch
from torch import nn

class InputGateModule(nn.Module):
  def __init__(self, input_channel1, input_channel2):
    super().__init__()
    self.conv1 = nn.Conv2d(input_channel1, input_channel1, 3, padding=1, bias=True)
    self.conv2 = nn.Conv2d(input_channel1, input_channel1, 3, padding=1, bias=True)
    self.conv3 = nn.Conv2d(input_channel1+input_channel2, input_channel2, 3, padding=1, bias=True)

  def forward(self, X1, X2):
    in1 = torch.mul(self.conv1(X1),nn.functional.sigmoid(self.conv2(X1)))
    return self.conv3(torch.cat((in1, X2), dim=1)) # 1 means on channel dimension
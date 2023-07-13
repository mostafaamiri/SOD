import torch
from torch import nn

class MLA(nn.Module):
  def __init__(self, input_channel1 , input_channel2):
    super().__init__()
    self.conv1 = torch.nn.Conv2d(input_channel1, 1, 1)
    self.conv2 = nn.Conv2d(input_channel2, 1, 1, bias=True)
    self.input_channel2 = input_channel2
  def forward(self, X1, X2):
    alpha = torch.reshape(torch.nn.functional.softmax(torch.flatten(torch.nn.functional.tanh(self.conv2(X2)), start_dim=1),dim=1),(X2.shape[0], 1, X2.shape[-2], X2.shape[-1]))
    d_X1  =torch.nn.functional.interpolate(self.conv1(X1), (X2.shape[-2], X2.shape[-1]), align_corners=False, antialias=True, mode='bilinear')
    return X2 + torch.mul(d_X1, alpha).repeat(1,self.input_channel2,1,1)
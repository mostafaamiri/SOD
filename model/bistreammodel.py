from . import InputGateModule, OutputGateModule, MLA
from torchvision.models import resnet50, vgg16
import torch
from torch import nn

class BiStreamModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.R_model = resnet50(weights='IMAGENET1K_V1').cuda()
    self.V_model = vgg16(weights='IMAGENET1K_V1').features.cuda()
    self.igs = [InputGateModule(64, 64).cuda(),
                InputGateModule(128, 256).cuda(),
                InputGateModule(256, 512).cuda(),
                InputGateModule(512, 1024).cuda(),
                InputGateModule(512, 2048).cuda()]
    self.ogs = [OutputGateModule(64).cuda(),
                OutputGateModule(256).cuda(),
                OutputGateModule(512).cuda(),
                OutputGateModule(1024).cuda()]
    self.mlas = [MLA(256, 1024).cuda(),
                 MLA(64, 2048).cuda()]
    self.last_conv = nn.Conv2d(2048, 128, 1)
    self.classifier = nn.Sequential(
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(2048, 16384)
    ).cuda()

  def forward(self, X):
    X_r1 = self.R_model.conv1(X)
    X_v1 = self.V_model[:5](X)
    out1 = self.igs[0](X_v1, X_r1)

    X_r2 = self.R_model.layer1(self.R_model.maxpool(self.R_model.relu(self.R_model.bn1(out1))))
    X_r2 = self.ogs[0](out1, X_r2)
    X_v2 = self.V_model[5:10](X_v1)
    out2 = self.igs[1](X_v2, X_r2)

    X_r3 = self.R_model.layer2(out2)
    X_r3 = self.ogs[1](out2, X_r3)
    X_v3 = self.V_model[10:17](X_v2)
    out3 = self.igs[2](X_v3, X_r3)

    X_r4 = self.R_model.layer3(out3)
    X_r4 = self.ogs[2](out3, X_r4)
    X_v4 = self.V_model[17:24](X_v3)
    out4 = self.igs[3](X_v4, X_r4)
    out4 = self.mlas[0](out2, out4)

    X_r5 = self.R_model.layer4(out4)
    X_r5 = self.ogs[3](out4, X_r5)
    X_v5 = self.V_model[24:31](X_v4)
    out5 = self.igs[4](X_v5, X_r5)

    out  = torch.flatten(self.last_conv(self.mlas[1](out1, out5)), start_dim=1)
    return self.classifier(out)
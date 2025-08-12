# methods/models.py
import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet, BasicBlock
from torch.quantization import QuantStub, DeQuantStub

class TinyResNet18(ResNet):
    def __init__(self, num_classes=200):
        super().__init__(block=BasicBlock, layers=[2,2,2,2], num_classes=num_classes)
        self.conv1 = nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1,bias=False)
        self.maxpool = nn.Identity()
    def forward_features(self, x):
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)
    def forward(self, x):
        x = self.forward_features(x)
        return self.fc(x)

class QATWrapper(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.quant = QuantStub(); self.backbone = backbone; self.dequant = DeQuantStub()
    def forward(self, x):
        x = self.quant(x)
        x = self.backbone(x)
        x = self.dequant(x)
        return x

class QIRPWrapper(nn.Module):
    def __init__(self, backbone: TinyResNet18, survive_idx=None):
        super().__init__()
        self.quant = QuantStub(); self.backbone = backbone; self.dequant = DeQuantStub()
        self.register_buffer("survive_idx", None)
        if survive_idx is not None: self.set_survive_idx(survive_idx)
    def set_survive_idx(self, idx):
        if not isinstance(idx, torch.Tensor):
            idx = torch.as_tensor(idx, dtype=torch.long)
        self.survive_idx = idx.to(torch.long)
    def forward(self, x):
        x = self.quant(x)
        feats = self.backbone.forward_features(x)
        if self.survive_idx is not None:
            feats = feats[:, self.survive_idx]
        logits = self.backbone.fc(feats)
        return self.dequant(logits)
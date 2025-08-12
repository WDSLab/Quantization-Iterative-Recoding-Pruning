# methods/qpsnn.py
import os, copy, torch
import torch.nn as nn, torch.optim as optim
from torchvision.models import resnet18
from torch.quantization import prepare_qat, convert, get_default_qat_qconfig
from .common import evaluate, get_flops, get_params, resolve_device_for_method

class QPResNet18(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        from torch.quantization import QuantStub, DeQuantStub
        self.quant, self.dequant = QuantStub(), DeQuantStub()
    def forward(self, x):
        x = self.quant(x); x = self.model(x); x = self.dequant(x); return x

def run(args, train_loader, val_loader, num_classes):
    torch.backends.quantized.engine = args.qat_backend
    device = resolve_device_for_method(is_quant=True)  # CPU
    base = os.path.join(args.out_dir, "base.pth"); assert os.path.exists(base)

    m = QPResNet18(num_classes).to(device)
    m.model.load_state_dict(torch.load(base, map_location="cpu"))
    m.qconfig = get_default_qat_qconfig(args.qat_backend)
    prepare_qat(m, inplace=True)

    opt = optim.SGD(m.parameters(), lr=0.01, momentum=0.9)
    crit = nn.CrossEntropyLoss()
    best=0.0; best_sd=copy.deepcopy(m.state_dict())

    for ep in range(100):
        m.train()
        for xb,yb in train_loader:
            xb,yb = xb.to(device), yb.to(device)
            opt.zero_grad(); loss = crit(m(xb), yb); loss.backward(); opt.step()
        a1,_ = evaluate(m.eval(), val_loader, device)
        if a1>best: best=a1; best_sd=copy.deepcopy(m.state_dict())

    m.load_state_dict(best_sd)
    last_out = None
    for mod in m.modules():
        if isinstance(mod, nn.Conv2d):
            W = mod.weight.data.cpu().view(mod.out_channels, -1)
            _, S, _ = torch.svd(W)
            k = int(mod.out_channels*args.prune_frac)
            keep = torch.topk(S, mod.out_channels - k).indices
            mod.weight.data = mod.weight.data[keep].to(device)
            mod.out_channels = keep.numel()
            last_out = keep.numel()
    m.model.fc = nn.Linear(last_out, num_classes).to(device)

    for ep in range(100):
        m.train()
        for xb,yb in train_loader:
            xb,yb = xb.to(device), yb.to(device)
            opt.zero_grad(); loss = crit(m(xb), yb); loss.backward(); opt.step()

    int8 = convert(copy.deepcopy(m).cpu(), inplace=False)
    a1,a5 = evaluate(int8.to(device), val_loader, device)
    fl,pa = get_flops(int8), get_params(int8)
    out = os.path.join(args.out_dir, "qpsnn.pth"); torch.save(int8.state_dict(), out)
    print(f"[QPSNN Final] Top1={a1*100:.2f}% FLOPs={fl:.2f}G Params={pa:,} ckpt={out}")

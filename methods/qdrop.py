# methods/qdrop.py
import os, copy, torch
import torch.nn as nn, torch.optim as optim
from torch.quantization import prepare_qat, convert, get_default_qat_qconfig
from .models import TinyResNet18, QATWrapper
from .common import evaluate, get_flops, get_params, resolve_device_for_method, load_float_only

class QDrop(nn.Module):
    def __init__(self, backbone: nn.Module, drop_prob=0.3):
        super().__init__()
        self.model = QATWrapper(backbone)
        self.drop_prob = drop_prob
    def forward(self, x):
        with torch.no_grad():
            for m in self.model.modules():
                if isinstance(m, nn.Conv2d) and torch.rand(1).item() < self.drop_prob:
                    m.weight.mul_(0)
        return self.model(x)

def run(args, train_loader, val_loader, num_classes):
    torch.backends.quantized.engine = args.qat_backend
    device = resolve_device_for_method(is_quant=True)
    base = os.path.join(args.out_dir, "base.pth"); assert os.path.exists(base)

    bb = TinyResNet18(num_classes=num_classes)
    bb.load_state_dict(torch.load(base, map_location="cpu"))
    model = QDrop(bb, drop_prob=0.3).to(device)
    model.model.qconfig = get_default_qat_qconfig(args.qat_backend)
    prepare_qat(model.model, inplace=True)
    load_float_only(model.model, torch.load(base, map_location="cpu"))

    opt = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    crit = nn.CrossEntropyLoss()
    best=0.0; best_sd=copy.deepcopy(model.state_dict()); out_best=os.path.join(args.out_dir, "qdrop_best.pth")

    it = iter(train_loader)
    for step in range(200):
        try: xb,yb = next(it)
        except StopIteration: it = iter(train_loader); xb,yb = next(it)
        xb,yb = xb.to(device), yb.to(device)
        opt.zero_grad(); loss = crit(model(xb), yb); loss.backward(); opt.step()
        if step % 100 == 0:
            a1,a5 = evaluate(model, val_loader, device)
            print(f"[QDrop] step{step} Top1={a1*100:.2f}%")
            if a1>best: best=a1; best_sd=copy.deepcopy(model.state_dict()); torch.save(best_sd, out_best)

    for step in range(100):
        try: xb,yb = next(it)
        except StopIteration: it = iter(train_loader); xb,yb = next(it)
        xb,yb = xb.to(device), yb.to(device)
        opt.zero_grad(); loss = crit(model(xb), yb); loss.backward(); opt.step()

    model.load_state_dict(best_sd)
    int8 = convert(copy.deepcopy(model.model).cpu(), inplace=False)
    a1,a5 = evaluate(int8.to(device), val_loader, device)
    fl,pa = get_flops(int8), get_params(int8)
    out = os.path.join(args.out_dir, "qdrop_int8.pth"); torch.save(int8.state_dict(), out)
    print(f"[QDrop Final] Top1={a1*100:.2f}% FLOPs={fl:.2f}G Params={pa:,} ckpt={out}")

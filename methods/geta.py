# methods/geta.py
import os, copy, torch
import torch.nn as nn, torch.optim as optim
from torchvision.models import resnet18
from .common import evaluate, get_flops, get_params, resolve_device_for_method

def run(args, train_loader, val_loader, num_classes):
    device = resolve_device_for_method(is_quant=False)
    base = os.path.join(args.out_dir, "base.pth")
    assert os.path.exists(base)
    m = resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    m.load_state_dict(torch.load(base, map_location="cpu"))
    m = m.to(device)

    m.train()
    xb, yb = next(iter(train_loader))
    xb, yb = xb.to(device), yb.to(device)
    loss = nn.CrossEntropyLoss()(m(xb), yb); loss.backward()

    for mod in m.modules():
        if isinstance(mod, nn.Conv2d) and mod.weight.grad is not None:
            imp = (mod.weight.grad * mod.weight).abs().sum(dim=(1,2,3))
            k = int(imp.numel()*args.prune_frac)
            keep = imp.topk(imp.numel()-k).indices
            mod.weight.data = mod.weight.data[keep]
            mod.out_channels = keep.numel()

    last_out = [mod for mod in m.modules() if isinstance(mod, nn.Conv2d)][-1].out_channels
    m.fc = nn.Linear(last_out, num_classes).to(device)

    opt = optim.SGD(m.parameters(), lr=0.01, momentum=0.9)
    best = copy.deepcopy(m.state_dict()); best_acc=0.0
    for ep in range(100):
        m.train()
        for xb,yb in train_loader:
            xb,yb = xb.to(device), yb.to(device)
            opt.zero_grad(); l = nn.CrossEntropyLoss()(m(xb), yb); l.backward(); opt.step()
        a1,a5 = evaluate(m.eval(), val_loader, device)
        if a1>best_acc: best_acc=a1; best=copy.deepcopy(m.state_dict())
    m.load_state_dict(best)
    out = os.path.join(args.out_dir, "geta.pth"); torch.save(m.state_dict(), out)
    print(f"[GETA] FLOPs={get_flops(m):.2f}G Params={get_params(m):,} ckpt={out}")

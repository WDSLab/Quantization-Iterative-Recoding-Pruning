# methods/structalign.py
import os, copy, torch
import torch.nn as nn, torch.optim as optim
from .models import TinyResNet18
from .common import evaluate, get_flops, get_params, resolve_device_for_method

def apply_structalign(m: nn.Module, ratio: float):
    prev_idx = None
    with torch.no_grad():
        for module in m.modules():
            if isinstance(module, nn.Conv2d):
                W = module.weight.data.abs()
                imp = W.view(W.size(0), -1).sum(dim=1)
                k = int(W.size(0)*ratio)
                _, idx = torch.topk(imp, W.size(0)-k, largest=True)
                if prev_idx is not None and module.in_channels == len(prev_idx):
                    module.weight.data = module.weight.data[idx][:, prev_idx]
                else:
                    module.weight.data = module.weight.data[idx]
                module.out_channels = len(idx)
                prev_idx = idx
            elif isinstance(module, nn.BatchNorm2d):
                if prev_idx is not None and module.num_features == len(prev_idx):
                    module.weight.data = module.weight.data[prev_idx]
                    module.bias.data   = module.bias.data[prev_idx]
                    module.running_mean = module.running_mean[prev_idx]
                    module.running_var  = module.running_var[prev_idx]
                    module.num_features = len(prev_idx)

def rebuild_fc(model, num_classes, device):
    with torch.no_grad():
        dummy = torch.randn(1,3,64,64).to(device)
        feats = model.forward_features(dummy)
        in_ch = feats.shape[1]
        model.fc = nn.Linear(in_ch, num_classes).to(device)

def run(args, train_loader, val_loader, num_classes):
    device = resolve_device_for_method(is_quant=False)
    base = os.path.join(args.out_dir, "base.pth")
    assert os.path.exists(base)
    m = TinyResNet18(num_classes=num_classes).to(device)
    m.load_state_dict(torch.load(base, map_location=device))
    apply_structalign(m, args.prune_frac)
    rebuild_fc(m, num_classes, device)

    opt = optim.SGD(m.parameters(), lr=0.01, momentum=0.9)
    crit = nn.CrossEntropyLoss()
    best = copy.deepcopy(m.state_dict()); best_acc=0.0
    for ep in range(100):
        m.train()
        for xb,yb in train_loader:
            xb,yb = xb.to(device), yb.to(device)
            opt.zero_grad(); loss = crit(m(xb), yb); loss.backward(); opt.step()
        a1,a5 = evaluate(m.eval(), val_loader, device)
        if a1>best_acc: best_acc=a1; best=copy.deepcopy(m.state_dict())
    m.load_state_dict(best)
    out = os.path.join(args.out_dir, "structalign.pth")
    torch.save(m.state_dict(), out)
    print(f"[StructAlign] FLOPs={get_flops(m):.2f}G Params={get_params(m):,} ckpt={out}")

# methods/prune.py
import os, copy, torch
import torch.nn as nn, torch.optim as optim
import torch.nn.utils.prune as prune
from .models import TinyResNet18
from .common import evaluate, get_flops, get_params, resolve_device_for_method, save_if_best

def _apply_structured_pruning(model: nn.Module, amount: float):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            prune.ln_structured(m, name="weight", amount=amount, n=2, dim=0)
            prune.remove(m, "weight")

def run(args, train_loader, val_loader, num_classes):
    device = resolve_device_for_method(is_quant=False)
    base_path = os.path.join(args.out_dir, "base.pth")
    assert os.path.exists(base_path), "Run baseline first."
    model = TinyResNet18(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(base_path, map_location=device))

    _apply_structured_pruning(model, amount=args.prune_frac)

    opt = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    crit = nn.CrossEntropyLoss()
    best = {"acc1":0.0, "sd":None}
    out = os.path.join(args.out_dir, f"pruned_ft_{int(args.prune_frac*100)}.pth")

    for ep in range(100):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); loss = crit(model(xb), yb); loss.backward(); opt.step()
        if (ep+1) % 25 == 0:
            acc1, acc5 = evaluate(model, val_loader, device)
            print(f"[Prune] {ep+1}/100  Top1={acc1*100:.2f}%  Top5={acc5*100:.2f}%")
            save_if_best(model, acc1, best, out)

    model.load_state_dict(torch.load(out, map_location=device))
    print(f"[Prune Summary] FLOPs={get_flops(model):.2f}G  Params={get_params(model):,}  ckpt={out}")


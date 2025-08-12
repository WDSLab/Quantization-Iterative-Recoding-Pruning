# methods/baseline.py
import os, torch
import torch.nn as nn, torch.optim as optim
from .models import TinyResNet18
from .common import evaluate, get_flops, get_params, resolve_device_for_method, save_if_best

def run(args, train_loader, val_loader, num_classes):
    device = resolve_device_for_method(is_quant=False)
    model = TinyResNet18(num_classes=num_classes).to(device)
    opt = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
    sch = optim.lr_scheduler.MultiStepLR(opt, milestones=[60,120,160], gamma=0.2)
    crit = nn.CrossEntropyLoss()
    best = {"acc1":0.0, "sd":None}
    out = os.path.join(args.out_dir, "base.pth")
    for ep in range(args.epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); loss = crit(model(xb), yb); loss.backward(); opt.step()
        sch.step()
        acc1, acc5 = evaluate(model, val_loader, device)
        print(f"[Baseline] {ep+1}/{args.epochs}  Top1={acc1*100:.2f}%  Top5={acc5*100:.2f}%")
        save_if_best(model, acc1, best, out)
    model.load_state_dict(torch.load(out, map_location=device))
    print(f"[Baseline Summary] FLOPs={get_flops(model):.2f}G  Params={get_params(model):,}  ckpt={out}")

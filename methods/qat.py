# methods/qat.py
import os, copy, torch
import torch.nn as nn, torch.optim as optim
from torch.quantization import prepare_qat, convert, get_default_qat_qconfig
from .models import TinyResNet18, QATWrapper
from .common import evaluate, get_flops, get_params, resolve_device_for_method, load_float_only

def run(args, train_loader, val_loader, num_classes):
    torch.backends.quantized.engine = args.qat_backend
    device = resolve_device_for_method(is_quant=True)
    base = os.path.join(args.out_dir, "base.pth"); assert os.path.exists(base)

    backbone = TinyResNet18(num_classes=num_classes)
    backbone.load_state_dict(torch.load(base, map_location="cpu"))
    model = QATWrapper(backbone).to(device)
    model.qconfig = get_default_qat_qconfig(args.qat_backend)
    prepare_qat(model, inplace=True)
    load_float_only(model, torch.load(base, map_location="cpu"))

    opt = optim.SGD(model.parameters(), lr=args.qat_lr, momentum=0.9, weight_decay=5e-4)
    crit = nn.CrossEntropyLoss()
    best_acc=0.0; best_sd=None; best_path = os.path.join(args.out_dir, "qat_best.pth")

    for ep in range(args.qat_epochs):
        model.train()
        for xb,yb in train_loader:
            xb,yb = xb.to(device), yb.to(device)
            opt.zero_grad(); loss = crit(model(xb), yb); loss.backward(); opt.step()
        a1,a5 = evaluate(model, val_loader, device)
        print(f"[QAT] {ep+1}/{args.qat_epochs} Top1={a1*100:.2f}% Top5={a5*100:.2f}%")
        if a1>best_acc: best_acc=a1; best_sd=copy.deepcopy(model.state_dict()); torch.save(best_sd, best_path)

    model.load_state_dict(best_sd)
    int8 = convert(copy.deepcopy(model).cpu(), inplace=False)
    a1,a5 = evaluate(int8.to(device), val_loader, device)
    fl = get_flops(int8); pa = get_params(int8)
    out = os.path.join(args.out_dir, "qat_int8.pth"); torch.save(int8.state_dict(), out)
    print(f"[QAT Final] Top1={a1*100:.2f}% Top5={a5*100:.2f}% FLOPs={fl:.2f}G Params={pa:,} ckpt={out}")

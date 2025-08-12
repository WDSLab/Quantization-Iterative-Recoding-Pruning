# methods/qirp.py
import os, copy, torch, numpy as np
import torch.nn as nn, torch.optim as optim
from torch.quantization import prepare_qat, convert, get_default_qat_qconfig
from .models import TinyResNet18, QIRPWrapper
from .common import evaluate, get_flops, get_params, resolve_device_for_method

def run(args, train_loader, val_loader, num_classes):
    torch.backends.quantized.engine = args.qat_backend
    device = resolve_device_for_method(is_quant=True)  # CPU
    base = os.path.join(args.out_dir, "base.pth"); assert os.path.exists(base)

    bb = TinyResNet18(num_classes=num_classes)
    bb.load_state_dict(torch.load(base, map_location="cpu"))
    dyn = QIRPWrapper(bb).to(device)
    dyn.qconfig = get_default_qat_qconfig(args.qat_backend)
    prepare_qat(dyn, inplace=True)

    opt = optim.SGD(dyn.parameters(), lr=args.qirp_lr, momentum=0.9)
    crit = nn.CrossEntropyLoss()
    counts = np.zeros(dyn.backbone.fc.in_features, dtype=int)
    it = iter(train_loader)

    for cycle in range(args.qirp_cycles):
        try: xb,yb = next(it)
        except StopIteration: it = iter(train_loader); xb,yb = next(it)
        xb,yb = xb.to(device), yb.to(device)
        dyn.train(); opt.zero_grad(); loss = crit(dyn(xb), yb); loss.backward(); opt.step()

        with torch.no_grad():
            W = dyn.backbone.fc.weight.detach().cpu().numpy()
            imp = np.abs(W).sum(axis=0)
            k = int(W.shape[1]*args.prune_frac)
            idx = np.argsort(imp)[::-1][:k]
            for n,v in enumerate(idx): counts[v] += n

        if (cycle+1)%50==0:
            a1,a5 = evaluate(dyn.eval(), val_loader, device)
            print(f"[QIRP] {cycle+1}/{args.qirp_cycles} Top1={a1*100:.2f}%")

    order = np.argsort(counts)[::-1]
    pruned = int(len(order)*args.prune_frac)
    keep = order[:-pruned]

    pruned_model = QIRPWrapper(TinyResNet18(num_classes=num_classes)).to(device)
    pruned_model.set_survive_idx(keep)

    old_fc = pruned_model.backbone.fc
    new_fc = nn.Linear(len(keep), num_classes, bias=(old_fc.bias is not None)).to(device)
    with torch.no_grad():
        new_fc.weight.copy_(old_fc.weight[:, torch.as_tensor(keep)])
        if old_fc.bias is not None: new_fc.bias.copy_(old_fc.bias)
    pruned_model.backbone.fc = new_fc

    pruned_model.qconfig = get_default_qat_qconfig(args.qat_backend)
    prepare_qat(pruned_model, inplace=True)
    opt_ft = optim.SGD(pruned_model.parameters(), lr=args.qirp_lr, momentum=0.9)
    for ep in range(100):
        pruned_model.train()
        for xb,yb in train_loader:
            xb,yb = xb.to(device), yb.to(device)
            opt_ft.zero_grad(); loss = crit(pruned_model(xb), yb); loss.backward(); opt_ft.step()

    int8 = convert(copy.deepcopy(pruned_model).cpu(), inplace=False)
    a1,a5 = evaluate(int8.to(device), val_loader, device)
    fl,pa = get_flops(int8), get_params(int8)
    out = os.path.join(args.out_dir, f"qirp_{int(args.prune_frac*100)}.pth")
    torch.save(int8.state_dict(), out)
    print(f"[QIRP Final] Top1={a1*100:.2f}% FLOPs={fl:.2f}G Params={pa:,} ckpt={out}")
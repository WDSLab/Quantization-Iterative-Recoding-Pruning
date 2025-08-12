# methods/spsrc.py
import os, copy, torch
import torch.nn as nn, torch.optim as optim
from .models import TinyResNet18
from .common import evaluate, get_flops, get_params, resolve_device_for_method

def apply_spsrc(model, ratio, device):
    with torch.no_grad():
        for name, m in model.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                gamma = m.weight.abs().clone()
                C = gamma.numel()
                k = int(C * ratio)
                _, keep = torch.topk(gamma, C - k, largest=True, sorted=True)
                m.weight.data = m.weight.data[keep]
                m.bias.data   = m.bias.data[keep]
                m.running_mean = m.running_mean[keep]
                m.running_var  = m.running_var[keep]
                
                for n2, c in model.named_modules():
                    if isinstance(c, nn.Conv2d) and c.out_channels == C:
                        new_conv = nn.Conv2d(c.in_channels, keep.numel(), c.kernel_size, c.stride,
                                             c.padding, c.dilation, bias=(c.bias is not None)).to(device)
                        new_conv.weight.data = c.weight.data[keep]
                        if c.bias is not None:
                            new_conv.bias.data = c.bias.data[keep]
                        setattr(model, n2, new_conv); break

def run(args, train_loader, val_loader, num_classes):
    device = resolve_device_for_method(is_quant=False)
    base_path = os.path.join(args.out_dir, "base.pth")
    assert os.path.exists(base_path)
    m = TinyResNet18(num_classes=num_classes).to(device)
    m.load_state_dict(torch.load(base_path, map_location=device))
    apply_spsrc(m, args.prune_frac, device)

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
    out = os.path.join(args.out_dir, "spsrc.pth"); torch.save(m.state_dict(), out)
    print(f"[SPSRC] FLOPs={get_flops(m):.2f}G Params={get_params(m):,} ckpt={out}")

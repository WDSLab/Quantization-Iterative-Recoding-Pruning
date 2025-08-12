# methods/sms.py
import os, copy, torch
import torch.nn as nn, torch.optim as optim
from .models import TinyResNet18
from .common import evaluate, get_flops, get_params, resolve_device_for_method

def _prune_fc_inputs(model, frac, device):
    with torch.no_grad():
        W = model.fc.weight.abs().sum(dim=0)
        k = int(W.numel()*frac)
        keep = torch.argsort(W)[k:]
        new_fc = nn.Linear(keep.numel(), model.fc.out_features).to(device)
        new_fc.weight.copy_(model.fc.weight[:, keep])
        new_fc.bias.copy_(model.fc.bias)
        model.fc = new_fc
    return model, keep

def _avg_fc(models, device):
    with torch.no_grad():
        W = sum(m.fc.weight.data for m in models)/len(models)
        b = sum(m.fc.bias.data  for m in models)/len(models)
        d_out, d_in = W.size(0), W.size(1)
        new_fc = nn.Linear(d_in, d_out).to(device)
        new_fc.weight.data = W; new_fc.bias.data = b
        return new_fc

def run(args, train_loader, val_loader, num_classes):
    device = resolve_device_for_method(is_quant=False)
    base_ckpt = os.path.join(args.out_dir, "base.pth")
    assert os.path.exists(base_ckpt), "Run baseline first."

    base = TinyResNet18(num_classes=num_classes).to(device)
    base.load_state_dict(torch.load(base_ckpt, map_location=device))

    soup = []
    for seed in range(args.sms_models):
        torch.manual_seed(seed)
        m = TinyResNet18(num_classes=num_classes).to(device)
        m.load_state_dict(copy.deepcopy(base.state_dict()))
        m,_ = _prune_fc_inputs(m, args.prune_frac, device)
        opt = optim.SGD(m.parameters(), lr=0.01, momentum=0.9)
        crit = nn.CrossEntropyLoss()
        best = copy.deepcopy(m.state_dict()); best_acc=0.0
        for ep in range(args.sms_epochs):
            m.train()
            for xb,yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad(); loss = crit(m(xb), yb); loss.backward(); opt.step()
            a1,a5 = evaluate(m.eval(), val_loader, device)
            if a1>best_acc: best_acc=a1; best=copy.deepcopy(m.state_dict())
        m.load_state_dict(best); soup.append(m)

    sms = copy.deepcopy(soup[0]).to(device)
    sms.fc = _avg_fc(soup, device)
    out = os.path.join(args.out_dir, "sms.pth")
    torch.save(sms.state_dict(), out)
    print(f"[SMS Summary] Top1={evaluate(sms, val_loader, device)[0]*100:.2f}% FLOPs={get_flops(sms):.2f}G Params={get_params(sms):,} ckpt={out}")

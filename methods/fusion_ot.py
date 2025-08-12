# methods/fusion_ot.py
import os, copy, torch
import torch.nn as nn, torch.optim as optim
from .models import TinyResNet18
from .common import evaluate, get_flops, get_params, resolve_device_for_method

def _random_l1_prune_fc(model, prune_frac=0.3, seed=0, device="cpu"):
    torch.manual_seed(seed)
    with torch.no_grad():
        W = model.fc.weight.abs().sum(dim=0)
        k = int(W.numel() * prune_frac)
        keep = torch.argsort(W)[k:]
        new_fc = nn.Linear(keep.numel(), model.fc.out_features).to(device)
        new_fc.weight.copy_(model.fc.weight[:, keep])
        new_fc.bias.copy_(model.fc.bias)
        model.fc = new_fc
    return model, keep

def _sinkhorn_ot(cost, eps=0.01, iters=50):
    K = torch.exp(-cost/eps)
    u = torch.ones(K.size(0), device=K.device)/K.size(0)
    v = torch.ones(K.size(1), device=K.device)/K.size(1)
    for _ in range(iters):
        u = 1.0/(K @ v)
        v = 1.0/(K.t() @ u)
    return torch.diag(u) @ K @ torch.diag(v)

def _fuse_fc(fcA, fcB, device):
    WA, WB = fcA.weight.data, fcB.weight.data
    d_out = fcA.out_features
    d_in = max(WA.size(1), WB.size(1))
    A = torch.zeros(d_out, d_in, device=device); B = torch.zeros(d_out, d_in, device=device)
    A[:,:WA.size(1)] = WA; B[:,:WB.size(1)] = WB
    cost = torch.cdist(A.t(), B.t(), p=2)
    T = _sinkhorn_ot(cost)
    Wf = (T @ B.t()).t()
    fused = nn.Linear(Wf.size(1), d_out).to(device)
    fused.weight.data = Wf
    fused.bias.data = (fcA.bias.data + fcB.bias.data)/2
    return fused

def run(args, train_loader, val_loader, num_classes):
    device = resolve_device_for_method(is_quant=False)
    base_path = os.path.join(args.out_dir, "base.pth")
    assert os.path.exists(base_path), "Run baseline first."

    base = TinyResNet18(num_classes=num_classes).to(device)
    base.load_state_dict(torch.load(base_path, map_location=device))
    base.eval()

    A, _ = _random_l1_prune_fc(copy.deepcopy(base).to(device), args.prune_frac, seed=42, device=device)
    B, _ = _random_l1_prune_fc(copy.deepcopy(base).to(device), args.prune_frac, seed=99, device=device)

    fused = copy.deepcopy(base).to(device)
    fused.fc = _fuse_fc(A.fc, B.fc, device)

    opt = optim.SGD(fused.parameters(), lr=0.01, momentum=0.9)
    crit = nn.CrossEntropyLoss()
    step = 0; best_acc = 0.0; best = copy.deepcopy(fused.state_dict())
    out = os.path.join(args.out_dir, "fusion.pth")

    for ep in range(args.fusion_epochs):
        fused.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); loss = crit(fused(xb), yb); loss.backward(); opt.step()
            if step % 50 == 0:
                a1,a5 = evaluate(fused.eval(), val_loader, device)
                print(f"[Fusion] step{step} Top1={a1*100:.2f}% Top5={a5*100:.2f}%")
                if a1>best_acc: best_acc=a1; best=copy.deepcopy(fused.state_dict()); torch.save(best, out)
                fused.train()
            step += 1

    fused.load_state_dict(torch.load(out, map_location=device))
    print(f"[Fusion Summary] FLOPs={get_flops(fused):.2f}G Params={get_params(fused):,} ckpt={out}")

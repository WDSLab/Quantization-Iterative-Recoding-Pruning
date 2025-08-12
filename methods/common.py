# methods/common.py
import os, copy, torch
import torch.nn as nn
from thop import profile

def resolve_device_for_method(is_quant: bool) -> torch.device:
    """ 양자화 메소드는 CPU 고정, 그 외는 GPU 있으면 GPU."""
    if is_quant:
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(model: nn.Module, loader, device: torch.device):
    model.eval()
    correct1 = correct5 = total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            _, pred = out.topk(5, 1, True, True)
            total += yb.size(0)
            correct1 += (pred[:,0] == yb).sum().item()
            correct5 += (pred == yb.view(-1,1)).sum().item()
    return correct1/total, correct5/total

def get_params(model: nn.Module):
    return sum(p.numel() for p in model.parameters())

def get_flops(model: nn.Module, input_size=(1,3,64,64)):
    m = copy.deepcopy(model).cpu().eval()
    dummy = torch.randn(*input_size)
    flops, _ = profile(m, inputs=(dummy,), verbose=False)
    return flops/1e9

def save_if_best(model, acc1, best, path):
    if acc1 > best["acc1"]:
        best["acc1"] = acc1
        best["sd"] = copy.deepcopy(model.state_dict())
        torch.save(best["sd"], path)
        print(f"  ↳ Saved {path} (Top1 {acc1*100:.2f}%)")

def load_float_only(model: nn.Module, state_dict: dict):
    filt = {k:v for k,v in state_dict.items()
            if (k.endswith(".weight") or k.endswith(".bias"))
            and ("activation_post_process" not in k)
            and ("weight_fake_quant" not in k)}
    model.load_state_dict(filt, strict=False)
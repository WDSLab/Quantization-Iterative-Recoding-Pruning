# main.py
import argparse, os, torch, random, numpy as np
from methods import METHODS
from data.dataset import create_loaders

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--method", type=str, default="baseline", choices=list(METHODS.keys()))
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="./checkpoints")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--img_size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--prune_frac", type=float, default=0.3)
    p.add_argument("--qat_epochs", type=int, default=100)
    p.add_argument("--qat_lr", type=float, default=0.01)
    p.add_argument("--qat_backend", type=str, default="fbgemm") 
    p.add_argument("--qirp_cycles", type=int, default=200)
    p.add_argument("--qirp_lr", type=float, default=0.01)
    p.add_argument("--sms_models", type=int, default=3)
    p.add_argument("--fusion_epochs", type=int, default=30)
    p.add_argument("--sms_epochs", type=int, default=30)

    return p.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    train_loader, val_loader, num_classes = create_loaders(
        root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
    )

    METHODS[args.method](args=args,
                         train_loader=train_loader,
                         val_loader=val_loader,
                         num_classes=num_classes)

if __name__ == "__main__":
    main()


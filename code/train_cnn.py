"""Supervised ImageNet-pretrained baselines (ResNet-50, ViT-B/16), fully
fine-tuned on the PlantVillage training split and evaluated on the
PlantVillage test split and on PlantDoc (cross-domain).

  python train_cnn.py --arch resnet50 --pv_root ... --pd_root ... --out results
  python train_cnn.py --arch vit_b_16 --pv_root ... --pd_root ... --out results
"""
import argparse, os, json, torch, torch.nn as nn, torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from classes import CLASS_NAMES
from data import index_plantvillage, index_plantdoc, stratified_split, ImageList
from sava import metrics

p = argparse.ArgumentParser()
p.add_argument("--arch", choices=["resnet50", "vit_b_16"], required=True)
p.add_argument("--pv_root", required=True); p.add_argument("--pd_root", required=True)
p.add_argument("--out", default="results"); p.add_argument("--epochs", type=int, default=10)
p.add_argument("--bs", type=int, default=64); p.add_argument("--lr", type=float, default=None)
args = p.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

M, S = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
ttf = T.Compose([T.RandomResizedCrop(224, scale=(0.6, 1)), T.RandomHorizontalFlip(),
                 T.RandomVerticalFlip(), T.ColorJitter(0.2, 0.2, 0.1), T.ToTensor(), T.Normalize(M, S)])
etf = T.Compose([T.Resize(224), T.CenterCrop(224), T.ToTensor(), T.Normalize(M, S)])

pv = index_plantvillage(args.pv_root)
tr, va, te = stratified_split(pv, seed=42)
pd_items = index_plantdoc(args.pd_root)

if args.arch == "resnet50":
    net = torchvision.models.resnet50(weights="IMAGENET1K_V2")
    net.fc = nn.Linear(net.fc.in_features, len(CLASS_NAMES)); lr = args.lr or 1e-3
else:
    net = torchvision.models.vit_b_16(weights="IMAGENET1K_V1")
    net.heads.head = nn.Linear(net.heads.head.in_features, len(CLASS_NAMES)); lr = args.lr or 1e-4
net = net.to(dev)
opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.05)
dl = DataLoader(ImageList(tr, ttf), args.bs, shuffle=True, num_workers=8, drop_last=True)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=args.epochs * len(dl), pct_start=0.1)
scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda")
for ep in range(args.epochs):
    net.train()
    for x, y in dl:
        with torch.autocast(dev, enabled=dev == "cuda"):
            loss = nn.functional.cross_entropy(net(x.to(dev)), y.to(dev), label_smoothing=0.1)
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
    print(f"epoch {ep + 1}/{args.epochs} loss {loss.item():.3f}")


@torch.no_grad()
def run(items):
    net.eval(); P, Y = [], []
    for x, y in DataLoader(ImageList(items, etf), 128, num_workers=8):
        with torch.autocast(dev, enabled=dev == "cuda"):
            P.append(net(x.to(dev)).float().cpu()); Y.append(y)
    return metrics(torch.cat(P), torch.cat(Y))


key = {"resnet50": "resnet", "vit_b_16": "vit"}[args.arch]
os.makedirs(args.out, exist_ok=True)
rp = os.path.join(args.out, "results.json")
R = json.load(open(rp)) if os.path.exists(rp) else {}
R[key] = {"pv": run(te), "pd": run(pd_items)}
R.setdefault("params", {})[key] = sum(p.numel() for p in net.parameters())
json.dump(R, open(rp, "w"), indent=1)
print(key, R[key])

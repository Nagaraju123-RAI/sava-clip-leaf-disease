"""Runs every CLIP-based experiment reported in the paper and writes
results/results.json (merged with any existing entries).

Usage:
  python run_all.py --pv_root PlantVillage-Dataset/raw/color \
                    --pd_root PlantDoc-Dataset --out results
Then (optional) python train_cnn.py ... and python generative_vlm.py ...
Finally:  python write_tex.py --out results   &&   python make_figures.py --out results
"""
import argparse, os, json, random, math
import numpy as np, torch
from classes import CLASS_NAMES, name_prompts, sape_prompts
from data import (index_plantvillage, index_plantdoc, stratified_split, eval_tf,
                  train_tf, save_split)
from sava import (CLIPBackbone, metrics, zero_shot, linear_probe, SAVA, CLIPAdapter,
                  TipAdapterF, train_head, predict, count_trainable)

p = argparse.ArgumentParser()
p.add_argument("--pv_root", required=True)
p.add_argument("--pd_root", required=True)
p.add_argument("--out", default="results")
p.add_argument("--backbone", default="openai/clip-vit-base-patch16")
p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
p.add_argument("--shots", type=int, nargs="+", default=[1, 2, 4, 8, 16])
p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
p.add_argument("--n_aug", type=int, default=10)
p.add_argument("--fs_steps", type=int, default=1000)
p.add_argument("--full_epochs", type=int, default=20)
p.add_argument("--skip_full", action="store_true")
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)
dev = args.device
C = len(CLASS_NAMES)
HP = dict(ratio=0.2, crop_w=0.5, reg_w=5.0, lr=1e-3)

# ------------------------------------------------------------- data
pv = index_plantvillage(args.pv_root)
pd_items = index_plantdoc(args.pd_root)
tr, va, te = stratified_split(pv, seed=42)
save_split(os.path.join(args.out, "split.json"), train=tr, val=va, test=te)
print(f"PlantVillage train/val/test = {len(tr)}/{len(va)}/{len(te)} ; PlantDoc = {len(pd_items)}")

bb = CLIPBackbone(args.backbone, dev)
cache = os.path.join(args.out, "feats.pt")
if os.path.exists(cache):
    F_ = torch.load(cache)
else:
    F_ = {}
    F_["Wname"] = bb.encode_text(name_prompts()).cpu()
    F_["Wsape"] = bb.encode_text(sape_prompts()).cpu()
    for k, items in (("val", va), ("test", te), ("pd", pd_items)):
        F_[k] = bb.encode_images(items, eval_tf())
    torch.save(F_, cache)
Wn, Ws = F_["Wname"], F_["Wsape"]
vaF, vaY = F_["val"]; teF, teY = F_["test"]; pdF, pdY = F_["pd"]

res_path = os.path.join(args.out, "results.json")
R = json.load(open(res_path)) if os.path.exists(res_path) else {}
R["meta"] = dict(n_train=len(tr), n_val=len(va), n_test=len(te), n_pd=len(pd_items),
                 backbone=args.backbone, hp=HP, shots=args.shots, seeds=args.seeds)


def evaluate(fn):
    return {"pv": metrics(fn(teF), teY), "pd": metrics(fn(pdF), pdY)}


def dump():
    json.dump(R, open(res_path, "w"), indent=1)


# ------------------------------------------------------------- zero-shot
R["zsname"] = evaluate(lambda f: zero_shot(f, Wn))
R["zssape"] = evaluate(lambda f: zero_shot(f, Ws))
print("zero-shot", R["zsname"], R["zssape"]); dump()


def build(method, Ftr, Ytr, Fkeys, Ykeys, epochs, seed, variant=None):
    """Returns (trained model or predictor, trainable-parameter count)."""
    if method == "clipad":
        m = train_head(CLIPAdapter(Wn, HP["ratio"]), Ftr, Ytr, epochs, HP["lr"], device=dev, seed=seed)
    elif method == "tipf":
        m = train_head(TipAdapterF(Wn, Fkeys, Ykeys, C), Ftr, Ytr, epochs, HP["lr"] * 1e-0,
                       device=dev, seed=seed)
    elif method == "sava":
        v = variant or dict(init="sape", adapter=True, text=True, crop=True, reg=True)
        W0 = Ws if v["init"] == "sape" else Wn
        m = train_head(SAVA(W0, v["adapter"], v["text"], HP["ratio"]), Ftr, Ytr, epochs,
                       HP["lr"], device=dev, seed=seed,
                       crop_w=HP["crop_w"] if v["crop"] else 0.0,
                       reg_w=HP["reg_w"] if v["reg"] else 0.0)
    return m, count_trainable(m)


def run_eval(m):
    return evaluate(lambda f: predict(m, f, dev))


# ------------------------------------------------------------- few-shot
ABL = {
    "a": None,  # zero-shot, class name prompts      (reported from zsname)
    "b": None,  # zero-shot, SAPE prompts            (reported from zssape)
    "c": dict(init="sape", adapter=False, text=True, crop=False, reg=False),
    "d": dict(init="sape", adapter=True, text=True, crop=False, reg=False),
    "e": dict(init="sape", adapter=True, text=True, crop=True, reg=False),
    "f": dict(init="sape", adapter=True, text=True, crop=True, reg=True),   # full SAVA
    "g": dict(init="name", adapter=True, text=True, crop=True, reg=True),   # name-init
}
trn_rng_items = tr
fs = {}
abl = {k: [] for k in "cdefg"}
for seed in args.seeds:
    rng = random.Random(seed)
    per = {}
    for i, (_, y) in enumerate(tr):
        per.setdefault(y, []).append(i)
    order = {y: rng.sample(v, len(v)) for y, v in per.items()}
    kmax = max(args.shots)
    sel = [order[y][j] for y in range(C) for j in range(min(kmax, len(order[y])))]
    sel_items = [tr[i] for i in sel]
    sel_rank = [j for y in range(C) for j in range(min(kmax, len(order[y])))]
    cf = os.path.join(args.out, f"fs_feats_seed{seed}.pt")
    if os.path.exists(cf):
        clean, aug = torch.load(cf)
    else:
        torch.manual_seed(seed)
        clean = bb.encode_images(sel_items, eval_tf())
        aug = [bb.encode_images(sel_items, train_tf()) for _ in range(args.n_aug)]
        torch.save((clean, aug), cf)
    rank = torch.tensor(sel_rank)
    for k in args.shots:
        mask = rank < k
        cF, cY = clean[0][mask], clean[1][mask]
        aF = torch.cat([a[0][mask] for a in aug] + [cF]); aY = torch.cat([a[1][mask] for a in aug] + [cY])
        ep = max(1, round(args.fs_steps * 256 / len(aF)))
        out = {}
        lp = linear_probe(cF, cY, vaF, vaY)
        out["lp"] = evaluate(lp)
        for mth in ("clipad", "tipf", "sava"):
            m, npar = build(mth, aF, aY, cF, cY, ep, seed)
            out[mth] = run_eval(m)
            R.setdefault("params", {})[mth + f"_{k}shot"] = npar
        fs.setdefault(str(k), []).append(out)
        if k == 16:
            for a in "cdefg":
                m, _ = build("sava", aF, aY, cF, cY, ep, seed, ABL[a])
                abl[a].append(run_eval(m))
        print(f"seed {seed} k={k}", {m: round(v['pv']['acc'], 2) for m, v in out.items()})


def mean_over(lst):
    keys = lst[0].keys()
    return {m: {d: {s: float(np.mean([x[m][d][s] for x in lst])) for s in ("acc", "f1")}
                for d in ("pv", "pd")} for m in keys} | \
           {m + "_std": {d: {s: float(np.std([x[m][d][s] for x in lst])) for s in ("acc", "f1")}
                         for d in ("pv", "pd")} for m in keys}


R["fewshot"] = {k: mean_over(v) for k, v in fs.items()}
R["ablation"] = {a: {d: {s: float(np.mean([x[d][s] for x in v])) for s in ("acc", "f1")}
                     for d in ("pv", "pd")} for a, v in abl.items() if v}
R["ablation"]["a"] = R["zsname"]; R["ablation"]["b"] = R["zssape"]
dump()

# ------------------------------------------------------------- full data
if not args.skip_full:
    cf = os.path.join(args.out, "full_feats.pt")
    if os.path.exists(cf):
        trC, trA = torch.load(cf)
    else:
        trC = bb.encode_images(tr, eval_tf())
        trA = bb.encode_images(tr, train_tf())
        torch.save((trC, trA), cf)
    aF, aY = torch.cat([trC[0], trA[0]]), torch.cat([trC[1], trA[1]])
    R["lp"] = evaluate(linear_probe(trC[0], trC[1], vaF, vaY))
    for mth in ("clipad", "tipf", "sava"):
        m, npar = build(mth, aF, aY, trC[0], trC[1], args.full_epochs, 0)
        R[mth] = run_eval(m); R.setdefault("params", {})[mth] = npar
        if mth == "sava":
            torch.save(m.state_dict(), os.path.join(args.out, "sava_full.pt"))
            np.savez(os.path.join(args.out, "sava_preds.npz"),
                     pv_pred=predict(m, teF, dev).argmax(-1).numpy(), pv_true=teY.numpy(),
                     pd_pred=predict(m, pdF, dev).argmax(-1).numpy(), pd_true=pdY.numpy(),
                     zs_pv_pred=zero_shot(teF, Ws).argmax(-1).numpy(),
                     zs_pd_pred=zero_shot(pdF, Ws).argmax(-1).numpy())
        print(mth, R[mth]); dump()
dump()
print("done ->", res_path)

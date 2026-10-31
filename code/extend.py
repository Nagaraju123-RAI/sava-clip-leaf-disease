"""Journal-extension experiments for SAVA-CLIP. Writes results/ext/results_ext.json.

  python extend.py --pv_root ... --pd_root ... --out results/ext \
                   --backbones openai/clip-vit-base-patch16 openai/clip-vit-large-patch14

Per backbone:
  * zero-shot (class names, SAPE)
  * few-shot k in {1,2,4,8,16} x seeds {1,2,3}: linear probe, CLIP-Adapter,
    Tip-Adapter-F with alpha/beta searched on the validation set (as in the
    original paper), CoOp (ViT-B/16 only), SAVA-CLIP, and SAVA/zero-shot
    probability ensembles
  * full data: linear probe, CLIP-Adapter, tuned Tip-Adapter-F, SAVA-CLIP, ensembles
First backbone only: sensitivity to lambda, gamma and number of templates K
(16 shots), and exact McNemar tests on the full-data predictions.
"""
import argparse, os, json, random, time, math
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from scipy.stats import binomtest
from transformers.modeling_attn_mask_utils import _create_4d_causal_attention_mask
from classes import CLASS_NAMES, CLASSES, name_prompts, sape_prompts
from data import index_plantvillage, index_plantdoc, stratified_split, eval_tf, train_tf
from sava import (CLIPBackbone, metrics, zero_shot, linear_probe, SAVA, CLIPAdapter,
                  TipAdapterF, train_head, predict, count_trainable)

p = argparse.ArgumentParser()
p.add_argument("--pv_root", required=True); p.add_argument("--pd_root", required=True)
p.add_argument("--out", default="results/ext")
p.add_argument("--backbones", nargs="+",
               default=["openai/clip-vit-base-patch16", "openai/clip-vit-large-patch14"])
p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
p.add_argument("--shots", type=int, nargs="+", default=[1, 2, 4, 8, 16])
p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
p.add_argument("--n_aug", type=int, default=10)
p.add_argument("--fs_steps", type=int, default=1000)
p.add_argument("--full_epochs", type=int, default=20)
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)
dev, C = args.device, len(CLASS_NAMES)
HP = dict(ratio=0.2, crop_w=0.5, reg_w=5.0, lr=1e-3)
ENS_W = [0.0, 0.25, 0.5, 0.75, 1.0]
ALPHAS = [0.25, 0.5, 1, 2, 4, 8, 16, 32, 64]
BETAS = [1.0, 2.0, 3.5, 5.5, 7.5, 10.0]
T0 = time.time()


def log(*a):
    print(f"[{(time.time() - T0) / 60:6.1f} min]", *a, flush=True)


res_path = os.path.join(args.out, "results_ext.json")
R = json.load(open(res_path)) if os.path.exists(res_path) else {}


def dump():
    json.dump(R, open(res_path, "w"), indent=1)


pv = index_plantvillage(args.pv_root)
pd_items = index_plantdoc(args.pd_root)
tr, va, te = stratified_split(pv, seed=42)
log(f"train/val/test {len(tr)}/{len(va)}/{len(te)}  PlantDoc {len(pd_items)}")


# ------------------------------------------------------------------ CoOp
class CoOp(nn.Module):
    """Zhou et al. CoOp: 16 learnable context vectors shared by all classes,
    class name at the end, frozen CLIP text encoder."""

    def __init__(self, bb, n_ctx=16):
        super().__init__()
        object.__setattr__(self, "_bb", bb)  # not registered: keeps CLIP frozen and uncounted
        tm = bb.model.text_model
        names = [f"{CLASSES[c][0]} leaf with {CLASSES[c][1]}" if CLASSES[c][1] != "healthy"
                 else f"healthy {CLASSES[c][0]} leaf" for c in CLASS_NAMES]
        prompts = [" ".join(["X"] * n_ctx) + " " + n + "." for n in names]
        t = bb.tok(prompts, padding="max_length", max_length=77, truncation=True, return_tensors="pt")
        ids = t["input_ids"].to(bb.device)
        eos = getattr(bb.tok, "eos_token_id", None)
        self.register_buffer("eos_pos", (ids == eos).int().argmax(-1) if eos is not None
                             else ids.argmax(-1))
        with torch.no_grad():
            emb = tm.embeddings.token_embedding(ids).float()
        self.register_buffer("prefix", emb[:, :1]); self.register_buffer("suffix", emb[:, 1 + n_ctx:])
        self.ctx = nn.Parameter(torch.randn(n_ctx, emb.shape[-1], device=bb.device) * 0.02)

    def text_features(self):
        bb = self._bb; tm = bb.model.text_model
        x = torch.cat([self.prefix, self.ctx.unsqueeze(0).expand(self.prefix.shape[0], -1, -1),
                       self.suffix], 1)
        h = tm.embeddings(inputs_embeds=x)
        mask = _create_4d_causal_attention_mask(x.shape[:2], h.dtype, device=h.device)
        out = tm.final_layer_norm(tm.encoder(inputs_embeds=h, causal_attention_mask=mask).last_hidden_state)
        pooled = out[torch.arange(out.shape[0], device=out.device), self.eos_pos]
        return F.normalize(bb.model.text_projection(pooled), dim=-1)

    def forward(self, f):
        return 100.0 * f @ self.text_features().t()


# ------------------------------------------------------------------ helpers
def tipf_tuned(Wn, kF, kY, aF, aY, vaF, vaY, epochs, seed):
    m = train_head(TipAdapterF(Wn, kF, kY, C), aF, aY, epochs, HP["lr"], device=dev, seed=seed)
    best = (-1, 1.0, 5.5)
    for al in ALPHAS:
        for be in BETAS:
            m.alpha, m.beta = al, be
            a = (predict(m, vaF, dev).argmax(-1) == vaY).float().mean().item()
            if a > best[0]:
                best = (a, al, be)
    m.alpha, m.beta = best[1], best[2]
    return m, best[1:]


def ens_eval(logit_fn, Ws, feats):
    """probability ensemble of an adapted model with the SAPE zero-shot classifier"""
    out = {}
    for d, (f, y) in feats.items():
        pa = torch.softmax(logit_fn(f), -1); pz = torch.softmax(zero_shot(f, Ws), -1)
        for w in ENS_W:
            out.setdefault(str(w), {})[d] = metrics((1 - w) * pa + w * pz, y)
    return out


def mean_std(lst):
    """lst: list of nested dicts {method: {dom: {acc, f1}}} -> mean and std"""
    def walk(xs):
        if isinstance(xs[0], dict):
            return {k: walk([x[k] for x in xs]) for k in xs[0]}
        return {"mean": float(np.mean(xs)), "std": float(np.std(xs))}
    return walk(lst)


# ------------------------------------------------------------------ main loop
for bi, name in enumerate(args.backbones):
    tag = name.split("/")[-1]
    if R.get(tag, {}).get("done"):
        log("skip", tag); continue
    log("==== backbone", tag)
    bb = CLIPBackbone(name, dev)
    cdir = os.path.join(args.out, tag); os.makedirs(cdir, exist_ok=True)
    cf = os.path.join(cdir, "feats.pt")
    if os.path.exists(cf):
        Fd = torch.load(cf)
    else:
        Fd = {"Wname": bb.encode_text(name_prompts()).cpu(), "Wsape": bb.encode_text(sape_prompts()).cpu()}
        for K in (1, 3):
            Fd[f"Wsape{K}"] = bb.encode_text([p[:K] for p in sape_prompts()]).cpu()
        for k, items in (("val", va), ("test", te), ("pd", pd_items)):
            Fd[k] = bb.encode_images(items, eval_tf())
        torch.save(Fd, cf)
    log("features ready")
    Wn, Ws = Fd["Wname"], Fd["Wsape"]
    vaF, vaY = Fd["val"]; teF, teY = Fd["test"]; pdF, pdY = Fd["pd"]
    EV = {"pv": (teF, teY), "pd": (pdF, pdY)}
    ev = lambda fn: {d: metrics(fn(f), y) for d, (f, y) in EV.items()}
    B = R.setdefault(tag, {})
    B["dim"] = int(Ws.shape[1])
    B["zsname"] = ev(lambda f: zero_shot(f, Wn)); B["zssape"] = ev(lambda f: zero_shot(f, Ws))
    for K in (1, 3):
        B[f"zssape_K{K}"] = ev(lambda f: zero_shot(f, Fd[f"Wsape{K}"]))
    log("zero-shot", {k: round(v["pd"]["acc"], 2) for k, v in B.items() if k.startswith("zs")}); dump()

    # ---------------- few-shot
    fs, sens = {}, {}
    for seed in args.seeds:
        rng = random.Random(seed)
        per = {}
        for i, (_, y) in enumerate(tr):
            per.setdefault(y, []).append(i)
        order = {y: rng.sample(v, len(v)) for y, v in per.items()}
        kmax = max(args.shots)
        sel = [order[y][j] for y in range(C) for j in range(min(kmax, len(order[y])))]
        rank = torch.tensor([j for y in range(C) for j in range(min(kmax, len(order[y])))])
        ff = os.path.join(cdir, f"fs_seed{seed}.pt")
        if os.path.exists(ff):
            clean, aug = torch.load(ff)
        else:
            torch.manual_seed(seed)
            items = [tr[i] for i in sel]
            clean = bb.encode_images(items, eval_tf())
            aug = [bb.encode_images(items, train_tf()) for _ in range(args.n_aug)]
            torch.save((clean, aug), ff)
        for k in args.shots:
            mk = rank < k
            cF, cY = clean[0][mk], clean[1][mk]
            aF = torch.cat([a[0][mk] for a in aug] + [cF]); aY = torch.cat([a[1][mk] for a in aug] + [cY])
            ep = max(1, round(args.fs_steps * 256 / len(aF)))
            o = {"lp": ev(linear_probe(cF, cY, vaF, vaY))}
            m = train_head(CLIPAdapter(Wn, HP["ratio"]), aF, aY, ep, HP["lr"], device=dev, seed=seed)
            o["clipad"] = ev(lambda f: predict(m, f, dev))
            m, ab = tipf_tuned(Wn, cF, cY, aF, aY, vaF, vaY, ep, seed)
            o["tipf_tuned"] = ev(lambda f: predict(m, f, dev))
            B.setdefault("tipf_ab", {})[f"{k}_{seed}"] = ab
            if bi == 0:
                m = train_head(CoOp(bb), aF, aY, ep, 2e-3, device=dev, seed=seed)
                o["coop"] = ev(lambda f: predict(m, f, dev))
            m = train_head(SAVA(Ws, ratio=HP["ratio"]), aF, aY, ep, HP["lr"], device=dev, seed=seed,
                           crop_w=HP["crop_w"], reg_w=HP["reg_w"])
            o["sava"] = ev(lambda f: predict(m, f, dev))
            for w, r in ens_eval(lambda f: predict(m, f, dev), Ws, EV).items():
                o[f"ens{w}"] = r
            fs.setdefault(str(k), []).append(o)
            # sensitivity (first backbone, 16 shots)
            if bi == 0 and k == 16:
                for lam in (0.05, 0.1, 0.4, 0.6):
                    m = train_head(SAVA(Ws, ratio=lam), aF, aY, ep, HP["lr"], device=dev, seed=seed,
                                   crop_w=HP["crop_w"], reg_w=HP["reg_w"])
                    sens.setdefault(f"lambda_{lam}", []).append(ev(lambda f: predict(m, f, dev)))
                for g in (0.0, 1.0, 10.0, 20.0):
                    m = train_head(SAVA(Ws, ratio=HP["ratio"]), aF, aY, ep, HP["lr"], device=dev, seed=seed,
                                   crop_w=HP["crop_w"], reg_w=g)
                    sens.setdefault(f"gamma_{g}", []).append(ev(lambda f: predict(m, f, dev)))
                for K in (1, 3):
                    m = train_head(SAVA(Fd[f"Wsape{K}"], ratio=HP["ratio"]), aF, aY, ep, HP["lr"], device=dev,
                                   seed=seed, crop_w=HP["crop_w"], reg_w=HP["reg_w"])
                    sens.setdefault(f"K_{K}", []).append(ev(lambda f: predict(m, f, dev)))
            log(tag, f"seed {seed} k={k}", {m_: round(v["pv"]["acc"], 2) for m_, v in o.items()},
                "| PD", {m_: round(v["pd"]["acc"], 2) for m_, v in o.items()})
        B["fewshot"] = {k: mean_std(v) for k, v in fs.items()}
        if sens:
            B["sensitivity"] = {k: mean_std(v) for k, v in sens.items()}
        dump()

    # ---------------- full data
    ff = os.path.join(cdir, "full.pt")
    if os.path.exists(ff):
        trC, trA = torch.load(ff)
    else:
        trC = bb.encode_images(tr, eval_tf()); trA = bb.encode_images(tr, train_tf())
        torch.save((trC, trA), ff)
    log("full-data features ready")
    aF, aY = torch.cat([trC[0], trA[0]]), torch.cat([trC[1], trA[1]])
    full, preds = {}, {}
    lp = linear_probe(trC[0], trC[1], vaF, vaY)
    full["lp"] = ev(lp)
    m = train_head(CLIPAdapter(Wn, HP["ratio"]), aF, aY, args.full_epochs, HP["lr"], device=dev, seed=0)
    full["clipad"] = ev(lambda f: predict(m, f, dev)); full.setdefault("params", {})["clipad"] = count_trainable(m)
    preds["clipad"] = {d: predict(m, f, dev).argmax(-1).numpy() for d, (f, _) in EV.items()}
    m, ab = tipf_tuned(Wn, trC[0], trC[1], aF, aY, vaF, vaY, args.full_epochs, 0)
    full["tipf_tuned"] = ev(lambda f: predict(m, f, dev)); B["tipf_ab_full"] = ab
    full["params"]["tipf"] = count_trainable(m)
    m = train_head(SAVA(Ws, ratio=HP["ratio"]), aF, aY, args.full_epochs, HP["lr"], device=dev, seed=0,
                   crop_w=HP["crop_w"], reg_w=HP["reg_w"])
    full["sava"] = ev(lambda f: predict(m, f, dev)); full["params"]["sava"] = count_trainable(m)
    preds["sava"] = {d: predict(m, f, dev).argmax(-1).numpy() for d, (f, _) in EV.items()}
    preds["zssape"] = {d: zero_shot(f, Ws).argmax(-1).numpy() for d, (f, _) in EV.items()}
    for w, r in ens_eval(lambda f: predict(m, f, dev), Ws, EV).items():
        full[f"ens{w}"] = r
    pe = {d: (0.5 * torch.softmax(predict(m, f, dev), -1) + 0.5 * torch.softmax(zero_shot(f, Ws), -1)).argmax(-1).numpy()
          for d, (f, _) in EV.items()}
    preds["ens0.5"] = pe
    B["full"] = full
    log(tag, "full", {k: (round(v["pv"]["acc"], 2), round(v["pd"]["acc"], 2)) for k, v in full.items() if k != "params"})

    # ---------------- McNemar (exact) on full-data predictions
    def mcnemar(a, b, y):
        ca, cb = a == y, b == y
        n01, n10 = int((ca & ~cb).sum()), int((~ca & cb).sum())
        pval = binomtest(min(n01, n10), n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
        return {"a_only": n01, "b_only": n10, "p": float(pval)}
    yv = {"pv": teY.numpy(), "pd": pdY.numpy()}
    B["mcnemar"] = {f"{a}_vs_{b}": {d: mcnemar(preds[a][d], preds[b][d], yv[d]) for d in yv}
                    for a, b in (("sava", "clipad"), ("sava", "zssape"), ("ens0.5", "zssape"), ("ens0.5", "sava"))}
    np.savez(os.path.join(cdir, "preds.npz"), pv_true=yv["pv"], pd_true=yv["pd"],
             **{f"{k}_{d}": v for k, dd in preds.items() for d, v in dd.items()})
    B["done"] = True
    dump()
    del bb; torch.cuda.empty_cache()
log("all done ->", res_path)

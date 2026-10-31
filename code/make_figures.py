"""Result figures for the paper, written to ../fig/:
  fig_fewshot.pdf    accuracy vs. shots (PlantVillage and PlantDoc)
  fig_confusion.pdf  row-normalised confusion matrix of SAVA-CLIP on PlantVillage test
  fig_perclass.pdf   per-class F1 on PlantDoc: zero-shot SAPE vs. SAVA-CLIP
  fig_tsne.pdf       t-SNE of test embeddings before / after the visual adapter
  fig_heatmaps.pdf   patch-text similarity maps (needs the CLIP model; --heatmaps)
"""
import argparse, json, os, numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score
from classes import CLASS_NAMES, CLASSES

p = argparse.ArgumentParser()
p.add_argument("--out", default="results"); p.add_argument("--fig", default="../fig")
p.add_argument("--heatmaps", nargs="*", default=None, help="image paths (4 recommended)")
a = p.parse_args()
os.makedirs(a.fig, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})
COL = {"lp": "#8c8c8c", "clipad": "#4c78a8", "tipf": "#f58518", "sava": "#2e8b57", "zssape": "#b279a2"}
LAB = {"lp": "Linear probe", "clipad": "CLIP-Adapter", "tipf": "Tip-Adapter-F",
       "sava": "SAVA-CLIP (ours)", "zssape": "Zero-shot SAPE"}
R = json.load(open(os.path.join(a.out, "results.json")))
short = [f"{CLASSES[c][0][:6]}:{CLASSES[c][1][:10]}" for c in CLASS_NAMES]

# ---- few-shot curves
if "fewshot" in R:
    ks = sorted(map(int, R["fewshot"]))
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.3))
    for ax, dom, title in zip(axs, ("pv", "pd"), ("PlantVillage (in-domain)", "PlantDoc (cross-domain)")):
        for m in ("lp", "clipad", "tipf", "sava"):
            y = [R["fewshot"][str(k)][m][dom]["acc"] for k in ks]
            sd = [R["fewshot"][str(k)][m + "_std"][dom]["acc"] for k in ks]
            ax.errorbar(ks, y, yerr=sd, marker="o", ms=3, lw=1.6 if m == "sava" else 1.1,
                        color=COL[m], label=LAB[m], capsize=2)
        ax.axhline(R["zssape"][dom]["acc"], ls="--", lw=1, color=COL["zssape"], label=LAB["zssape"])
        ax.set_xscale("log", base=2); ax.set_xticks(ks); ax.set_xticklabels(ks)
        ax.set_xlabel("shots per class"); ax.set_ylabel("top-1 accuracy (%)"); ax.set_title(title)
        ax.grid(alpha=.3, lw=.5)
    axs[1].legend(frameon=False, fontsize=6.5, loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(a.fig, "fig_fewshot.pdf")); plt.close(fig)

npz = os.path.join(a.out, "sava_preds.npz")
if os.path.exists(npz):
    P = np.load(npz)
    cm = confusion_matrix(P["pv_true"], P["pv_pred"], labels=range(len(CLASS_NAMES))).astype(float)
    cm /= np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(3.4, 3.4))
    im = ax.imshow(cm, cmap="Greens", vmin=0, vmax=1)
    ax.set_xticks(range(38)); ax.set_yticks(range(38))
    ax.set_xticklabels(range(1, 39), fontsize=4); ax.set_yticklabels(range(1, 39), fontsize=4)
    ax.set_xlabel("predicted class"); ax.set_ylabel("true class")
    fig.colorbar(im, fraction=.046, pad=.03); fig.tight_layout()
    fig.savefig(os.path.join(a.fig, "fig_confusion.pdf")); plt.close(fig)

    labs = sorted(set(P["pd_true"].tolist()))
    f_zs = f1_score(P["pd_true"], P["zs_pd_pred"], labels=labs, average=None, zero_division=0)
    f_sv = f1_score(P["pd_true"], P["pd_pred"], labels=labs, average=None, zero_division=0)
    o = np.argsort(f_sv)
    fig, ax = plt.subplots(figsize=(3.4, 3.6))
    yy = np.arange(len(labs))
    ax.barh(yy - .2, 100 * f_zs[o], .4, color=COL["zssape"], label=LAB["zssape"])
    ax.barh(yy + .2, 100 * f_sv[o], .4, color=COL["sava"], label=LAB["sava"])
    ax.set_yticks(yy); ax.set_yticklabels([short[labs[i]] for i in o], fontsize=5)
    ax.set_xlabel("F1 on PlantDoc (%)"); ax.legend(frameon=False, fontsize=6, loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(a.fig, "fig_perclass.pdf")); plt.close(fig)

feats, ckpt = os.path.join(a.out, "feats.pt"), os.path.join(a.out, "sava_full.pt")
if os.path.exists(feats) and os.path.exists(ckpt):
    from sklearn.manifold import TSNE
    from sava import SAVA
    Fd = torch.load(feats); teF, teY = Fd["test"]
    m = SAVA(Fd["Wsape"]); m.load_state_dict(torch.load(ckpt, map_location="cpu")); m.eval()
    idx = np.random.RandomState(0).choice(len(teF), min(3000, len(teF)), replace=False)
    with torch.no_grad():
        f2 = torch.nn.functional.normalize(m.ratio * m.adapter(teF[idx]) + (1 - m.ratio) * teF[idx], dim=-1)
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.0))
    for ax, X, t in zip(axs, (teF[idx], f2), ("frozen CLIP embedding", "after SAVA visual adapter")):
        Z = TSNE(2, init="pca", perplexity=30, random_state=0).fit_transform(X.numpy())
        ax.scatter(Z[:, 0], Z[:, 1], c=teY[idx].numpy(), cmap="tab20", s=2, lw=0)
        ax.set_title(t); ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); fig.savefig(os.path.join(a.fig, "fig_tsne.pdf"), dpi=300); plt.close(fig)

if a.heatmaps:
    from PIL import Image
    from sava import CLIPBackbone
    from data import eval_tf
    Fd = torch.load(feats)
    bb = CLIPBackbone(R["meta"]["backbone"], "cuda" if torch.cuda.is_available() else "cpu")
    n = len(a.heatmaps)
    fig, axs = plt.subplots(2, n, figsize=(1.75 * n, 3.6))
    torch.set_grad_enabled(False)
    for j, path in enumerate(a.heatmaps):
        img = Image.open(path).convert("RGB")
        x = eval_tf()(img).unsqueeze(0)
        y = int(np.argmax((Fd["Wsape"] @ torch.nn.functional.normalize(
            bb.model.get_image_features(pixel_values=x.to(bb.device)).float().cpu(), dim=-1).t()).numpy()))
        s = bb.patch_similarity(x, Fd["Wsape"][y])
        s = (s - s.min()) / (s.max() - s.min() + 1e-8)
        im = np.asarray(img.resize((224, 224)))
        axs[0, j].imshow(im); axs[0, j].set_title(short[y], fontsize=6)
        axs[1, j].imshow(im); axs[1, j].imshow(np.kron(s, np.ones((16, 16))), cmap="jet", alpha=.45)
        for r in (0, 1):
            axs[r, j].axis("off")
    fig.tight_layout(); fig.savefig(os.path.join(a.fig, "fig_heatmaps.pdf"), dpi=300); plt.close(fig)
print("figures written to", a.fig)

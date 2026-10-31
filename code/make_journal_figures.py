"""Figures for the journal version from results_ext.json:
fig_fewshot_j.pdf, fig_ensemble.pdf, fig_sensitivity.pdf, fig_backbones.pdf"""
import argparse, json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--ext", required=True); p.add_argument("--fig", required=True)
p.add_argument("--b16", default="clip-vit-base-patch16"); p.add_argument("--l14", default="clip-vit-large-patch14")
a = p.parse_args()
os.makedirs(a.fig, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})
R = json.load(open(a.ext)); B = R[a.b16]
COL = {"lp": "#8c8c8c", "coop": "#9467bd", "clipad": "#4c78a8", "tipf_tuned": "#f58518",
       "sava": "#2e8b57", "ens0.5": "#17becf", "zssape": "#b279a2"}
LAB = {"lp": "Linear probe", "coop": "CoOp", "clipad": "CLIP-Adapter", "tipf_tuned": "Tip-Adapter-F (tuned)",
       "sava": "SAVA-CLIP", "ens0.5": "SAVA-CLIP-E ($w$=0.5)", "zssape": "Zero-shot SAPE"}
ks = sorted(map(int, B["fewshot"]))

# few-shot curves
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.6))
for ax, d, t in zip(axs, ("pv", "pd"), ("PlantVillage (in-domain)", "PlantDoc (field, unseen)")):
    for m in ("lp", "coop", "clipad", "tipf_tuned", "sava", "ens0.5"):
        if m not in B["fewshot"][str(ks[0])]:
            continue
        y = [B["fewshot"][str(k)][m][d]["acc"]["mean"] for k in ks]
        e = [B["fewshot"][str(k)][m][d]["acc"]["std"] for k in ks]
        ax.errorbar(ks, y, yerr=e, marker="o", ms=3, capsize=2, color=COL[m], label=LAB[m],
                    lw=1.7 if m.startswith(("sava", "ens")) else 1.1)
    ax.axhline(B["zssape"][d]["acc"], ls="--", lw=1, color=COL["zssape"], label=LAB["zssape"])
    ax.set_xscale("log", base=2); ax.set_xticks(ks); ax.set_xticklabels(ks)
    ax.set_xlabel("labelled images per class"); ax.set_ylabel("top-1 accuracy (%)"); ax.set_title(t)
    ax.grid(alpha=.3, lw=.5)
axs[1].legend(frameon=False, fontsize=6, loc="lower right", ncol=2)
fig.tight_layout(); fig.savefig(os.path.join(a.fig, "fig_fewshot_j.pdf")); plt.close(fig)

# ensemble trade-off (full data and 16-shot), both backbones
ws = ["0.0", "0.25", "0.5", "0.75", "1.0"]
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.7))
for ax, (title, getter) in zip(axs, (("Full data", lambda X, w, d: X["full"][f"ens{w}"][d]["acc"]),
                                     ("16 shots", lambda X, w, d: X["fewshot"]["16"][f"ens{w}"][d]["acc"]["mean"]))):
    for bk, mk, nm in ((a.b16, "o", "ViT-B/16"), (a.l14, "s", "ViT-L/14")):
        if bk not in R or "full" not in R[bk]:
            continue
        X = R[bk]
        xs = [getter(X, w, "pv") for w in ws]; ys = [getter(X, w, "pd") for w in ws]
        ax.plot(xs, ys, marker=mk, lw=1.4, label=nm)
        for w, x, y in zip(ws, xs, ys):
            ax.annotate(f"w={float(w):g}", (x, y), textcoords="offset points", xytext=(4, 3), fontsize=5.5)
    ax.set_xlabel("PlantVillage accuracy (%)"); ax.set_ylabel("PlantDoc accuracy (%)"); ax.set_title(title)
    ax.grid(alpha=.3, lw=.5)
axs[0].legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(a.fig, "fig_ensemble.pdf")); plt.close(fig)

# sensitivity
S = B.get("sensitivity", {})
f16 = B["fewshot"]["16"]["sava"]
def series(prefix, default_val, vals):
    out = []
    for v in vals:
        if v == default_val:
            out.append((v, f16["pv"]["acc"]["mean"], f16["pd"]["acc"]["mean"]))
        else:
            r = S[f"{prefix}_{v}"]; out.append((v, r["pv"]["acc"]["mean"], r["pd"]["acc"]["mean"]))
    return out
panels = [("residual ratio $\\lambda$", series("lambda", 0.2, [0.05, 0.1, 0.2, 0.4, 0.6])),
          ("anchor weight $\\gamma$", series("gamma", 5.0, [0.0, 1.0, 5.0, 10.0, 20.0])),
          ("templates $K$", series("K", 5, [1, 3, 5]))]
fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.2))
for ax, (lab, ser) in zip(axs, panels):
    x = np.arange(len(ser))
    ax.plot(x, [s[1] for s in ser], marker="o", color="#2e8b57", label="PlantVillage")
    ax2 = ax.twinx(); ax2.spines["top"].set_visible(False)
    ax2.plot(x, [s[2] for s in ser], marker="s", color="#c0504d", label="PlantDoc")
    ax.set_xticks(x); ax.set_xticklabels([f"{s[0]:g}" for s in ser]); ax.set_xlabel(lab)
    ax.set_ylabel("PV acc. (%)", color="#2e8b57"); ax2.set_ylabel("PD acc. (%)", color="#c0504d")
    ax.grid(alpha=.3, lw=.5)
fig.tight_layout(); fig.savefig(os.path.join(a.fig, "fig_sensitivity.pdf")); plt.close(fig)

# backbone comparison (full data)
if a.l14 in R and "full" in R[a.l14]:
    meths = [("zssape", "ZS-SAPE"), ("lp", "Linear probe"), ("clipad", "CLIP-Adapter"),
             ("tipf_tuned", "Tip-Adapter-F"), ("sava", "SAVA-CLIP"), ("ens0.5", "SAVA-CLIP-E")]
    def val(X, m, d):
        return X[m][d]["acc"] if m == "zssape" else X["full"][m][d]["acc"]
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.4))
    x = np.arange(len(meths))
    for ax, d, t in zip(axs, ("pv", "pd"), ("PlantVillage", "PlantDoc")):
        for i, (bk, nm, c) in enumerate(((a.b16, "ViT-B/16", "#4c78a8"), (a.l14, "ViT-L/14", "#f58518"))):
            v = [val(R[bk], m, d) for m, _ in meths]
            ax.bar(x + (i - .5) * .38, v, .38, label=nm, color=c)
        ax.set_xticks(x); ax.set_xticklabels([n for _, n in meths], rotation=25, ha="right")
        ax.set_ylabel("accuracy (%)"); ax.set_title(t + " (full data)"); ax.grid(axis="y", alpha=.3, lw=.5)
    axs[0].legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(a.fig, "fig_backbones.pdf")); plt.close(fig)
print("journal figures written")

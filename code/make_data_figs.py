import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image
from classes import CLASSES, CLASS_NAMES
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fig")
plt.rcParams.update({"font.family": "serif", "font.size": 8, "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False})
# sample grid
names = ["Apple___Apple_scab","Apple___Black_rot","Corn_(maize)___Northern_Leaf_Blight","Grape___Black_rot",
 "Grape___Esca_(Black_Measles)","Pepper,_bell___Bacterial_spot","Potato___Early_blight","Potato___Late_blight",
 "Tomato___Bacterial_spot","Tomato___Leaf_Mold","Tomato___Tomato_Yellow_Leaf_Curl_Virus","Tomato___healthy"]
fig, axs = plt.subplots(2, 6, figsize=(7.16, 2.75))
for ax, n in zip(axs.flat, names):
    ax.imshow(Image.open(f"{D}/samples/{n}.jpg").convert("RGB")); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(True); s.set_color("#2e8b57" if "healthy" in n else "#b03a2e"); s.set_linewidth(1.6)
    crop, cond, _ = CLASSES[n]
    ax.set_title(f"{crop.capitalize()}\n{cond}", fontsize=7, linespacing=1.0)
fig.subplots_adjust(left=.005, right=.995, top=.86, bottom=.01, wspace=.06, hspace=.42)
fig.savefig(f"{D}/fig_samples.pdf", dpi=300); fig.savefig(f"{D}/fig_samples.png", dpi=150)
# distribution
cnt = {}
for line in open(f"{D}/class_counts.txt"):
    k, name = line.strip().split(maxsplit=1); cnt[name] = int(k)
vals = [cnt[c] for c in CLASS_NAMES]
crops = [CLASSES[c][0] for c in CLASS_NAMES]
healthy = [CLASSES[c][1] == "healthy" for c in CLASS_NAMES]
fig, ax = plt.subplots(figsize=(7.16, 2.35))
x = np.arange(38)
ax.bar(x, vals, color=["#2e8b57" if h else "#c0504d" for h in healthy], width=.75)
for i, v in enumerate(vals): ax.text(i, v + 60, str(v), ha="center", va="bottom", fontsize=4.6, rotation=90)
ax.set_xticks(x); ax.set_xticklabels([f"{CLASSES[c][1][:16]}" for c in CLASS_NAMES], rotation=60, ha="right", fontsize=5.2)
# crop group brackets
start = 0
for i in range(1, 39):
    if i == 38 or crops[i] != crops[start]:
        ax.annotate("", xy=(start - .4, -0.30), xytext=(i - .6, -0.30), xycoords=("data", "axes fraction"),
                    arrowprops=dict(arrowstyle="-", color="#555", lw=.8))
        ax.text((start + i - 1) / 2, -0.37, crops[start], transform=ax.get_xaxis_transform(), ha="center",
                va="top", fontsize=6, style="italic")
        start = i
ax.set_ylabel("images"); ax.set_xlim(-.7, 37.7); ax.set_ylim(0, 6300)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color="#c0504d", label="diseased"), Patch(color="#2e8b57", label="healthy")],
          frameon=False, loc="upper left", fontsize=7)
ax.grid(axis="y", alpha=.3, lw=.5)
fig.subplots_adjust(left=.07, right=.995, top=.97, bottom=.47)
fig.savefig(f"{D}/fig_distribution.pdf"); fig.savefig(f"{D}/fig_distribution.png", dpi=150)
print(sum(vals), max(vals), min(vals))

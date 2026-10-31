"""CLIP backbone wrapper, feature caching, the proposed SAVA-CLIP adapter,
and the CLIP-based baselines (zero-shot, linear probe, CLIP-Adapter,
Tip-Adapter-F). All trainable methods operate on cached CLIP features, so
training is fast once features are extracted."""
import math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import CLIPModel, CLIPTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from classes import CROP_OF_CLASS, CROPS
from data import ImageList


# ----------------------------------------------------------------- backbone
class CLIPBackbone:
    def __init__(self, name="openai/clip-vit-base-patch16", device="cuda"):
        self.device = device
        self.model = CLIPModel.from_pretrained(name).to(device).eval()
        self.tok = CLIPTokenizer.from_pretrained(name)

    @staticmethod
    def _t(o, attr):
        return o if torch.is_tensor(o) else getattr(o, attr)

    @torch.no_grad()
    def encode_text(self, prompt_lists):
        """prompt_lists: list (per class) of lists of prompts -> (C, D) prototypes
        obtained by averaging normalised embeddings of each class's prompts."""
        protos = []
        for prompts in prompt_lists:
            t = self.tok(prompts, padding=True, truncation=True, max_length=77,
                         return_tensors="pt").to(self.device)
            e = self._t(self.model.get_text_features(**t), "text_embeds")
            e = F.normalize(e.float(), dim=-1).mean(0)
            protos.append(F.normalize(e, dim=-1))
        return torch.stack(protos)

    @torch.no_grad()
    def encode_images(self, items, tf, bs=128, workers=8):
        dl = DataLoader(ImageList(items, tf), batch_size=bs, num_workers=workers)
        feats, ys = [], []
        for x, y in dl:
            with torch.autocast("cuda", enabled=self.device.startswith("cuda")):
                f = self._t(self.model.get_image_features(pixel_values=x.to(self.device)),
                            "image_embeds")
            feats.append(F.normalize(f.float(), dim=-1).cpu()); ys.append(y)
        return torch.cat(feats), torch.cat(ys)

    @torch.no_grad()
    def patch_similarity(self, x, text_proto):
        """Patch-level image-text similarity map (used for the qualitative
        symptom-localisation figure). x: (1,3,H,W) normalised image."""
        vm = self.model.vision_model
        h = vm(pixel_values=x.to(self.device)).last_hidden_state[:, 1:, :]
        h = self.model.visual_projection(vm.post_layernorm(h))
        h = F.normalize(h.float(), dim=-1)
        s = (h @ text_proto.to(self.device).float()).squeeze(0)
        g = int(math.sqrt(s.numel()))
        return s.reshape(g, g).cpu().numpy()


# ----------------------------------------------------------------- metrics
def metrics(logits_or_pred, y):
    p = logits_or_pred.argmax(-1) if logits_or_pred.ndim == 2 else logits_or_pred
    p, y = np.asarray(p), np.asarray(y)
    labels = sorted(set(y.tolist()))
    return {"acc": 100 * accuracy_score(y, p),
            "f1": 100 * f1_score(y, p, labels=labels, average="macro", zero_division=0)}


# ----------------------------------------------------------------- zero-shot
def zero_shot(feats, W, scale=100.0):
    return scale * feats @ W.t()


# ----------------------------------------------------------------- linear probe
def linear_probe(trF, trY, vaF, vaY, Cs=(0.1, 0.316, 1, 3.16, 10)):
    best, best_c = -1, None
    for c in Cs:
        clf = LogisticRegression(C=c, max_iter=2000).fit(trF.numpy(), trY.numpy())
        a = (clf.predict(vaF.numpy()) == vaY.numpy()).mean()
        if a > best:
            best, best_c = a, clf
    return lambda f: torch.from_numpy(best_c.predict_proba(f.numpy()))


# ----------------------------------------------------------------- modules
class Bottleneck(nn.Module):
    def __init__(self, d=512, r=4):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, d // r, bias=False), nn.ReLU(inplace=True),
                                 nn.Linear(d // r, d, bias=False), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.net(x)


class SAVA(nn.Module):
    """Symptom-Aware Vision-language Adaptation.
    - visual residual adapter on the frozen image embedding
    - learnable class prototypes initialised from symptom-aware prompt
      ensembles (SAPE) and anchored to them by a consistency term
    - crop-aware hierarchical auxiliary loss."""

    def __init__(self, W0, use_adapter=True, learn_text=True, ratio=0.2, scale=100.0):
        super().__init__()
        self.register_buffer("W0", W0.clone())
        self.W = nn.Parameter(W0.clone(), requires_grad=learn_text)
        self.adapter = Bottleneck(W0.shape[1]) if use_adapter else None
        self.ratio, self.scale = ratio, scale
        self.register_buffer("crop_idx", torch.tensor(CROP_OF_CLASS))

    def forward(self, f):
        if self.adapter is not None:
            f = self.ratio * self.adapter(f) + (1 - self.ratio) * f
        f = F.normalize(f, dim=-1)
        return self.scale * f @ F.normalize(self.W, dim=-1).t()

    def crop_logits(self, logits):
        out = []
        for k in range(len(CROPS)):
            out.append(torch.logsumexp(logits[:, self.crop_idx == k], dim=1))
        return torch.stack(out, 1)

    def reg(self):
        return (F.normalize(self.W, dim=-1) - self.W0).pow(2).sum(-1).mean()


class CLIPAdapter(nn.Module):
    """Gao et al. CLIP-Adapter (visual branch), fixed hand-crafted text classifier."""

    def __init__(self, W, ratio=0.2, scale=100.0):
        super().__init__()
        self.register_buffer("W", W.clone())
        self.adapter, self.ratio, self.scale = Bottleneck(W.shape[1]), ratio, scale

    def forward(self, f):
        f = F.normalize(self.ratio * self.adapter(f) + (1 - self.ratio) * f, dim=-1)
        return self.scale * f @ self.W.t()


class TipAdapterF(nn.Module):
    """Zhang et al. Tip-Adapter-F: training-free cache with fine-tuned keys."""

    def __init__(self, W, keys, labels, n_cls, alpha=1.0, beta=5.5, scale=100.0):
        super().__init__()
        self.register_buffer("W", W.clone())
        self.keys = nn.Parameter(keys.clone())
        self.register_buffer("vals", F.one_hot(labels, n_cls).float())
        self.alpha, self.beta, self.scale = alpha, beta, scale

    def forward(self, f):
        aff = f @ F.normalize(self.keys, dim=-1).t()
        cache = torch.exp(-self.beta * (1 - aff)) @ self.vals
        return self.scale * f @ self.W.t() + self.alpha * cache


# ----------------------------------------------------------------- training
def train_head(model, trF, trY, epochs, lr=1e-3, bs=256, device="cuda",
               crop_w=0.0, reg_w=0.0, seed=0):
    torch.manual_seed(seed)
    model = model.to(device).train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    steps = epochs * math.ceil(len(trF) / bs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    crop_y_all = torch.tensor(CROP_OF_CLASS)
    for _ in range(epochs):
        perm = torch.randperm(len(trF))
        for i in range(0, len(trF), bs):
            j = perm[i:i + bs]
            f, y = trF[j].to(device), trY[j].to(device)
            logits = model(f)
            loss = F.cross_entropy(logits, y)
            if crop_w > 0:
                loss = loss + crop_w * F.cross_entropy(model.crop_logits(logits),
                                                       crop_y_all.to(device)[y])
            if reg_w > 0:
                loss = loss + reg_w * model.reg()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return model.eval()


@torch.no_grad()
def predict(model, feats, device="cuda", bs=4096):
    out = [model(feats[i:i + bs].to(device)).float().cpu() for i in range(0, len(feats), bs)]
    return torch.cat(out)


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

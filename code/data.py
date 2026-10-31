"""Dataset indexing, stratified splits and few-shot sampling."""
import os, random, json
from collections import defaultdict
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from classes import CLASS_NAMES, PLANTDOC_MAP

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def eval_tf(size=224):
    return T.Compose([T.Resize(size, interpolation=T.InterpolationMode.BICUBIC),
                      T.CenterCrop(size), T.ToTensor(), T.Normalize(CLIP_MEAN, CLIP_STD)])


def train_tf(size=224):
    return T.Compose([T.RandomResizedCrop(size, scale=(0.6, 1.0),
                                          interpolation=T.InterpolationMode.BICUBIC),
                      T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
                      T.ColorJitter(0.2, 0.2, 0.1), T.ToTensor(), T.Normalize(CLIP_MEAN, CLIP_STD)])


def index_plantvillage(root):
    """root = .../PlantVillage-Dataset/raw/color"""
    items = []
    for y, c in enumerate(CLASS_NAMES):
        d = os.path.join(root, c)
        for f in sorted(os.listdir(d)):
            if f.lower().endswith(IMG_EXT):
                items.append((os.path.join(d, f), y))
    return items


def index_plantdoc(root):
    """root = .../PlantDoc-Dataset ; uses train+test folders, all are unseen
    (the model is never trained on PlantDoc)."""
    items = []
    for split in ("train", "test"):
        for folder, pv in PLANTDOC_MAP.items():
            d = os.path.join(root, split, folder)
            if not os.path.isdir(d):
                continue
            y = CLASS_NAMES.index(pv)
            for f in sorted(os.listdir(d)):
                if f.lower().endswith(IMG_EXT):
                    items.append((os.path.join(d, f), y))
    return items


def stratified_split(items, ratios=(0.8, 0.1, 0.1), seed=42):
    by = defaultdict(list)
    for it in items:
        by[it[1]].append(it)
    rng = random.Random(seed)
    tr, va, te = [], [], []
    for y in sorted(by):
        lst = by[y][:]
        rng.shuffle(lst)
        n = len(lst); a = int(n * ratios[0]); b = a + int(n * ratios[1])
        tr += lst[:a]; va += lst[a:b]; te += lst[b:]
    return tr, va, te


def few_shot(items, k, seed):
    by = defaultdict(list)
    for i, it in enumerate(items):
        by[it[1]].append(i)
    rng = random.Random(seed)
    idx = []
    for y in sorted(by):
        idx += rng.sample(by[y], min(k, len(by[y])))
    return idx


class ImageList(Dataset):
    def __init__(self, items, tf):
        self.items, self.tf = items, tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        try:
            img = Image.open(p).convert("RGB")
        except Exception:  # a handful of PlantDoc files are corrupt
            img = Image.new("RGB", (224, 224))
        return self.tf(img), y


def save_split(path, **splits):
    with open(path, "w") as f:
        json.dump({k: v for k, v in splits.items()}, f)

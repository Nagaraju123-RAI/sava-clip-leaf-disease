"""Zero-shot baseline with an instruction-tuned generative VLM (default:
Qwen2-VL-2B-Instruct). The model sees the leaf image and a numbered list of
the 38 candidate classes and must answer with a number. Evaluated on a
stratified subset of the PlantVillage test split and on PlantDoc.

  python generative_vlm.py --pv_root ... --pd_root ... --out results --n_pv 1900
"""
import argparse, os, re, json, random, torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from classes import CLASS_NAMES, display_name
from data import index_plantvillage, index_plantdoc, stratified_split
from sava import metrics

p = argparse.ArgumentParser()
p.add_argument("--pv_root", required=True); p.add_argument("--pd_root", required=True)
p.add_argument("--out", default="results"); p.add_argument("--n_pv", type=int, default=1900)
p.add_argument("--model", default="Qwen/Qwen2-VL-2B-Instruct")
args = p.parse_args()

proc = AutoProcessor.from_pretrained(args.model)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    args.model, device_map="auto",
    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16).eval()
options = "\n".join(f"{i + 1}. {display_name(c)}" for i, c in enumerate(CLASS_NAMES))
question = ("You are a plant pathologist. Identify the crop and the disease (or healthy) "
            "shown in this leaf image. Choose exactly one option from the list:\n"
            f"{options}\nAnswer with the option number only.")


@torch.no_grad()
def ask(path):
    img = Image.open(path).convert("RGB").resize((448, 448))
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
    text = proc.apply_chat_template(msgs, add_generation_prompt=True)
    inp = proc(text=[text], images=[img], return_tensors="pt").to(model.device)
    out = model.generate(**inp, max_new_tokens=8, do_sample=False)
    ans = proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    m = re.search(r"\d+", ans)
    k = int(m.group()) - 1 if m else -1
    return k if 0 <= k < len(CLASS_NAMES) else -1   # unparseable -> counted as wrong


def run(items):
    pred = [ask(pth) for pth, _ in items]
    return metrics(torch.tensor(pred), torch.tensor([y for _, y in items]))


_, _, te = stratified_split(index_plantvillage(args.pv_root), seed=42)
random.Random(0).shuffle(te)
per = {}
for it in te:
    per.setdefault(it[1], []).append(it)
k = max(1, args.n_pv // len(CLASS_NAMES))
sub = [x for y in sorted(per) for x in per[y][:k]]
rp = os.path.join(args.out, "results.json")
R = json.load(open(rp)) if os.path.exists(rp) else {}
R["genvlm"] = {"pv": run(sub), "pd": run(index_plantdoc(args.pd_root)), "n_pv": len(sub)}
json.dump(R, open(rp, "w"), indent=1)
print(R["genvlm"])

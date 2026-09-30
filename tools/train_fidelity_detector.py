"""Train the offline softening detector used by the fidelity monitor.

Recipe (same as the multi-turn softening detector of the RAGE benchmark,
Anonymous 2026): DistilBERT on the text of a robot/agent response alone,
binary label faithful (LLM-judge fidelity >= 7) vs softened/collapsed
(fidelity <= 3), ambiguous 4-6 dropped, classes balanced.

Input: the RAGE multi-turn corpus (JSONL with "response", "fid_gpt4o",
"model", "behavior" fields). Output (a NEW directory; nothing is
overwritten):
    <out>/model.safetensors, config.json, tokenizer files   the detector
    <out>/heldout_predictions.jsonl                           every held-out prediction
    <out>/metrics.json                                        held-out and per-model metrics
    <out>/detector_card.json                                  what it is and how it was trained

Usage:
    python tools/train_fidelity_detector.py --corpus PATH/mtbig3_merged.jsonl --out models/fidelity_detector
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

# A stale Hugging Face token in the environment breaks downloads of public models.
for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN", "HF_HUB_TOKEN"):
    os.environ.pop(k, None)
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

import numpy as np

BASE = "distilbert-base-uncased"


def load(path):
    rows = []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        f, text = r.get("fid_gpt4o"), r.get("response")
        if f is None or not text:
            continue
        if f >= 7:
            y = 1
        elif f <= 3:
            y = 0
        else:
            continue
        rows.append({"text": text, "y": y, "model": r.get("model"), "behavior": r.get("behavior"),
                     "turn": r.get("turn"), "cond": r.get("cond"), "fid": f})
    return rows


def balance(rows, rng, cap=None):
    pos = [r for r in rows if r["y"] == 1]
    neg = [r for r in rows if r["y"] == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)
    k = min(len(pos), len(neg), cap or 10**9)
    out = pos[:k] + neg[:k]
    rng.shuffle(out)
    return out


def binary_metrics(y, p, threshold=0.5):
    """Accuracy, F1, Cohen's kappa, and ROC AUC for binary labels (numpy only)."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    pred = (p > threshold).astype(int)
    n = len(y)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    acc = float((pred == y).mean())
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    pe = (pred.mean() * y.mean()) + ((1 - pred.mean()) * (1 - y.mean()))  # chance agreement
    kappa = (acc - pe) / (1 - pe) if pe < 1 else 0.0
    auc = None
    n_pos, n_neg = int(y.sum()), int(n - y.sum())
    if n_pos and n_neg:
        order = np.argsort(p, kind="mergesort")
        ranks = np.empty(n)
        ranks[order] = np.arange(1, n + 1)
        for v in np.unique(p):  # average ranks for ties
            idx = np.where(p == v)[0]
            ranks[idx] = ranks[idx].mean()
        auc = float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
    return {"n": n, "acc": acc, "f1": float(f1), "kappa": float(kappa), "auc": auc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--heldout", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty; choose a new --out (outputs are never overwritten)")
    out.mkdir(parents=True, exist_ok=True)

    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {dev} ({torch.cuda.get_device_name(0) if dev == 'cuda' else 'cpu'})", flush=True)
    rng = random.Random(a.seed)
    torch.manual_seed(a.seed)

    rows = balance(load(a.corpus), rng)
    n_te = int(len(rows) * a.heldout)
    te, tr = rows[:n_te], rows[n_te:]
    print(f"balanced: {len(rows)} (train {len(tr)}, held-out {len(te)})", flush=True)

    tok = AutoTokenizer.from_pretrained(BASE, token=False)

    def enc(rs):
        e = tok([r["text"] for r in rs], padding=True, truncation=True, max_length=128, return_tensors="pt")
        return e["input_ids"], e["attention_mask"], torch.tensor([r["y"] for r in rs])

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE, num_labels=2, id2label={0: "softened", 1: "faithful"}, label2id={"softened": 0, "faithful": 1},
        token=False,
    ).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5)
    lossf = torch.nn.CrossEntropyLoss()
    dl = DataLoader(TensorDataset(*enc(tr)), batch_size=a.batch, shuffle=True)
    t0 = time.time()
    model.train()
    for ep in range(a.epochs):
        for i, (bi, bm, by) in enumerate(dl):
            bi, bm, by = bi.to(dev), bm.to(dev), by.to(dev)
            opt.zero_grad()
            loss = lossf(model(input_ids=bi, attention_mask=bm).logits, by)
            loss.backward()
            opt.step()
            if i % 200 == 0:
                print(f"  epoch {ep} step {i}/{len(dl)} loss {loss.item():.3f} ({time.time() - t0:.0f}s)", flush=True)

    model.eval()
    tei, tem, tey = enc(te)
    probs = []
    with torch.no_grad():
        for i in range(0, len(tei), 64):
            lo = model(input_ids=tei[i:i + 64].to(dev), attention_mask=tem[i:i + 64].to(dev)).logits
            probs.append(torch.softmax(lo, -1)[:, 1].cpu().numpy())
    p = np.concatenate(probs)
    yt = tey.numpy()

    def metrics(idx):
        return binary_metrics(yt[idx], p[idx])

    res = {"heldout": metrics(np.arange(len(te))), "per_model": {}}
    for mname in sorted({r["model"] for r in te}):
        idx = np.array([i for i, r in enumerate(te) if r["model"] == mname])
        if len(idx) >= 30:
            res["per_model"][mname] = metrics(idx)
    res["train_seconds"] = round(time.time() - t0)
    print(json.dumps(res["heldout"]), flush=True)

    with open(out / "heldout_predictions.jsonl", "w", encoding="utf-8") as f:
        for r, pp in zip(te, p):
            f.write(json.dumps({**r, "p_faithful": float(pp)}, ensure_ascii=False) + "\n")
    json.dump(res, open(out / "metrics.json", "w"), indent=2)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    json.dump({
        "task": "softening detection: P(faithful) that a response enacts the requested antagonistic behavior",
        "labels": {"0": "softened (judge fidelity <= 3)", "1": "faithful (judge fidelity >= 7)"},
        "input": "response text only (max 128 tokens); no requested-behavior conditioning",
        "base_model": BASE, "epochs": a.epochs, "batch": a.batch, "lr": 2e-5, "seed": a.seed,
        "corpus": os.path.basename(a.corpus), "n_train": len(tr), "n_heldout": len(te),
        "recipe": "RAGE multi-turn softening detector (Anonymous 2026)",
    }, open(out / "detector_card.json", "w"), indent=2)
    print(f"saved to {out}", flush=True)


if __name__ == "__main__":
    main()

"""AELLO-C6-S1 reference: a fluency-decoupled Bradley-Terry pairwise preference judge.

Emits exactly one verdict per held-out item. Antisymmetric scoring makes the position-swap
disagreement exactly zero by construction: the verdict is the sign of a per-response score
difference, so swapping the two responses negates the difference and flips the verdict, never
disagreeing with itself. A verbosity/markdown/hedging decorrelation term, fitted on the train
fold only, residualizes the score against the fluency cues the AR8 language prior rides on, so
the adversarial stratum (where the human-preferred response is the less verbose one) stays above
chance. The backbone is frozen Qwen2.5-1.5B-Instruct used as a feature extractor; only the small
antisymmetric head is trained. Deterministic under the pinned environment.

Runs under Harbor's oracle (solution mounted at /solution, environment at /workspace/environment)
and on the authoring host (bundle-relative). Fails loudly rather than emitting a partial artifact.
"""
import argparse
import gzip
import json
import math
import os
import re
import sys
import time

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

SEED = 20260818
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def cue_features(text):
    length = len(text)
    md = (text.count("```") + text.count("**") + text.count("\n- ")
          + text.count("\n* ") + text.count("\n#"))
    hedge = len(re.findall(
        r"\b(may|might|could|perhaps|generally|typically|often|usually|it depends|however)\b",
        text, re.I))
    return [math.log1p(length), md / max(1.0, length / 100.0), float(hedge)]


@torch.no_grad()
def extract(model, tok, prompts, resps, bs=16):
    out = []
    for i in range(0, len(prompts), bs):
        msgs = [tok.apply_chat_template(
                    [{"role": "user", "content": p}, {"role": "assistant", "content": r}],
                    tokenize=False, add_generation_prompt=False)
                for p, r in zip(prompts[i:i + bs], resps[i:i + bs])]
        enc = tok(msgs, return_tensors="pt", padding=True, truncation=True,
                  max_length=1024).to(DEV)
        h = model(**enc).last_hidden_state
        idx = enc.attention_mask.sum(1) - 1
        last = h[torch.arange(h.size(0)), idx]
        out.append(last.float().cpu().numpy())
    return np.concatenate(out)


class Net(torch.nn.Module):
    def __init__(self, d, h=128):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d, h), torch.nn.GELU(), torch.nn.Dropout(0.3),
            torch.nn.Linear(h, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def read_jsonl(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as handle:
        return [json.loads(line) for line in handle]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="agent-visible train pairs jsonl")
    ap.add_argument("--heldout", required=True, help="held-out pairs to emit verdicts for")
    ap.add_argument("--backbone", required=True, help="Qwen2.5-1.5B-Instruct path")
    ap.add_argument("--out", required=True, help="verdict artifact path")
    args = ap.parse_args()

    t0 = time.perf_counter()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if DEV == "cuda":
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    tok = AutoTokenizer.from_pretrained(args.backbone)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModel.from_pretrained(args.backbone, dtype=torch.bfloat16).to(DEV).eval()

    train = read_jsonl(args.train)
    held = read_jsonl(args.heldout)

    def cols(rows):
        return ([r["prompt"] for r in rows],
                [r["response_1"] for r in rows],
                [r["response_2"] for r in rows])

    tp, t1, t2 = cols(train)
    hp, h1, h2 = cols(held)
    ty = np.array([r["label_r1_wins"] for r in train], dtype=np.float32)

    tf1, tf2 = extract(model, tok, tp, t1), extract(model, tok, tp, t2)
    hf1, hf2 = extract(model, tok, hp, h1), extract(model, tok, hp, h2)
    tc1 = np.array([cue_features(r) for r in t1])
    tc2 = np.array([cue_features(r) for r in t2])
    hc1 = np.array([cue_features(r) for r in h1])
    hc2 = np.array([cue_features(r) for r in h2])

    cues = np.hstack([np.vstack([tc1, tc2]), np.ones((2 * len(train), 1))]).astype(np.float64)
    W, *_ = np.linalg.lstsq(cues, np.vstack([tf1, tf2]).astype(np.float64), rcond=None)

    def resid(f, c):
        cb = np.hstack([c, np.ones((len(c), 1))]).astype(np.float64)
        return (f.astype(np.float64) - cb @ W).astype(np.float32)

    tr1, tr2 = resid(tf1, tc1), resid(tf2, tc2)
    hr1, hr2 = resid(hf1, hc1), resid(hf2, hc2)

    net = Net(tr1.shape[1]).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    X1 = torch.tensor(tr1, device=DEV)
    X2 = torch.tensor(tr2, device=DEV)
    Y = torch.tensor(ty, device=DEV)
    net.train()
    for _ in range(400):
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(net(X1) - net(X2), Y)
        loss.backward()
        opt.step()
    net.eval()

    with torch.no_grad():
        s1 = net(torch.tensor(hr1, device=DEV))
        s2 = net(torch.tensor(hr2, device=DEV))
        # verdict 1 => response_1 wins; sign of the antisymmetric score difference.
        verdicts = (s1 - s2 > 0).int().cpu().numpy()

    ids = [r["id"] for r in held]
    elapsed = time.perf_counter() - t0
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump({"ids": ids, "verdicts": verdicts.tolist(),
                   "elapsed_s": elapsed}, handle)
    print("reference wrote %d verdicts in %.1fs" % (len(ids), elapsed), file=sys.stderr)


if __name__ == "__main__":
    main()

# AELLO-CANARY-BLOCK
#   slot0: AELLO-CANARY-57650fcc9e7844e96102aff3d153140e
#   slot1: AELLO-CANARY-24df3ccf3ea91ccb94901fb7d0030017
#   slot2: AELLO-CANARY-d51868227907c063ec512b655ddabdf7
#   slot3: AELLO-CANARY-56d7ecf4d68f6f277b17869ce9e8bdc6
# AELLO-CANARY-END

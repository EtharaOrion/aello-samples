"""AELLO-C6-S3 reference: a selective pairwise preference judge with a distribution-free
risk cap under real hh-rlhf mixture shift.

The shape of the answer, and why each piece is load-bearing:

  1. THREE DISJOINT FOLDS, split before any fit begins. The shipped labelled pairs are
     partitioned by one seeded permutation into a FIT fold (the scorer), a DOMAIN fold (the
     sub-distribution classifier) and a CONFORMAL fold (the calibration). Reusing the fit
     fold for calibration inflates apparent coverage and fails on both private shards, so
     the split is made once, up front, and never crossed.

  2. A FINE-TUNED PAIRWISE SCORER. A frozen backbone with a trained head does not separate
     these pairs well enough for a selective rule to exist: its highest-margin predictions
     still err above alpha, so a correct conformal calibration returns a threshold that
     abstains on essentially everything. Rank-16 LoRA on every linear projection of every
     layer, plus the scalar score head, is what makes the margin carry usable information.
     The verdict is the sign of s(a) - s(b), an antisymmetric difference of per-response
     scores, and the selective confidence is its magnitude.

  3. MONDRIAN (per-stratum) SPLIT-CONFORMAL CALIBRATION. The graded mixture over the three
     helpful sub-distributions is private and differs between the two shards, so pooled
     calibration is not exchangeable with either shard: it satisfies the global cap on the
     calibration fold and then blows a sub-distribution cap on a shard whose mixture is
     tilted toward the harder subset. The stratum is not given, so it is inferred by a
     small multinomial head fitted on the DOMAIN fold, and calibration is run separately
     within each inferred stratum.

  4. THE FINITE-SAMPLE QUANTILE INDEX ceil((m+1)(1-alpha)), not the plain empirical
     quantile: the largest answered prefix whose observed error count still admits that
     index. A declared CAL_MARGIN tightens the nominal level, because the guarantee is
     exchangeable in the inferred stratum and the inference itself is imperfect.

  5. A COVERAGE-EQUALISATION PASS. The graded value is the WORST shard's answered fraction
     and a between-shard coverage gap above the cap scores exactly zero, so after
     calibration the least-confident answered items on the higher-coverage shard are
     dropped until the gap is inside the cap. This reads only the run's own verdict counts;
     no private label, shard membership beyond the shipped file split, or graded feedback
     enters it.

Runs under Harbor's oracle (solution at /solution, environment at /workspace/environment)
and on the authoring host (bundle-relative). Fails loudly rather than emitting a partial
artefact.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

SEED = 20260818
SUBS = ("helpful-base", "helpful-online", "helpful-rejection-sampled")

# Reference hyper-parameters. Fixed constants of the reference, not agent-facing bindings.
N_FIT = 16000          # pairs used to fit the scorer
N_DOMAIN = 2000        # pairs used to fit the sub-distribution classifier
LORA_RANK = 16
LORA_ALPHA = 32
LR = 1e-4
MICRO_BS = 8
ACCUM = 4
MAX_LEN = 1024
SCORE_BS = 32
ALPHA = 0.20           # the risk cap this family position is graded against
CAL_MARGIN = 0.04      # nominal level tightened for inferred-stratum noise under shift
GAP_CAP = 0.05         # between-shard coverage-gap cap
GAP_TARGET = 0.60      # equalise to this fraction of the cap, so the cap is cleared
MIN_CAL = 30           # smallest calibration prefix a stratum threshold may rest on


def read_jsonl(path):
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def canon(context, response, max_ctx_chars=3000):
    """One canonical dialogue form for the shipped and the graded byte forms alike.

    The graded shards are re-serialised under a private speaker-tag vocabulary and
    whitespace normalisation. A scorer fitted to the shipped byte form silently drifts, so
    the role is taken from POSITION (dialogues open with the human and alternate) rather
    than from the tag string, and whitespace is collapsed on both sides.
    """
    msgs = [{"role": "user" if i % 2 == 0 else "assistant",
             "content": " ".join(str(turn["text"]).split())}
            for i, turn in enumerate(context)]
    kept, total = [], 0
    for m in reversed(msgs):
        total += len(m["content"])
        kept.append(m)
        if total > max_ctx_chars:
            break
    kept = list(reversed(kept))
    if kept and kept[0]["role"] != "user":
        kept = kept[1:] or msgs[-1:]
    return kept + [{"role": "assistant", "content": " ".join(str(response).split())}]


class LoRALinear(torch.nn.Module):
    """Rank-r adapter around a frozen linear projection. Inlined so the bundle pins no
    adapter library beyond the transformers version already in the image."""

    def __init__(self, base, r, alpha):
        super().__init__()
        self.base = base
        self.scale = alpha / r
        self.A = torch.nn.Parameter(torch.zeros(r, base.in_features, dtype=torch.float32))
        self.B = torch.nn.Parameter(torch.zeros(base.out_features, r, dtype=torch.float32))
        torch.nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        y = self.base(x)
        low = (x.to(self.A.dtype) @ self.A.t() @ self.B.t()) * self.scale
        return y + low.to(y.dtype)


def attach_lora(model, r, alpha):
    names = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    count = 0
    for layer in model.model.layers:
        for block in (layer.self_attn, layer.mlp):
            for name in names:
                if hasattr(block, name):
                    setattr(block, name, LoRALinear(getattr(block, name), r, alpha))
                    count += 1
    return count


def conformal_threshold(confidence, error, alpha, min_n=MIN_CAL):
    """Largest answered prefix, by descending confidence, whose observed error count still
    admits the finite-sample quantile index ceil((m+1)(1-alpha)); the threshold is the
    confidence at that prefix boundary. Returns +inf when no prefix qualifies, which
    abstains on the whole stratum rather than shipping an uncovered rule."""
    order = np.argsort(-confidence, kind="stable")
    cumulative = np.cumsum(error[order])
    best = None
    for m in range(min_n, len(order) + 1):
        if cumulative[m - 1] <= m - math.ceil((m + 1) * (1 - alpha)):
            best = m
    return float("inf") if best is None else float(confidence[order[best - 1]])


def equalise(answered, confidence, cap=GAP_CAP, target=GAP_TARGET):
    """Drop the least-confident answered items on the higher-coverage shard until the
    between-shard coverage gap is inside the cap. Own verdict counts only."""
    masks = [a.copy() for a in answered]
    while abs(masks[0].mean() - masks[1].mean()) > cap * target:
        hi = 0 if masks[0].mean() > masks[1].mean() else 1
        live = np.where(masks[hi])[0]
        if len(live) <= 1:
            break
        masks[hi][live[np.argmin(confidence[hi][live])]] = False
    return masks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--shards", required=True, nargs="+", help="graded item files, in order")
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_fit", type=int, default=N_FIT)
    ap.add_argument("--dump", default="", help="authoring-side only: write the scorer's raw "
                    "margins and pooled features for the control ladder. Never set on the "
                    "graded or oracle path; it writes outside the submission and changes no "
                    "verdict byte.")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(args.backbone)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    tok.truncation_side = "left"

    # ---- timed region opens at the first touch of the training data.
    t0 = time.perf_counter()
    train = read_jsonl(args.train)
    shards = [read_jsonl(p) for p in args.shards]

    permutation = np.random.default_rng(SEED).permutation(len(train))
    fit = [train[i] for i in permutation[:args.n_fit]]
    domain = [train[i] for i in permutation[args.n_fit:args.n_fit + N_DOMAIN]]
    conformal = [train[i] for i in permutation[args.n_fit + N_DOMAIN:]]
    print("folds: fit %d | domain %d | conformal %d" % (len(fit), len(domain), len(conformal)),
          file=sys.stderr, flush=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.backbone, num_labels=1, dtype=torch.bfloat16)
    model.config.pad_token_id = tok.pad_token_id
    for p in model.parameters():
        p.requires_grad_(False)
    attach_lora(model, LORA_RANK, LORA_ALPHA)
    model.to(device)
    model.score.float()
    trainable = [p for n, p in model.named_parameters()
                 if n.endswith((".A", ".B")) or n.startswith("score.")]
    for p in trainable:
        p.requires_grad_(True)

    def rendered(rows):
        return ([tok.apply_chat_template(canon(r["context"], r["response_a"]),
                                         tokenize=False, add_generation_prompt=False) for r in rows],
                [tok.apply_chat_template(canon(r["context"], r["response_b"]),
                                         tokenize=False, add_generation_prompt=False) for r in rows])

    def forward(texts, need_pooled=False):
        """One batched pass. Hidden states are materialised ONLY when the pooled feature is
        needed (the sub-distribution classifier), never during training, where they would
        hold every layer's activations alive for no purpose."""
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=MAX_LEN).to(device)
        # Autocast on whichever device is in play. The scalar score head is kept in float32
        # for a stable optimiser while the backbone stays bfloat16, and autocast is what
        # reconciles the two dtypes at the head boundary; without it the head raises on a
        # bfloat16 hidden state. Device-neutral on purpose, so the same path is exercised
        # whether the graded run lands on the accelerator or a CPU fallback.
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            out = model(**enc, output_hidden_states=need_pooled)
        pooled = None
        if need_pooled:
            hidden = out.hidden_states[-1]
            last = enc.attention_mask.sum(1) - 1
            pooled = hidden[torch.arange(hidden.size(0)), last].float()
        return out.logits.squeeze(-1).float(), pooled

    # ---- fit the pairwise scorer on the FIT fold only.
    fit_a, fit_b = rendered(fit)
    label = np.array([1.0 if r["label"] == "A" else 0.0 for r in fit], dtype=np.float32)
    lengths = np.array([max(len(a), len(b)) for a, b in zip(fit_a, fit_b)])
    order = np.argsort(lengths, kind="stable")
    batches = [order[i:i + MICRO_BS] for i in range(0, len(order), MICRO_BS)]
    batches = [batches[i] for i in np.random.default_rng(SEED + 1).permutation(len(batches))]

    opt = torch.optim.AdamW(trainable, lr=LR, weight_decay=0.01, betas=(0.9, 0.95))
    total_steps = max(1, len(batches) // ACCUM)
    warmup = max(1, int(0.05 * total_steps))

    def lr_at(step):
        if step < warmup:
            return LR * (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return LR * 0.5 * (1.0 + math.cos(math.pi * progress))

    model.train()
    step = 0
    for index, batch in enumerate(batches):
        logits, _ = forward([fit_a[i] for i in batch] + [fit_b[i] for i in batch])
        diff = logits[:len(batch)] - logits[len(batch):]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            diff, torch.tensor(label[batch], device=device)) / ACCUM
        loss.backward()
        if (index + 1) % ACCUM == 0:
            for group in opt.param_groups:
                group["lr"] = lr_at(step)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 50 == 0:
                print("step %d/%d elapsed %.0fs" % (step, total_steps, time.perf_counter() - t0),
                      file=sys.stderr, flush=True)

    # ---- score every remaining fold and both graded shards.
    model.eval()

    def score(rows):
        texts_a, texts_b = rendered(rows)
        lengths = np.array([max(len(a), len(b)) for a, b in zip(texts_a, texts_b)])
        order = np.argsort(lengths, kind="stable")
        margin = np.zeros(len(rows), np.float32)
        pooled = None
        with torch.no_grad():
            for i in range(0, len(order), SCORE_BS):
                ids = order[i:i + SCORE_BS]
                logits, feats = forward([texts_a[j] for j in ids] + [texts_b[j] for j in ids],
                                        need_pooled=True)
                n = len(ids)
                if pooled is None:
                    pooled = np.zeros((len(rows), feats.shape[1]), np.float32)
                margin[ids] = (logits[:n] - logits[n:]).cpu().numpy()
                pooled[ids] = ((feats[:n] + feats[n:]) / 2).cpu().numpy()
        return margin, pooled

    dom_margin, dom_feats = score(domain)
    conf_margin, conf_feats = score(conformal)
    shard_margin, shard_feats = [], []
    for rows in shards:
        m, f = score(rows)
        shard_margin.append(m)
        shard_feats.append(f)

    # ---- sub-distribution classifier, fitted on the DOMAIN fold only.
    mu, sd = dom_feats.mean(0), dom_feats.std(0) + 1e-6

    def standardise(f):
        return torch.tensor((f - mu) / sd, dtype=torch.float32)

    X = standardise(dom_feats)
    Y = torch.tensor([SUBS.index(r["subset"]) for r in domain])
    W = torch.zeros(X.shape[1], len(SUBS), requires_grad=True)
    bias = torch.zeros(len(SUBS), requires_grad=True)
    lbfgs = torch.optim.LBFGS([W, bias], lr=0.5, max_iter=200)

    def closure():
        lbfgs.zero_grad()
        objective = torch.nn.functional.cross_entropy(X @ W + bias, Y) + 1e-3 * (W ** 2).sum()
        objective.backward()
        return objective

    lbfgs.step(closure)

    def infer(feats):
        with torch.no_grad():
            return (standardise(feats) @ W + bias).argmax(1).numpy()

    # ---- Mondrian split-conformal calibration on the CONFORMAL fold.
    conf_strata = infer(conf_feats)
    conf_error = ((conf_margin > 0).astype(int)
                  != np.array([1 if r["label"] == "A" else 0 for r in conformal])).astype(int)
    conf_confidence = np.abs(conf_margin)
    nominal = ALPHA - CAL_MARGIN
    taus = {}
    for k in range(len(SUBS)):
        mask = conf_strata == k
        taus[k] = (conformal_threshold(conf_confidence[mask], conf_error[mask], nominal)
                   if mask.sum() >= MIN_CAL else float("inf"))
    print("thresholds %s" % {SUBS[k]: round(v, 4) for k, v in taus.items()},
          file=sys.stderr, flush=True)

    confidences = [np.abs(m) for m in shard_margin]
    answered = []
    for margin, feats in zip(shard_margin, shard_feats):
        strata = infer(feats)
        answered.append(np.abs(margin) >= np.array([taus[int(k)] for k in strata]))
    if len(answered) == 2:
        answered = equalise(answered, confidences)

    ids, verdicts = [], []
    for rows, margin, mask in zip(shards, shard_margin, answered):
        for row, m, a in zip(rows, margin, mask):
            ids.append(row["id"])
            verdicts.append(("A" if m > 0 else "B") if a else "ABSTAIN")

    elapsed = time.perf_counter() - t0
    reported = {"cov_p%d" % (i + 1): float(a.mean()) for i, a in enumerate(answered)}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump({"ids": ids, "verdicts": verdicts, "elapsed_s": elapsed,
                   "reported": reported}, handle)
    print("reference: %d verdicts, coverage %s, %.1fs"
          % (len(ids), {k: round(v, 4) for k, v in reported.items()}, elapsed),
          file=sys.stderr, flush=True)

    # Authoring-side dump, AFTER the timed region closes and AFTER the artefact is written, so
    # it can never enter the graded elapsed seconds or move a verdict byte. Never set on the
    # graded or oracle path. It also scores a slice of the FIT fold, which the control ladder
    # needs to build the calibrate-on-the-training-fold control.
    if args.dump:
        fit_slice = fit[:3000]
        fit_margin, _ = score(fit_slice)
        np.savez(args.dump, conf_margin=conf_margin, conf_feats=conf_feats,
                 conf_label=np.array([1 if r["label"] == "A" else 0 for r in conformal]),
                 conf_subset=np.array([SUBS.index(r["subset"]) for r in conformal]),
                 dom_margin=dom_margin, dom_feats=dom_feats,
                 dom_subset=np.array([SUBS.index(r["subset"]) for r in domain]),
                 s0_margin=shard_margin[0], s0_feats=shard_feats[0],
                 s1_margin=shard_margin[1], s1_feats=shard_feats[1],
                 fit_margin=fit_margin,
                 fit_label=np.array([1 if r["label"] == "A" else 0 for r in fit_slice]),
                 fit_subset=np.array([SUBS.index(r["subset"]) for r in fit_slice]),
                 taus=np.array([taus[k] for k in range(len(SUBS))]))
        print("dump written to %s (outside the timed region)" % args.dump,
              file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()

# AELLO-CANARY-BLOCK
#   slot0: AELLO-CANARY-5a2a398da0b1caaab4680dfe60f25158
#   slot1: AELLO-CANARY-e63e4a2ffec892f6e66052747d2ca73b
#   slot2: AELLO-CANARY-d5486f12956ce7d142627e039e421dc1
#   slot3: AELLO-CANARY-28761ed455783590174668c81e473ff2
# AELLO-CANARY-END

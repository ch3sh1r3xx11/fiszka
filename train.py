"""Train Fiszka from the public dataset: same network, seed and threshold rule as the released Fiszka 0.3.

    pip install numpy torch
    python train.py --data fiszka-dataset --out my-fiszka
    python fiszka.py --model my-fiszka --title "..." --text "..."     # use the result like the released model

The dataset folder holds train.jsonl (one text per line, chronological), vectors.npy (EmbeddingGemma 2
vectors, same row order) and dataset_info.json (topic list and embedding settings). Training runs on CPU,
one thread, about 2 minutes. With the defaults it reproduces the released weights of Fiszka 0.3.
"""
import argparse
import json
import struct
from datetime import datetime
from pathlib import Path

import numpy as np

SEED = 138
VALIDATION_SHARE = 0.2             # threshold is chosen on the newest 20% of the training texts
THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.05), 2)
MAX_TOPICS = 3


class DeepNet:
    """768 -> 256 -> 128 -> 64 -> topics (ReLU) plus a linear skip 768 -> topics, sigmoid outputs.

    The skip is a logistic regression inside the network; hidden layers add corrections to it.
    Loss: BCE with a per-topic positive weight sqrt(neg/pos), capped at 10. Early stopping on the
    newest 10% of the rows it is given. After training the weights are kept as numpy arrays.
    """
    LAYERS = (256, 128, 64)
    INPUT_DROPOUT, DROPOUT = 0.1, 0.3
    EPOCHS, PATIENCE, BATCH, LR, WD = 300, 20, 128, 1e-3, 1e-4

    def __init__(self, seed=SEED, threads=1):
        self.seed, self.threads = seed, threads

    def fit(self, X, Y):
        import torch
        from torch import nn
        torch.manual_seed(self.seed)
        torch.set_num_threads(self.threads)
        X = torch.tensor(np.asarray(X, dtype=np.float32))
        Y = torch.tensor(np.asarray(Y, dtype=np.float32))
        n_val = max(1, int(len(X) * 0.1))
        Xt, Yt, Xv, Yv = X[:-n_val], Y[:-n_val], X[-n_val:], Y[-n_val:]
        pos = Yt.sum(0).clamp(min=1)
        pos_weight = ((len(Yt) - pos) / pos).sqrt().clamp(1, 10)

        dims = (X.shape[1], *self.LAYERS)
        hidden = [nn.Linear(a, b) for a, b in zip(dims, dims[1:])]
        output, skip = nn.Linear(dims[-1], Y.shape[1]), nn.Linear(X.shape[1], Y.shape[1])
        net = nn.ModuleList([*hidden, output, skip])   # creation order matters: it fixes the random init
        drop_in, drop = nn.Dropout(self.INPUT_DROPOUT), nn.Dropout(self.DROPOUT)

        def forward(x, train):
            net.train(train)
            x = drop_in(x) if train else x
            h = x
            for layer in hidden:
                h = torch.relu(layer(h))
                h = drop(h) if train else h
            return output(h) + skip(x)

        opt = torch.optim.AdamW(net.parameters(), lr=self.LR, weight_decay=self.WD)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        best, state, waited = float("inf"), None, 0
        gen = torch.Generator().manual_seed(self.seed)
        self.epochs = 0
        for _ in range(self.EPOCHS):
            self.epochs += 1
            for idx in torch.randperm(len(Xt), generator=gen).split(self.BATCH):
                opt.zero_grad()
                loss_fn(forward(Xt[idx], True), Yt[idx]).backward()
                opt.step()
            with torch.no_grad():
                loss = loss_fn(forward(Xv, False), Yv).item()
            if loss < best - 1e-4:
                best, waited = loss, 0
                state = [p.detach().clone() for p in net.parameters()]
            else:
                waited += 1
                if waited >= self.PATIENCE:
                    break
        with torch.no_grad():
            for p, s in zip(net.parameters(), state):
                p.copy_(s)
        # (W, b) per layer: hidden layers, then output, then skip; W is input x output
        self.weights = [(m.weight.detach().numpy().T.copy(), m.bias.detach().numpy().copy()) for m in [*hidden, output, skip]]
        return self

    def predict_proba(self, X):
        x = np.asarray(X, dtype=np.float32)
        h = x
        for W, b in self.weights[:-2]:
            h = np.maximum(0, h @ W + b)
        logit = h @ self.weights[-2][0] + self.weights[-2][1] + x @ self.weights[-1][0] + self.weights[-1][1]
        return 1 / (1 + np.exp(-logit))


def decide(P, threshold, max_topics=MAX_TOPICS):
    """Topics with probability >= threshold, at most max_topics most confident per text."""
    top = np.argsort(-P, axis=1, kind="stable")[:, :max_topics]
    D = np.zeros(P.shape, dtype=bool)
    np.put_along_axis(D, top, True, axis=1)
    return D & (P >= threshold)


def micro_f1(Y, D):
    tp = int((Y & D).sum())
    fp, fn = int((~Y & D).sum()), int((Y & ~D).sum())
    return 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)


def pick_threshold(P, Y):
    """Threshold with the best micro-F1 on validation; ties go to the lowest threshold."""
    scores = [micro_f1(Y, decide(P, t)) for t in THRESHOLDS]
    return float(THRESHOLDS[int(np.argmax(scores))])


def load(folder):
    folder = Path(folder)
    info = json.loads((folder / "dataset_info.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (folder / "train.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    X = np.load(folder / "vectors.npy")
    if len(X) != len(rows):
        raise SystemExit(f"vectors.npy has {len(X)} rows, train.jsonl has {len(rows)}")
    order = sorted(range(len(rows)), key=lambda i: (datetime.fromisoformat(rows[i]["created_at"]), rows[i]["id"]))
    if order != list(range(len(rows))):
        raise SystemExit("train.jsonl is not in chronological order; the validation split depends on it")
    keys = [t["key"] for t in info["topics"]]
    index = {k: i for i, k in enumerate(keys)}
    Y = np.zeros((len(rows), len(keys)), dtype=bool)
    for r, row in enumerate(rows):
        for t in row["topics"]:
            Y[r, index[t]] = True
    return info, rows, X, Y


def save_safetensors(path, tensors, metadata):
    """safetensors without the library: 8-byte little-endian header length, JSON header, raw F32 bytes."""
    header, chunks, offset = {"__metadata__": metadata}, [], 0
    for name, t in tensors.items():
        b = np.ascontiguousarray(t, dtype="<f4").tobytes()
        header[name] = {"dtype": "F32", "shape": list(t.shape), "data_offsets": [offset, offset + len(b)]}
        chunks.append(b)
        offset += len(b)
    h = json.dumps(header, separators=(",", ":")).encode()
    h += b" " * (-len(h) % 8)
    Path(path).write_bytes(struct.pack("<Q", len(h)) + h + b"".join(chunks))


def save(out, model, threshold, info, n_texts, seed, version):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    names = [f"hidden.{i}" for i in range(len(model.weights) - 2)] + ["output", "skip"]
    tensors = {}
    for n, (W, b) in zip(names, model.weights):
        tensors[f"{n}.weight"] = W.T   # torch.nn.Linear layout: output x input
        tensors[f"{n}.bias"] = b
    count = sum(t.size for t in tensors.values())
    save_safetensors(out / "model.safetensors", tensors, {"format": "pt", "name": "fiszka", "version": version})
    config = {
        "name": "fiszka",
        "version": version,
        "license": "Apache-2.0",
        "parameters": count,
        "topics": info["topics"],
        "threshold": threshold,
        "max_topics": MAX_TOPICS,
        "min_chars": info["min_chars"],
        "embedding": info["embedding"],
        "hidden_layers": list(DeepNet.LAYERS),
        "architecture": f"{info['embedding']['dim']} -> 256 -> 128 -> 64 -> {len(info['topics'])} (ReLU) plus a linear skip "
                        f"{info['embedding']['dim']} -> {len(info['topics'])}, sigmoid outputs",
        "training": {"texts": n_texts, "seed": seed, "labels": info["labels"],
                     "validation": f"threshold chosen on the newest {VALIDATION_SHARE:.0%} of training texts"},
    }
    (out / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=1), encoding="utf-8")
    return count


def train(X, Y, seed=SEED, threads=1):
    """Threshold from a model trained on the oldest 80%, then the final model on everything."""
    split = int(len(Y) * (1 - VALIDATION_SHARE))
    first = DeepNet(seed, threads).fit(X[:split], Y[:split])
    threshold = pick_threshold(first.predict_proba(X[split:]), Y[split:])
    return DeepNet(seed, threads).fit(X, Y), threshold, len(Y) - split


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", type=Path, required=True, help="dataset folder (train.jsonl, vectors.npy, dataset_info.json)")
    p.add_argument("--out", type=Path, required=True, help="folder for model.safetensors and config.json")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--threads", type=int, default=1, help="more threads train faster but change the weights slightly")
    p.add_argument("--version", default="0.3-retrained")
    a = p.parse_args()

    info, rows, X, Y = load(a.data)
    print(f"{len(rows)} texts, {Y.shape[1]} topics, {X.shape[1]}-d vectors")
    model, threshold, n_val = train(X, Y, a.seed, a.threads)
    count = save(a.out, model, threshold, info, len(rows), a.seed, a.version)
    print(f"threshold {threshold} (chosen on {n_val} newest texts), {model.epochs} epochs, {count} weights -> {a.out}")


if __name__ == "__main__":
    main()

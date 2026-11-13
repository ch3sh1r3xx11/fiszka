"""Fiszka: multi-label topic classifier for AI news and papers (17 topics, 252,194 weights).

Standalone inference: numpy + Python standard library. Embeddings come from a local Ollama server
running EmbeddingGemma 2 (`ollama pull embeddinggemma-2:270m`).

    python fiszka.py --title "Some paper title" --text "Abstract or article body ..."
    echo '{"title": "...", "text": "..."}' | python fiszka.py          # one JSON object per line
    python fiszka.py --vectors vectors.npy                              # precomputed 768-d embeddings

Set OLLAMA_HOST if Ollama is not on http://localhost:11434.
"""
import argparse
import html
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

# Text preparation: must match training exactly, otherwise the embeddings drift.
ARXIV_HEADER = re.compile(r"^\s*arXiv:\S+\s+Announce Type:\s*\S+\s*(Abstract:)?\s*")
HTML_TAG = re.compile(
    r"</?(?:a|abbr|article|aside|blockquote|br|button|code|dd|div|dl|dt|em|figcaption|figure|footer|form|h[1-6]|header|hr|"
    r"iframe|img|input|li|link|main|meta|nav|ol|p|picture|pre|section|source|span|strong|sub|sup|svg|table|tbody|td|th|"
    r"thead|time|tr|ul|video)(?:\s[^>]*)?/?>|</?[biu]>", re.IGNORECASE)
NO_TEXT_BLOCK = re.compile(r"<(script|style|svg)\b.*?</\1>", re.IGNORECASE | re.DOTALL)


def clean(text):
    """Strip the arXiv announcement header and HTML markup; plain text passes through unchanged."""
    text = ARXIV_HEADER.sub("", text or "")
    if not HTML_TAG.search(text):
        return text
    text = html.unescape(HTML_TAG.sub(" ", NO_TEXT_BLOCK.sub(" ", text)))
    return re.sub(r"\s*\n\s*", "\n", re.sub(r"[^\S\n]+", " ", text)).strip()


def document(title, text, limit):
    """EmbeddingGemma document format used in training."""
    return f"title: {(title or '').strip() or 'none'} | text: {clean(text)[:limit]}"


def load_safetensors(path):
    """Minimal reader for float32 safetensors files (no extra dependency)."""
    raw = Path(path).read_bytes()
    n = int.from_bytes(raw[:8], "little")
    header = json.loads(raw[8:8 + n])
    body = raw[8 + n:]
    tensors = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        if info["dtype"] != "F32":
            raise ValueError(f"{name}: expected F32, got {info['dtype']}")
        start, end = info["data_offsets"]
        tensors[name] = np.frombuffer(body[start:end], dtype="<f4").reshape(info["shape"])
    return tensors


class Fiszka:
    def __init__(self, folder=HERE):
        folder = Path(folder)
        self.config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        t = load_safetensors(folder / "model.safetensors")
        # weights are stored like torch.nn.Linear: (out_features, in_features)
        self.hidden = [(t[f"hidden.{i}.weight"].T, t[f"hidden.{i}.bias"]) for i in range(len(self.config["hidden_layers"]))]
        self.output = (t["output.weight"].T, t["output.bias"])
        self.skip = (t["skip.weight"].T, t["skip.bias"])
        self.topics = [x["key"] for x in self.config["topics"]]

    def probabilities(self, vectors):
        """vectors: (n, 768) EmbeddingGemma 2 embeddings -> (n, 17) topic probabilities."""
        x = np.asarray(vectors, dtype=np.float32)
        h = x
        for W, b in self.hidden:
            h = np.maximum(0, h @ W + b)
        logit = h @ self.output[0] + self.output[1] + x @ self.skip[0] + self.skip[1]
        return 1 / (1 + np.exp(-logit))

    def decide(self, P):
        """Topics with probability >= threshold, at most max_topics most confident per text."""
        best = np.argsort(-P, axis=1, kind="stable")[:, :self.config["max_topics"]]
        D = np.zeros(P.shape, dtype=bool)
        np.put_along_axis(D, best, True, axis=1)
        D &= P >= self.config["threshold"]
        return [[k for k, yes in zip(self.topics, row) if yes] for row in D]

    def embed(self, texts, batch=16):
        """texts: list of {"title", "text"} -> (n, 768) embeddings from Ollama."""
        emb = self.config["embedding"]
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        host = host if host.startswith("http") else f"http://{host}"
        inputs = [document(t.get("title"), t.get("text"), emb["char_limit"]) for t in texts]
        out = []
        for i in range(0, len(inputs), batch):
            body = json.dumps({"model": emb["ollama_model"], "input": inputs[i:i + batch], "truncate": True}).encode()
            req = urllib.request.Request(f"{host}/api/embed", body, {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as r:
                vecs = json.load(r)["embeddings"]
            if len(vecs) != len(inputs[i:i + batch]) or any(len(v) != emb["dim"] for v in vecs):
                raise RuntimeError("Ollama returned a wrong number of embeddings or a wrong dimension")
            out += vecs
        return np.asarray(out, dtype=np.float32)

    def tag(self, texts):
        """texts: list of {"title", "text"} -> list of {"topics", "probabilities"} (None for too short texts)."""
        results = [None] * len(texts)
        long_idx = [i for i, t in enumerate(texts) if len(clean(t.get("text"))) >= self.config["min_chars"]]
        if long_idx:
            P = self.probabilities(self.embed([texts[i] for i in long_idx]))
            for i, p, topics in zip(long_idx, P, self.decide(P)):
                results[i] = {"topics": topics,
                              "probabilities": {k: round(float(v), 4) for k, v in zip(self.topics, p)}}
        for i, r in enumerate(results):
            if r is None:
                results[i] = {"topics": None, "reason": f"text shorter than {self.config['min_chars']} characters"}
        return results


def main():
    p = argparse.ArgumentParser(description="Fiszka topic classifier")
    p.add_argument("--title")
    p.add_argument("--text")
    p.add_argument("--vectors", type=Path, help=".npy file with precomputed (n, 768) EmbeddingGemma 2 embeddings")
    p.add_argument("--model", type=Path, default=HERE, help="folder with config.json and model.safetensors")
    a = p.parse_args()
    f = Fiszka(a.model)
    if a.vectors:
        P = f.probabilities(np.load(a.vectors))
        for topics in f.decide(P):
            print(json.dumps({"topics": topics}))
        return
    if a.text is not None:
        texts = [{"title": a.title, "text": a.text}]
    else:
        texts = [json.loads(line) for line in sys.stdin if line.strip()]
    for t, r in zip(texts, f.tag(texts)):
        print(json.dumps({"title": t.get("title"), **r}, ensure_ascii=False))


if __name__ == "__main__":
    main()

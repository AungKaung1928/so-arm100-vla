"""Sentence encoders. Two, on purpose.

    MiniLMEncoder   sentence-transformers all-MiniLM-L6-v2, frozen, 384-d.
                    22M parameters, a real language model, one download.
    HashEncoder     hashed bag of words + bigrams into 384 buckets, L2
                    normalised. No parameters, no download, no semantics
                    beyond token identity.

The second one is the ablation that answers "does the language model
matter": if a policy conditioned on hashed tokens generalises to unseen
paraphrases as well as one conditioned on MiniLM, the paraphrase split was
not testing language understanding, only template lookup. Both produce the
same 384-d shape so the policy is identical downstream.

Embeddings are cached to disk keyed by a hash of the sentence, so the
encoder runs once per unique sentence, ever.
"""
import hashlib
import json
import os
import re

import numpy as np

EMBED_DIM = 384
_WORD = re.compile(r"[a-z]+")


class TextEncoder:
    name = "base"
    dim = EMBED_DIM

    def embed_all(self, sentences):
        raise NotImplementedError

    def embed(self, sentence):
        return self.embed_all([sentence])[0]


class HashEncoder(TextEncoder):
    name = "hash"

    def __init__(self, dim=EMBED_DIM, seed="so-arm100-vla"):
        self.dim = int(dim)
        self.seed = seed

    def _bucket(self, token):
        h = hashlib.sha1(f"{self.seed}|{token}".encode()).digest()
        idx = int.from_bytes(h[:4], "little") % self.dim
        sign = 1.0 if h[4] % 2 == 0 else -1.0
        return idx, sign

    def embed_all(self, sentences):
        out = np.zeros((len(sentences), self.dim), np.float32)
        for i, s in enumerate(sentences):
            words = _WORD.findall(s.lower())
            toks = words + [f"{a}_{b}" for a, b in zip(words, words[1:])]
            for t in toks:
                j, sgn = self._bucket(t)
                out[i, j] += sgn
            n = np.linalg.norm(out[i])
            if n > 0:
                out[i] /= n
        return out


class MiniLMEncoder(TextEncoder):
    name = "minilm"
    MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, cache_dir="data/embeddings", device="cpu"):
        self.cache_dir = cache_dir
        self.device = device
        self._model = None
        os.makedirs(cache_dir, exist_ok=True)
        self._index_path = os.path.join(cache_dir, "minilm_index.json")
        self._vec_path = os.path.join(cache_dir, "minilm_vectors.npy")
        self._index, self._vectors = self._load_cache()

    def _load_cache(self):
        if os.path.exists(self._index_path) and os.path.exists(self._vec_path):
            with open(self._index_path) as f:
                index = json.load(f)
            return index, np.load(self._vec_path)
        return {}, np.zeros((0, EMBED_DIM), np.float32)

    def _save_cache(self):
        with open(self._index_path, "w") as f:
            json.dump(self._index, f)
        np.save(self._vec_path, self._vectors)

    @staticmethod
    def key(sentence):
        return hashlib.sha1(sentence.strip().lower().encode()).hexdigest()

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.MODEL, device=self.device)
            self._model.eval()
        return self._model

    def embed_all(self, sentences):
        keys = [self.key(s) for s in sentences]
        missing = sorted({k for k in keys if k not in self._index})
        if missing:
            by_key = {self.key(s): s for s in sentences}
            texts = [by_key[k] for k in missing]
            vecs = self._load_model().encode(texts, batch_size=32, convert_to_numpy=True,
                                             normalize_embeddings=True).astype(np.float32)
            base = self._vectors.shape[0]
            self._vectors = np.concatenate([self._vectors, vecs], axis=0)
            for i, k in enumerate(missing):
                self._index[k] = base + i
            self._save_cache()
        return self._vectors[[self._index[k] for k in keys]].copy()


def make_encoder(name, **kw):
    if name == "hash":
        return HashEncoder(**kw)
    if name == "minilm":
        return MiniLMEncoder(**kw)
    raise KeyError(f"unknown encoder {name!r}; have hash, minilm")

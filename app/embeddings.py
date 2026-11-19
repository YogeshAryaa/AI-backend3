import asyncio
import logging
import re
import threading

import httpx
import numpy as np

from .config import cfg

log = logging.getLogger("rag.embeddings")

_model = None
_reranker = None
_load_lock = threading.Lock()

# Prefixes the local model family was trained with.
_KNOWN_PREFIXES = [
    ("e5", "query: ", "passage: "),
    ("bge-small-en", "Represent this sentence for searching relevant passages: ", ""),
    ("bge-base-en", "Represent this sentence for searching relevant passages: ", ""),
    ("bge-large-en", "Represent this sentence for searching relevant passages: ", ""),
]


def _prefix(value: str, kind: int) -> str:
    if value.lower() == "none":
        return ""
    if value:
        return value
    name = cfg.embedding_model.lower()
    for key, q, d in _KNOWN_PREFIXES:
        if key in name:
            return q if kind == 0 else d
    return ""


QUERY_PREFIX = _prefix(cfg.query_prefix, 0)
DOC_PREFIX = _prefix(cfg.doc_prefix, 1)


def fingerprint() -> str:
    """Identifies how stored vectors were produced. If it changes, everything is re-embedded."""
    backend = cfg.embedding_backend.lower()
    if backend in ("gemini", "openai"):
        return f"{backend}|{cfg.embedding_model}"
    return f"{cfg.embedding_model}|{DOC_PREFIX}"


def dim() -> int:
    backend = cfg.embedding_backend.lower()
    if backend == "gemini":
        return 768
    if backend == "openai":
        return 3072 if "large" in cfg.embedding_model else 1536
    m = _load_torch()
    get = getattr(m, "get_embedding_dimension", None) or m.get_sentence_embedding_dimension
    return int(get())


# ------------------------------------------------------------ Gemini Cloud Embeddings (0 MB RAM)
async def _embed_gemini(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, dim()), np.float32)
    import os
    key = (cfg.gemini_api_key or os.environ.get("GEMINI_API_KEY", "")).strip().strip('"').strip("'")
    if "=" in key and key.split("=", 1)[0].strip().upper() in ("GEMINI_API_KEY", "API_KEY"):
        key = key.split("=", 1)[1].strip().strip('"').strip("'")

    if not key or key.startswith("your-") or key.startswith("AIzaSy..."):
        raise RuntimeError(
            "GEMINI_API_KEY is missing or invalid in Render Environment Variables. "
            "Go to Render Dashboard -> Environment, set Key: GEMINI_API_KEY and Value: your actual key from https://aistudio.google.com/apikey"
        )

    raw_model = cfg.embedding_model.strip() or "gemini-embedding-001"
    if "text-embedding" in raw_model or "e5" in raw_model or not raw_model:
        raw_model = "gemini-embedding-001"
    model_name = raw_model.replace("models/", "")
    model_path = f"models/{model_name}"

    url = f"https://generativelanguage.googleapis.com/v1beta/{model_path}:batchEmbedContents?key={key}"
    out: list[list[float]] = []
    batch_size = max(1, min(cfg.embedding_batch_size, 100))

    async with httpx.AsyncClient(timeout=30.0) as client:
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            payload = {
                "requests": [
                    {
                        "model": model_path,
                        "content": {"parts": [{"text": t[:8000]}]},
                        "outputDimensionality": 768,
                    }
                    for t in chunk
                ]
            }
            res = await client.post(url, json=payload)
            if res.status_code != 200:
                raise RuntimeError(f"Gemini Embedding API error ({res.status_code}): {res.text[:300]}")
            data = res.json()
            for item in data.get("embeddings", []):
                out.append(item.get("values", []))

    if not out:
        return np.zeros((0, dim()), np.float32)

    arr = np.array(out, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (arr / norms).astype(np.float32)


# ------------------------------------------------------------ OpenAI Cloud Embeddings
async def _embed_openai(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, dim()), np.float32)
    key = cfg.openai_api_key.strip()
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    url = cfg.embedding_api_url.strip() or "https://api.openai.com/v1/embeddings"
    model = cfg.embedding_model or "text-embedding-3-small"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    out: list[list[float]] = []
    batch_size = max(1, min(cfg.embedding_batch_size, 100))

    async with httpx.AsyncClient(timeout=30.0) as client:
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            payload = {"model": model, "input": [t[:8000] for t in chunk]}
            res = await client.post(url, headers=headers, json=payload)
            if res.status_code != 200:
                raise RuntimeError(f"OpenAI embedding API error ({res.status_code}): {res.text[:300]}")
            data = res.json()
            items = sorted(data.get("data", []), key=lambda x: x.get("index", 0))
            for item in items:
                out.append(item.get("embedding", []))

    arr = np.array(out, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (arr / norms).astype(np.float32)


# ------------------------------------------------------------ Local PyTorch Fallback
def _load_torch():
    global _model
    if _model is None:
        with _load_lock:
            if _model is None:
                try:
                    if cfg.embedding_threads > 0:
                        import torch
                        torch.set_num_threads(cfg.embedding_threads)
                    from sentence_transformers import SentenceTransformer
                except ImportError as e:
                    raise RuntimeError(
                        "Local sentence-transformers is not installed. "
                        "Set EMBEDDING_BACKEND=gemini and GEMINI_API_KEY in your .env or environment variables."
                    ) from e

                _model = SentenceTransformer(cfg.embedding_model)
    return _model


def _encode_torch(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, dim()), np.float32)
    return _load_torch().encode(
        texts, normalize_embeddings=True, batch_size=cfg.embedding_batch_size, show_progress_bar=False
    ).astype(np.float32)


# ------------------------------------------------------------ Public embedding interface
async def embed_documents(texts: list[str]) -> np.ndarray:
    backend = cfg.embedding_backend.lower()
    if backend == "gemini":
        return await _embed_gemini(texts)
    if backend == "openai":
        return await _embed_openai(texts)

    # local fallback
    out = []
    step = max(cfg.embedding_batch_size * 4, 64)
    for i in range(0, len(texts), step):
        out.append(await asyncio.to_thread(_encode_torch, [DOC_PREFIX + t for t in texts[i:i + step]]))
    return np.vstack(out) if out else np.zeros((0, dim()), np.float32)


async def embed_query(text: str) -> np.ndarray:
    backend = cfg.embedding_backend.lower()
    if backend == "gemini":
        arr = await _embed_gemini([text])
        return arr[0]
    if backend == "openai":
        arr = await _embed_openai([text])
        return arr[0]

    out = await asyncio.to_thread(_encode_torch, [QUERY_PREFIX + text])
    return out[0]


async def embed_queries(texts: list[str]) -> np.ndarray:
    backend = cfg.embedding_backend.lower()
    if backend == "gemini":
        return await _embed_gemini(texts)
    if backend == "openai":
        return await _embed_openai(texts)

    return await asyncio.to_thread(_encode_torch, [QUERY_PREFIX + t for t in texts])


# ------------------------------------------------------------ optional re-ranker
def _load_reranker():
    global _reranker
    if _reranker is None and cfg.reranker_model:
        with _load_lock:
            if _reranker is None:
                try:
                    from sentence_transformers import CrossEncoder
                    _reranker = CrossEncoder(cfg.reranker_model, max_length=512)
                except ImportError:
                    log.warning("sentence-transformers not installed, re-ranker disabled")
    return _reranker


def reranker_enabled() -> bool:
    return bool(cfg.reranker_model)


async def rerank(query: str, texts: list[str]) -> list[float]:
    model = _load_reranker()
    if model is None or not texts:
        return []
    scores = await asyncio.to_thread(model.predict, [(query, t) for t in texts], show_progress_bar=False)
    return [float(s) for s in scores]


# ------------------------------------------------------------ chunking
def split_text(text: str, size: int | None = None, overlap: int | None = None) -> list[str]:
    size = size or cfg.chunk_chars
    overlap = min(overlap if overlap is not None else cfg.chunk_overlap, size // 2)
    text = re.sub(r"[ \t]+", " ", text).strip()
    if not text:
        return []
    # "।" is the Hindi full stop
    sentences = re.split(r"(?<=[.!?।])\s+|\n{2,}", text)
    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        while len(s) > size:  # very long sentence: hard split
            if cur:
                chunks.append(" ".join(cur))
                cur, cur_len = [], 0
            chunks.append(s[:size])
            s = s[size - overlap:]
        if cur and cur_len + len(s) + 1 > size:
            chunks.append(" ".join(cur))
            tail: list[str] = []
            length = 0
            for prev in reversed(cur):
                if length + len(prev) > overlap:
                    break
                tail.insert(0, prev)
                length += len(prev)
            cur = tail
            cur_len = sum(len(x) + 1 for x in cur)
        cur.append(s)
        cur_len += len(s) + 1
    if cur:
        chunks.append(" ".join(cur))
    return chunks

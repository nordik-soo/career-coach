"""Sentence-transformer embedding wrapper for matching v2 step 5.

Soft dependency: when `sentence-transformers` isn't installed (e.g. on
a stripped-down dev box that didn't pull torch), `get_embedder()`
returns None and the engine falls back to lexical-only matching. No
crash, no infrastructure dependency forced.

The wrapper is a thin singleton:
  - First call to `get_embedder()` lazy-loads the model into memory
    (~80MB for all-MiniLM-L6-v2)
  - Subsequent calls return the cached instance
  - Encoding is batched-internal; callers can pass one string or many
  - All vectors are L2-normalized so cosine similarity reduces to dot
    product downstream

The model is OFFLINE-CAPABLE: sentence-transformers caches model
weights under ~/.cache/huggingface/. First load downloads the
~80MB weights once; subsequent invocations work without network.
"""
from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

from config import EMBEDDING_MODEL_NAME, EMBEDDING_MODEL_VERSION

if TYPE_CHECKING:
    import numpy as np


log = logging.getLogger(__name__)


class EmbeddingUnavailable(RuntimeError):
    """Raised when callers need to know the embedder is unavailable but
    only when None-returning isn't possible (e.g. inside a generator
    that already committed to returning vectors). The common path is
    get_embedder() returning None -- callers should prefer the
    None-check."""


class _Embedder:
    """Thin wrapper around a sentence-transformers SentenceTransformer.

    Encapsulates: lazy model load, output dtype (float32), L2 normalization,
    single-vs-batch encode helpers. Keeps the import of
    sentence-transformers inside this class so importing the module
    itself doesn't trigger the ~700MB torch chain.
    """

    def __init__(self, model_name: str, model_version: str) -> None:
        self.model_name = model_name
        self.model_version = model_version
        self._model = None  # type: ignore[var-annotated]
        self._load_lock = threading.Lock()

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            # Local import so this whole module can be imported without
            # paying the torch cost on processes that don't use embeddings.
            from sentence_transformers import SentenceTransformer
            log.info(
                "Loading sentence-transformer model %s (version=%s) into memory",
                self.model_name, self.model_version,
            )
            self._model = SentenceTransformer(self.model_name)
            log.info(
                "Loaded %s (dim=%d)",
                self.model_name, self._model.get_sentence_embedding_dimension(),
            )

    def encode_one(self, text: str) -> "np.ndarray":
        """Embed a single string. Returns a 1-D L2-normalized float32 vector."""
        self._ensure_loaded()
        import numpy as np
        vec = self._model.encode(  # type: ignore[union-attr]
            text, normalize_embeddings=True, convert_to_numpy=True,
        )
        return vec.astype(np.float32, copy=False)

    def encode_many(self, texts: list[str]) -> "np.ndarray":
        """Embed a list of strings in one batched call. Returns a 2-D
        (N, dim) L2-normalized float32 array. Empty input returns a
        zero-row array so downstream `np.dot(...)` works without a
        special case."""
        self._ensure_loaded()
        import numpy as np
        if not texts:
            dim = self._model.get_sentence_embedding_dimension()  # type: ignore[union-attr]
            return np.zeros((0, dim), dtype=np.float32)
        vecs = self._model.encode(  # type: ignore[union-attr]
            texts, normalize_embeddings=True, convert_to_numpy=True,
            batch_size=32, show_progress_bar=False,
        )
        return vecs.astype(np.float32, copy=False)

    def dim(self) -> int:
        self._ensure_loaded()
        return int(self._model.get_sentence_embedding_dimension())  # type: ignore[union-attr]


# Module-level singleton + thread-safe lazy construction.
_INSTANCE: _Embedder | None = None
_INSTANCE_LOCK = threading.Lock()
_UNAVAILABLE_LOGGED = False


def get_embedder() -> _Embedder | None:
    """Return the shared embedder, or None if sentence-transformers is
    not installed.

    Logs the "unavailable" warning exactly once per process so the engine
    can call this on every match without flooding logs.
    """
    global _INSTANCE, _UNAVAILABLE_LOGGED
    if _INSTANCE is not None:
        return _INSTANCE
    # Cheap probe: try the import without instantiating the model.
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        if not _UNAVAILABLE_LOGGED:
            log.warning(
                "sentence-transformers not installed; semantic re-ranker "
                "(matching v2 step 5) is disabled. Engine will fall back "
                "to lexical-only matching. pip install sentence-transformers "
                "to enable."
            )
            _UNAVAILABLE_LOGGED = True
        return None
    with _INSTANCE_LOCK:
        if _INSTANCE is None:
            _INSTANCE = _Embedder(EMBEDDING_MODEL_NAME, EMBEDDING_MODEL_VERSION)
    return _INSTANCE


def reset_embedder_for_tests() -> None:
    """Drop the singleton + warning-latch so tests can re-exercise the
    cold-path (e.g. simulating a fresh process). Production code MUST
    NOT call this."""
    global _INSTANCE, _UNAVAILABLE_LOGGED
    _INSTANCE = None
    _UNAVAILABLE_LOGGED = False

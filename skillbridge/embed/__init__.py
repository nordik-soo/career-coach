"""Embedding service for matching v2 step 5 (semantic re-ranker).

The engine consumes a single function from this package:

    from skillbridge.embed.service import get_embedder, EmbeddingUnavailable

    embedder = get_embedder()    # None if sentence-transformers not installed
    if embedder is not None:
        vec = embedder.encode_one("welding & fabrication")

See skillbridge/embed/service.py for the wrapper and graceful-failure
contract.
"""

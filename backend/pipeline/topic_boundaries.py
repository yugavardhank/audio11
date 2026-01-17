from sklearn.metrics.pairwise import cosine_similarity

_model = None


def _get_model():
    """Lazy load embedding model to avoid heavy import at module load."""
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            _model = False
    return _model

def detect_boundaries(blocks, threshold=0.65):
    model = _get_model()
    if not model:
        return [0]

    texts = [b["text"] for b in blocks]
    embeddings = model.encode(texts)

    boundaries = [0]

    for i in range(1, len(embeddings)):
        sim = cosine_similarity(
            [embeddings[i - 1]],
            [embeddings[i]]
        )[0][0]

        if sim < threshold:
            boundaries.append(i)

    return boundaries
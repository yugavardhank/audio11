from transformers import pipeline

_summarizer = None

def _get_summarizer():
    global _summarizer
    if _summarizer is None:
        _summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            device=-1
        )
    return _summarizer


def generate_summary(text):
    if not text or len(text.split()) < 120:
        return text.strip()

    try:
        summarizer = _get_summarizer()
        out = summarizer(
            text[:3500],
            max_length=150,
            min_length=60,
            do_sample=False
        )
        return out[0]["summary_text"]
    except Exception:
        # fallback: explain first + last context
        sentences = text.split(". ")
        return ". ".join(sentences[:3] + sentences[-3:])


def generate_title(text):
    """
    Generate a semantic topic title (NOT subtitle)
    """
    try:
        summarizer = _get_summarizer()
        out = summarizer(
            "Generate a concise topic title: " + text[:1200],
            max_length=20,
            min_length=6,
            do_sample=False
        )
        return out[0]["summary_text"].replace(".", "")
    except Exception:
        # fallback heuristic
        words = text.split()
        return " ".join(words[:6]).capitalize()
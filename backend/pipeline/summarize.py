"""
Topic Summarization Module
Generates concise summaries for each topic using HuggingFace transformers
"""

import logging

logger = logging.getLogger(__name__)

# Initialize summarization pipeline (lazy-loaded)
_summarizer = None

def _get_summarizer():
    """Lazy load summarization model"""
    global _summarizer
    if _summarizer is None:
        try:
            from transformers import pipeline
            # Use BART for summarization
            _summarizer = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                device=-1  # Use CPU; set to 0 for GPU
            )
            logger.info("Summarization model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load summarization model: {e}")
            _summarizer = False
    return _summarizer if _summarizer else None

def summarize_text(text, max_length=60, min_length=20):
    """
    Generate a summary of given text
    
    Args:
        text: Text to summarize
        max_length: Maximum length of summary in tokens
        min_length: Minimum length of summary in tokens
    
    Returns:
        Summary text or original text if summarization fails
    """
    if not text or len(text.strip()) < 50:
        return text[:100]  # Too short to summarize
    
    summarizer = _get_summarizer()
    if not summarizer:
        # Fallback: return first sentence or truncate
        sentences = text.split('.')
        return sentences[0].strip() if sentences else text[:100]
    
    try:
        # Limit input to 1024 tokens (BART max input)
        words = text.split()
        if len(words) > 200:
            text = ' '.join(words[:200])
        
        summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
        return summary[0]['summary_text'] if summary else text[:100]
    except Exception as e:
        logger.warning(f"Summarization failed: {e}")
        # Fallback: return first few sentences
        sentences = text.split('.')
        return '. '.join(sentences[:2])[:100]

def summarize_topics(segments):
    """
    Generate summaries for all topic segments
    
    Args:
        segments: List of topic segment dictionaries
    
    Returns:
        List of segments with 'summary' field added
    """
    for segment in segments:
        if 'text' in segment:
            segment['summary'] = summarize_text(segment['text'])
        else:
            segment['summary'] = ""
    
    return segments

def extract_key_points(text, num_points=3):
    """
    Extract key points from text by identifying important sentences
    
    Args:
        text: Input text
        num_points: Number of key points to extract
    
    Returns:
        List of key point sentences
    """
    if not text:
        return []
    
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    
    if len(sentences) <= num_points:
        return sentences
    
    # Simple heuristic: return sentences with most keywords
    from collections import Counter
    
    # Tokenize and count words
    words = text.lower().split()
    word_freq = Counter(word for word in words if len(word) > 4)
    
    # Score sentences by keyword frequency
    sentence_scores = []
    for sent in sentences:
        score = sum(word_freq.get(word.lower(), 0) for word in sent.split())
        sentence_scores.append((sent, score))
    
    # Sort by score and return top sentences
    sentence_scores.sort(key=lambda x: x[1], reverse=True)
    top_sentences = [sent for sent, _ in sentence_scores[:num_points]]
    
    # Return in original order
    return [sent for sent in sentences if sent in top_sentences]

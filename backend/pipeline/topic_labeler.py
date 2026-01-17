"""
Topic Labeler Module
Generates descriptive titles for topics using embeddings and keyword extraction
"""

import logging
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

# Load embedding model
_model = None

def _get_model():
    """Lazy load embedding model"""
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Embedding model loaded for topic labeling")
        except Exception as e:
            logger.warning(f"Failed to load embedding model: {e}")
            _model = False
    return _model if _model else None

def extract_keywords(text, num_keywords=5):
    """
    Extract important keywords from text using TF-IDF
    
    Args:
        text: Input text
        num_keywords: Number of keywords to extract
    
    Returns:
        List of keywords
    """
    if not text or len(text.strip()) == 0:
        return []
    
    try:
        vectorizer = TfidfVectorizer(
            max_features=20,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1
        )
        
        # If text is too short, just split by spaces
        words = text.lower().split()
        if len(words) < 10:
            # Filter out very short words
            keywords = [w for w in words if len(w) > 3]
            return keywords[:num_keywords]
        
        # Use TF-IDF for longer texts
        matrix = vectorizer.fit_transform([text])
        features = vectorizer.get_feature_names_out()
        
        # Get top features by TF-IDF score
        scores = matrix.toarray()[0]
        top_indices = np.argsort(scores)[-num_keywords:][::-1]
        
        keywords = [features[i] for i in top_indices if scores[i] > 0]
        return keywords
    except Exception as e:
        logger.warning(f"Keyword extraction failed: {e}")
        # Fallback: return first few significant words
        words = text.lower().split()
        return [w for w in words if len(w) > 4][:num_keywords]

def generate_topic_label(text, embeddings=None, context=""):
    """
    Generate a descriptive label for a topic
    
    Args:
        text: Topic text content
        embeddings: Pre-computed embeddings (optional)
        context: Additional context about the topic
    
    Returns:
        Descriptive title for the topic
    """
    if not text or len(text.strip()) == 0:
        return "Unknown Topic"
    
    # Extract keywords
    keywords = extract_keywords(text, num_keywords=5)
    
    if not keywords:
        # Fallback: use first few words
        words = text.split()[:3]
        return ' '.join(words).title()
    
    # Create label from top keywords
    label = ' '.join(keywords[:3]).title()
    
    # Clean up the label
    label = label.replace('_', ' ')
    
    # Ensure it's not too long
    if len(label) > 60:
        label = label[:60].rsplit(' ', 1)[0] + '...'
    
    return label if label else "Topic Discussion"

def rank_keywords_by_relevance(text, keywords):
    """
    Rank keywords by relevance to the text using embeddings
    
    Args:
        text: Text content
        keywords: List of candidate keywords
    
    Returns:
        Sorted list of keywords by relevance
    """
    model = _get_model()
    if not model:
        return keywords
    
    try:
        # Embed text and keywords
        text_embedding = model.encode(text)
        keyword_embeddings = model.encode(keywords)
        
        # Calculate similarity scores
        from sklearn.metrics.pairwise import cosine_similarity
        scores = cosine_similarity([text_embedding], keyword_embeddings)[0]
        
        # Sort keywords by score
        ranked = sorted(zip(keywords, scores), key=lambda x: x[1], reverse=True)
        return [kw for kw, _ in ranked]
    except Exception as e:
        logger.warning(f"Embedding-based ranking failed: {e}")
        return keywords

def generate_topic_labels_batch(segments):
    """
    Generate labels for multiple topic segments
    
    Args:
        segments: List of topic segment dictionaries with 'text' field
    
    Returns:
        List of segments with 'title' field added
    """
    for i, segment in enumerate(segments):
        if 'text' in segment:
            segment['title'] = generate_topic_label(segment['text'])
        else:
            segment['title'] = f"Topic {i + 1}"
    
    return segments

def refine_label_with_context(label, speaker_info=None, timestamp=None):
    """
    Refine a topic label with additional context
    
    Args:
        label: Current topic label
        speaker_info: Information about speakers in this segment
        timestamp: Timestamp of the topic
    
    Returns:
        Refined label
    """
    if speaker_info and isinstance(speaker_info, dict):
        dominant_speaker = speaker_info.get('dominant_speaker')
        if dominant_speaker:
            # Add speaker context to label
            label = f"{label} (by {dominant_speaker})"
    
    return label

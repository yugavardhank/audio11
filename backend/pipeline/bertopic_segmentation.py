"""
Advanced Topic Modeling using BERTopic
Implements the professional approach from the Medium article:
- Embeddings with sentence-transformers
- Dimensionality reduction with UMAP
- Clustering with HDBSCAN
- Topic modeling with BERTopic
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)

# Global models
_bertopic_model = None
_embedding_model = None
_umap_model = None
_hdbscan_model = None

def _get_embedding_model():
    """Lazy load sentence transformer for embeddings"""
    global _embedding_model
    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Embedding model loaded")
        except Exception as e:
            logger.warning(f"Failed to load embedding model: {e}")
            return None
    return _embedding_model

def _get_umap_model():
    """Lazy load UMAP for dimensionality reduction"""
    global _umap_model
    if _umap_model is None:
        try:
            from umap import UMAP
            _umap_model = UMAP(
                n_components=5,
                min_dist=0.0,
                metric='cosine',
                random_state=42
            )
            logger.info("UMAP model initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize UMAP: {e}")
            return None
    return _umap_model

def _get_hdbscan_model():
    """Lazy load HDBSCAN for clustering"""
    global _hdbscan_model
    if _hdbscan_model is None:
        try:
            from hdbscan import HDBSCAN
            _hdbscan_model = HDBSCAN(
                min_cluster_size=3,
                metric="euclidean",
                cluster_selection_method="eom"
            )
            logger.info("HDBSCAN model initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize HDBSCAN: {e}")
            return None
    return _hdbscan_model

def _get_bertopic_model():
    """Lazy load BERTopic model"""
    global _bertopic_model
    if _bertopic_model is None:
        try:
            from bertopic import BERTopic
            
            embedding_model = _get_embedding_model()
            umap_model = _get_umap_model()
            hdbscan_model = _get_hdbscan_model()
            
            if not all([embedding_model, umap_model, hdbscan_model]):
                logger.warning("Required models unavailable for BERTopic")
                return None
            
            _bertopic_model = BERTopic(
                embedding_model=embedding_model,
                umap_model=umap_model,
                hdbscan_model=hdbscan_model,
                language="english",
                top_n_words=10,
                verbose=False
            )
            logger.info("BERTopic model initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize BERTopic: {e}")
            return None
    return _bertopic_model

def bertopic_segment_sentences(sentences, embeddings=None):
    """
    Segment sentences into topics using BERTopic
    
    Args:
        sentences: List of sentence texts
        embeddings: Pre-computed embeddings (optional)
    
    Returns:
        List of topic segments with metadata
    """
    if len(sentences) < 3:
        # Too few sentences for meaningful clustering
        return [{
            'text': ' '.join(sentences),
            'topic': 0,
            'keywords': [],
            'sentences': sentences
        }]
    
    try:
        # Get or compute embeddings
        if embeddings is None:
            embedding_model = _get_embedding_model()
            if not embedding_model:
                logger.warning("Cannot compute embeddings, falling back")
                return None
            embeddings = embedding_model.encode(sentences)
        
        embeddings = np.array(embeddings)
        
        # Get BERTopic model
        topic_model = _get_bertopic_model()
        if not topic_model:
            logger.warning("BERTopic unavailable, using fallback")
            return None
        
        # Fit and transform
        logger.info(f"Fitting BERTopic on {len(sentences)} sentences...")
        topics, probs = topic_model.fit_transform(sentences, embeddings)
        
        # Get topic info
        topic_info = topic_model.get_topic_info()
        
        # Create segments by topic
        segments = {}
        for idx, (sent, topic) in enumerate(zip(sentences, topics)):
            if topic not in segments:
                segments[topic] = {
                    'text': [],
                    'indices': [],
                    'topic': topic,
                    'probability': float(probs[idx].max()) if hasattr(probs[idx], 'max') else 0.0
                }
            segments[topic]['text'].append(sent)
            segments[topic]['indices'].append(idx)
        
        # Build result with keywords
        result = []
        for topic_id in sorted(segments.keys()):
            seg = segments[topic_id]
            
            # Get keywords for this topic
            try:
                topic_keywords = topic_model.get_topic(topic_id)
                keywords = [word for word, _ in topic_keywords] if topic_keywords else []
            except:
                keywords = []
            
            result.append({
                'text': ' '.join(seg['text']),
                'topic': topic_id,
                'keywords': keywords[:5],
                'sentences': seg['text'],
                'sentence_indices': seg['indices'],
                'probability': seg['probability']
            })
        
        logger.info(f"BERTopic identified {len(result)} topics")
        return result
    
    except Exception as e:
        logger.warning(f"BERTopic segmentation failed: {e}")
        return None

def extract_topic_keywords(text, num_keywords=5):
    """
    Extract keywords from topic text
    
    Args:
        text: Topic text content
        num_keywords: Number of keywords to extract
    
    Returns:
        List of keywords
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    try:
        vectorizer = TfidfVectorizer(
            max_features=20,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        words = text.lower().split()
        if len(words) < 10:
            keywords = [w for w in words if len(w) > 3]
            return keywords[:num_keywords]
        
        matrix = vectorizer.fit_transform([text])
        features = vectorizer.get_feature_names_out()
        scores = matrix.toarray()[0]
        
        top_indices = np.argsort(scores)[-num_keywords:][::-1]
        keywords = [features[i] for i in top_indices if scores[i] > 0]
        
        return keywords
    except Exception as e:
        logger.warning(f"Keyword extraction failed: {e}")
        return []

def generate_topic_summary_from_keywords(keywords, text=None):
    """
    Generate a summary/label from keywords and optionally the text
    
    Args:
        keywords: List of important keywords
        text: Original text (optional)
    
    Returns:
        Summary string
    """
    if not keywords:
        return "Topic Discussion"
    
    # Clean up keywords (remove special chars)
    clean_keywords = []
    for kw in keywords[:4]:
        kw = str(kw).strip()
        if kw and len(kw) > 2:
            clean_keywords.append(kw)
    
    if clean_keywords:
        summary = ' • '.join(clean_keywords[:3])
        return summary if len(summary) < 60 else summary[:57] + "..."
    
    return "Topic Discussion"

def segment_with_bertopic(sentences, embeddings=None, timestamps=None):
    """
    Main function: Segment sentences into topics using BERTopic
    
    Args:
        sentences: List of sentence texts
        embeddings: Pre-computed embeddings (optional)
        timestamps: List of timestamp dicts (optional)
    
    Returns:
        List of topic segments or None if BERTopic fails
    """
    # Try BERTopic first
    bertopic_result = bertopic_segment_sentences(sentences, embeddings)
    
    if bertopic_result:
        # Convert BERTopic results to segment format
        segments = []
        for topic_data in bertopic_result:
            segment = {
                'text': topic_data['text'],
                'topic_id': topic_data['topic'],
                'keywords': topic_data['keywords'],
                'title': generate_topic_summary_from_keywords(topic_data['keywords']),
                'start_idx': topic_data['sentence_indices'][0] if topic_data['sentence_indices'] else 0,
                'end_idx': topic_data['sentence_indices'][-1] if topic_data['sentence_indices'] else 0,
                'sentence_count': len(topic_data['sentences']),
            }
            
            # Add timestamps if available
            if timestamps:
                start_ts = topic_data['sentence_indices'][0]
                end_ts = topic_data['sentence_indices'][-1]
                
                if start_ts < len(timestamps) and end_ts < len(timestamps):
                    segment['start_time'] = timestamps[start_ts].get('start', 0.0)
                    segment['end_time'] = timestamps[end_ts].get('end', 0.0)
                else:
                    segment['start_time'] = 0.0
                    segment['end_time'] = 0.0
            
            segments.append(segment)
        
        return segments
    
    return None

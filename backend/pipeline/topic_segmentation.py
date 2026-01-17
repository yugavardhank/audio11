"""
Improved Topic Segmentation Module
Uses clustering and dynamic thresholding for better topic boundary detection
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from sklearn.cluster import DBSCAN
import logging

logger = logging.getLogger(__name__)

def detect_topic_boundaries_clustering(embeddings, sentences, window_size=3, min_cluster_size=2):
    """
    Detect topic boundaries using clustering approach with sliding window
    
    Args:
        embeddings: List of sentence embeddings (numpy arrays)
        sentences: List of sentence texts
        window_size: Sliding window size for local context
        min_cluster_size: Minimum cluster size for DBSCAN
    
    Returns:
        List of boundary indices where topic changes occur
    """
    if len(embeddings) < 2:
        return []
    
    boundaries = []
    embeddings_array = np.array(embeddings)
    
    # Calculate local coherence using sliding window
    coherence_scores = []
    
    for i in range(len(embeddings_array)):
        # Get window of embeddings around current position
        start = max(0, i - window_size)
        end = min(len(embeddings_array), i + window_size + 1)
        window = embeddings_array[start:end]
        
        if len(window) > 1:
            # Calculate average pairwise similarity within window
            distances = cosine_distances(window, window)
            # Convert distances to similarities (1 - distance)
            similarities = 1 - distances
            # Average similarity (excluding diagonal)
            mask = ~np.eye(len(similarities), dtype=bool)
            avg_similarity = similarities[mask].mean() if mask.sum() > 0 else 0
            coherence_scores.append(avg_similarity)
        else:
            coherence_scores.append(1.0)  # Single element has perfect coherence
    
    # Detect significant drops in coherence (topic boundaries)
    if len(coherence_scores) > 1:
        coherence_array = np.array(coherence_scores)
        
        # Calculate dynamic threshold based on coherence distribution
        mean_coherence = coherence_array.mean()
        std_coherence = coherence_array.std()
        threshold = mean_coherence - (0.5 * std_coherence)  # Adaptive threshold
        
        # Detect boundaries where coherence drops below threshold
        for i in range(1, len(coherence_scores)):
            if coherence_scores[i] < threshold and coherence_scores[i-1] >= threshold:
                boundaries.append(i)
    
    return boundaries

def detect_topic_boundaries_embeddings(embeddings, sentences, threshold=0.4, min_distance=3):
    """
    Detect topic boundaries using embedding similarity with improved threshold
    
    Args:
        embeddings: List of sentence embeddings
        sentences: List of sentence texts
        threshold: Similarity threshold for topic change (lower = more boundaries)
        min_distance: Minimum sentences between boundaries
    
    Returns:
        List of boundary indices
    """
    if len(embeddings) < 2:
        return []
    
    boundaries = []
    embeddings_array = np.array(embeddings)
    
    for i in range(1, len(embeddings_array)):
        # Calculate similarity between consecutive sentences
        sim = 1 - cosine_distances(
            embeddings_array[i-1:i], 
            embeddings_array[i:i+1]
        )[0][0]
        
        # Check if this is a topic boundary
        if sim < threshold:
            # Ensure minimum distance between boundaries
            if not boundaries or (i - boundaries[-1]) >= min_distance:
                boundaries.append(i)
    
    return boundaries

def refine_boundaries_with_context(boundaries, sentences, embeddings, context_window=2):
    """
    Refine boundary positions by looking at context
    
    Args:
        boundaries: List of boundary indices
        sentences: List of sentence texts
        embeddings: List of sentence embeddings
        context_window: Number of sentences to look around boundary
    
    Returns:
        Refined list of boundary indices
    """
    if not boundaries or len(embeddings) < 2:
        return boundaries
    
    refined_boundaries = []
    embeddings_array = np.array(embeddings)
    
    for boundary_idx in boundaries:
        # Get context around boundary
        start = max(0, boundary_idx - context_window)
        end = min(len(embeddings_array), boundary_idx + context_window + 1)
        
        if start >= end - 1:
            refined_boundaries.append(boundary_idx)
            continue
        
        # Find the position with maximum distance change
        max_distance_change = 0
        best_idx = boundary_idx
        
        context_embeddings = embeddings_array[start:end]
        
        for j in range(1, len(context_embeddings) - 1):
            # Distance from previous to current
            dist_prev = cosine_distances(
                context_embeddings[j-1:j], 
                context_embeddings[j:j+1]
            )[0][0]
            
            # Distance from current to next
            dist_next = cosine_distances(
                context_embeddings[j:j+1], 
                context_embeddings[j+1:j+2]
            )[0][0]
            
            # Total distance change
            distance_change = abs(dist_next - dist_prev)
            
            if distance_change > max_distance_change:
                max_distance_change = distance_change
                best_idx = start + j
        
        refined_boundaries.append(best_idx)
    
    # Remove duplicates and sort
    refined_boundaries = sorted(set(refined_boundaries))
    
    return refined_boundaries

def get_topic_segments(boundaries, sentences, timestamps=None):
    """
    Create topic segments from boundaries
    
    Args:
        boundaries: List of boundary indices
        sentences: List of sentence texts
        timestamps: List of sentence timestamps (optional)
    
    Returns:
        List of topic segments with text and metadata
    """
    segments = []
    
    if not boundaries:
        # Single segment if no boundaries
        if sentences:
            segment = {
                'text': ' '.join(sentences),
                'start_idx': 0,
                'end_idx': len(sentences) - 1,
                'sentence_count': len(sentences),
                'start_time': timestamps[0]['start'] if timestamps else 0.0,
                'end_time': timestamps[-1]['end'] if timestamps else 0.0,
            }
            segments.append(segment)
        return segments
    
    # Create segments between boundaries
    start_idx = 0
    boundaries_list = list(boundaries)
    
    for boundary_idx in boundaries_list:
        segment = {
            'text': ' '.join(sentences[start_idx:boundary_idx]),
            'start_idx': start_idx,
            'end_idx': boundary_idx - 1,
            'sentence_count': boundary_idx - start_idx,
            'start_time': timestamps[start_idx]['start'] if timestamps else 0.0,
            'end_time': timestamps[boundary_idx - 1]['end'] if timestamps else 0.0,
        }
        if segment['text'].strip():
            segments.append(segment)
        start_idx = boundary_idx
    
    # Add final segment
    if start_idx < len(sentences):
        segment = {
            'text': ' '.join(sentences[start_idx:]),
            'start_idx': start_idx,
            'end_idx': len(sentences) - 1,
            'sentence_count': len(sentences) - start_idx,
            'start_time': timestamps[start_idx]['start'] if timestamps else 0.0,
            'end_time': timestamps[-1]['end'] if timestamps else 0.0,
        }
        if segment['text'].strip():
            segments.append(segment)
    
    return segments

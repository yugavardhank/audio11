"""
Evaluation metrics for topic segmentation.

Implements Pk, WinDiff, and SPCF (Segmentation Purity and Coverage F-score) metrics
as described in "Advancing Topic Segmentation of Broadcasted Speech with Multilingual 
Semantic Embeddings" (arxiv:2409.06222v1).
"""

import numpy as np
from typing import List, Tuple


def segment_to_boundaries(segments: List[dict], total_segments: int = None) -> List[int]:
    """
    Convert segment list to binary boundary vector.
    
    Args:
        segments: List of segment dicts with 'start' and 'end' positions
        total_segments: Total number of segments in transcript
        
    Returns:
        Binary vector where 1 indicates topic boundary
    """
    if total_segments is None:
        total_segments = max(seg['end'] for seg in segments) + 1
    
    boundaries = np.zeros(total_segments, dtype=int)
    for seg in segments:
        if seg['start'] > 0:
            boundaries[seg['start']] = 1
    
    return boundaries.tolist()


def boundaries_to_segments(boundaries: List[int]) -> List[Tuple[int, int]]:
    """
    Convert binary boundary vector to segment spans.
    
    Args:
        boundaries: Binary vector where 1 = boundary
        
    Returns:
        List of (start, end) tuples for each segment
    """
    segments = []
    start = 0
    
    for i, is_boundary in enumerate(boundaries[1:], 1):
        if is_boundary:
            segments.append((start, i - 1))
            start = i
    
    if start < len(boundaries):
        segments.append((start, len(boundaries) - 1))
    
    return segments


def calculate_pk(ref_boundaries: List[int], hyp_boundaries: List[int], k: int = None) -> float:
    """
    Calculate Pk (probability of error) metric.
    
    Lower is better. Pk measures the probability that, while traversing a sliding
    window of size k across sentences, the sentences at the window's edges will be
    inaccurately classified as belonging to the same segment.
    
    Args:
        ref_boundaries: Reference binary boundary vector
        hyp_boundaries: Hypothesis binary boundary vector
        k: Window size (default: half of average reference segment size)
        
    Returns:
        Pk score (float between 0 and 1)
    """
    ref_boundaries = np.array(ref_boundaries)
    hyp_boundaries = np.array(hyp_boundaries)
    
    if len(ref_boundaries) != len(hyp_boundaries):
        raise ValueError("Reference and hypothesis must have same length")
    
    n = len(ref_boundaries)
    
    if k is None:
        # Default k = half of average segment size
        ref_segments = boundaries_to_segments(ref_boundaries)
        if ref_segments:
            avg_segment_size = n / len(ref_segments)
            k = max(1, int(np.round(avg_segment_size / 2)))
        else:
            k = max(1, n // 2)
    
    if k >= n:
        return 0.0
    
    errors = 0
    total = 0
    
    for i in range(n - k):
        # Get samples at window edges
        ref_at_edges = ref_boundaries[i] == ref_boundaries[i + k]
        hyp_at_edges = hyp_boundaries[i] == hyp_boundaries[i + k]
        
        # Error if prediction doesn't match reference
        if ref_at_edges != hyp_at_edges:
            errors += 1
        total += 1
    
    return float(errors) / float(total) if total > 0 else 0.0


def calculate_windiff(ref_boundaries: List[int], hyp_boundaries: List[int], k: int = None) -> float:
    """
    Calculate WinDiff metric.
    
    Lower is better. Counts instances where predicted and reference segment boundaries
    differ within a sliding window.
    
    Args:
        ref_boundaries: Reference binary boundary vector
        hyp_boundaries: Hypothesis binary boundary vector
        k: Window size (default: half of average reference segment size)
        
    Returns:
        WinDiff score (float between 0 and 1)
    """
    ref_boundaries = np.array(ref_boundaries)
    hyp_boundaries = np.array(hyp_boundaries)
    
    if len(ref_boundaries) != len(hyp_boundaries):
        raise ValueError("Reference and hypothesis must have same length")
    
    n = len(ref_boundaries)
    
    if k is None:
        ref_segments = boundaries_to_segments(ref_boundaries)
        if ref_segments:
            avg_segment_size = n / len(ref_segments)
            k = max(1, int(np.round(avg_segment_size / 2)))
        else:
            k = max(1, n // 2)
    
    if k >= n:
        return 0.0
    
    differences = 0
    
    for i in range(n - k):
        # Count boundaries in windows
        ref_boundaries_in_window = np.sum(ref_boundaries[i:i+k])
        hyp_boundaries_in_window = np.sum(hyp_boundaries[i:i+k])
        
        # Check if counts differ
        if ref_boundaries_in_window != hyp_boundaries_in_window:
            differences += 1
    
    return float(differences) / float(n - k) if (n - k) > 0 else 0.0


def calculate_spcf(ref_segments: List[Tuple[int, int]], 
                   hyp_segments: List[Tuple[int, int]]) -> Tuple[float, float, float]:
    """
    Calculate SPCF (Segmentation Purity and Coverage F-score).
    
    Based on speaker diarization metrics, measures how well hypothesis segments
    match reference segments while accounting for partial overlaps.
    
    Args:
        ref_segments: List of (start, end) tuples for reference segments
        hyp_segments: List of (start, end) tuples for hypothesis segments
        
    Returns:
        Tuple of (purity, coverage, f_score)
    """
    def overlap_percentage(seg1: Tuple[int, int], seg2: Tuple[int, int]) -> float:
        """Calculate overlap percentage between two segments."""
        start = max(seg1[0], seg2[0])
        end = min(seg1[1], seg2[1])
        
        if start > end:
            return 0.0
        
        overlap = end - start + 1
        length = seg2[1] - seg2[0] + 1
        return overlap / length
    
    # Calculate purity (for each hypothesis segment, find best matching reference)
    purity_scores = []
    for hyp_seg in hyp_segments:
        best_overlap = max(
            (overlap_percentage(hyp_seg, ref_seg) for ref_seg in ref_segments),
            default=0.0
        )
        purity_scores.append(best_overlap)
    
    purity = np.mean(purity_scores) if purity_scores else 0.0
    
    # Calculate coverage (for each reference segment, find best matching hypothesis)
    coverage_scores = []
    for ref_seg in ref_segments:
        best_overlap = max(
            (overlap_percentage(ref_seg, hyp_seg) for hyp_seg in hyp_segments),
            default=0.0
        )
        coverage_scores.append(best_overlap)
    
    coverage = np.mean(coverage_scores) if coverage_scores else 0.0
    
    # Calculate F-score
    if purity + coverage == 0:
        f_score = 0.0
    else:
        f_score = 2 * purity * coverage / (purity + coverage)
    
    return float(purity), float(coverage), float(f_score)


def evaluate_segmentation(ref_segments: List[dict], 
                         hyp_segments: List[dict],
                         total_blocks: int = None) -> dict:
    """
    Comprehensive evaluation of segmentation quality.
    
    Args:
        ref_segments: Reference segments with 'start' and 'end' keys
        hyp_segments: Hypothesis segments with 'start' and 'end' keys
        total_blocks: Total number of text blocks/sentences
        
    Returns:
        Dictionary with Pk, WinDiff, and SPCF scores
    """
    if total_blocks is None:
        all_ends = []
        if ref_segments:
            all_ends.extend(seg['end'] for seg in ref_segments)
        if hyp_segments:
            all_ends.extend(seg['end'] for seg in hyp_segments)
        total_blocks = max(all_ends) + 1 if all_ends else 1
    
    # Convert to boundary vectors
    ref_boundaries = segment_to_boundaries(ref_segments, total_blocks)
    hyp_boundaries = segment_to_boundaries(hyp_segments, total_blocks)
    
    # Calculate metrics
    pk = calculate_pk(ref_boundaries, hyp_boundaries)
    windiff = calculate_windiff(ref_boundaries, hyp_boundaries)
    
    # Convert back to segment tuples for SPCF
    ref_segs_tuple = boundaries_to_segments(ref_boundaries)
    hyp_segs_tuple = boundaries_to_segments(hyp_boundaries)
    purity, coverage, spcf = calculate_spcf(ref_segs_tuple, hyp_segs_tuple)
    
    return {
        'pk': pk,
        'windiff': windiff,
        'purity': purity,
        'coverage': coverage,
        'spcf': spcf,
        'total_ref_segments': len(ref_segments),
        'total_hyp_segments': len(hyp_segments),
    }


if __name__ == "__main__":
    # Example usage
    ref_segments = [
        {'start': 0, 'end': 4},
        {'start': 5, 'end': 9},
        {'start': 10, 'end': 14},
    ]
    
    hyp_segments = [
        {'start': 0, 'end': 3},
        {'start': 4, 'end': 8},
        {'start': 9, 'end': 14},
    ]
    
    results = evaluate_segmentation(ref_segments, hyp_segments, 15)
    print("Segmentation Evaluation Results:")
    for metric, value in results.items():
        print(f"  {metric}: {value:.4f}" if isinstance(value, float) else f"  {metric}: {value}")

"""
Visualization module for audio analysis and topic segmentation.
Generates mel spectrograms and cosine similarity heatmaps.
"""

import os
import base64
from io import BytesIO
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import librosa
import librosa.display


def generate_mel_spectrogram(audio_path, output_dir):
    """
    Generate mel spectrogram visualization from audio file.
    
    Args:
        audio_path: Path to audio file
        output_dir: Directory to save visualization
        
    Returns:
        Path to saved spectrogram image
    """
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=None)
        
        # Generate mel spectrogram
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_db = librosa.power_to_db(S, ref=np.max)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Display spectrogram
        img = librosa.display.specshow(
            S_db,
            sr=sr,
            hop_length=512,
            x_axis='time',
            y_axis='mel',
            ax=ax,
            cmap='viridis'
        )
        
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        ax.set_title('Mel Spectrogram - Frequency Analysis', fontsize=14, fontweight='bold')
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Frequency (Hz)')
        
        # Save figure
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'mel_spectrogram.png')
        plt.tight_layout()
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        return output_path
        
    except Exception as e:
        print(f"Error generating mel spectrogram: {e}")
        return None


def generate_cosine_similarity_graph(embeddings, sentence_timestamps, output_dir):
    """
    Generate cosine similarity heatmap for topic segmentation.
    Shows similarity between consecutive segments.
    
    Args:
        embeddings: Array of sentence embeddings (N, D)
        sentence_timestamps: List of timestamp dicts with 'start', 'end', 'text'
        output_dir: Directory to save visualization
        
    Returns:
        Path to saved similarity graph
    """
    try:
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Calculate cosine similarity matrix
        similarity_matrix = cosine_similarity(embeddings)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 1, figsize=(16, 10))
        
        # Plot 1: Full similarity heatmap
        im = axes[0].imshow(similarity_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        axes[0].set_title('Cosine Similarity Heatmap - Full Matrix', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Segment Index')
        axes[0].set_ylabel('Segment Index')
        fig.colorbar(im, ax=axes[0], label='Similarity Score')
        
        # Plot 2: Diagonal similarity (similarity of each segment with next)
        diagonal_sim = np.array([
            similarity_matrix[i, i+1] if i < len(similarity_matrix) - 1 else 0
            for i in range(len(similarity_matrix))
        ])
        
        times = np.array([ts.get('start', 0) for ts in sentence_timestamps])
        
        axes[1].plot(times, diagonal_sim, linewidth=2, color='steelblue', marker='o', markersize=4)
        axes[1].fill_between(times, diagonal_sim, alpha=0.3, color='steelblue')
        axes[1].set_title('Segment Continuity - Similarity with Next Segment', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Time (seconds)')
        axes[1].set_ylabel('Cosine Similarity')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([0, 1])
        
        # Add topic boundary markers (low similarity = potential boundary)
        threshold = 0.5
        boundaries = np.where(diagonal_sim < threshold)[0]
        for boundary in boundaries[:10]:  # Limit to 10 most significant boundaries
            axes[1].axvline(x=times[boundary], color='red', linestyle='--', alpha=0.5, linewidth=1)
        
        # Save figure
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'cosine_similarity.png')
        plt.tight_layout()
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        return output_path
        
    except Exception as e:
        print(f"Error generating cosine similarity graph: {e}")
        return None


def generate_topic_boundaries_visualization(segments, topics, output_dir):
    """
    Generate visualization of detected topic boundaries.
    
    Args:
        segments: Transcript segments with timing
        topics: Detected topic segments with boundaries
        output_dir: Directory to save visualization
        
    Returns:
        Path to saved visualization
    """
    try:
        fig, ax = plt.subplots(figsize=(16, 6))
        
        # Get total duration
        total_duration = max(seg.get('end', 0) for seg in segments) if segments else 100
        
        # Plot timeline
        ax.barh([0], [total_duration], height=0.5, color='lightgray', label='Total Duration')
        
        # Plot topic segments
        colors = plt.cm.Set3(np.linspace(0, 1, len(topics)))
        for i, topic in enumerate(topics):
            start = topic.get('start_time', 0)
            end = topic.get('end_time', start + 10)
            duration = end - start
            
            ax.barh([0], [duration], left=[start], height=0.3, color=colors[i], 
                   label=f"Topic {i+1}: {topic.get('title', 'Unknown')}")
        
        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_title('Topic Segmentation Timeline', fontsize=14, fontweight='bold')
        ax.set_ylim([-0.5, 0.5])
        ax.set_yticks([])
        
        # Add legend
        if len(topics) <= 10:
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=10)
        
        # Save figure
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'topic_timeline.png')
        plt.tight_layout()
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        return output_path
        
    except Exception as e:
        print(f"Error generating topic boundaries visualization: {e}")
        return None


def image_to_base64(image_path):
    """
    Convert image file to base64 string for embedding in HTML.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Base64 encoded string or None if failed
    """
    try:
        if not os.path.exists(image_path):
            return None
            
        with open(image_path, 'rb') as f:
            image_data = f.read()
            base64_str = base64.b64encode(image_data).decode('utf-8')
            ext = os.path.splitext(image_path)[1].lower()
            mime_type = 'image/png' if ext == '.png' else 'image/jpeg'
            return f"data:{mime_type};base64,{base64_str}"
    except Exception as e:
        print(f"Error converting image to base64: {e}")
        return None

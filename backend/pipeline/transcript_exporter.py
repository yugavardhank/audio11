"""
Transcript exporter for multiple formats.

Supports:
- WebVTT (subtitle format with timestamps)
- DOTE JSON (structured transcript format)
- Plain text with timestamps
"""

import json
import os
from typing import List, Dict, Any, Optional
from datetime import timedelta


def seconds_to_vtt_timestamp(seconds: float) -> str:
    """Convert seconds to VTT timestamp format (HH:MM:SS.mmm)."""
    td = timedelta(seconds=seconds)
    hours, remainder = divmod(int(td.total_seconds()), 3600)
    minutes, seconds_part = divmod(remainder, 60)
    milliseconds = int((td.total_seconds() % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds_part:02d}.{milliseconds:03d}"


def seconds_to_dote_timestamp(seconds: float) -> str:
    """Convert seconds to ISO 8601 duration format for DOTE."""
    td = timedelta(seconds=seconds)
    total_seconds = td.total_seconds()
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    secs = total_seconds % 60
    return f"PT{hours:02d}H{minutes:02d}M{secs:05.2f}S"


class TranscriptExporter:
    """Exports transcripts in multiple formats."""
    
    @staticmethod
    def export_webvtt(segments: List[Dict[str, Any]], output_path: str) -> str:
        """
        Export transcript as WebVTT subtitle file.
        
        Args:
            segments: List of segment dicts with 'text', 'start_time', 'end_time'
            output_path: Path to save .vtt file
            
        Returns:
            Path to saved file
        """
        lines = ["WEBVTT", ""]
        
        for i, segment in enumerate(segments):
            start = seconds_to_vtt_timestamp(segment.get('start_time', 0))
            end = seconds_to_vtt_timestamp(segment.get('end_time', 0))
            text = segment.get('text', '')
            
            # Add speaker info if available
            if 'speaker' in segment:
                text = f"{segment['speaker']}: {text}"
            
            lines.append(f"{start} --> {end}")
            lines.append(text)
            lines.append("")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))
        
        return output_path
    
    @staticmethod
    def export_dote_json(segments: List[Dict[str, Any]], 
                        topics: List[Dict[str, Any]],
                        metadata: Optional[Dict[str, Any]] = None,
                        output_path: str = None) -> Dict[str, Any]:
        """
        Export transcript in DOTE JSON format (Podlove-compatible).
        
        DOTE = Digital Object Transcript Format (Extended)
        
        Args:
            segments: Transcript segments with timing
            topics: Topic segments with boundaries and summaries
            metadata: Optional metadata (title, duration, etc.)
            output_path: Optional path to save JSON file
            
        Returns:
            DOTE JSON dict (also saves to file if output_path provided)
        """
        # Build transcript entries
        transcript_entries = []
        for i, segment in enumerate(segments):
            entry = {
                "start": seconds_to_dote_timestamp(segment.get('start_time', 0)),
                "end": seconds_to_dote_timestamp(segment.get('end_time', 0)),
                "text": segment.get('text', ''),
                "id": f"t{i}",
            }
            
            if 'speaker' in segment:
                entry['speaker'] = segment['speaker']
            
            if 'confidence' in segment:
                entry['confidence'] = segment['confidence']
            
            transcript_entries.append(entry)
        
        # Build chapters from topics
        chapters = []
        for i, topic in enumerate(topics):
            chapter = {
                "start": seconds_to_dote_timestamp(topic.get('start_time', 0)),
                "title": topic.get('title', f'Topic {i+1}'),
                "id": f"ch{i}",
            }
            
            if 'summary' in topic:
                chapter['summary'] = topic['summary']
            
            if 'keywords' in topic:
                chapter['tags'] = topic['keywords']
            
            chapters.append(chapter)
        
        # Build DOTE structure
        dote = {
            "version": "1.2",
            "transcript": transcript_entries,
            "chapters": chapters,
        }
        
        # Add metadata
        if metadata:
            dote['metadata'] = {
                "title": metadata.get('title', ''),
                "description": metadata.get('description', ''),
                "duration": metadata.get('duration', ''),
                "publicationDate": metadata.get('publication_date', ''),
                "author": metadata.get('author', ''),
            }
        
        # Save to file if path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(dote, f, indent=2, ensure_ascii=False)
        
        return dote
    
    @staticmethod
    def export_plain_text(segments: List[Dict[str, Any]], 
                         output_path: str,
                         include_timestamps: bool = True) -> str:
        """
        Export transcript as plain text.
        
        Args:
            segments: Transcript segments
            output_path: Path to save .txt file
            include_timestamps: Whether to include timing information
            
        Returns:
            Path to saved file
        """
        lines = []
        
        for segment in segments:
            if include_timestamps:
                start = seconds_to_vtt_timestamp(segment.get('start_time', 0))
                speaker = segment.get('speaker', 'SPEAKER')
                text = segment.get('text', '')
                lines.append(f"[{start}] {speaker}: {text}")
            else:
                text = segment.get('text', '')
                if 'speaker' in segment:
                    text = f"{segment['speaker']}: {text}"
                lines.append(text)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))
        
        return output_path
    
    @staticmethod
    def export_all_formats(segments: List[Dict[str, Any]],
                          topics: List[Dict[str, Any]],
                          output_dir: str,
                          metadata: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """
        Export transcript in all supported formats.
        
        Args:
            segments: Transcript segments
            topics: Topic segments
            output_dir: Directory to save all files
            metadata: Optional metadata
            
        Returns:
            Dictionary mapping format names to file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        
        base_name = metadata.get('title', 'transcript') if metadata else 'transcript'
        base_name = base_name.replace(' ', '_').replace('/', '_')
        
        results = {}
        
        # WebVTT
        vtt_path = os.path.join(output_dir, f"{base_name}.vtt")
        TranscriptExporter.export_webvtt(segments, vtt_path)
        results['webvtt'] = vtt_path
        
        # DOTE JSON
        dote_path = os.path.join(output_dir, f"{base_name}.dote.json")
        TranscriptExporter.export_dote_json(segments, topics, metadata, dote_path)
        results['dote_json'] = dote_path
        
        # Plain text with timestamps
        txt_path = os.path.join(output_dir, f"{base_name}.txt")
        TranscriptExporter.export_plain_text(segments, txt_path, include_timestamps=True)
        results['plain_text'] = txt_path
        
        # Plain text without timestamps
        txt_simple_path = os.path.join(output_dir, f"{base_name}_simple.txt")
        TranscriptExporter.export_plain_text(segments, txt_simple_path, include_timestamps=False)
        results['plain_text_simple'] = txt_simple_path
        
        return results


if __name__ == "__main__":
    # Example usage
    example_segments = [
        {
            'text': 'This is the first segment',
            'start_time': 0.0,
            'end_time': 5.0,
            'speaker': 'SPEAKER_00'
        },
        {
            'text': 'This is the second segment',
            'start_time': 5.0,
            'end_time': 10.5,
            'speaker': 'SPEAKER_01'
        },
    ]
    
    example_topics = [
        {
            'title': 'Introduction',
            'start_time': 0.0,
            'summary': 'Speaker introduces the topic',
            'keywords': ['intro', 'welcome']
        },
        {
            'title': 'Main Discussion',
            'start_time': 5.0,
            'summary': 'Discussion of main points',
            'keywords': ['discussion', 'important']
        }
    ]
    
    example_metadata = {
        'title': 'Example Audio',
        'duration': '00:10:30',
        'author': 'Test Author'
    }
    
    # Test exports
    exporter = TranscriptExporter()
    
    # Create temp directory for testing
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = exporter.export_all_formats(
            example_segments,
            example_topics,
            tmpdir,
            example_metadata
        )
        
        print("Exported files:")
        for fmt, path in paths.items():
            print(f"  {fmt}: {path}")
            if os.path.exists(path):
                print(f"    ✓ File exists, size: {os.path.getsize(path)} bytes")

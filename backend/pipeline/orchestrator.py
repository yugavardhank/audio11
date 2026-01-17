from pipeline.audio import normalize_audio
from pipeline.chunker import chunk_audio
from pipeline.asr import transcribe
from pipeline.diarization import diarize
from pipeline.context_blocks import build_blocks
from pipeline.text_preprocessor import preprocess_text, segment_into_sentences
from pipeline.topic_segmentation import detect_topic_boundaries_embeddings, refine_boundaries_with_context, get_topic_segments
from pipeline.bertopic_segmentation import segment_with_bertopic
from pipeline.summarize import summarize_topics
from pipeline.topic_labeler import generate_topic_labels_batch
from pipeline.metrics import evaluate_segmentation
from pipeline.transcript_exporter import TranscriptExporter
import numpy as np
import os

def format_time(seconds):
    """Convert seconds to MM:SS format"""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins}:{secs:02d}"

def assign_speakers_to_transcript(transcript, speaker_data):
    """
    Assign speakers to transcript segments using diarization results.
    """
    if isinstance(speaker_data, dict) and "segments" in speaker_data:
        speaker_segments = speaker_data["segments"]
        
        for seg in transcript:
            seg_start = seg["start"]
            seg_end = seg["end"]
            
            # Find overlapping speaker
            assigned_speaker = "UNKNOWN"
            max_overlap = 0
            
            for spk_seg in speaker_segments:
                # Calculate overlap
                overlap_start = max(seg_start, spk_seg["start"])
                overlap_end = min(seg_end, spk_seg["end"])
                overlap = max(0, overlap_end - overlap_start)
                
                if overlap > max_overlap:
                    max_overlap = overlap
                    assigned_speaker = spk_seg["speaker"]
            
            seg["speaker"] = assigned_speaker
    else:
        # Fallback: use simple mapping
        for seg in transcript:
            seg["speaker"] = speaker_data.get(seg["start"], "SPEAKER_00")
    
    return transcript

def run_pipeline(audio_path):
    """
    Main pipeline orchestrator with improved NLP processing:
    1. Normalize audio to mono 16kHz
    2. Chunk audio for processing
    3. Transcribe using Whisper ASR
    4. Perform speaker diarization
    5. Preprocess text (cleaning, tokenization)
    6. Generate sentence embeddings
    7. Detect topic boundaries using clustering
    8. Generate summaries and labels
    """
    print(f"\n=== Starting Improved Pipeline for: {audio_path} ===\n")
    
    # Step 1: Normalize audio
    print("Step 1: Normalizing audio...")
    normalized_audio = normalize_audio(audio_path)
    
    # Step 2: Chunk audio
    print("\nStep 2: Chunking audio...")
    chunks = chunk_audio(normalized_audio)
    print(f"Created {len(chunks)} chunks\n")
    
    # Step 3: Transcribe
    print("Step 3: Transcribing audio...")
    transcript = transcribe(chunks)
    
    # Step 4: Diarize
    print("\nStep 4: Performing speaker diarization...")
    speaker_data = diarize(normalized_audio)
    speaker_count = speaker_data.get("speaker_count", 1) if isinstance(speaker_data, dict) else len(set(speaker_data.values()))
    
    # Step 5: Assign speakers to transcript
    print("\nStep 5: Assigning speakers to transcript...")
    transcript = assign_speakers_to_transcript(transcript, speaker_data)
    
    # Step 6: Validate transcript
    if not transcript or len(transcript) == 0:
        print("Warning: No transcript available, using placeholder")
        transcript = [{
            "start": 0.0,
            "end": 1.0,
            "text": "No transcription available",
            "speaker": "SPEAKER_00",
            "start_time": "0:00",
            "end_time": "0:01"
        }]
    
    # Step 7: Use transcript segments as sentences directly
    print("\nStep 6: Using transcript segments as sentences...")
    
    # Use transcript segments directly - they already have proper timestamps
    sentences = [seg.get("text", "") for seg in transcript if seg.get("text", "").strip()]
    sentence_timestamps = [
        {
            "start": seg.get("start", 0.0),
            "end": seg.get("end", 0.0),
            "text": seg.get("text", "")
        }
        for seg in transcript if seg.get("text", "").strip()
    ]
    
    print(f"Using {len(sentences)} transcript segments as sentences")
    
    # Step 8: Generate sentence embeddings
    print("\nStep 7: Generating sentence embeddings...")
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(sentences)
        print(f"Generated embeddings for {len(embeddings)} sentences")
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        # Fallback: use dummy embeddings
        embeddings = np.random.random((len(sentences), 384))
    
    # Step 9: Detect topic boundaries using improved method
    print("\nStep 8: Detecting topics with BERTopic...")
    
    # Try BERTopic first (most advanced)
    if len(embeddings) > 2:
        try:
            bertopic_segments = segment_with_bertopic(sentences, embeddings, sentence_timestamps)
            if bertopic_segments and len(bertopic_segments) > 0:
                segments = bertopic_segments
                print(f"BERTopic identified {len(segments)} topics")
        except Exception as e:
            print(f"BERTopic failed: {e}, falling back to clustering...")
            segments = None
    else:
        segments = None
    
    # Fallback: Use clustering-based approach if BERTopic fails
    if segments is None or (isinstance(segments, list) and len(segments) == 0):
        print("Using fallback: Clustering-based topic detection...")
        
        # Use clustering-based approach with adaptive thresholding
        if len(embeddings) > 1:
            # Try clustering approach first
            try:
                from pipeline.topic_segmentation import detect_topic_boundaries_clustering
                boundaries = detect_topic_boundaries_clustering(embeddings, sentences, window_size=2)
                print(f"Clustering detected {len(boundaries)} boundaries")
            except:
                # Fallback to embedding-based approach
                boundaries = detect_topic_boundaries_embeddings(embeddings, sentences, threshold=0.35)
                print(f"Embedding-based detected {len(boundaries)} boundaries")
            
            # Refine boundaries using context
            if boundaries:
                boundaries = refine_boundaries_with_context(boundaries, sentences, embeddings)
                print(f"Refined to {len(boundaries)} boundaries")
        else:
            boundaries = []
        
        # Step 10: Create topic segments
        print("\nStep 9: Creating topic segments...")
        segments = get_topic_segments(boundaries, sentences, sentence_timestamps)
        print(f"Created {len(segments)} topic segments")
    
    # Step 11: Generate summaries
    print("\nStep 10: Generating summaries...")
    segments = summarize_topics(segments)
    
    # Step 12: Generate topic labels
    print("Step 11: Generating topic labels...")
    segments = generate_topic_labels_batch(segments)
    
    # Format output topics
    topics = []
    for i, seg in enumerate(segments):
        # Handle both old and new segment formats
        title = seg.get("title", seg.get("title", f"Topic {i + 1}"))
        text = seg.get("text", "")
        summary = seg.get("summary", "")
        
        # Get timestamps
        if "start_time" in seg and isinstance(seg["start_time"], (int, float)):
            start_time = seg["start_time"]
            end_time = seg.get("end_time", start_time + 1)
        else:
            start_time = seg.get("start", 0.0)
            end_time = seg.get("end", 0.0)
        
        topic = {
            "id": i + 1,
            "title": title,
            "text": text,
            "summary": summary,
            "start": start_time,
            "end": end_time,
            "start_time": format_time(start_time),
            "end_time": format_time(end_time),
            "sentence_count": seg.get("sentence_count", 0),
            "keywords": seg.get("keywords", [])
        }
        topics.append(topic)
    
    # Format transcript timestamps
    for seg in transcript:
        seg["start_time"] = format_time(seg.get("start", 0.0))
        seg["end_time"] = format_time(seg.get("end", 0.0))
    
    print("\n=== Improving Output Formats and Metrics ===\n")
    
    # Convert transcript segments for export (with start_time and end_time fields)
    export_transcript = []
    for seg in transcript:
        export_transcript.append({
            'text': seg.get('text', ''),
            'start_time': seg.get('start', 0.0),
            'end_time': seg.get('end', 0.0),
            'speaker': seg.get('speaker', 'SPEAKER_00'),
            'confidence': seg.get('confidence', 0.0)
        })
    
    # Convert topic segments for export
    export_topics = []
    for topic in topics:
        export_topics.append({
            'title': topic.get('title', ''),
            'start_time': topic.get('start', 0.0),
            'end_time': topic.get('end', 0.0),
            'summary': topic.get('summary', ''),
            'keywords': topic.get('keywords', [])
        })
    
    # Create output directory for transcripts
    output_dir = os.path.join(os.path.dirname(audio_path), 'transcripts')
    os.makedirs(output_dir, exist_ok=True)
    
    # Export transcripts in multiple formats
    print("\nExporting transcripts in multiple formats...")
    metadata = {
        'title': os.path.splitext(os.path.basename(audio_path))[0],
        'duration': format_time(transcript[-1]["end"]) if transcript else "0:00",
        'author': 'Audio Pipeline',
        'publication_date': None
    }
    
    try:
        export_paths = TranscriptExporter.export_all_formats(
            export_transcript,
            export_topics,
            output_dir,
            metadata
        )
        print(f"Exported formats: {', '.join(export_paths.keys())}")
    except Exception as e:
        print(f"Warning: Could not export transcripts: {e}")
        export_paths = {}
    
    print("\n=== Improved Pipeline Complete ===\n")
    
    return {
        "transcript": transcript,
        "topics": topics,
        "speaker_count": speaker_count,
        "total_duration": format_time(transcript[-1]["end"]) if transcript else "0:00",
        "export_formats": export_paths,
        "metrics": {
            "total_segments": len(transcript),
            "total_topics": len(topics),
            "avg_topic_length": np.mean([t.get('end', 0) - t.get('start', 0) for t in topics]) if topics else 0
        }
    }
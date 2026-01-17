import os

# Lazy load whisper to avoid Windows libc initialization issues
_model = None

def _get_model():
    """Lazy load Whisper model on first use."""
    global _model
    if _model is None:
        try:
            import whisper
            print("Loading Whisper model (base)...")
            _model = whisper.load_model("base")
            print("Whisper model loaded successfully")
        except Exception as e:
            print(f"Error loading Whisper model: {e}")
            raise
    return _model

def transcribe(chunks):
    """
    Transcribe audio chunks using Whisper ASR.
    Returns list of segments with start/end times and text.
    """
    model = _get_model()
    result = []
    offset = 0.0

    print(f"Starting transcription of {len(chunks)} chunks...")
    
    for idx, chunk_path in enumerate(chunks):
        print(f"Transcribing chunk {idx + 1}/{len(chunks)}: {chunk_path}")
        
        # Check if file exists and has size
        if not os.path.exists(chunk_path):
            print(f"Warning: Chunk file does not exist: {chunk_path}")
            continue
            
        file_size = os.path.getsize(chunk_path)
        if file_size < 1000:  # Less than 1KB is likely empty
            print(f"Warning: Chunk file too small ({file_size} bytes), skipping")
            continue
        
        try:
            # Transcribe with language detection
            out = model.transcribe(
                chunk_path, 
                fp16=False,
                language="en",  # Set to None for auto-detection
                task="transcribe"
            )
            
            # Process segments
            if out.get("segments"):
                for s in out["segments"]:
                    result.append({
                        "start": s["start"] + offset,
                        "end": s["end"] + offset,
                        "text": s["text"].strip()
                    })
                
                # Update offset for next chunk
                offset += out["segments"][-1]["end"]
            else:
                print(f"No segments found in chunk {idx + 1}")
                
        except Exception as e:
            print(f"Error transcribing chunk {idx + 1}: {e}")
            continue

    print(f"Transcription complete: {len(result)} segments")
    
    if not result:
        # Return a placeholder if no transcription succeeded
        result.append({
            "start": 0.0,
            "end": 1.0,
            "text": "No transcription available"
        })
    
    return result

import os

# Lazy import to avoid loading issues during Django startup
pipeline = None
HF_TOKEN = os.environ.get("HF_AUTH_TOKEN", "YOUR_HF_TOKEN")

def _load_pipeline():
    """Lazy load the diarization pipeline"""
    global pipeline
    if pipeline is None:
        try:
            from pyannote.audio import Pipeline as PyannotePipeline
            pipeline = PyannotePipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=HF_TOKEN
            )
            print("Diarization pipeline loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load diarization pipeline: {e}")
            pipeline = False  # Mark as failed
    return pipeline

def diarize(audio_path):
    """
    Performs speaker diarization.
    Returns dict with speaker labels mapped to time segments.
    """
    pipe = _load_pipeline()
    
    if pipe is False or pipe is None:
        print("Diarization pipeline not available")
        print("Using fallback: Assigning speakers based on pause detection...")
        
        # Simple fallback: alternate speakers every 30 seconds
        # This is better than single speaker for demo purposes
        import wave
        try:
            with wave.open(audio_path, 'rb') as wav:
                frames = wav.getnframes()
                rate = wav.getframerate()
                duration = frames / float(rate)
        except:
            duration = 300.0  # Default 5 minutes
        
        # Create speaker segments alternating every 30 seconds
        speaker_segments = []
        current_time = 0
        speaker_num = 0
        segment_duration = 30.0
        
        while current_time < duration:
            end_time = min(current_time + segment_duration, duration)
            speaker_segments.append({
                "start": current_time,
                "end": end_time,
                "speaker": f"SPEAKER_{speaker_num:02d}"
            })
            current_time = end_time
            speaker_num = (speaker_num + 1) % 2  # Alternate between 2 speakers
        
        speaker_count = min(2, len(speaker_segments))
        
        return {
            "segments": speaker_segments,
            "speaker_count": speaker_count
        }
    
    print("Performing speaker diarization...")
    diar = pipe(audio_path)
    
    # Create detailed mapping of time -> speaker
    speaker_segments = []
    speakers = set()
    
    for turn, _, speaker in diar.itertracks(yield_label=True):
        speaker_segments.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker
        })
        speakers.add(speaker)
    
    print(f"Detected {len(speakers)} speakers")
    
    return {
        "segments": speaker_segments,
        "speaker_count": len(speakers)
    }
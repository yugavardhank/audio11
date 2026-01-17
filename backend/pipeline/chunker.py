import os
import subprocess

def chunk_audio(audio_path, chunk_duration=300):
    """
    Split audio into chunks of specified duration.
    Returns list of chunk file paths.
    """
    output_dir = "media/chunks"
    os.makedirs(output_dir, exist_ok=True)
    
    # Clean up old chunks
    for f in os.listdir(output_dir):
        if f.endswith(".wav"):
            os.remove(os.path.join(output_dir, f))

    pattern = os.path.join(output_dir, "chunk_%03d.wav")

    cmd = [
        "ffmpeg",
        "-y",
        "-i", audio_path,
        "-f", "segment",
        "-segment_time", str(chunk_duration),
        "-acodec", "pcm_s16le",  # Re-encode to PCM instead of copy
        "-ar", "16000",  # Ensure 16kHz
        "-ac", "1",  # Ensure mono
        pattern
    ]

    try:
        result = subprocess.run(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            check=True
        )
        print(f"Audio chunked successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error chunking audio: {e.stderr.decode()}")
        # If chunking fails, return the original file as single chunk
        return [audio_path]

    chunks = sorted([
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if f.endswith(".wav") and f.startswith("chunk_")
    ])
    
    # If no chunks created, use original file
    if not chunks:
        print("No chunks created, using original file")
        return [audio_path]
    
    print(f"Created {len(chunks)} chunks")
    return chunks
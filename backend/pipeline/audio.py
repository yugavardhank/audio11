import os
import subprocess

def normalize_audio(input_path):
    """
    Converts audio to:
    - mono
    - 16kHz sample rate
    - wav format
    Returns normalized audio path
    """
    
    # Create output path with _normalized suffix
    base_name = os.path.splitext(input_path)[0]
    output_path = f"{base_name}_normalized.wav"

    # FFmpeg command to convert to mono 16kHz
    cmd = [
        "ffmpeg",
        "-y",  # Overwrite output file
        "-i", input_path,
        "-ac", "1",  # mono (1 audio channel)
        "-ar", "16000",  # 16kHz sample rate
        "-acodec", "pcm_s16le",  # PCM 16-bit
        output_path
    ]

    try:
        subprocess.run(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            check=True
        )
        print(f"Audio normalized successfully: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error normalizing audio: {e.stderr.decode()}")
        raise

    return output_path
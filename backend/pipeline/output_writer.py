import os

def _fmt(seconds):
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"

def write_transcript(transcript, out_dir):
    path = os.path.join(out_dir, "transcript.txt")
    with open(path, "w", encoding="utf-8") as f:
        for seg in transcript:
            start = _fmt(seg["start"])
            end = _fmt(seg["end"])
            text = seg["text"]
            speaker = seg.get("speaker", "Speaker")
            f.write(f"[{start} - {end}] {speaker}: {text}\n")
    return path

def write_topics_timeline(topics, out_dir):
    path = os.path.join(out_dir, "topics_timeline.txt")
    with open(path, "w", encoding="utf-8") as f:
        for t in topics:
            start = _fmt(t["start_time"])
            end = _fmt(t["end_time"])
            title = t.get("title", "Untitled Topic")
            f.write(f"[{start} - {end}] {title}\n")
    return path

def write_topics_with_summaries(topics, out_dir):
    path = os.path.join(out_dir, "topics_with_summaries.txt")

    with open(path, "w", encoding="utf-8") as f:
        for i, t in enumerate(topics, 1):
            title = t.get("title", "Untitled Topic")
            summary = t.get("summary", "[Summary missing]")
            start = t.get("start_time", 0)
            end = t.get("end_time", 0)

            f.write(f"Topic {i}: {title}\n")
            f.write(f"Time: {_fmt(start)} → {_fmt(end)}\n\n")
            f.write(f"Summary:\n{summary}\n\n")
            f.write("=" * 60 + "\n\n")
    
    return path

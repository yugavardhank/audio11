def build_blocks(segments, window=150):
    """
    Build context blocks from transcript segments.
    Groups segments within a time window for topic analysis.
    """
    blocks = []
    current = []
    start = segments[0]["start"] if segments else 0

    for s in segments:
        if s["start"] - start <= window:
            current.append(s)
        else:
            blocks.append({
                "start": start,
                "end": current[-1]["end"] if current else start,
                "text": " ".join(x["text"] for x in current)
            })
            start = s["start"]
            current = [s]

    # Add final block
    if current:
        blocks.append({
            "start": start,
            "end": current[-1]["end"],
            "text": " ".join(x["text"] for x in current)
        })

    return blocks
def build_topic_segments(blocks, boundaries):
    """
    Build topic segments with start and end timestamps.
    """
    topics = []

    for i, start_idx in enumerate(boundaries):
        end_idx = boundaries[i + 1] if i + 1 < len(boundaries) else len(blocks)

        text = " ".join(blocks[j]["text"] for j in range(start_idx, end_idx))
        
        # Calculate end time from the last block in this segment
        end_time = blocks[end_idx - 1].get("end", blocks[end_idx - 1]["start"] + 150)

        topics.append({
            "start": blocks[start_idx]["start"],
            "end": end_time,
            "text": text
        })

    return topics
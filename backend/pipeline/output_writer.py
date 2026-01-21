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

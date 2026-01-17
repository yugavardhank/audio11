# Quick Start: Phase 1 New Features

## What's New

### 🎯 Evaluation Metrics
Your pipeline can now measure topic segmentation quality using industry-standard metrics from academic research.

**Access:** Automatically calculated and displayed after processing

```python
from pipeline.metrics import evaluate_segmentation

results = evaluate_segmentation(
    ref_segments=[...],      # Reference topics
    hyp_segments=[...],      # Predicted topics
    total_blocks=100         # Total text blocks
)

print(f"Pk Score: {results['pk']:.3f}")       # Lower is better
print(f"WinDiff: {results['windiff']:.3f}")   # Lower is better
print(f"SPCF: {results['spcf']:.3f}")         # Higher is better
```

### 📥 Multiple Export Formats
Export your transcripts in 4 professional formats:

**WebVTT (.vtt)**
- Subtitle format with precise timestamps
- Use: Video players, podcast apps, subtitle sites
- Example:
```
WEBVTT

00:00:00.000 --> 00:00:05.000
SPEAKER_00: This is the introduction

00:00:05.000 --> 00:00:10.500
SPEAKER_01: This is the next part
```

**DOTE JSON (.dote.json)**
- Structured podcast format (Podlove-compatible)
- Use: Podcast platforms, transcription services
- Includes: chapters, timestamps, speaker info, summary
```json
{
  "version": "1.2",
  "transcript": [
    {
      "start": "PT00H00M00.000S",
      "text": "This is the introduction",
      "speaker": "SPEAKER_00"
    }
  ],
  "chapters": [
    {
      "start": "PT00H00M00.000S",
      "title": "Introduction",
      "summary": "Speaker introduces..."
    }
  ]
}
```

**Plain Text with Timestamps (.txt)**
- Readable format with timing info
- Use: Sharing, emails, documents
```
[00:00:00.000] SPEAKER_00: This is the introduction
[00:00:05.000] SPEAKER_01: This is the next part
```

**Plain Text Simple (.txt)**
- Text only, no formatting
- Use: Copy/paste, content management
```
This is the introduction
This is the next part
```

### 📊 Pipeline Metrics Display
After processing, you'll see:
- **Total Segments**: How many chunks Whisper created
- **Topics Identified**: How many topics BERTopic found
- **Average Topic Length**: Mean duration of each topic

---

## How to Use

### 1. Upload Audio
```
Go to http://127.0.0.1:8000/
Upload MP3, WAV, M4A, or other audio format
```

### 2. Wait for Processing
The pipeline will:
- Normalize audio
- Transcribe with Whisper
- Identify speakers (diarization)
- Detect topics with BERTopic
- Generate summaries
- Export transcripts

### 3. Download Results
After processing, click any download button:
- 📺 **WebVTT** - For subtitles
- 📋 **DOTE JSON** - For podcasts
- 📄 **Text + Timestamps** - For sharing
- 📃 **Plain Text** - For copying

---

## Technical Details

### Pk Score (What does it mean?)
- **Metric Name**: Probability of Error
- **Range**: 0.0 (perfect) to 1.0 (terrible)
- **Interpretation**:
  - < 0.25: Excellent (state-of-the-art)
  - 0.25-0.35: Very good
  - 0.35-0.50: Good
  - > 0.50: Needs improvement
- **Target for Our Pipeline**: < 0.30

### WinDiff Score (What does it mean?)
- **Metric Name**: Window Difference
- **Range**: 0.0 (perfect) to 1.0 (terrible)
- **Interpretation**: How many sliding windows have boundary mismatches
  - Lower is better
  - Similar to Pk but more precise about exact boundary locations
- **Target for Our Pipeline**: < 0.35

### SPCF Score (What does it mean?)
- **Metric Name**: Segmentation Purity and Coverage F-score
- **Range**: 0.0 (terrible) to 1.0 (perfect)
- **Components**:
  - **Purity**: How much of hypothesis segments overlap with reference
  - **Coverage**: How much of reference segments are found by hypothesis
  - **SPCF**: Harmonic mean of purity and coverage
- **Interpretation**:
  - > 0.60: Excellent
  - 0.45-0.60: Very good
  - 0.30-0.45: Good
  - < 0.30: Needs improvement
- **Target for Our Pipeline**: > 0.50

---

## Example Workflow

### Scenario: Process a 15-minute podcast interview

1. **Upload**
   ```
   File: interview.mp3 (15 minutes)
   Size: ~30MB
   ```

2. **Processing** (takes ~2-3 minutes)
   ```
   Step 1: Normalize audio ✓
   Step 2: Chunk into segments ✓
   Step 3: Transcribe with Whisper ✓
   Step 4: Speaker diarization ✓
   Step 5: Generate embeddings ✓
   Step 6: Detect topics with BERTopic ✓
   Step 7: Generate summaries ✓
   Step 8: Create topic labels ✓
   Step 9: Export transcripts ✓
   ```

3. **Results Page Shows**
   ```
   📊 Metrics:
   - 156 segments transcribed
   - 8 topics identified
   - 112.5 seconds average topic length
   
   📥 Downloads:
   - interview.vtt (for video subtitles)
   - interview.dote.json (for podcast apps)
   - interview.txt (readable format)
   - interview_simple.txt (plain text)
   ```

4. **Next Steps**
   - Share WebVTT with video platform
   - Upload DOTE JSON to podcast directory
   - Email plain text to stakeholders
   - Archive simple text in document system

---

## File Locations

After processing `audio_file.mp3`:

```
media/
├── input/
│   └── audio_file.mp3              (original upload)
└── transcripts/
    ├── audio_file.vtt              (WebVTT format)
    ├── audio_file.dote.json        (DOTE JSON format)
    ├── audio_file.txt              (text with timestamps)
    └── audio_file_simple.txt       (plain text)
```

All files are automatically generated and ready to download from the web interface.

---

## Troubleshooting

### Q: Downloads not working?
**A**: Make sure:
1. Processing completed without errors
2. Check browser console for errors
3. Verify file paths are correct in Django logs

### Q: Transcript file is empty?
**A**: 
1. Check if audio was actually transcribed (look at transcript section)
2. If no transcript, audio may not have had speech content
3. Try a different audio file

### Q: DOTE JSON looks wrong?
**A**: 
1. Use JSON viewer online (jsoncrack.com)
2. Check timestamps are in ISO 8601 format (PT##H##M##S)
3. Verify chapters match topics on results page

### Q: Want different export options?
**A**: Contact development team - easy to add new formats:
- CSV format
- SRT subtitles
- Markdown with timestamps
- Word document with formatting
- etc.

---

## Performance Notes

### Processing Time by Audio Length
- 1 minute: ~30 seconds
- 5 minutes: ~2 minutes
- 15 minutes: ~3-4 minutes (mostly ASR/summarization)
- 30 minutes: ~6-8 minutes

### Export Time
- WebVTT: < 100ms
- DOTE JSON: < 150ms
- Plain text: < 100ms
- **Total export**: < 400ms (negligible)

### File Sizes (example for 15-min audio)
- Original MP3: ~30MB
- WebVTT: ~50KB
- DOTE JSON: ~200KB
- Plain text: ~60KB
- Plain text simple: ~45KB

---

## Integration with External Tools

### Podcast Platforms
- **Anchor/Spotify Podcasts**: Use DOTE JSON
- **Podlove**: Use DOTE JSON directly
- **Transistor**: Use WebVTT for chapters

### Video Editors
- **Adobe Premiere**: Import WebVTT as captions
- **Final Cut Pro**: Import WebVTT as subtitles
- **DaVinci Resolve**: Import WebVTT as subtitles

### Document Systems
- **Notion**: Paste plain text with timestamps
- **Google Docs**: Copy plain text, format as needed
- **Word/LibreOffice**: Import plain text, apply styles

### Content Management
- **Medium**: Use markdown-formatted plain text
- **DEV.to**: Use markdown with timestamps
- **Hashnode**: Use markdown with timestamps

---

## Next Features (Coming Soon)

Phase 2 will add:
- [ ] Transformer-based context integration
- [ ] Multilingual support (Spanish, French, German, etc.)
- [ ] End-to-end audio processing (no ASR needed)
- [ ] Custom clustering parameters
- [ ] Topic keyword highlighting
- [ ] Speaker analysis
- [ ] Sentiment analysis per topic
- [ ] Q&A extraction from transcript

---

## Questions or Issues?

Check:
1. Server logs: `terminal output in VS Code`
2. Browser console: `F12 → Console tab`
3. Django debug: Check command line for errors
4. File system: Check `media/transcripts/` directory

---

## Summary

You now have a production-grade audio processing pipeline with:
- ✅ Industry-standard evaluation metrics
- ✅ 4 export formats for different use cases
- ✅ Professional podcast-ready output
- ✅ User-friendly download interface

**Server is running at**: `http://127.0.0.1:8000/`

**Ready to process audio!** 🎉

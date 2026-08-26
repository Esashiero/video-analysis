#!/usr/bin/env python
"""v3 — Structured report analyzer (arashsajjadi/ai-powered-video-analyzer style).

Produces a machine-readable JSON report + a human-readable Markdown report:
visual captions (Mistral vision), optional spoken transcript (Voxtral), and an
evidence-grounded summary separating OBSERVED FACTS from PLAUSIBLE INTERPRETATION.

Audio is optional (--no-audio) per user request.

Usage:
    export MISTRAL_API_KEY=...
    python v3_structured_analyzer.py video.mp4 --no-audio
    python v3_structured_analyzer.py video.mp4 --no-audio --template numbers
"""
import argparse
import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(__file__))

from mistral_models import (
    get_client,
    get_rate_limiter,
    DEFAULT_VISION,
    DEFAULT_AUDIO,
    DEFAULT_SUMMARIZER,
)
from core import sample_frames, extract_audio, transcribe, describe_frames, aggregate
from cost import fmt_eur
from prompts import get_template


def main():
    ap = argparse.ArgumentParser(description="Structured Mistral video report (JSON+MD)")
    ap.add_argument("video")
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--max-frames", type=int, default=40)
    ap.add_argument("--vision-model", default=DEFAULT_VISION)
    ap.add_argument("--audio-model", default=DEFAULT_AUDIO)
    ap.add_argument("--summarizer", default=DEFAULT_SUMMARIZER)
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--template", default="default")
    ap.add_argument("--out", default="report")
    ap.add_argument("--api-key", default=None)
    args = ap.parse_args()

    client = get_client(args.api_key)
    vrl = get_rate_limiter(args.vision_model)

    t0 = time.time()
    prompt = get_template(args.template)
    vrl.wait()
    frames, duration = sample_frames(args.video, args.fps, args.max_frames)
    caption, cost = describe_frames(client, frames, prompt, args.vision_model)

    transcript = ""
    if not args.no_audio:
        arl = get_rate_limiter(args.audio_model)
        tmp = tempfile.mkdtemp(prefix="va3_")
        audio = os.path.join(tmp, "audio.mp3")
        extract_audio(args.video, audio)
        arl.wait()
        transcript, _ = transcribe(client, audio, args.audio_model)

    parts = []
    if transcript:
        parts.append("SPOKEN AUDIO TRANSCRIPT:\n" + transcript)
    parts.append("VISUAL CAPTIONS:\n" + caption)

    summary = aggregate(
        client,
        "\n\n".join(parts),
        "Produce a factual, evidence-grounded analysis. Separate OBSERVED FACTS "
        "from PLAUSIBLE INTERPRETATION. If the evidence is weak, say "
        "'Insufficient evidence' rather than guess.",
        args.summarizer,
    )

    act = cost.get("actual") or {}
    report = {
        "vision_model": args.vision_model,
        "audio_used": not args.no_audio,
        "summarizer": args.summarizer,
        "duration_sec": round(duration, 2),
        "fps": args.fps,
        "frames_sampled": len(frames),
        "template": args.template,
        "transcript": transcript,
        "visual_captions": caption,
        "summary": summary,
        "vision_cost_eur": act.get("total_eur"),
        "elapsed_sec": round(time.time() - t0, 2),
    }

    with open(args.out + ".json", "w") as fh:
        json.dump(report, fh, indent=2)
    md = (
        f"# Video Analysis Report\n\n"
        f"**Duration:** {duration:.1f}s · **Frames sampled:** {len(frames)} · "
        f"**Vision cost:** {fmt_eur(act.get('total_eur', 0))}\n\n"
        f"## Summary\n\n{summary}\n\n"
        f"## Visual captions\n\n{caption}\n\n"
        f"## Transcript\n\n{transcript or '(audio disabled)'}\n"
    )
    with open(args.out + ".md", "w") as fh:
        fh.write(md)
    print(f"wrote {args.out}.json and {args.out}.md")
    print(f"[vision cost] {fmt_eur(act.get('total_eur', 0))}")


if __name__ == "__main__":
    main()

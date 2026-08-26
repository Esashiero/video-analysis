#!/usr/bin/env python
"""v2 — Chunked analyzer (kdoronin/video_analyzer style, ported to Mistral).

Splits the video into fixed-length chunks, transcribes each chunk's audio with
Voxtral (optional), describes sampled frames per chunk with a Mistral vision
model, then asks a Mistral text model to write a structured markdown analysis.

Audio is optional (--no-audio) per user request; vision is always on.

Usage:
    export MISTRAL_API_KEY=...
    python v2_chunked_analyzer.py video.mp4 --no-audio
    python v2_chunked_analyzer.py video.mp4 --no-audio --chunk-seconds 60 --fps 1
"""
import argparse
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(__file__))

from mistral_models import (
    get_client,
    get_rate_limiter,
    DEFAULT_VISION,
    DEFAULT_AUDIO,
    DEFAULT_SUMMARIZER,
)
from core import sample_frames, chunk_video, extract_audio, transcribe, describe_frames, aggregate
from cost import fmt_eur
from prompts import get_template


def main():
    ap = argparse.ArgumentParser(description="Chunked Mistral video analyzer")
    ap.add_argument("video")
    ap.add_argument("--chunk-seconds", type=int, default=60, dest="chunk_seconds")
    ap.add_argument("--vision-model", default=DEFAULT_VISION)
    ap.add_argument("--audio-model", default=DEFAULT_AUDIO)
    ap.add_argument("--summarizer", default=DEFAULT_SUMMARIZER)
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--max-frames-per-chunk", type=int, default=240,
                    help="sampling cap per chunk; describe_frames auto-batches "
                         "past the 8-image API limit, so this can stay high")
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--template", default="timeline",
                    help="prompt template for per-chunk vision calls")
    ap.add_argument("--language", default=None, help="ISO code for Voxtral, e.g. en")
    ap.add_argument("--out", default="analysis.md")
    ap.add_argument("--api-key", default=None)
    args = ap.parse_args()

    client = get_client(args.api_key)
    vrl = get_rate_limiter(args.vision_model)

    prompt = get_template(args.template)
    tmp = tempfile.mkdtemp(prefix="va2_")
    chunks = chunk_video(args.video, max(args.chunk_seconds / 60.0, 1 / 60), tmp)

    total_cost = 0.0
    sections = []
    for i, ch in enumerate(chunks, 1):
        block = f"## Chunk {i}\n"

        if not args.no_audio:
            arl = get_rate_limiter(args.audio_model)
            audio = os.path.join(tmp, f"chunk_{i:03d}.mp3")
            extract_audio(ch, audio)
            arl.wait()
            text, _ = transcribe(client, audio, args.audio_model, language=args.language)
            block += f"\n### Audio transcript\n\n{text or '(no speech detected)'}\n"

        vrl.wait()
        fr, _ = sample_frames(ch, args.fps, args.max_frames_per_chunk)
        cap, cost = describe_frames(client, fr, prompt, args.vision_model)
        act = (cost.get("actual") or {}).get("total_eur") or 0.0
        total_cost += act
        block += f"\n### Visual\n\n{cap}\n"
        sections.append(block)

    combined = "\n\n".join(sections)
    summary = aggregate(
        client,
        combined,
        "Write a structured markdown analysis of this video from the per-chunk "
        "notes. Include: an overall summary, key topics, and a timeline of events.",
        args.summarizer,
    )

    md = (
        f"# Video Analysis\n\n"
        f"**Total vision cost:** {fmt_eur(total_cost)} · "
        f"**Chunks:** {len(chunks)}\n\n{summary}\n\n---\n\n## Per-chunk detail\n\n{combined}\n"
    )
    with open(args.out, "w") as fh:
        fh.write(md)
    print(f"wrote {args.out} ({len(chunks)} chunks)")
    print(f"[vision cost] {fmt_eur(total_cost)}")


if __name__ == "__main__":
    main()

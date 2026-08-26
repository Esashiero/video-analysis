#!/usr/bin/env python
"""v1 — Minimal frame describer (closest to the original ~/Pictures/vision.py).

Ports the image-only Mistral pattern to video: sample frames with OpenCV,
base64-encode, batch them to a Mistral vision model, print the description.
Vision-only; audio ignored per user request.

Prompt quality dominates results here (see prompts.py header for the proof),
so pick a template with --template or override fully with --prompt.

Usage:
    export MISTRAL_API_KEY=...
    python v1_frame_describer.py video.mp4
    python v1_frame_describer.py video.mp4 --template numbers --fps 2
    python v1_frame_describer.py video.mp4 --prompt "your own prompt"
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from mistral_models import get_client, get_rate_limiter, DEFAULT_VISION
from core import sample_frames, describe_frames
from cost import fmt_eur
from prompts import get_template


def main():
    ap = argparse.ArgumentParser(description="Minimal Mistral video frame describer")
    ap.add_argument("video")
    ap.add_argument("--fps", type=float, default=2.0,
                    help="frames per second to sample (2.0 default: catches on-screen scores)")
    ap.add_argument("--max-frames", type=int, default=40)
    ap.add_argument("--model", default=DEFAULT_VISION)
    ap.add_argument("--template", default="default", choices=sorted(__import__("prompts").TEMPLATES),
                    help="named prompt template")
    ap.add_argument("--prompt", default=None, help="full prompt override (beats --template)")
    ap.add_argument("--out", default=None, help="optional JSON output path")
    ap.add_argument("--api-key", default=None)
    args = ap.parse_args()

    client = get_client(args.api_key)
    rl = get_rate_limiter(args.model)

    prompt = args.prompt or get_template(args.template)
    frames, duration = sample_frames(args.video, args.fps, args.max_frames)
    print(f"Sampled {len(frames)} frames from {duration:.1f}s video "
          f"(fps={args.fps}, template={args.template})", file=sys.stderr)

    desc, cost = describe_frames(client, frames, prompt, args.model)
    print(desc)

    est, act = cost["estimate"], cost["actual"]
    if est:
        print(f"[cost estimate] {len(frames)} imgs x {est['tokens_per_image']} tok/img = "
              f"{est['input_tokens']} in-tok; ~{fmt_eur(est['total_eur'])}", file=sys.stderr)
    if act:
        print(f"[cost actual]   {act['input_tokens']} in ({act['cached_input_tokens']} cached) "
              f"+ {act['output_tokens']} out = {fmt_eur(act['total_eur'])}", file=sys.stderr)

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(
                {
                    "model": args.model,
                    "duration_sec": round(duration, 2),
                    "fps": args.fps,
                    "num_frames": len(frames),
                    "template": None if args.prompt else args.template,
                    "description": desc,
                    "cost": cost,
                },
                fh,
                indent=2,
            )
        print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()

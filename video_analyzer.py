#!/usr/bin/env python
"""video_analyzer.py — single consolidated Mistral vision video analyzer.

Distilled from three prototype versions (v1/v2/v3) after live testing on an
18.3s Snapchat clip with mistral-small-2603. What survived and why:

- fps=2.0        : at 1fps the model MISSED on-screen scores ("902"); 2fps caught
                   them. Sampling rate matters more than model choice.
- auto-batching  : API caps 8 images/request; frames split into <=8 batches,
                   partial descriptions merged in a final text pass.
- two-stage      : stage 1 = strict extraction per batch (exact text, people,
                   actions); stage 2 = merge + facts-vs-interpretation report
                   with evidence citation to suppress hallucinated specifics.
- chunking       : only engages when the frame count exceeds what fits in a
                   reasonable number of batches; keeps long-video cost linear.
- cost tracking  : pre-call estimate from Vision.md formula + actual usage
                   from the API, printed every run.

Audio is intentionally not part of this pipeline.

Usage:
    export MISTRAL_API_KEY=...
    python video_analyzer.py video.mp4                    # full report
    python video_analyzer.py video.mp4 --numbers-only     # just extract text/scores
    python video_analyzer.py video.mp4 --fps 1 --max-frames 20   # cheap mode
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

from mistral_models import get_client, get_rate_limiter, DEFAULT_VISION
from core import sample_frames, describe_frames
from cost import fmt_eur

EXTRACT_PROMPT = (
    "These are consecutive sampled frames from ONE video, in chronological order.\n"
    "Extract ONLY what is visibly present. For each distinct moment:\n"
    "1. SETTING: where (indoor/outdoor, venue type), lighting, time of day.\n"
    "2. PEOPLE: count, adult/child, clothing colors, action being performed.\n"
    "3. TEXT & NUMBERS: transcribe ALL visible text EXACTLY - scores, timers, "
    "prices, labels, brand logos, signage. Copy digits verbatim, never "
    "paraphrase or guess missing digits.\n"
    "4. OBJECTS: notable objects/devices/machines and their state.\n"
    "Do not speculate. If unsure whether something is present, omit it."
)

MERGE_PROMPT = (
    "You are given extraction notes from consecutive batches of frames of ONE "
    "video, in order. Produce a final report:\n\n"
    "## Summary\n(3-5 sentences, chronological)\n\n"
    "## Timeline\n(markdown table: time-range | scene | people | key action)\n\n"
    "## On-screen text & numbers\n(every exact string found, with where it appeared)\n\n"
    "## Observed facts\n(bullet list)\n\n"
    "## Plausible interpretation\n(short - clearly labeled as inference)\n\n"
    "Rules: only include facts supported by the notes. If the notes disagree or "
    "a detail appears in only one ambiguous note, mark it '(uncertain)'. "
    "Never invent numbers."
)


def main():
    ap = argparse.ArgumentParser(description="Consolidated Mistral video analyzer")
    ap.add_argument("video")
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--max-frames", type=int, default=60)
    ap.add_argument("--model", default=DEFAULT_VISION)
    ap.add_argument("--numbers-only", action="store_true",
                    help="skip the report; just dump every piece of on-screen text")
    ap.add_argument("--out", default=None, help="JSON output path")
    ap.add_argument("--api-key", default=None)
    args = ap.parse_args()

    client = get_client(args.api_key)
    rl = get_rate_limiter(args.model)
    t0 = time.time()

    frames, duration = sample_frames(args.video, args.fps, args.max_frames)
    print(f"video: {duration:.1f}s -> {len(frames)} frames @ {args.fps}fps", file=sys.stderr)

    rl.wait()
    if args.numbers_only:
        from prompts import get_template
        desc, cost = describe_frames(client, frames, get_template("numbers"), args.model)
        print(desc)
    else:
        # Stage 1 happens inside describe_frames (batched extraction+merge);
        # we then run one grounding pass over the merged description.
        merged, cost = describe_frames(client, frames, EXTRACT_PROMPT, args.model,
                                       est_output_tokens=800)
        act = (cost.get("actual") or {}).get("total_eur") or 0.0
        print(f"[stage 1 cost] {fmt_eur(act)}", file=sys.stderr)

        srl = get_rate_limiter(args.model)
        srl.wait()
        from core import _chat_with_retry
        resp = _chat_with_retry(client, args.model, [{"role": "user", "content": [
                {"type": "text", "text": MERGE_PROMPT},
                {"type": "text", "text": merged},
            ]}])
        report = resp.choices[0].message.content
        mu = getattr(resp, "usage", None)
        extra = 0.0
        if mu is not None:
            from cost import cost_from_usage
            extra = cost_from_usage(mu, args.model)["total_eur"]
        print(report)
        print(f"\n[total cost] {fmt_eur(act + extra)}", file=sys.stderr)
        cost["actual"]["total_eur"] = act + extra

    if args.out:
        payload = {
            "video": args.video,
            "model": args.model,
            "fps": args.fps,
            "num_frames": len(frames),
            "mode": "numbers" if args.numbers_only else "report",
            "cost": cost,
        }
        if args.numbers_only:
            payload["extracted_text"] = desc
        else:
            payload["report"] = report
        with open(args.out, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""video_analyzer.py — consolidated Mistral vision video analyzer.

fps=2.0 + explicit OCR instructions beat model choice on real footage
(verified live: caught "902"/"978" scores that 1fps missed). Frames are sent
in batches of <=8 with inline timestamps; batch notes are merged into a final
facts-vs-interpretation report.

Every run writes a full audit trail under .runs/:
  run.json, frames/*.jpg + manifest.json,
  turns/batch_NN.json (instruction + frame refs + response),
  turns/merge.json (instructions + received prompt + merged description).

Usage:
    export MISTRAL_API_KEY=...
    python video_analyzer.py video.mp4                    # full report
    python video_analyzer.py video.mp4 --numbers-only     # just extract text/scores
    python video_analyzer.py video.mp4 --fps 1 --max-frames 20   # cheap mode
    python video_analyzer.py video.mp4 --no-trace         # skip .runs/ audit trail
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

from mistral_models import get_client, get_rate_limiter, DEFAULT_VISION
from core import sample_frames, describe_frames, _chat_with_retry
from cost import fmt_eur, cost_from_usage
from prompts import get_template
from trace import RunTrace

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
    ap.add_argument("--no-trace", action="store_true",
                    help="do not write the .runs/ audit trail")
    args = ap.parse_args()

    client = get_client(args.api_key)
    rl = get_rate_limiter(args.model)
    trace = None if args.no_trace else RunTrace(args.video)
    t0 = time.time()

    if trace is not None:
        trace.set_meta(video=os.path.abspath(args.video), model=args.model,
                       fps=args.fps, max_frames=args.max_frames,
                       mode="numbers" if args.numbers_only else "report",
                       started_at=time.strftime("%Y-%m-%dT%H:%M:%S"))

    frames, duration = sample_frames(args.video, args.fps, args.max_frames,
                                     trace=trace)
    print(f"video: {duration:.1f}s -> {len(frames)} frames @ {args.fps}fps", file=sys.stderr)
    if trace is not None:
        trace.write_manifest()

    desc = report = None
    if args.numbers_only:
        merged, cost = describe_frames(client, frames, get_template("numbers"),
                                       args.model, rate_limiter=rl, trace=trace)
        desc = merged
        print(desc)
    else:
        merged, cost = describe_frames(client, frames, EXTRACT_PROMPT, args.model,
                                       rate_limiter=rl, est_output_tokens=800,
                                       trace=trace)
        act = (cost.get("actual") or {}).get("total_eur") or 0.0
        print(f"[stage 1 cost] {fmt_eur(act)}", file=sys.stderr)

        resp = _chat_with_retry(client, args.model, [{"role": "user", "content": [
                {"type": "text", "text": MERGE_PROMPT},
                {"type": "text", "text": merged},
            ]}], rate_limiter=get_rate_limiter(args.model))
        report = resp.choices[0].message.content
        mu = getattr(resp, "usage", None)
        extra = 0.0
        if mu is not None:
            extra = cost_from_usage(mu, args.model)["total_eur"]
        print(report)
        print(f"\n[total cost] {fmt_eur(act + extra)}", file=sys.stderr)
        cost["actual"]["total_eur"] = act + extra

        # record the report pass as an extra merge-style turn for full replay
        if trace is not None:
            trace.record_merge_turn(
                MERGE_PROMPT, merged, report, mu)

    if trace is not None:
        wall = round(time.time() - t0, 1)
        trace.finalize(extra={"wall_seconds": wall, "cost": cost})
        print(f"trace: {trace.dir}", file=sys.stderr)

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

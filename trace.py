"""Run trace store: persist every frame, request, and response of an analysis run.

One RunTrace = one folder under .runs/<timestamp>_<videoname>/ containing:
  run.json      — run metadata (args, model, fps, cost)
  frames/       — the exact resized JPEGs sent to the API
  frames/manifest.json — per-frame index: ts, width, height, file
  turns/batch_NN.json  — one entry per API turn: instruction + frame refs
                         + raw response text + token usage
  turns/merge.json     — the final merge turn (instructions + full received
                         prompt + merged output + usage)

Images live as files; JSON holds references only.
"""
import json
import os
import re
from datetime import datetime


class RunTrace:
    def __init__(self, video_path, root=".runs"):
        base = os.path.splitext(os.path.basename(video_path))[0]
        safe = re.sub(r"[^A-Za-z0-9_-]", "_", base)[:60]
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.dir = os.path.join(root, f"{stamp}_{safe}")
        self.frames_dir = os.path.join(self.dir, "frames")
        self.turns_dir = os.path.join(self.dir, "turns")
        os.makedirs(self.frames_dir, exist_ok=True)
        os.makedirs(self.turns_dir, exist_ok=True)
        self.manifest = []
        self.turn_count = 0
        self.meta = {}

    # ------------------------------------------------------------------
    # Frames
    # ------------------------------------------------------------------
    def save_frame(self, idx, ts, jpeg_bytes, width, height):
        """Write the exact JPEG sent to the API; record it in the manifest."""
        name = f"f{idx:04d}.jpg"
        with open(os.path.join(self.frames_dir, name), "wb") as fh:
            fh.write(jpeg_bytes)
        self.manifest.append({
            "idx": idx, "ts": round(ts, 3), "width": width,
            "height": height, "file": name,
        })

    def write_manifest(self):
        path = os.path.join(self.frames_dir, "manifest.json")
        with open(path, "w") as fh:
            json.dump(self.manifest, fh, indent=2)

    # ------------------------------------------------------------------
    # Turns
    # ------------------------------------------------------------------
    def record_turn(self, kind, prompt, image_refs, response_text, usage):
        """Record one API turn. kind: 'batch' or 'merge'.

        usage: dict or object with prompt_tokens/completion_tokens/cached.
        """
        rec = {
            "kind": kind,
            "instruction": prompt,
            "image_refs": image_refs,
            "response": response_text,
            "usage": _usage_dict(usage),
        }
        if kind == "merge":
            path = os.path.join(self.turns_dir, "merge.json")
        else:
            path = os.path.join(self.turns_dir, f"batch_{self.turn_count:02d}.json")
            self.turn_count += 1
        with open(path, "w") as fh:
            json.dump(rec, fh, indent=2)

    def record_merge_turn(self, instructions, received_prompt, response_text, usage):
        """The final merge turn: full prompt it received + its output."""
        rec = {
            "kind": "merge",
            "instructions": instructions,
            "received_prompt": received_prompt,
            "response": response_text,
            "usage": _usage_dict(usage),
        }
        with open(os.path.join(self.turns_dir, "merge.json"), "w") as fh:
            json.dump(rec, fh, indent=2)

    # ------------------------------------------------------------------
    # Run metadata
    # ------------------------------------------------------------------
    def set_meta(self, **kwargs):
        self.meta.update(kwargs)

    def finalize(self, extra=None):
        if extra:
            self.meta.update(extra)
        self.meta["finished_at"] = datetime.now().isoformat(timespec="seconds")
        with open(os.path.join(self.dir, "run.json"), "w") as fh:
            json.dump(self.meta, fh, indent=2)


def _usage_dict(usage):
    if usage is None:
        return None
    if isinstance(usage, dict):
        return usage
    return {
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "cached_tokens": int(getattr(getattr(usage, "prompt_tokens_details", None),
                                     "cached_tokens", 0) or 0),
    }

"""Shared video-processing helpers for the Mistral video analyzer.

Backend logic lives here: frame sampling (ffmpeg pipe), Mistral calls with
rate limiting + retry, cost accounting. Callers decide models and prompts.

Sampling uses ffmpeg's fps filter (fast seek + only decoded frames we keep)
piped as raw JPEGs; frames are pre-resized client-side to the model's max
resolution so we never ship oversized payloads.
"""
import base64
import io
import json
import os
import subprocess
import sys
import time
from types import SimpleNamespace

from mistralai.client.errors import SDKError
from mistralai.client.models.file import File


# --------------------------------------------------------------------------
# Frame sampling — ffmpeg pipe, resized client-side
# --------------------------------------------------------------------------
def sample_frames(video_path, fps=2.0, max_frames=None, max_dim=1540,
                  jpeg_quality=85, trace=None):
    """Sample frames at ~fps via ffmpeg, resized to fit max_dim x max_dim.

    Returns (frames, duration_sec) where frames is a list of dicts:
      {"ts": float, "b64": str, "width": int, "height": int}
    If `trace` (RunTrace) is given, each JPEG is also persisted with its ts.
    """
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=r_frame_rate,nb_frames,duration",
         "-of", "json", video_path],
        capture_output=True, text=True, check=True,
    )
    info = json.loads(probe.stdout)["streams"][0]
    num, den = info.get("r_frame_rate", "25/1").split("/")
    src_fps = float(num) / float(den or 1) if float(den or 1) else 0.0
    duration = float(info["duration"]) if info.get("duration") else 0.0

    # scale to fit inside max_dim box, preserving aspect ratio
    vf = f"fps={fps},scale='min({max_dim},iw)':-2:flags=lanczos"
    cmd = [
        "ffmpeg", "-nostdin", "-loglevel", "error",
        "-i", video_path,
        "-vf", vf,
        "-q:v", str(max(2, min(10, int((100 - jpeg_quality) / 10) + 1))),
        "-f", "image2pipe", "-vcodec", "mjpeg", "-",
    ]
    # stderr -> DEVNULL: leaving it as PIPE undrained deadlocks once the
    # 64KB pipe buffer fills and ffmpeg blocks on log writes.
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    frames = []
    idx = 0
    try:
        while True:
            data = _read_jpeg(proc.stdout)
            if data is None:
                break
            w, h = _jpeg_size(data)
            b64 = base64.b64encode(data).decode("utf-8")
            ts = idx / fps
            frames.append({"ts": ts, "b64": b64, "width": w, "height": h})
            if trace is not None:
                trace.save_frame(idx, ts, data, w, h)
            idx += 1
            if max_frames and len(frames) >= max_frames:
                # kill BEFORE wait(): ffmpeg is still blocked writing into the
                # full pipe; waiting without draining would deadlock.
                proc.kill()
                proc.wait()
                return frames, duration
        proc.wait()
    finally:
        if proc.poll() is None:
            proc.kill()
    return frames, duration


def _read_jpeg(stream):
    """Read one JPEG from a multipart mjpeg stream.

    Delimit on SOI (FFD8FF): each new SOI closes the previous image. This
    avoids hunting for EOI, which can appear inside entropy-coded data.
    """
    soi = b"\xff\xd8\xff"
    buf = bytearray()
    started = False
    while True:
        chunk = stream.read(65536)
        if not chunk:
            # stream end: flush whatever remains as the final image
            return bytes(buf) if buf else None
        buf.extend(chunk)
        if not started:
            pos = buf.find(soi)
            if pos == -1:
                buf.clear()
                continue
            del buf[:pos]
            started = True
        # look for the NEXT soi after the current image's start
        nxt = buf.find(soi, 3)
        if nxt != -1:
            data = bytes(buf[:nxt])
            del buf[:nxt]
            return data


def _jpeg_size(data):
    """Parse width/height from a JPEG buffer without full decode."""
    i = 2
    while i < len(data) - 8:
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker in (0xC0, 0xC1, 0xC2, 0xC3):
            h = (data[i + 5] << 8) | data[i + 6]
            w = (data[i + 7] << 8) | data[i + 8]
            return w, h
        seg_len = (data[i + 2] << 8) | data[i + 3]
        i += 2 + seg_len
    return 1540, 1540  # conservative fallback


def resize_to_fit_b64(b64_str, max_dim=1540):
    """Resize an existing base64 JPEG to fit max_dim (used by callers holding
    already-encoded frames). Returns (new_b64, w, h). Rarely needed now that
    sample_frames resizes at encode time."""
    import cv2
    import numpy as np
    buf = np.frombuffer(base64.b64decode(b64_str), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    if max(h, w) <= max_dim:
        return b64_str, w, h
    scale = max_dim / float(max(h, w))
    new_w, new_h = int(w * scale), int(h * scale)
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    ok, out = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(out.tobytes()).decode("utf-8"), new_w, new_h


# --------------------------------------------------------------------------
# FFmpeg wrappers (system ffmpeg assumed present)
# --------------------------------------------------------------------------
def chunk_video(video_path, chunk_minutes, outdir):
    """Split a video into fixed-length .mp4 chunks with ffmpeg."""
    os.makedirs(outdir, exist_ok=True)
    base = os.path.splitext(os.path.basename(video_path))[0]
    pattern = os.path.join(outdir, f"{base}_%03d.mp4")
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-f", "segment", "-segment_time", str(chunk_minutes * 60),
        "-reset_timestamps", "1", pattern,
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return sorted(
        os.path.join(outdir, f)
        for f in os.listdir(outdir)
        if f.startswith(base) and f.endswith(".mp4")
    )


def extract_audio(video_path, out_path, sample_rate=16000):
    """Extract mono 16 kHz mp3 audio for transcription."""
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-ac", "1", "-ar", str(sample_rate),
        "-f", "mp3", out_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return out_path


# --------------------------------------------------------------------------
# Mistral API calls
# --------------------------------------------------------------------------
def transcribe(client, audio_path, model, language=None, timestamps=False):
    """Transcribe audio with a Voxtral model. Returns (text, segments)."""
    with open(audio_path, "rb") as fh:
        content = fh.read()
    kwargs = dict(
        model=model,
        file=File(
            file_name=os.path.basename(audio_path),
            content=content,
            content_type="audio/mpeg",
        ),
    )
    if language:
        kwargs["language"] = language
    if timestamps:
        kwargs["timestamp_granularities"] = ["segment"]

    resp = client.audio.transcriptions.complete(**kwargs)
    text = getattr(resp, "text", None)
    if text is None and hasattr(resp, "model_dump"):
        text = resp.model_dump().get("text")
    segments = getattr(resp, "segments", None) if timestamps else None
    return text or "", segments


def _chat_with_retry(client, model, messages, rate_limiter=None, max_retries=5):
    """client.chat.complete with RPS gating + exponential backoff on 429/5xx.

    TPM limits (e.g. 50k/min on mistral-small-2603) can trip even when RPS is
    respected because one batched vision request carries thousands of image
    tokens — backing off and retrying is the correct handling there.
    """
    if rate_limiter is not None:
        rate_limiter.wait()
    delay = 4.0
    for attempt in range(max_retries + 1):
        try:
            return client.chat.complete(model=model, messages=messages)
        except SDKError as e:
            status = getattr(getattr(e, "http_response", None), "status_code", None)
            if status not in (429, 500, 502, 503) or attempt == max_retries:
                raise
            wait = delay * (2 ** attempt)
            print(f"    [rate-limited] retrying in {wait:.0f}s...", file=sys.stderr)
            time.sleep(wait)


MERGE_INSTRUCTIONS = (
    "The following are separate partial descriptions of frames from one "
    "continuous video, in order. Merge them into ONE coherent, chronological "
    "description of the whole video. Remove redundancy."
)


def describe_frames(client, frames, prompt, model, rate_limiter=None,
                    est_output_tokens=400, max_images=8, trace=None):
    """Send sampled frames to a vision model and return (description, cost).

    Frames are sent in batches of <=max_images with their timestamps labeled
    inline, so the model can anchor events in time. Batch descriptions are
    merged in a final pass. Returns both a pre-call estimate and live usage.

    VISION-ONLY: no audio.
    """
    from cost import estimate_request_cost, cost_from_usage

    if not frames:
        return "", {"estimate": None, "actual": None, "batches": 0}

    w = max(f["width"] for f in frames)
    h = max(f["height"] for f in frames)
    est = estimate_request_cost(len(frames), w, h, est_output_tokens, model)

    acc = {"prompt": 0, "completion": 0, "cached": 0}
    batch_texts = []
    n_batches = 0

    for i in range(0, len(frames), max_images):
        batch = frames[i:i + max_images]
        content = [{"type": "text", "text": prompt}]
        refs = []
        for item in batch:
            label = f"[Time: {item['ts']:.2f}s]"
            content.append({"type": "text", "text": label})
            content.append({"type": "image_url",
                            "image_url": f"data:image/jpeg;base64,{item['b64']}"})
            refs.append(label)
        resp = _chat_with_retry(client, model,
                                [{"role": "user", "content": content}],
                                rate_limiter=rate_limiter)
        n_batches += 1
        batch_texts.append(resp.choices[0].message.content)
        _accumulate(acc, getattr(resp, "usage", None))
        if trace is not None:
            trace.record_turn("batch", prompt, refs,
                              batch_texts[-1], getattr(resp, "usage", None))

    received_prompt = "\n\n".join(f"[Batch {j+1}]\n{t}" for j, t in enumerate(batch_texts))
    if n_batches > 1:
        merge_resp = _chat_with_retry(client, model,
                                      [{"role": "user", "content": [
                                          {"type": "text", "text": MERGE_INSTRUCTIONS},
                                          {"type": "text", "text": received_prompt},
                                      ]}],
                                      rate_limiter=rate_limiter)
        desc = merge_resp.choices[0].message.content
        merge_usage = getattr(merge_resp, "usage", None)
        _accumulate(acc, merge_usage)
        if trace is not None:
            trace.record_merge_turn(MERGE_INSTRUCTIONS, received_prompt,
                                    desc, merge_usage)
    else:
        desc = batch_texts[0]

    usage_obj = SimpleNamespace(
        prompt_tokens=acc["prompt"],
        completion_tokens=acc["completion"],
        prompt_tokens_details=SimpleNamespace(cached_tokens=acc["cached"]),
    )
    live = cost_from_usage(usage_obj, model)
    return desc, {"estimate": est, "actual": live, "batches": n_batches}


def _accumulate(acc, usage):
    if usage is None:
        return
    acc["prompt"] += int(getattr(usage, "prompt_tokens", 0) or 0)
    acc["completion"] += int(getattr(usage, "completion_tokens", 0) or 0)
    det = getattr(usage, "prompt_tokens_details", None)
    if det is not None:
        acc["cached"] += int(getattr(det, "cached_tokens", 0) or 0)


def aggregate(client, text, prompt, model):
    """Ask a text model to synthesize `text` using `prompt` as instructions."""
    resp = client.chat.complete(
        model=model,
        messages=[{"role": "user", "content": [{"type": "text", "text": f"{prompt}\n\n{text}"}]}],
    )
    return resp.choices[0].message.content

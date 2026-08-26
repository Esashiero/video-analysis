"""Shared video-processing helpers for the Mistral video-analyzer versions.

All three version scripts (v1/v2/v3) import from here so the backend logic
(decoding, frame sampling, chunking, audio extraction, Mistral calls) lives
in one place. No model decisions are made here beyond what callers pass in.
"""
import base64
import os
import subprocess

import cv2

from mistralai.client.models.file import File


# --------------------------------------------------------------------------
# Video decode / frame sampling
# --------------------------------------------------------------------------
def sample_frames(video_path, fps=1.0, max_frames=None):
    """Sample frames from a video at ~`fps`.

    Returns (frames, duration_sec) where frames is a list of
    (timestamp_sec, base64_jpeg_string).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = (total / src_fps) if src_fps else 0.0
    step = max(1, int(round(src_fps / fps))) if src_fps else 1

    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            ok, buf = cv2.imencode(".jpg", frame)
            if ok:
                b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
                ts = (idx / src_fps) if src_fps else 0.0
                frames.append((ts, b64))
                if max_frames and len(frames) >= max_frames:
                    break
        idx += 1
    cap.release()
    return frames, duration


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


def _chat_with_retry(client, model, messages, max_retries=5):
    """client.chat.complete with exponential backoff on 429/5xx.

    TPM limits (50k/min on mistral-small-2603) can trip even when RPS is
    respected, because one batched vision request carries thousands of image
    tokens. Backing off and retrying is the correct handling.
    """
    import time as _t
    from mistralai.client.errors import SDKError
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
            _t.sleep(wait)


def describe_frames(client, frames, prompt, model, est_output_tokens=400, max_images=8,
                    batch_pause=8.0):
    """Send sampled frames to a vision model and return (description, cost).

    Vision models cap images/request (mistral-small-2603 = 8). Frames are sent
    in batches of <=max_images; descriptions from each batch are concatenated,
    then a final pass merges them into one coherent description. Returns both a
    pre-call estimate and the live usage cost.

    VISION-ONLY: ignores audio per user request.
    """
    from cost import estimate_request_cost, cost_from_usage

    if not frames:
        return "", {"estimate": None, "actual": None, "batches": 0}

    # decode image sizes for the token estimate (Pillow already installed)
    import base64 as _b64
    import io
    from PIL import Image
    img_sizes = []
    for _ts, b64 in frames:
        try:
            with Image.open(io.BytesIO(_b64.b64decode(b64))) as im:
                img_sizes.append(im.size)
        except Exception:
            img_sizes.append((1540, 1540))
    w = max((s[0] for s in img_sizes), default=1540)
    h = max((s[1] for s in img_sizes), default=1540)
    est = estimate_request_cost(len(frames), w, h, est_output_tokens, model)

    batch_texts = []
    usage_acc = {"prompt_tokens": 0, "completion_tokens": 0, "cached": 0}
    n_batches = 0
    for i in range(0, len(frames), max_images):
        batch = frames[i:i + max_images]
        content = [{"type": "text", "text": prompt}]
        for _ts, b64 in batch:
            content.append({"type": "image_url", "image_url": f"data:image/jpeg;base64,{b64}"})
        resp = _chat_with_retry(client, model, [{"role": "user", "content": content}])
        n_batches += 1
        batch_texts.append(resp.choices[0].message.content)
        # pace batches: ~5k image tokens per batch vs 50k TPM ceiling
        import time as _pt
        _pt.sleep(batch_pause)
        u = getattr(resp, "usage", None)
        if u is not None:
            usage_acc["prompt_tokens"] += int(getattr(u, "prompt_tokens", 0) or 0)
            usage_acc["completion_tokens"] += int(getattr(u, "completion_tokens", 0) or 0)
            det = getattr(u, "prompt_tokens_details", None)
            if det is not None:
                usage_acc["cached"] += int(getattr(det, "cached_tokens", 0) or 0)

    merged = "\n\n".join(f"[Batch {j+1}]\n{t}" for j, t in enumerate(batch_texts))
    if n_batches > 1:
        # final merge pass so the output is one coherent description, not chunks
        merge_resp = _chat_with_retry(client, model, [{"role": "user", "content": [
                {"type": "text", "text":
                 "The following are separate partial descriptions of frames from one "
                 "continuous video, in order. Merge them into ONE coherent, chronological "
                 "description of the whole video. Remove redundancy."},
                {"type": "text", "text": merged},
            ]}])
        n_batches += 1
        desc = merge_resp.choices[0].message.content
        mu = getattr(merge_resp, "usage", None)
        if mu is not None:
            usage_acc["prompt_tokens"] += int(getattr(mu, "prompt_tokens", 0) or 0)
            usage_acc["completion_tokens"] += int(getattr(mu, "completion_tokens", 0) or 0)
            mdet = getattr(mu, "prompt_tokens_details", None)
            if mdet is not None:
                usage_acc["cached"] += int(getattr(mdet, "cached_tokens", 0) or 0)
    else:
        desc = batch_texts[0]

    class _U:
        prompt_tokens = usage_acc["prompt_tokens"]
        completion_tokens = usage_acc["completion_tokens"]

        class _D:
            cached_tokens = usage_acc["cached"]
        prompt_tokens_details = _D()

    live = cost_from_usage(_U(), model)
    return desc, {"estimate": est, "actual": live, "batches": n_batches}


def aggregate(client, text, prompt, model):
    """Ask a text model to synthesize `text` using `prompt` as instructions."""
    resp = client.chat.complete(
        model=model,
        messages=[{"role": "user", "content": [{"type": "text", "text": f"{prompt}\n\n{text}"}]}],
    )
    return resp.choices[0].message.content

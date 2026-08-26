"""Shared Mistral model catalog + rate-limited client.

Catalog is sourced from the admin.mistral.ai limits doc the user provided.
Rate limits are per-model Requests-per-Second (RPS) and Tokens-per-Minute (TPM).
The RateLimiter enforces the RPS ceiling so we don't trip 429s.

Vision-capable: mistral-small-2603, mistral-large-2512
Audio (Voxtral): voxtral-mini-2602, voxtral-small-2507 (transcribe),
                 voxtral-mini-transcribe-realtime-2602 (realtime),
                 voxtral-mini-tts-2603 (TTS)
Text-only aggregation: ministral-3b/8b/14b
Audio global limits: 3600 audio-seconds/minute, unlimited/month.
OCR global limits: 625 pages/minute, 250 MB max doc.
"""
import os
import time

from mistralai.client import Mistral

MODELS = {
    # vision + text
    "mistral-small-2603": {"type": "vision+text", "tpm": 50000, "rps": 0.83},
    "mistral-large-2512": {"type": "vision+text", "tpm": 250000, "rps": 0.07},
    # audio transcription
    "voxtral-mini-2602": {"type": "audio-transcribe", "tpm": 50000, "rps": 1.0},
    "voxtral-small-2507": {"type": "audio-transcribe", "tpm": 50000, "rps": 1.0},
    "voxtral-mini-transcribe-realtime-2602": {"type": "audio-realtime", "tpm": 50000, "rps": 1.0},
    # tts
    "voxtral-mini-tts-2603": {"type": "tts", "tpm": 50000, "rps": 1.0},
    # text-only summarizers
    "ministral-3b-2512": {"type": "text", "tpm": 1300000, "rps": 12.5},
    "ministral-8b-2512": {"type": "text", "tpm": 625000, "rps": 3.13},
    "ministral-14b-2512": {"type": "text", "tpm": 937500, "rps": 0.5},
}

# Sensible defaults for a video-analysis pipeline
DEFAULT_VISION = "mistral-small-2603"      # cheap, vision-capable
DEFAULT_AUDIO = "voxtral-mini-2602"         # WORKING Voxtral transcription model (verified live)
DEFAULT_SUMMARIZER = "ministral-8b-2512"    # fast, cheap text aggregation
#
# NOTE on the audio catalog: the admin.mistral.ai limits doc lists
# voxtral-small-2507 and voxtral-mini-transcribe-realtime-2602 as audio models,
# but the live transcription endpoint ONLY accepts `voxtral-mini-2602`
# (and its alias voxtral-mini-latest). voxtral-small-2507 is a chat model that
# ingests audio inline, not a `client.audio.transcriptions` endpoint model.
# voxtral-mini-transcribe-realtime-2602 is for the realtime websocket API.
# Verified 2026-08-23 against the live API.

# Global audio ceiling (not per-model)
AUDIO_SECONDS_PER_MINUTE = 3600


class RateLimiter:
    """Sleeps just enough to stay at or below a model's RPS limit."""

    def __init__(self, rps):
        self.min_interval = 1.0 / rps if rps and rps > 0 else 0.0
        self._last = 0.0

    def wait(self):
        if not self.min_interval:
            return
        now = time.monotonic()
        gap = self.min_interval - (now - self._last)
        if gap > 0:
            time.sleep(gap)
        self._last = time.monotonic()


def get_client(api_key=None):
    api_key = api_key or os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY not set (pass --api-key or export it)")
    return Mistral(api_key=api_key)


def get_rate_limiter(model):
    if model not in MODELS:
        raise KeyError(f"Unknown model '{model}'. Known: {', '.join(MODELS)}")
    return RateLimiter(MODELS[model]["rps"])

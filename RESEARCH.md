# Video Analysis / AI Video Understanding — Research Notes

Scope: how AI systems analyze video on the backend, plus reference repos cloned into `ressources/`.

## 1. The universal backend pipeline

Every video-capable AI does some version of this (no provider "natively" ingests raw video efficiently):

1. **Decode** — ffmpeg / PyAV / DeepStream / av / cv2.VideoCapture into frames + audio track.
2. **Sample** — uniform (Gemini default 1 FPS), keyframe selection (CLIP/BLIP relevance scoring), or shot segmentation (TransNetV2).
3. **Encode** — each frame through a frozen vision encoder (ViT / CLIP / SigLIP / ImageBind) → visual tokens.
4. **Compress** — the real engineering problem. Token count M = (T/Δt) × (HW/P²). An hour at 1 FPS hits 1M+ tokens. Methods:
   - Temporal pooling / averaging across frames
   - Mamba state-space temporal encoder (STORM)
   - Token merging (ToMe) + hierarchical compression (VideoChat-Flash: ~16 tokens/frame, 1/50 ratio)
   - Query-aware token selection (QTSplus: cross-attention gate, shrinks KV-cache up to 89%)
   - Keyframe selection as bandit problem (FOCUS: <2% frames, +11.9% on 20min+ video)
5. **Reason** — feed tokens (+ Whisper transcript) to LLM/VLM. For long video: chunk → per-chunk VLM captions → LLM aggregation (CA-RAG) → knowledge graph (Graph-RAG) for Q&A.

## 2. Provider / framework backend approaches

| System | Backend | Notes |
|---|---|---|
| Gemini | Native video input; samples 1 FPS, audio 1 Kbps; ~300 tok/sec (258/frame + 32 audio) | 1M ctx = ~1h video. Context-cache for long reuse. `media_resolution` trades tokens for detail. Free 2GB / paid 20GB. |
| OpenAI (GPT-4o/4.1) | NO native video. Frame-sampling pattern: OpenCV extract → base64 JPEG → send as image parts to 1M-ctx LLM | Official position (openai-node#1778): video input is NOT supported in Responses API, only the frame-extraction workaround. Loses audio + temporal context. |
| Mistral (Pixtral) | Image-only API. Video = frame-sampling loop | `vision.py` already does single-image. Loop per frame + concatenate. |
| TwelveLabs Marengo/Pegasus | Native video embedding model; segment-level embeddings stored in "index" | Marengo = any-to-any embedding (visual/audio/transcription), 512-dim, dynamic OR fixed segmentation. Pegasus = text generation/summarization. Embedding-first. |
| AWS Bedrock (Nova) | Frame-based, shot-based, embedding workflows via Step Functions | Nova Multimodal Embedding for search; Bedrock LLMs for analysis. |
| ByteDance | Custom multimodal LLM, cross-modal attention fusion, INT8 quant on Inferentia2, billions/day | Enterprise scale; ~50% cost cut via Neuron + static batching. |
| NVIDIA VSS Blueprint | VLM pipeline (DeepStream) → Milvus VectorDB → CA-RAG (LLM aggregation) → Graph-RAG (Neo4J/ArangoDB) | Production reference arch; chunk+caption+aggregate; streamed via SSE. |

## 3. Token math (why it matters)

Gemini published rates are the clearest reference: ~300 tokens/sec at default resolution, ~100/sec at low.
A 1-hour video ≈ 1.08M tokens. That's why every backend either (a) aggressively downsamples,
(b) compresses tokens post-encoder, or (c) chunks-and-aggregates. Context window is the bottleneck, not the model.

## 4. Key open-source token-compression research

- **VideoChat-Flash (ICLR 2026)** — HiCo hierarchical compression: clip-level (ToMe token merging, ~16 tok/frame) + video-level (progressive visual dropout in LLM). 99.1% on 10,000 frames (NIAH). 5-10× faster.
- **STORM (ICCV 2025)** — Mamba temporal projector between ViT and LLM; 8× compute cut.
- **AKS** — adaptive keyframe sampling balancing relevance vs coverage (Ripley's K-function).
- **FOCUS** — training-free keyframe selection as multi-armed bandit; <2% of frames.
- **QTSplus** — query-aware token selector; cross-attention gates tokens, shrinks KV-cache up to 89%.
- **VideoLLaMA3** — modern MLLM (Qwen2.5 base), inference notebooks for video + long video + temporal grounding.

## 5. Self-host reference architectures (cloned in ressources/)

- `video-search-and-summarization` (NVIDIA) — production blueprint, microservices, MCP agent.
- `VideoLLaMA3` (DAMO) — frontier open video MLLM.
- `Video-LLaVA` (PKU) — unified image/video representation, HF model, clean inference snippet.
- `Video_RAG_Pipeline` (Microsoft) — Azure microservices: index → chunk → summarize (two-pass).
- `mini_videorag` (bdallard) — lightweight DAG pipeline, Whisper/OCR/YOLO/Brand/NSFW, LiteLLM, Temporal+MinIO.
- `fennec-search` (JasonMakes801) — self-host video search: CLIP + Whisper + ArcFace + pgvector, Docker.
- `tldw_server` (rmusser01) — API-first media analysis + RAG, 16+ LLM providers, VLM backends.
- `vidi` (bytedance) — video understanding + editing, STG/temporal retrieval.
- `VideoChat-Flash` (OpenGVLab) — SOTA long-context compression.
- `InternVideo` / `Ask-Anything` (OpenGVLab) — video foundation models + VideoChat chat-centric system.

## 6. Direct extension of the user's vision.py (Mistral)

Since Mistral has no native video endpoint, the path is:
- `cv2` extract frames (stride e.g. every Nth frame),
- for each frame: base64 encode → Mistral chat complete (image_url) → caption,
- collect captions → final LLM call to aggregate into a description/summary.

That is exactly the OpenAI cookbook frame-loop, ported to Mistral.

## 7. Built toolset in this project (multiple versions)

Three working scripts in `~/projects/video-analysis/`, plus shared `mistral_models.py`
(model catalog + RPS rate limiter from the limits doc) and `core.py` (ffmpeg/cv2/Mistral wrappers).

- `v1_frame_describer.py` — minimal port of `vision.py` to video. Frames → Mistral vision → description. **VERIFIED LIVE.**
- `v2_chunked_analyzer.py` — kdoronin/video_analyzer style. ffmpeg chunk → Voxtral transcript per chunk (+ optional Mistral vision) → Mistral text summary → markdown.
- `v3_structured_analyzer.py` — arashsajjadi/ai-powered-video-analyzer style. JSON + Markdown report: visual captions, transcript, evidence-grounded summary (facts vs interpretation).

### VERIFIED Mistral API facts (2026-08-23, live calls)
- Vision: `client.chat.complete(model="mistral-small-2603", messages=[{role,content:[{type:text},{type:image_url,image_url:"data:image/jpeg;base64,..."}]}])` — WORKS.
- Audio transcription: `client.audio.transcriptions.complete(model="voxtral-mini-2602", file=File(file_name, content=bytes, content_type="audio/mpeg"))` — WORKS.
- **CRITICAL**: the admin limits doc lists `voxtral-small-2507` and `voxtral-mini-transcribe-realtime-2602` as audio models, but the `transcriptions` endpoint REJECTS both (Status 400 invalid_model). Only `voxtral-mini-2602` (alias `voxtral-mini-latest`) is accepted as a transcription endpoint model. `voxtral-small-2507` is a chat model that ingests audio inline (different call path), and the `-realtime-` one is the websocket realtime API.
- TTS: `client.audio.speech.complete(input=, model="voxtral-mini-tts-2603", voice_id=, response_format=)` — signature confirmed, not yet live-tested.

## 8. Rate-limit reality (from limits doc) — drives design
| Model | RPS | TPM | Role |
|---|---|---|---|
| mistral-large-2512 | 0.07 | 250k | vision, very slow RPS |
| mistral-small-2603 | 0.83 | 50k | vision (default) |
| voxtral-mini-2602 | 1.0 | 50k | audio transcribe |
| ministral-3b-2512 | 12.5 | 1.3M | text aggregation (fast) |
| ministral-8b-2512 | 3.13 | 625k | text aggregation (default) |
| Audio global | — | 3600 audio-sec/min | all audio combined |

Implication: vision RPS is the bottleneck (0.07–0.83 req/s). Per-frame calls are impractical at scale; batch frames into ONE multimodal message (as v1/v3 do) to stay under RPS. Text aggregation is effectively free (12.5 RPS), so chunk-then-aggregate is the right shape.

"""Cost estimation for Mistral vision requests.

Image tokenization formula from Downloads/Vision.md, per-model pricing from
Downloads/Pricing.md. Live `resp.usage` is the ground-truth cost; the formula
below is the pre-call estimate.

Tokenization (Vision.md):
  Mistral Large 3 / Small 3.2 / Medium 3 / Ministral 3: max 1540x1540, /784, cap ~3025
  Pixtral Large / Pixtral 12B:                       max 1024x1024, /256, cap ~4096
  Images larger than max are downscaled first. Max 8 images/request, 10 MB/file.

Pricing (Pricing.md), EUR per 1M tokens for vision flagships:
  mistral-large-2512  (Large 3):    0.44 in / 1.30 out
  mistral-small-2603  (Small 4*):   0.12 in / 0.50 out
  ministral-3b-2512   (Ministral 3B): 0.088 in / 0.088 out
  ministral-8b-2512   (Ministral 8B): 0.13 in / 0.13 out
  ministral-14b-2512  (Ministral 14B): 0.18 in / 0.18 out
  * limits doc id mistral-small-2603 maps to pricing "Mistral Small 4" (both 2603).
"""
from math import ceil

# (max_res, divisor, cap) per model family
TOKEN_CONFIG = {
    "mistral-large-2512":   {"max_res": 1540, "divisor": 784, "cap": 3025},
    "mistral-small-2603":   {"max_res": 1540, "divisor": 784, "cap": 3025},
    "ministral-3b-2512":    {"max_res": 1540, "divisor": 784, "cap": 3025},
    "ministral-8b-2512":    {"max_res": 1540, "divisor": 784, "cap": 3025},
    "ministral-14b-2512":   {"max_res": 1540, "divisor": 784, "cap": 3025},
    "pixtral-large-latest": {"max_res": 1024, "divisor": 256, "cap": 4096},
    "pixtral-12b":          {"max_res": 1024, "divisor": 256, "cap": 4096},
}

# EUR per 1M tokens: input, cached_input, output
PRICING_EUR = {
    "mistral-large-2512":   {"input": 0.44,  "cached": 0.044,  "output": 1.3},
    "mistral-small-2603":   {"input": 0.12,  "cached": 0.012,  "output": 0.5},
    "ministral-3b-2512":    {"input": 0.088, "cached": 0.0088, "output": 0.088},
    "ministral-8b-2512":    {"input": 0.13,  "cached": 0.013,  "output": 0.13},
    "ministral-14b-2512":   {"input": 0.18,  "cached": 0.018,  "output": 0.18},
    "pixtral-large-latest": {"input": 0.0,   "cached": 0.0,    "output": 0.0},
    "pixtral-12b":          {"input": 0.0,   "cached": 0.0,    "output": 0.0},
}


def estimate_image_tokens(width, height, model):
    """Estimate tokens for one image of (width, height) per Vision.md formula."""
    cfg = TOKEN_CONFIG.get(model, {"max_res": 1540, "divisor": 784, "cap": 3025})
    w = min(width, cfg["max_res"])
    h = min(height, cfg["max_res"])
    raw = (w * h) / cfg["divisor"]
    return int(min(round(raw), cfg["cap"]))


def estimate_request_cost(num_images, img_w, img_h, est_output_tokens, model):
    """Pre-call cost estimate (EUR) for a request of `num_images` at img size."""
    per = estimate_image_tokens(img_w, img_h, model)
    in_tokens = per * num_images
    p = PRICING_EUR.get(model, {"input": 0.0, "output": 0.0})
    in_cost = in_tokens / 1_000_000 * p["input"]
    out_cost = est_output_tokens / 1_000_000 * p["output"]
    return {
        "tokens_per_image": per,
        "input_tokens": in_tokens,
        "output_tokens_est": est_output_tokens,
        "input_cost_eur": in_cost,
        "output_cost_eur": out_cost,
        "total_eur": in_cost + out_cost,
        "price_input_per_M": p["input"],
        "price_output_per_M": p["output"],
    }


def cost_from_usage(usage, model):
    """Ground-truth cost from the API response usage object."""
    if usage is None:
        return None
    in_t = int(getattr(usage, "prompt_tokens", 0) or 0)
    out_t = int(getattr(usage, "completion_tokens", 0) or 0)
    cached_t = 0
    details = getattr(usage, "prompt_tokens_details", None)
    if details is not None:
        cached_t = int(getattr(details, "cached_tokens", 0) or 0)
    p = PRICING_EUR.get(model, {"input": 0.0, "cached": 0.0, "output": 0.0})
    uncached = max(in_t - cached_t, 0)
    in_cost = (uncached / 1e6 * p["input"]) + (cached_t / 1e6 * p["cached"])
    out_cost = out_t / 1e6 * p["output"]
    return {
        "input_tokens": in_t,
        "cached_input_tokens": cached_t,
        "output_tokens": out_t,
        "input_cost_eur": in_cost,
        "output_cost_eur": out_cost,
        "total_eur": in_cost + out_cost,
    }


def fmt_eur(x):
    return f"€{x:.6f}"

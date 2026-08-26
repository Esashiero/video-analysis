"""Named prompt templates for video analysis.

Motivation (proven empirically on Snapchat-1581139204.mp4):
the same model at 1fps with a vague prompt missed the on-screen "902" score;
at 2fps with an explicit "read all on-screen numbers exactly" instruction it
caught 902 AND 978. Prompt choice moves results more than model choice here.
Pick a template by name with --template, or pass --prompt to override fully.
"""

TEMPLATES = {
    # Rich general-purpose default: forces OCR of scores/labels, people,
    # actions, branding, chronological narrative.
    "default": (
        "These are sampled frames from ONE video, in chronological order.\n"
        "Analyze them step by step:\n"
        "1. Describe each distinct scene/setting.\n"
        "2. Identify every person (adult vs child, clothing, what they are doing).\n"
        "3. Read ALL on-screen text EXACTLY as written: scores, timers, prices, "
        "signage, brand logos, subtitles. Do not paraphrase numbers — copy them.\n"
        "4. Note camera movement / scene transitions.\n"
        "Then produce one coherent chronological description of the whole video."
    ),

    # Number/text-extraction focused: for scoreboards, receipts, UIs, signage.
    "numbers": (
        "These are sampled frames from ONE video, in chronological order.\n"
        "Your ONLY job: find and transcribe every piece of on-screen text and "
        "every visible number (scores, counters, prices, timestamps, license "
        "plates, labels). Output as a list: 'timestamp-ish position -> exact "
        "text'. If a number appears in multiple frames, report its final value. "
        "If nothing readable exists, say so explicitly."
    ),

    # Timeline / event segmentation: good base for v2/v3 structured reports.
    "timeline": (
        "These are sampled frames from ONE video, in chronological order.\n"
        "Segment the video into scenes. For each scene output:\n"
        "- approximate time range (based on frame order)\n"
        "- setting and who/what is present\n"
        "- key action\n"
        "- any on-screen text read EXACTLY\n"
        "End with a one-line summary of the overall video."
    ),

    # Minimal baseline: closest to the original vision.py behaviour.
    "basic": (
        "These are frames from a video in chronological order. "
        "Describe what happens."
    ),
}


def get_template(name):
    if name not in TEMPLATES:
        raise KeyError(f"Unknown template '{name}'. Available: {', '.join(TEMPLATES)}")
    return TEMPLATES[name]

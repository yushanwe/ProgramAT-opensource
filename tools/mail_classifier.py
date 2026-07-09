"""Mail Classification Tool for Blind Users

Classifies mail as important or junk based on a camera image.
Uses a two-stage pipeline: visual structure analysis followed by
reasoning about mail type (bills, government letters, etc.).
"""

from __future__ import annotations

from model_router_client import copilot_llm_call

TOOL_NAME = "mail_classifier"


def main(image, input_data=None):
    """Classify mail in the image as important or junk.

    Args:
        image: Camera frame (numpy array, BGR format).
        input_data: Optional dict (unused, reserved for future options).

    Returns:
        Audio-friendly string describing the mail classification result.
    """
    # Stage 1: Structured visual understanding — identify visible mail items
    # and extract observable characteristics (sender, logos, text snippets,
    # envelope style, postage markings, etc.).
    stage1 = copilot_llm_call(
        capability="structured_visual_understanding",
        goal="Determine if each mail item is important or junk",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a mail-sorting assistant for a blind user. "
                    "Examine the mail item(s) in the image. "
                    "Extract all visible cues: sender name or logo, return address, "
                    "subject line, envelope design, postage type, barcodes, and any "
                    "printed text. "
                    "Return a concise structured summary of these cues."
                ),
            },
            {
                "role": "user",
                "content": "Identify and describe the visible mail item(s) and their key features.",
            },
        ],
        images=[image],
        metadata={
            "tool_name": TOOL_NAME,
            "route_text": "extract structured visual cues from mail envelope or document",
        },
    )

    visual_artifact = stage1.get("artifact") or stage1.get("response", "")

    # Stage 2: General reasoning — classify each item as important or junk
    # based on the extracted cues.
    stage2 = copilot_llm_call(
        capability="general_reasoning",
        goal="Classify mail items based on characteristics like bills, government letters, etc.",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a mail-sorting assistant for a blind user. "
                    "Based on the mail features provided, decide whether each item is "
                    "important mail (bills, government correspondence, medical notices, "
                    "legal documents, bank statements, package notifications) or "
                    "junk mail (advertisements, unsolicited offers, coupons, bulk mailers). "
                    "State your answer clearly and concisely for text-to-speech. "
                    "Use phrases like 'important mail' or 'junk mail'. "
                    "Keep the total response under 20 words."
                ),
            },
            {
                "role": "user",
                "content": f"Mail features observed: {visual_artifact}\n\nIs this important mail or junk mail?",
            },
        ],
        images=[image],
        metadata={
            "tool_name": TOOL_NAME,
            "route_text": "classify mail as important or junk based on visual characteristics",
            "previous_stage_artifact": visual_artifact,
        },
    )

    response = stage2.get("response", "").strip()
    if not response:
        return "Unable to classify the mail. Please try again with a clearer image."
    return response

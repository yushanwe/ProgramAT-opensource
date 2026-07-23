from model_execution import execute_tool_policy

TOOL_NAME = "empty_seat_detection"
EXECUTION_MODE = "take_photo"
TOOL_PROMPT = (
    "You are guiding a blind or low-vision user. In this image, count how many clearly empty chairs or seats are visible, "
    "identify the closest empty one, and return one concise user-facing line that includes: the empty-seat count, then the "
    "closest empty seat's clock-face direction and rough distance in feet, without other visual landmarks. If no empty seat "
    "is visible, say exactly \"0 empty chairs. No empty seat is visible.\" Return only the final user-facing result."
)
TOOL_POLICY = {
    "strategy": "conditional",
    "condition": {"key": "needs_accuracy", "equals": True},
    "if_true": {"strategy": "single", "models": ["gpt-5"]},
    "if_false": {
        "strategy": "cascade",
        "models": ["gemini-3.1-flash-lite", "gpt-5"],
        "evaluator": "gpt-4o-mini",
        "stop_condition": "accepted",
    },
}


def main(image, input_data):
    if image is None:
        return "No camera image is available."
    metadata = {"needs_accuracy": (input_data or {}).get("needs_accuracy", False)}
    return execute_tool_policy(
        image=image,
        prompt=TOOL_PROMPT,
        policy=TOOL_POLICY,
        tool_name=TOOL_NAME,
        metadata=metadata,
    )

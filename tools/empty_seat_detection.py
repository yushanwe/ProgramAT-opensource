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
    "strategy": "parallel_aggregate",
    "models": ["gpt-4o-mini", "gemini-3.1-flash-lite", "gpt-5"],
    "aggregator": "gpt-4o-mini",
    "aggregation_prompt": (
        "You are combining three model answers in this order: GPT-4o-mini, Gemini, GPT-5. "
        "Return exactly three short spoken lines prefixed 'Quick:', 'Balanced:', and 'Detailed:' "
        "using each model's answer. Keep each line concise and audio-friendly."
    ),
}


def main(image, input_data):
    if image is None:
        return "No camera image is available."
    return execute_tool_policy(
        image=image,
        prompt=TOOL_PROMPT,
        policy=TOOL_POLICY,
        tool_name=TOOL_NAME,
    )

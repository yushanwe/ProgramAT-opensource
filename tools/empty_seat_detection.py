from model_execution import execute_tool_policy

TOOL_NAME = "empty_seat_detection"
EXECUTION_MODE = "take_photo"
TOOL_PROMPT = (
    "Find the nearest clearly empty chair or other seat in this image and give brief spoken guidance "
    "to reach it in crowded surroundings. If no empty seat is clearly visible, say exactly "
    "\"No clear empty seat is visible right now.\""
)
TOOL_POLICY = {"strategy": "single", "models": ["gemini-3.1-flash-lite"]}


def main(image, input_data):
    if image is None:
        return "No camera image is available."
    return execute_tool_policy(
        image=image, prompt=TOOL_PROMPT, policy=TOOL_POLICY, tool_name=TOOL_NAME
    )

from model_execution import execute_tool_policy

TOOL_NAME = "empty_seat_detection"
EXECUTION_MODE = "take_photo"
TOOL_PROMPT = (
    "You are guiding a blind or low-vision user. Follow this sequence in the same image: "
    "1) identify visible chairs or other seats; 2) determine which visible seats are clearly unoccupied; "
    "3) choose the nearest suitable empty seat and return one short, fast user-facing instruction using only "
    "clock-face direction (for example 11 o'clock) and rough distance in feet (for example 6 feet), "
    "without using other visual landmarks. If no empty seat is visible, say exactly "
    "\"No empty seat is visible.\" Return only the final user-facing result."
)
TOOL_POLICY = {"strategy": "single", "models": ["gemini-3.1-flash-lite"]}


def main(image, input_data):
    if image is None:
        return "No camera image is available."
    return execute_tool_policy(
        image=image, prompt=TOOL_PROMPT, policy=TOOL_POLICY, tool_name=TOOL_NAME
    )

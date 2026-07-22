from model_execution import execute_tool_policy

TOOL_NAME = "seating_locator"
EXECUTION_MODE = "take_photo"
TOOL_PROMPT = (
    "Find the nearest clearly empty chair or seat and give concise spoken guidance toward it, "
    "with simple directional wording suitable for crowded spaces. If no clearly empty nearby "
    "seat is visible, say so."
)
TOOL_POLICY = {"strategy": "single", "models": ["gemini-3.1-flash-lite"]}


def main(image, input_data):
    if image is None:
        return "No camera image is available."
    return execute_tool_policy(
        image=image, prompt=TOOL_PROMPT, policy=TOOL_POLICY, tool_name=TOOL_NAME
    )

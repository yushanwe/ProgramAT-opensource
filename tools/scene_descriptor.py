"""
Scene Descriptor Tool for Blind Users

Progressive scene description that provides brief object listing first,
then detailed descriptions when the user focuses on the same scene.

Features:
- Initial brief enumeration of objects in scene
- Detailed description on prolonged camera focus
- Scene change detection
- Audio-optimized output for blind and low-vision users

Example Usage:
First frame: "A desk, a cabinet, a printer, and a cat"
After prolonged focus: "A cat sitting on top of a white desk, looking out the window"

This tool implements both Take Photo and Streaming modes with shared prompt.
"""

import asyncio
from typing import Dict, Any, Optional
from litellm_utils import call_model, extract_text

TOOL_NAME = "scene_descriptor"
TOOL_PROMPT = (
    "Describe the scene for a blind user. "
    "1. List key objects briefly when the scene is first viewed. "
    "2. Provide detailed description with spatial relationships and actions when the same scene "
    "is viewed for several frames. "
    "Keep output concise and speech-ready. If the scene is unclear, state what cannot be determined."
)

# State tracking for progressive description
STATE_SCENE_HASH = "scene_hash"
STATE_FRAME_COUNT = "frame_count"
STATE_LAST_DESCRIPTION = "last_description"

# Configuration
FRAMES_FOR_DETAILED = 3  # Number of consistent frames before detailed description
SCENE_CHANGE_THRESHOLD = 0.4  # Scene similarity threshold (lower = more different)


def compute_scene_hash(image_base64: str) -> str:
    """
    Compute a simple hash of the image to detect scene changes.

    Args:
        image_base64: Base64 encoded image

    Returns:
        Hash string representing the scene
    """
    # Use first and last 100 characters as a simple hash proxy
    # This is a heuristic - real scene change would need more sophisticated comparison
    if len(image_base64) > 200:
        return image_base64[:100] + image_base64[-100:]
    return image_base64


def scene_similarity(hash1: str, hash2: str) -> float:
    """
    Estimate similarity between two scene hashes.

    Args:
        hash1: First scene hash
        hash2: Second scene hash

    Returns:
        Similarity score (0.0 to 1.0), where 1.0 means identical
    """
    if not hash1 or not hash2:
        return 0.0

    if hash1 == hash2:
        return 1.0

    # Simple character overlap similarity
    matches = sum(1 for c1, c2 in zip(hash1, hash2) if c1 == c2)
    max_len = max(len(hash1), len(hash2))

    return matches / max_len if max_len > 0 else 0.0


async def on_take_photo(runtime, image, input_data):
    """
    Take Photo mode: Provide brief object listing.

    Args:
        runtime: Tool runtime with state and emission
        image: Current image from camera
        input_data: Optional input configuration

    Returns:
        Brief scene description
    """
    # Use brief description for single photo
    prompt = (
        "Briefly list the main objects in this scene for a blind user. "
        "Format as a natural comma-separated list. Keep it under 15 words. "
        "Example: 'A desk, a chair, a laptop, and a coffee mug.'"
    )

    # Call model
    response = await asyncio.to_thread(
        call_model,
        "gemini/gemini-3.1-flash-lite-preview",
        [{"role": "user", "content": prompt}],
        [image],
    )

    description = extract_text(response)

    # Clean up and ensure concise
    description = description.strip()
    if len(description) > 200:
        description = description[:200].rsplit(' ', 1)[0] + '...'

    return description


async def on_stream_start(runtime, input_data):
    """
    Initialize streaming session.

    Args:
        runtime: Tool runtime
        input_data: Optional input configuration
    """
    # Clear state at start of new stream
    runtime.clear_state()


async def on_frame(runtime, frame):
    """
    Process each streaming frame with progressive description logic.

    Args:
        runtime: Tool runtime
        frame: Current frame object with image, timestamp, etc.
    """
    if runtime.is_cancelled():
        return

    # Get current scene hash
    current_hash = compute_scene_hash(frame.image_base64)

    # Get previous state
    prev_hash = runtime.get_state(STATE_SCENE_HASH)
    frame_count = runtime.get_state(STATE_FRAME_COUNT, 0)

    # Check if scene changed
    if prev_hash:
        similarity = scene_similarity(current_hash, prev_hash)
        if similarity < SCENE_CHANGE_THRESHOLD:
            # Scene changed - reset and give brief description
            runtime.clear_state()
            frame_count = 0

    # Increment frame count for this scene
    frame_count += 1
    runtime.set_state(STATE_FRAME_COUNT, frame_count)
    runtime.set_state(STATE_SCENE_HASH, current_hash)

    # Decide what type of description to provide
    if frame_count == 1:
        # First frame of this scene - brief object listing
        prompt = (
            "Briefly list the main objects in this scene for a blind user. "
            "Format as a natural comma-separated list. Keep it under 15 words. "
            "Example: 'A desk, a chair, a laptop, and a coffee mug.'"
        )
    elif frame_count >= FRAMES_FOR_DETAILED:
        # Prolonged focus - provide detailed description (only once)
        last_desc = runtime.get_state(STATE_LAST_DESCRIPTION)
        if last_desc and last_desc.startswith("DETAILED"):
            # Already gave detailed description, skip to avoid repetition
            return

        prompt = (
            "Provide a detailed scene description for a blind user. "
            "Include spatial relationships, object states, and visible actions. "
            "Keep it under 30 words and speech-ready. "
            "Example: 'A cat sitting on top of a white desk near the window, looking outside.'"
        )
    else:
        # In between - skip to avoid spam
        return

    # Call model
    response = await asyncio.to_thread(
        call_model,
        "gemini/gemini-3.1-flash-lite-preview",
        [{"role": "user", "content": prompt}],
        [frame.image],
    )

    description = extract_text(response)

    # Clean up
    description = description.strip()
    if len(description) > 300:
        description = description[:300].rsplit(' ', 1)[0] + '...'

    # Mark description type in state
    desc_type = "DETAILED" if frame_count >= FRAMES_FOR_DETAILED else "BRIEF"
    runtime.set_state(STATE_LAST_DESCRIPTION, f"{desc_type}:{description}")

    # Emit result
    await runtime.emit(description, final=True)


async def on_stream_stop(runtime):
    """
    Clean up streaming session.

    Args:
        runtime: Tool runtime
    """
    # Clear state on stream stop for fresh start next time
    runtime.clear_state()

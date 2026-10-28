"""Model-independent direct-action context; evaluation state is never serialized."""

import base64
import io
import json

from PIL import Image

PROMPT_VERSION = "local-navigation-v3-recovery"
ACTION_DESCRIPTIONS = {
    "MOVE_FORWARD": "Move forward 0.25 metres without changing heading.",
    "TURN_LEFT": "Rotate left in place by 15 degrees.",
    "TURN_RIGHT": "Rotate right in place by 15 degrees.",
    "STOP": "End this task now, only if the requested local destination has been reached.",
}
ACTION_QUESTION = (
    "Choose the next single action to achieve the local instruction, using the current image "
    "and any earlier images and executed-action feedback. Images are ordered oldest to newest; "
    "the final image is current. A rejected action did not move the robot. Do not assume an "
    "action succeeded. Stop only when the visual destination and instruction are satisfied."
)


def encode_images(view):
    images = []
    for frame in view.frames:
        buffer = io.BytesIO()
        Image.fromarray(frame.rgb).save(buffer, format="PNG")
        images.append("data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode())
    return images


def state_text(view, *, image_markers=False, instruction_label="Local instruction"):
    history = [
        {
            "proposed": x.proposed.value,
            "executed": x.executed.value if x.executed else None,
            "feedback": x.reason,
            "collided": x.collided,
        }
        for x in view.history
    ]
    frames = "\n".join(
        f"Image {i + 1}: step={f.step}" + (" <image>" if image_markers else "")
        for i, f in enumerate(view.frames)
    )
    return (
        f"{instruction_label}: {view.instruction}\n{frames}\n"
        f"Executed-action history: {json.dumps(history, ensure_ascii=False)}"
        + (f"\nRecovery feedback: {view.recovery_context}" if view.recovery_context else "")
    )


ROUTE_PROMPT_VERSION = "r2rce-direct-v1"


def action_context(view, prompt_version=PROMPT_VERSION):
    if prompt_version not in {PROMPT_VERSION, ROUTE_PROMPT_VERSION}:
        raise ValueError("Unknown direct-action prompt version")
    route = prompt_version == ROUTE_PROMPT_VERSION
    if route and (not 1 <= len(view.frames) <= 3 or len(view.history) > 4):
        raise ValueError("Route input supports current plus two prior images and four feedbacks")
    descriptions = dict(ACTION_DESCRIPTIONS)
    question = ACTION_QUESTION
    if route:
        descriptions["STOP"] = (
            "End the entire route now, only after all parts of the original route instruction "
            "have been completed, using visual evidence."
        )
        question = (
            "Choose the next single action to follow the entire original route instruction "
            "in order. Use the current image, earlier images, and executed-action feedback. "
            "Images are ordered oldest to newest; the final image is current. A rejected "
            "action did not move the robot. Do not assume an action succeeded. Select STOP "
            "only after the entire route instruction is satisfied, not at an intermediate landmark."
        )
    return {
        "state": state_text(
            view, instruction_label="Route instruction" if route else "Local instruction"
        ),
        "images": encode_images(view),
        "question": question,
        "criteria": descriptions,
    }

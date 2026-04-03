import os

PROMPTS_ROOT = os.path.join(os.path.dirname(__file__), "..", "prompts")


def load_prompt(prompt_name: str) -> tuple[str, str]:
    """
    Returns (prompt_text, version_string).

    Reads prompts/{prompt_name}/current.txt to find the active version,
    then loads prompts/{prompt_name}/{version}.txt.
    Windows-compatible — no symlinks used.
    """
    prompt_dir = os.path.normpath(os.path.join(PROMPTS_ROOT, prompt_name))
    current_file = os.path.join(prompt_dir, "current.txt")

    with open(current_file, "r", encoding="utf-8") as f:
        version = f.read().strip()  # e.g. "v1"

    prompt_file = os.path.join(prompt_dir, f"{version}.txt")
    with open(prompt_file, "r", encoding="utf-8") as f:
        prompt_text = f.read()

    return prompt_text, version

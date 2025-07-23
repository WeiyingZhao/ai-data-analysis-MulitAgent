import os

def load_prompt(name: str) -> str:
    """Load prompt text from the prompts directory."""
    path = os.path.join(os.path.dirname(__file__), name)
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

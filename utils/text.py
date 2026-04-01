import re

def clean_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def chunk_text(text: str, chunk_size: int = 500):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
"""Text preprocessing for fake news classification."""

import re


def clean_text_publisher_tag(text: str) -> str:
    """Remove publisher prefixes like 'WASHINGTON (Reuters) - '."""
    text = str(text)
    match = re.match(r"^(?:[A-Z\s]+(?:\([A-Za-z]+\))? - |\([A-Za-z]+\) - )", text)
    if match:
        first_hyphen_index = text.find("-")
        if first_hyphen_index != -1:
            return text[first_hyphen_index + 1 :].strip()
    return text

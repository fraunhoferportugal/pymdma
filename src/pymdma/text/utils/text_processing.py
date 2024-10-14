import re

from nltk import sent_tokenize


def tokenize_with_indexes(text, punctuation=True):
    if punctuation:
        word_pattern = r"\b\w+\b|[.,!?;]"
    else:
        word_pattern = r"\b\w+\b"

    tokens = []

    for match in re.finditer(word_pattern, text):
        start, end = match.span()
        token = match.group()
        tokens.append((token, start, end))

    return tokens


def tokenize_text_to_sentences(text: str):
    """Splits a chunk of text into sentences.

    Args:
        text (str): Input text.

    Returns:
        list: List of sentences.
    """
    return sent_tokenize(text)

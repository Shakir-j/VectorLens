def chunk_text(
    text: str,
    chunk_size: int = 250,
    overlap: int = 30
) -> list[str]:
    """
    Split a document into overlapping word-based chunks.

    Matches the original project's behavior:
    - 250 words per chunk
    - 30-word overlap
    """

    words = text.split()

    if not words:
        return []

    if overlap >= chunk_size:
        raise ValueError(
            "overlap must be smaller than chunk_size"
        )

    chunks = []

    step = chunk_size - overlap

    for start in range(
        0,
        len(words),
        step
    ):
        chunk_words = words[
            start:start + chunk_size
        ]

        if not chunk_words:
            break

        chunks.append(
            " ".join(chunk_words)
        )

        if start + chunk_size >= len(words):
            break

    return chunks
def get_strategy_explanation(strategy):
    if strategy == "Recursive":
        return """
        **Recursive (Fixed Size)** strategy splits the document into fixed-length chunks
        while preserving natural boundaries like paragraphs or sentences. It uses a hierarchy
        of separators (\\n\\n, \\n, ". ", " ") to find the best split points.

        - Good balance between completeness and coherence
        - Allows overlap to preserve context across chunks
        """

    elif strategy == "Markdown-aware":
        return """
        **Markdown-aware** strategy preserves document structure by chunking based on Markdown headers
        and paragraphs. Ideal for structured docs like technical blogs or books.

        - Maintains semantic groupings
        - Prevents breaking logical sections like headings or paragraphs
        """

    return "No explanation available."

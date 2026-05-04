"""
retrieval_eval.py  —  lives at project_root/evaluation/retrieval_eval.py

Retrieval quality metrics used by eval_runner.py.

Precision@K  — of the top-K retrieved chunks, what fraction are relevant?
               Relevance is determined by keyword presence in chunk text.

Hit Rate@K   — did at least one of the top-K chunks contain a relevant keyword?
               Binary: 1 if any hit, 0 if none.

These replace the simple keyword-in-concatenated-text hack that was in
eval_runner.py. The difference:
  - Old approach: concatenated ALL retrieved text and checked if keywords appeared
    anywhere in the blob. This always gave Precision = 1.0 as long as at least one
    chunk had the keyword, regardless of how many irrelevant chunks were retrieved.
  - New approach: checks each chunk individually, giving a true per-chunk precision.
"""

from typing import List


def precision_at_k(chunks: List[str], keywords: List[str], k: int = 6) -> float:
    """
    Compute Precision@K for a list of retrieved text chunks.

    A chunk is considered "relevant" if it contains at least one of the
    required keywords (case-insensitive). This matches how eval_queries.json
    defines relevance — via required_context_keywords.

    Args:
        chunks:   List of retrieved document strings (in retrieval-score order).
        keywords: List of keyword strings that define a relevant chunk.
        k:        Number of top chunks to evaluate. Defaults to 6 (matches
                  the k=6 used in SearchFinancialDatabaseTool).

    Returns:
        Float in [0.0, 1.0]. 1.0 means every top-K chunk was relevant.
    """
    if not chunks or not keywords:
        return 0.0

    top_k = chunks[:k]
    lower_keywords = [kw.lower() for kw in keywords]

    relevant_count = sum(
        1 for chunk in top_k
        if any(kw in chunk.lower() for kw in lower_keywords)
    )

    return round(relevant_count / len(top_k), 4)


def hit_rate_at_k(chunks: List[str], keywords: List[str], k: int = 6) -> int:
    """
    Compute Hit Rate@K — did at least one top-K chunk contain a keyword?

    Args:
        chunks:   List of retrieved document strings (in retrieval-score order).
        keywords: List of keyword strings that define a relevant chunk.
        k:        Number of top chunks to evaluate.

    Returns:
        1 if any top-K chunk contains at least one keyword, else 0.
    """
    if not chunks or not keywords:
        return 0

    top_k = chunks[:k]
    lower_keywords = [kw.lower() for kw in keywords]

    for chunk in top_k:
        if any(kw in chunk.lower() for kw in lower_keywords):
            return 1
    return 0


def keyword_hit_rate_flat(text: str, keywords: List[str]) -> float:
    """
    Legacy flat metric: fraction of keywords found anywhere in a text blob.
    Used as a fallback for refusal cases where no structured chunks exist —
    in those cases we check the pipeline's full string output instead.

    Args:
        text:     The full text to search (e.g. concatenated report + raw data).
        keywords: List of keyword strings.

    Returns:
        Float in [0.0, 1.0].
    """
    if not keywords:
        return 0.0
    lower_text = text.lower()
    hits = sum(1 for kw in keywords if kw.lower() in lower_text)
    return round(hits / len(keywords), 4)


if __name__ == "__main__":
    # ── Quick sanity check ────────────────────────────────────────────────────
    sample_chunks = [
        "[Source: Tesla | Ticker: TSLA | FY: 2023] Total automotive revenues were $82,419 million.",
        "[Source: Tesla | Ticker: TSLA | FY: 2023] Services and other revenues were $8,319 million.",
        "[Source: Tesla | Ticker: TSLA | FY: 2023] Energy generation revenue was $6,035 million.",
        "[Source: Apple | Ticker: AAPL | FY: 2025] iPhone net sales were $209,586 million.",  # irrelevant
        "[Source: Tesla | Ticker: TSLA | FY: 2023] Net income was $14,974 million.",
        "[Source: Tesla | Ticker: TSLA | FY: 2023] Total revenues were $96,773 million.",
    ]

    keywords = ["82,419", "automotive"]

    p_at_k = precision_at_k(sample_chunks, keywords, k=6)
    hr_at_k = hit_rate_at_k(sample_chunks, keywords, k=6)

    print(f"Precision@6 : {p_at_k}")   # 1/6 ≈ 0.1667 (only chunk[0] has both keywords)
    print(f"Hit Rate@6  : {hr_at_k}")  # 1 (chunk[0] is a hit)

    # Test with a refusal case
    refusal_output = "DATA_UNAVAILABLE: Query triggered safety filters."
    refusal_keywords = ["DATA_UNAVAILABLE", "out_of_scope"]
    flat_score = keyword_hit_rate_flat(refusal_output, refusal_keywords)
    print(f"Flat hit rate (refusal): {flat_score}")  # 0.5 — one of two keywords matched

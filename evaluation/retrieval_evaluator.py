from typing import Any, Dict, Iterable, Sequence, Set


class RetrievalEvaluator:
    """Precision@K, Recall@K, Reciprocal Rank and MRR."""

    @staticmethod
    def precision_at_k(retrieved_ids: Sequence[str], relevant_ids: Set[str], k: int) -> float:
        if k <= 0:
            return 0.0
        top_k = list(retrieved_ids[:k])
        if not top_k:
            return 0.0
        return sum(x in relevant_ids for x in top_k) / len(top_k)

    @staticmethod
    def recall_at_k(retrieved_ids: Sequence[str], relevant_ids: Set[str], k: int) -> float:
        if k <= 0 or not relevant_ids:
            return 0.0
        top_k = list(retrieved_ids[:k])
        return sum(x in relevant_ids for x in top_k) / len(relevant_ids)

    @staticmethod
    def reciprocal_rank(retrieved_ids: Sequence[str], relevant_ids: Set[str]) -> float:
        for rank, item_id in enumerate(retrieved_ids, start=1):
            if item_id in relevant_ids:
                return 1.0 / rank
        return 0.0

    @classmethod
    def evaluate_question(cls, retrieved_ids, relevant_ids, k=3) -> Dict[str, float]:
        relevant = set(relevant_ids)
        return {
            "precision_at_k": cls.precision_at_k(retrieved_ids, relevant, k),
            "recall_at_k": cls.recall_at_k(retrieved_ids, relevant, k),
            "reciprocal_rank": cls.reciprocal_rank(retrieved_ids, relevant),
        }

    @staticmethod
    def aggregate(results):
        if not results:
            return {"precision_at_k": 0.0, "recall_at_k": 0.0, "mrr": 0.0}
        return {
            "precision_at_k": sum(x["precision_at_k"] for x in results) / len(results),
            "recall_at_k": sum(x["recall_at_k"] for x in results) / len(results),
            "mrr": sum(x["reciprocal_rank"] for x in results) / len(results),
        }


def document_to_text(item: Any) -> str:
    if isinstance(item, tuple) and item:
        item = item[0]
    if hasattr(item, "page_content"):
        return str(item.page_content)
    return str(item)


def find_chunk_id(retrieved_item: Any, chunks) -> str | None:
    text = document_to_text(retrieved_item)
    for index, chunk in enumerate(chunks):
        if text == document_to_text(chunk):
            return f"chunk_{index}"
    return None

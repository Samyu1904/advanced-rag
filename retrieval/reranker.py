from sentence_transformers import CrossEncoder


class DocumentReranker:

    def __init__(
        self,
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        print("Loading reranker model...")

        self.model = CrossEncoder(model_name)

        print("Reranker model loaded successfully!")

    def rerank(
        self,
        question,
        documents,
        top_k=3
    ):
        if not documents:
            return []

        pairs = []

        for document in documents:
            pairs.append(
                (
                    question,
                    document.page_content
                )
            )

        scores = self.model.predict(pairs)

        ranked_documents = []

        for document, score in zip(documents, scores):
            ranked_documents.append(
                (
                    document,
                    float(score)
                )
            )

        ranked_documents.sort(
            key=lambda item: item[1],
            reverse=True
        )

        return ranked_documents[:top_k]
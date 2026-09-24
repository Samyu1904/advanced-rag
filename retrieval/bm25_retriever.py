from rank_bm25 import BM25Okapi


class BM25Retriever:

    def __init__(self, documents):

        self.documents = documents

        # Convert each document into tokens
        tokenized_documents = [
            document.page_content.lower().split()
            for document in documents
        ]

        # Create BM25 index
        self.bm25 = BM25Okapi(
            tokenized_documents
        )

        print("BM25 retriever created successfully!")

    def retrieve(self, question, top_k=3):

        # Convert question into tokens
        tokenized_question = (
            question.lower().split()
        )

        # Get BM25 scores
        scores = self.bm25.get_scores(
            tokenized_question
        )

        # Sort document indexes by score
        ranked_indexes = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )

        # Select top documents
        results = []

        for index in ranked_indexes[:top_k]:

            results.append(
                (
                    self.documents[index],
                    scores[index]
                )
            )

        return results
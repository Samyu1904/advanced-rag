class RAGRetriever:

    def __init__(self, vector_store, top_k=3, threshold=1.2):

        self.vector_store = vector_store
        self.top_k = top_k
        self.threshold = threshold

    def retrieve(self, question):

        results = self.vector_store.similarity_search_with_score(
            question,
            k=self.top_k
        )

        return results

    def is_relevant(self, results):

        if not results:
            return False

        best_score = results[0][1]

        print(f"Best FAISS score: {best_score}")

        if best_score <= self.threshold:
            return True

        return False

    def search(self, question):

        results = self.retrieve(question)

        relevant = self.is_relevant(results)

        return results, relevant
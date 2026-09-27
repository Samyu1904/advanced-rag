class ParentChildRetriever:

    def __init__(
        self,
        child_vector_store,
        parent_store,
        top_k=3,
        threshold=1.2
    ):

        self.child_vector_store = child_vector_store
        self.parent_store = parent_store
        self.top_k = top_k
        self.threshold = threshold

    def retrieve(self, question):

        # 1. Search child chunks
        child_results = (
            self.child_vector_store.search(
                question,
                top_k=self.top_k
            )
        )

        if not child_results:
            return []

        # 2. Check relevance using best child score
        best_score = child_results[0][1]

        print(
            f"Best child FAISS score: {best_score}"
        )

        if best_score > self.threshold:
            print(
                "Question is not relevant to the document."
            )
            return []

        # 3. Collect unique parent IDs
        parent_ids = []

        for document, score in child_results:

            parent_id = document.metadata.get(
                "parent_id"
            )

            if (
                parent_id
                and parent_id not in parent_ids
            ):
                parent_ids.append(parent_id)

        # 4. Retrieve complete parent documents
        parents = self.parent_store.get_parents(
            parent_ids
        )

        return parents
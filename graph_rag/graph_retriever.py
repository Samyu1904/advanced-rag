class GraphRetriever:
    def __init__(self, graph, top_k=10):
        self.nodes = graph["nodes"]
        self.edges = graph["edges"]
        self.top_k = top_k

        print(
            "Graph RAG retriever initialized successfully!"
        )

    def tokenize(self, text):
        punctuation = ".,!?;:()[]{}\"'"

        return set(
            word.strip(punctuation).lower()
            for word in text.split()
            if len(word.strip(punctuation)) > 2
        )

    def find_matching_nodes(self, question):
        question_words = self.tokenize(question)

        matched_nodes = []

        for node in self.nodes.values():

            node_words = self.tokenize(
                node["name"]
            )

            overlap = question_words & node_words

            if overlap:

                matched_nodes.append(
                    {
                        "node": node,
                        "score": len(overlap)
                    }
                )

        matched_nodes.sort(
            key=lambda item: item["score"],
            reverse=True
        )

        return matched_nodes[:self.top_k]

    def retrieve(self, question):
        print("\n")
        print("=" * 70)
        print("GRAPH RAG RETRIEVAL")
        print("=" * 70)

        print(
            f"\nSearching knowledge graph for:\n{question}"
        )

        matched_nodes = self.find_matching_nodes(
            question
        )

        print(
            f"\nMatched graph nodes: "
            f"{len(matched_nodes)}"
        )

        if not matched_nodes:
            print(
                "\nNo matching graph entities found."
            )

            return []

        matched_names = set(
            item["node"]["name"].lower()
            for item in matched_nodes
        )

        relevant_edges = []

        for edge in self.edges:

            source_match = (
                edge["source"].lower()
                in matched_names
            )

            target_match = (
                edge["target"].lower()
                in matched_names
            )

            if source_match or target_match:
                relevant_edges.append(edge)

        expanded_names = set(matched_names)

        for edge in relevant_edges:

            expanded_names.add(
                edge["source"].lower()
            )

            expanded_names.add(
                edge["target"].lower()
            )

        for edge in self.edges:

            connected = (
                edge["source"].lower()
                in expanded_names
                or
                edge["target"].lower()
                in expanded_names
            )

            if connected and edge not in relevant_edges:
                relevant_edges.append(edge)

        relevant_edges = relevant_edges[
            :self.top_k * 3
        ]

        print(
            f"Relevant relationships: "
            f"{len(relevant_edges)}"
        )

        return [
            {
                "source": edge["source"],
                "relationship": edge["relationship"],
                "target": edge["target"]
            }
            for edge in relevant_edges
        ]

    def is_relevant(self, results):
        return len(results) > 0

    def format_context(self, results):
        if not results:
            return ""

        lines = []

        for result in results:

            lines.append(
                f"{result['source']} "
                f"--[{result['relationship']}]--> "
                f"{result['target']}"
            )

        return "\n".join(lines)
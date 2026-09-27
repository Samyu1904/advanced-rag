from langchain_text_splitters import RecursiveCharacterTextSplitter


class ParentChildSplitter:

    def __init__(
        self,
        parent_chunk_size=2000,
        parent_chunk_overlap=200,
        child_chunk_size=500,
        child_chunk_overlap=100
    ):

        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_chunk_size,
            chunk_overlap=parent_chunk_overlap
        )

        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_chunk_size,
            chunk_overlap=child_chunk_overlap
        )

    def split_documents(self, documents):

        # Create parent chunks
        parents = self.parent_splitter.split_documents(
            documents
        )

        print("Parent splitting completed!")
        print(
            f"Number of parent chunks: {len(parents)}"
        )

        children = []

        # Create child chunks from each parent
        for parent_id, parent in enumerate(parents):

            parent_id_value = f"parent_{parent_id}"

            # Store ID in the parent itself
            parent.metadata["parent_id"] = (
                parent_id_value
            )

            child_chunks = self.child_splitter.split_documents(
                [parent]
            )

            for child in child_chunks:

                # Store the same parent ID in every child
                child.metadata["parent_id"] = (
                    parent_id_value
                )

                children.append(child)

        print("Child splitting completed!")
        print(
            f"Number of child chunks: {len(children)}"
        )

        return parents, children
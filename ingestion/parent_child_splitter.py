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

        parents = self.parent_splitter.split_documents(
            documents
        )

        children = []

        for parent_id, parent in enumerate(parents):

            child_chunks = self.child_splitter.split_documents(
                [parent]
            )

            for child in child_chunks:

                child.metadata["parent_id"] = parent_id

                children.append(child)

        print("Parent-child splitting completed!")

        print(
            f"Number of parent chunks: {len(parents)}"
        )

        print(
            f"Number of child chunks: {len(children)}"
        )

        return parents, children
class ParentStore:

    def __init__(self, parents):

        self.parents = {}

        for index, parent in enumerate(parents):

            parent_id = f"parent_{index}"

            self.parents[parent_id] = parent

        print("Parent store created successfully!")
        print(
            f"Number of parent documents: {len(self.parents)}"
        )

    def get_parent(self, parent_id):

        return self.parents.get(parent_id)

    def get_parents(self, parent_ids):

        results = []

        for parent_id in parent_ids:

            parent = self.get_parent(parent_id)

            if parent is not None:

                results.append(parent)

        return results
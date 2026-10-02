from memory.conversation_memory import ConversationMemory


class GraphQuestionHandler:

    def __init__(self, max_history=5):

        self.memory = ConversationMemory(
            max_history=max_history
        )

        self.last_search_question = None

        self.follow_up_prefixes = [
            "what about",
            "how about",
            "and what about",
            "what about them",
            "what about it",
            "how about them",
            "how about it",
            "tell me more",
            "and them",
            "and it",
            "what else",
            "why is that",
            "how does that",
            "how about children",
            "what about children"
        ]

        print(
            "Graph RAG question handler initialized successfully!"
        )

    def is_follow_up(self, question):

        normalized = question.strip().lower()

        for prefix in self.follow_up_prefixes:

            if normalized.startswith(prefix):
                return True

        return False

    def process_question(self, question):

        normalized = question.strip()

        if not normalized:

            return {
                "search_question": "",
                "needs_clarification": True
            }

        if self.is_follow_up(normalized):

            if self.last_search_question:

                search_question = (
                    f"{self.last_search_question} "
                    f"with specific focus on "
                    f"{normalized}"
                )

                self.last_search_question = (
                    search_question
                )

                return {
                    "search_question": search_question,
                    "needs_clarification": False
                }

            return {
                "search_question": normalized,
                "needs_clarification": True
            }

        self.last_search_question = normalized

        return {
            "search_question": normalized,
            "needs_clarification": False
        }

    def add_to_memory(
        self,
        question,
        answer
    ):

        self.memory.add_conversation(
            question,
            answer
        )

    def get_history(self):

        return self.memory.get_history()

    def clear_memory(self):

        self.memory.clear()

        self.last_search_question = None

        print(
            "\nGraph RAG memory cleared."
        )
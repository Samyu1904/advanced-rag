from memory.conversation_memory import ConversationMemory


class CRAGQuestionHandler:

    def __init__(self, max_history=5):

        self.memory = ConversationMemory(
            max_history=max_history
        )

        # Stores the complete resolved search context.
        self.last_search_question = None

        print(
            "CRAG question handler initialized successfully!"
        )

    def is_follow_up(self, question):

        follow_up_words = {
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
        }

        lower_question = question.lower().strip()

        return any(
            lower_question.startswith(word)
            for word in follow_up_words
        )

    def process_question(self, question):

        question = question.strip()

        if not question:

            return {
                "question": question,
                "search_question": question,
                "needs_clarification": True
            }

        is_follow_up = self.is_follow_up(
            question
        )

        if is_follow_up:

            if self.last_search_question is None:

                return {
                    "question": question,
                    "search_question": question,
                    "needs_clarification": True
                }

            search_question = (
                f"{self.last_search_question} "
                f"with specific focus on "
                f"{question}"
            )

            # Store the complete resolved context.
            self.last_search_question = search_question

            return {
                "question": question,
                "search_question": search_question,
                "needs_clarification": False
            }

        # New independent question.
        self.last_search_question = question

        return {
            "question": question,
            "search_question": question,
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

        # Clear the resolved search context too.
        self.last_search_question = None

        print(
            "CRAG conversation memory cleared."
        )
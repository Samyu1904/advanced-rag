from memory.conversation_memory import ConversationMemory


class AdvancedQuestionHandler:

    def __init__(self):

        self.memory = ConversationMemory()

        print(
            "Advanced question handler "
            "initialized successfully!"
        )

    def is_follow_up(self, question):

        follow_up_words = {
            "what about",
            "how about",
            "and what about",
            "what about them",
            "what about it",
            "how about them",
            "how about it"
        }

        lower_question = question.lower()

        return any(
            lower_question.startswith(word)
            for word in follow_up_words
        )

    def get_context_question(self):

        history = self.memory.get_history()

        if not history:
            return None

        for item in reversed(history):

            previous_question = item["question"]

            if not self.is_follow_up(
                previous_question
            ):
                return previous_question

        return None

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

            context_question = (
                self.get_context_question()
            )

            if context_question is None:

                return {
                    "question": question,
                    "search_question": question,
                    "needs_clarification": True
                }

            search_question = (
                f"{context_question} "
                f"with specific focus on "
                f"{question}"
            )

            return {
                "question": question,
                "search_question": search_question,
                "needs_clarification": False
            }

        return {
            "question": question,
            "search_question": question,
            "needs_clarification": False
        }

    def add_to_memory(self, question, answer):

        self.memory.add_conversation(
            question,
            answer
        )

    def clear_memory(self):

        self.memory.clear()

        print(
            "Advanced conversation memory "
            "cleared."
        )
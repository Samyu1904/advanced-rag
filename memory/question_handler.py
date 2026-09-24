class QuestionHandler:

    def __init__(self, conversation_memory):
        self.conversation_memory = conversation_memory

    def is_incomplete(self, question):

        question = question.strip().lower()

        incomplete_patterns = [
            "what about",
            "how about",
            "what are its",
            "what is its",
            "what are their",
            "what is their",
            "tell me more",
            "more about this",
            "more about that"
        ]

        for pattern in incomplete_patterns:

            if question.startswith(pattern):
                return True

        return False

    def resolve_follow_up(
        self,
        question,
        previous_question
    ):

        question_lower = question.lower().strip()

        # Example:
        # Previous: What is air pollution?
        # Follow-up: What are its health effects?

        if question_lower.startswith(
            "what are its"
        ):

            topic = previous_question

            if topic.lower().startswith(
                "what is "
            ):
                topic = topic[8:]

            elif topic.lower().startswith(
                "what are "
            ):
                topic = topic[9:]

            return question.replace(
                "its",
                topic,
                1
            )

        # Example:
        # Previous: What is Python?
        # Follow-up: What are its advantages?

        if question_lower.startswith(
            "what is its"
        ):

            topic = previous_question

            if topic.lower().startswith(
                "what is "
            ):
                topic = topic[8:]

            return question.replace(
                "its",
                topic,
                1
            )

        # Example:
        # Previous: What are the health effects
        # of air pollution?
        # Follow-up: What about children?

        if question_lower.startswith(
            "what about"
        ):

            topic = previous_question

            return (
                "What are the effects of "
                + topic.replace("What are ", "")
                + " on "
                + question[11:].strip()
                + "?"
            )

        # Example:
        # Previous: What is Python?
        # Follow-up: What are the keywords?

        if question_lower.startswith(
            "what are the"
        ):

            topic = previous_question

            if topic.lower().startswith(
                "what is "
            ):
                topic = topic[8:]

            return (
                question[:-1]
                + " of "
                + topic
                + "?"
            )

        return question

    def process_question(self, question):

        if not self.is_incomplete(question):

            return question, False

        previous_question = (
            self.conversation_memory.get_last_question()
        )

        if previous_question is None:

            return None, True

        resolved_question = self.resolve_follow_up(
            question,
            previous_question
        )

        return resolved_question, False
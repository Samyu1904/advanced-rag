
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

    def get_original_topic(self):

        history = self.conversation_memory.get_history()

        if not history:
            return None

        return history[0]["question"]

    def extract_topic(self, question):

        topic = question.strip()

        topic_lower = topic.lower()

        # Remove question prefixes
        prefixes = [
            "what is ",
            "what are ",
            "tell me about "
        ]

        for prefix in prefixes:

            if topic_lower.startswith(prefix):

                topic = topic[len(prefix):]

                break

        topic = topic.rstrip("?").strip()

        # -----------------------------------------------
        # Remove descriptive phrase "health effects of"
        # -----------------------------------------------

        topic_lower = topic.lower()

        if topic_lower.startswith(
            "the health effects of "
        ):

            topic = topic[
                len("the health effects of "):
            ]

        elif topic_lower.startswith(
            "health effects of "
        ):

            topic = topic[
                len("health effects of "):
            ]

        elif topic_lower.startswith(
            "the effects of "
        ):

            topic = topic[
                len("the effects of "):
            ]

        elif topic_lower.startswith(
            "effects of "
        ):

            topic = topic[
                len("effects of "):
            ]

        return topic.strip()

    def resolve_follow_up(
        self,
        question,
        previous_question
    ):

        question_lower = question.lower().strip()

        # ------------------------------------------------
        # Always extract the core topic from the original
        # question.
        # ------------------------------------------------

        topic = self.extract_topic(
            previous_question
        )

        # ------------------------------------------------
        # WHAT ABOUT
        # ------------------------------------------------

        if question_lower.startswith("what about"):

            subject = question[
                len("What about"):
            ].strip()

            subject = subject.rstrip("?").strip()

            return (
                "What are the effects of "
                + topic
                + " on "
                + subject
                + "?"
            )

        # ------------------------------------------------
        # HOW ABOUT
        # ------------------------------------------------

        if question_lower.startswith("how about"):

            subject = question[
                len("How about"):
            ].strip()

            subject = subject.rstrip("?").strip()

            return (
                "What are the effects of "
                + topic
                + " on "
                + subject
                + "?"
            )

        # ------------------------------------------------
        # WHAT ARE ITS
        # ------------------------------------------------

        if question_lower.startswith("what are its"):

            remaining = question[
                len("What are its"):
            ].strip()

            remaining = remaining.rstrip("?").strip()

            if remaining:

                return (
                    "What are "
                    + remaining
                    + " of "
                    + topic
                    + "?"
                )

            return (
                "What are the effects of "
                + topic
                + "?"
            )

        # ------------------------------------------------
        # WHAT IS ITS
        # ------------------------------------------------

        if question_lower.startswith("what is its"):

            remaining = question[
                len("What is its"):
            ].strip()

            remaining = remaining.rstrip("?").strip()

            if remaining:

                return (
                    "What is "
                    + remaining
                    + " of "
                    + topic
                    + "?"
                )

            return (
                "What is "
                + topic
                + "?"
            )

        # ------------------------------------------------
        # TELL ME MORE
        # ------------------------------------------------

        if question_lower.startswith("tell me more"):

            return (
                "Tell me more about "
                + topic
                + "."
            )

        # ------------------------------------------------
        # FALLBACK
        # ------------------------------------------------

        return question

    def process_question(self, question):

        # -----------------------------------------------
        # Normal complete question
        # -----------------------------------------------

        if not self.is_incomplete(question):

            return question, False

        # -----------------------------------------------
        # Get conversation history
        # -----------------------------------------------

        history = (
            self.conversation_memory.get_history()
        )

        if not history:

            return None, True

        # -----------------------------------------------
        # Use the original question as the reference.
        # -----------------------------------------------

        original_question = history[0]["question"]

        resolved_question = self.resolve_follow_up(
            question,
            original_question
        )

        return resolved_question, False


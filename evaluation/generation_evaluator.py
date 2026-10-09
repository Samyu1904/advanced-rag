import re


class GenerationEvaluator:

    def __init__(self):
        pass

    def normalize_text(self, text):

        if text is None:
            return ""

        text = text.lower()

        text = re.sub(
            r"[^a-z0-9\s]",
            " ",
            text
        )

        text = re.sub(
            r"\s+",
            " ",
            text
        )

        return text.strip()

    def word_set(self, text):

        normalized = self.normalize_text(text)

        if not normalized:
            return set()

        return set(normalized.split())

    def answer_relevance(
        self,
        question,
        answer
    ):

        question_words = self.word_set(question)
        answer_words = self.word_set(answer)

        if not question_words or not answer_words:
            return 0.0

        overlap = (
            question_words.intersection(
                answer_words
            )
        )

        return len(overlap) / len(question_words)

    def answer_correctness(
        self,
        answer,
        expected_answer
    ):

        answer_words = self.word_set(answer)
        expected_words = self.word_set(
            expected_answer
        )

        if not answer_words or not expected_words:
            return 0.0

        overlap = (
            answer_words.intersection(
                expected_words
            )
        )

        precision = (
            len(overlap) /
            len(answer_words)
        )

        recall = (
            len(overlap) /
            len(expected_words)
        )

        if precision + recall == 0:
            return 0.0

        return (
            2 * precision * recall /
            (precision + recall)
        )

    def faithfulness(
        self,
        answer,
        context
    ):

        answer_sentences = re.split(
            r"[.!?]+",
            answer
        )

        answer_sentences = [
            sentence.strip()
            for sentence in answer_sentences
            if sentence.strip()
        ]

        if not answer_sentences:
            return 0.0

        context_normalized = (
            self.normalize_text(context)
        )

        supported = 0

        for sentence in answer_sentences:

            sentence_words = self.word_set(
                sentence
            )

            if not sentence_words:
                continue

            context_words = self.word_set(
                context_normalized
            )

            overlap = (
                sentence_words.intersection(
                    context_words
                )
            )

            overlap_ratio = (
                len(overlap) /
                len(sentence_words)
            )

            if overlap_ratio >= 0.5:
                supported += 1

        return (
            supported /
            len(answer_sentences)
        )

    def evaluate(
        self,
        question,
        answer,
        context,
        expected_answer
    ):

        return {
            "faithfulness": round(
                self.faithfulness(
                    answer,
                    context
                ),
                3
            ),
            "answer_relevance": round(
                self.answer_relevance(
                    question,
                    answer
                ),
                3
            ),
            "answer_correctness": round(
                self.answer_correctness(
                    answer,
                    expected_answer
                ),
                3
            )
        }
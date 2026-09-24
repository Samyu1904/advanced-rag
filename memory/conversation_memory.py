class ConversationMemory:

    def __init__(self, max_history=5):

        self.max_history = max_history
        self.history = []

    def add_conversation(self, question, answer):

        self.history.append({
            "question": question,
            "answer": answer
        })

        if len(self.history) > self.max_history:

            self.history.pop(0)

    def get_history(self):

        return self.history

    def get_last_question(self):

        if not self.history:
            return None

        return self.history[-1]["question"]

    def clear(self):

        self.history = []
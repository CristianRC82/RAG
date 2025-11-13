from langchain_community.chat_message_histories import ChatMessageHistory

class ChatMemoryManager:

    def __init__(self):
        self.sessions = {}

    def get_session_history(self, session_id: str):
        if session_id not in self.sessions:
            self.sessions[session_id] = ChatMessageHistory()
        return self.sessions[session_id]


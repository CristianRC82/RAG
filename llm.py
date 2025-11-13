from langchain_google_genai import ChatGoogleGenerativeAI

class LLMManager:
    """Configura el modelo LLM (Gemini)."""

    def __init__(self, gcs):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
            google_api_key=gcs.api_key
        )

    def get_model(self):
        return self.llm

import os
from langchain_openai import AzureOpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

class EmbeddingFactory:
    """Crea y configura el modelo de embeddings según entorno."""
    @staticmethod
    def create_embeddings():
        provider = os.getenv("EMBEDDING_PROVIDER", "huggingface").lower()

        if provider == "azure":
            return AzureOpenAIEmbeddings(
                azure_deployment=os.getenv("AZURE_EMBBEDING_DEPLOYMENT"),
                openai_api_version=os.getenv("AZURE_API_VERSION_E"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_E"),
                api_key=os.getenv("AZURE_API_KEY_E")
            )
        elif provider == "huggingface":
            return HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")
        else:
            raise ValueError(f"Proveedor de embeddings no reconocido: {provider}")

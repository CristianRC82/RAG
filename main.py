from cloud import GoogleCloudStorageManager
from embedding import EmbeddingFactory
from vectorstore import VectorstoreManager
from llm import LLMManager
from retrieval_pipeline import RetrievalPipeline
from chat_history import ChatMemoryManager

def main():
    print("Chat iniciado (escribe 'salir' o 'exit' para terminar)\n")

    gcs = GoogleCloudStorageManager()
    embeddings = EmbeddingFactory.create_embeddings()
    vector_manager = VectorstoreManager(gcs, embeddings)
    vectorstore = vector_manager.load_or_create_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = LLMManager(gcs).get_model()

    memory_manager = ChatMemoryManager()
    pipeline = RetrievalPipeline(retriever, llm, memory_manager)

    session_id = "chat_usuario_001"

    while True:
        question = input("\nTú: ").strip()
        if question.lower() in ["exit", "salir"]:
            print("\nChat Terminado")
            break

        try:
            answer = pipeline.ask(question, session_id)
            print(f"Asistente: {answer}")
        except Exception as e:
            print(f"Error al procesar la pregunta: {e}")

if __name__ == "__main__":
    main()
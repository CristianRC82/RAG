from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser

class RetrievalPipeline:
    """Pipeline RAG con memoria basada en placeholder."""

    def __init__(self, retriever, llm, memory_manager):
        self.memory_manager = memory_manager

        self.prompt = ChatPromptTemplate.from_messages([
            ("system",
            "Eres un asistente financiero experto. "
            "Mantén coherencia con el historial previo y responde con base en él. "
            "Si el usuario hace referencia a algo anterior (por ejemplo 'ese seguro'), usa el historial para entenderlo."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "Basado en el siguiente contexto:\n{context}\n\nResponde a la pregunta:\n{question}")
        ])

        base_chain = (
            {
                "context": lambda x: self._format_docs(retriever.invoke(x["question"])),
                "question": lambda x: x["question"],
                "chat_history": lambda _: []
            }
            | self.prompt
            | llm
            | StrOutputParser()
        )

        self.chain = RunnableWithMessageHistory(
            base_chain,
            memory_manager.get_session_history,
            input_messages_key="question",
            history_messages_key="chat_history",
        )

    def _format_docs(self, docs):
        if not docs:
            return "No se encontró contexto relevante."
        return "\n\n".join(doc.page_content for doc in docs)


    def ask(self, question: str, session_id: str):
        history = self.memory_manager.get_session_history(session_id)
        print("\n🧠 Historial actual:")
        for msg in history.messages:
            print(f" - {msg.type}: {msg.content}")
        return self.chain.invoke(
            {"question": question},
            config={"configurable": {"session_id": session_id}}
        )


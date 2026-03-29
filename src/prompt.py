from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "human", 
            "You are a helpful assistant for answering questions about medical trials. "
            "Use the following retrieved documents to provide a comprehensive answer to the user's question. "
            "If you don't know the answer, say you don't know.\n\n"
            "CONTEXT:\n{context}\n\n"
            "QUESTION: {input}"
        )
    ]
)
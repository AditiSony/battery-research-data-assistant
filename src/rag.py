from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

from settings import DEVICE, EMBED_MODEL, LLM_MODEL, VECTOR_DB_DIR, K


def format_docs_with_id(docs):
    """Turns document objects into a numbered string for the LLM."""
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown")
        formatted.append(f"Source [{i}] (File: {source}):\n{doc.page_content}")
    return "\n\n".join(formatted)


def get_rag_chain():
    # initialize Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": DEVICE},
        encode_kwargs={"normalize_embeddings": True},
        query_encode_kwargs={
            "prompt": "Represent this sentence for searching relevant passages: "
        },
    )

    # Load the existing Vector DB and retrieve top-k abstracts
    vector_db = Chroma(persist_directory=VECTOR_DB_DIR, embedding_function=embeddings)
    retriever = vector_db.as_retriever(search_kwargs={"k": K})

    # Setup Local LLM (Ollama)
    llm = ChatOllama(model=LLM_MODEL, temperature=0)

    # re-write query (if needed) based on chat history
    context_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    context_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", context_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    question_generator = context_prompt | llm | RunnableLambda(lambda x: x.content)

    # Main RAG Prompt
    system_prompt = (
        "You are an expert battery researcher. Answer the question using ONLY the provided context. "
        "Every time you reference a fact from the context, you MUST cite the source number "
        "in square brackets immediately following the fact (e.g., 'LFP batteries have high safety [1].'). "
        "At the end of your response, provide a 'References' list mapping the numbers to filenames.\n\n"
        "Context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # Final RAG chain: Rephrase -> Retrieve -> Answer
    rag_chain = (
        RunnablePassthrough.assign(
            # re-write the question using history
            standalone_question=question_generator,
        ).assign(
            # retrieve docs using the standalone question and format them
            context=lambda x: format_docs_with_id(
                retriever.invoke(x["standalone_question"])
            )
        )
        | prompt
        | llm
    )

    return rag_chain


# function that stores chat history and runs chat loop until exit
def run_chat_loop():
    print(f"\n--- Research Assistant (Model: {LLM_MODEL}) ---")
    print("Type 'exit' to quit.\n")

    rag_chain = get_rag_chain()
    chat_history = []

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit", "q"]:
            print("Goodbye!")
            break

        # Process the query through the chain
        result = rag_chain.invoke({"input": user_input, "chat_history": chat_history})
        answer = result.content
        print(f"\nAssistant: {answer}\n")

        # Update chat history for the next turn
        chat_history.extend(
            [HumanMessage(content=user_input), AIMessage(content=answer)]
        )


if __name__ == "__main__":
    run_chat_loop()

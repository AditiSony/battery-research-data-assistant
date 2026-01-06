from langchain_chroma import Chroma
from langchain_classic.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

from settings import DEVICE, EMBED_MODEL, LLM_MODEL, VECTOR_DB_DIR, K


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

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, context_prompt
    )

    # Main RAG Prompt
    system_prompt = (
        "You are a Battery Research Assistant. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, say that you don't know.\n\n"
        "Context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # combine retrieved docs into answer
    qa_chain = create_stuff_documents_chain(llm, prompt)

    # Final RAG chain: Rephrase -> Retrieve -> Answer
    return create_retrieval_chain(history_aware_retriever, qa_chain)


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

        # Process the question through the chain
        result = rag_chain.invoke({"input": user_input, "chat_history": chat_history})

        answer = result["answer"]
        print(f"\nAssistant: {answer}\n")

        # Update chat history for the next turn
        chat_history.extend(
            [HumanMessage(content=user_input), AIMessage(content=answer)]
        )


if __name__ == "__main__":
    run_chat_loop()

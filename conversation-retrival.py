from operator import itemgetter

from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

llm = ChatOllama(model="qwen3:0.6b", temperature=0.4)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer the user's question based on the context provided. Context: {context}",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)

retriever_prompt = ChatPromptTemplate.from_messages(
    [
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        (
            "human",
            "Given the above chat history, generate a question to extract the most relevant information from the vector database. Only return the question, no other text.",
        ),
    ]
)


def get_documents_from_url(url: str):
    docs = WebBaseLoader(url).load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    return text_splitter.split_documents(docs)


def get_db():
    return FAISS.from_documents(
        get_documents_from_url(
            "https://blog.langchain.com/langchain-expression-language/"
        ),
        OllamaEmbeddings(model="qwen3-embedding:0.6b"),
    )


def format_docs(docs):
    """Format a list of documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)


def get_retrieval_chain():
    retriever = get_db().as_retriever()

    history_aware_retriever = (
        {"chat_history": itemgetter("chat_history"), "input": itemgetter("input")}
        | retriever_prompt
        | llm
        | StrOutputParser()
        | retriever
    )

    # Create the full retrieval chain
    return (
        {
            "context": RunnablePassthrough() | history_aware_retriever | format_docs,
            "input": itemgetter("input"),
            "chat_history": itemgetter("chat_history"),
        }
        | prompt
        | llm
        | StrOutputParser()
    )


def process_chat(user_input: str, chat_history: list[tuple[str, str]]):
    return get_retrieval_chain().invoke(
        {"input": user_input, "chat_history": chat_history}
    )


if __name__ == "__main__":
    chat_history = []
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        response = process_chat(user_input, chat_history)

        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response))

        print(f"Assistant: {response}")

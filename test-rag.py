import os

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

embeddings = OllamaEmbeddings(model="qwen3-embedding:0.6b")


def init_vectorstore():
    loader = PyPDFLoader("general-relativity.pdf")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=500)
    texts = text_splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(texts, embeddings)
    vectorstore.save_local("general-relativity-faiss")


if not os.path.exists("general-relativity-faiss"):
    init_vectorstore()

vectorstore = FAISS.load_local(
    "general-relativity-faiss", embeddings, allow_dangerous_deserialization=True
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that can answer using the context: {context}.",
        ),
        ("human", "{input}"),
    ]
)


def format_docs(docs):
    """Format a list of documents into a single string."""

    strings = []
    for doc in docs:
        strings.append(
            f"""
Page {doc.metadata["page_label"]}:
--------------------------------
{doc.page_content}
            """
        )
    return "\n\n".join(strings)


llm = ChatOllama(model="qwen3:0.6b")
chain = (
    {
        "input": RunnablePassthrough(),
        "context": vectorstore.as_retriever() | format_docs,
    }
    | prompt
    | llm
)

while True:
    user_input = input("Enter a question: ")
    if user_input == "exit":
        break
    print(chain.invoke(user_input).content)

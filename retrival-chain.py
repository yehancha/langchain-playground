from dotenv import load_dotenv
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings

load_dotenv()

llm = ChatOllama(model="qwen3:0.6b", temperature=0.4)

prompt = ChatPromptTemplate.from_template(
    """
    Answer the user's question based on the context provided.
    Context: {context}
    Question: {input}
    """
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


doc_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(get_db().as_retriever(), doc_chain)

print(retrieval_chain.invoke({"input": "What is LCEL?"}))

from dotenv import load_dotenv
from langchain_ollama import ChatOllama

load_dotenv()

llm = ChatOllama(model="qwen3:0.6b")

print(llm.invoke("Hello, how are you?").content)

from langchain_ollama import ChatOllama

llm = ChatOllama(model="qwen3:0.6b")

print(llm.invoke("Hello, how are you?").content)

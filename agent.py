from langchain_ollama import ChatOllama
from dotenv import load_dotenv

load_dotenv()

llm = ChatOllama(model="qwen3:0.6b", temperature=0.0)

def process_prompt(prompt: str):
    return llm.invoke(prompt)

if __name__ == "__main__":
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        print(process_prompt(user_input).content)
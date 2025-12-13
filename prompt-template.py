from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a chef. Create a recipe with a user given ingredients."),
        ("human", "I have {input}."),
    ]
)
llm = ChatOllama(model="qwen3:0.6b")
chain = prompt | llm

print(chain.invoke({"input": "carrot"}).content)

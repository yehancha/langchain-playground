from operator import itemgetter
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_tavily import TavilySearch
from dotenv import load_dotenv

load_dotenv()

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a friendly and helpful assistant called Max."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

tools = [TavilySearch()]

llm = ChatOllama(model="qwen3:0.6b", temperature=0.7)
llm.bind_tools(tools)
chain = (
    {
        "input": itemgetter("input"),
        "chat_history": itemgetter("chat_history"),
    }
    | prompt
    | llm
    | StrOutputParser()
)

# agent = create_agent(llm, tools, system_prompt=SystemMessage(content="You are a friendly and helpful assistant called Max. You can search anything inlcuding whether using search tools."))

def process_chat(user_input: str, chat_history: list[tuple[str, str]]):
    return chain.invoke({ "input": user_input, "chat_history": chat_history })

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
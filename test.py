from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

load_dotenv()


@tool
def check_weather(location: str):
    """
    Check the weather of a given location.
    Args:
        location: The location to check the weather of.
    Returns:
        The weather in the location.
    """
    print(f"Checking weather for {location}")
    return f"The weather in {location} is sunny."


@tool
def check_stock_price(symbol: str):
    """
    Check the latest stock price of a given symbol.
    Args:
        symbol: The symbol to check the stock price of.
    Returns:
        The latest stock price of the symbol.
    """
    print(f"Checking price for {symbol}")
    return f"The latest price of {symbol} is $15."


tools = {
    "check_weather": check_weather,
    "check_stock_price": check_stock_price,
}

prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content="You are an ai who talks with a human user. You have access to message history."
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)
llm = ChatOllama(model="qwen3:0.6b")
# llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
llm = llm.bind_tools(tools.values())
chain = prompt | llm
history = []


def process_tool_calls(response: BaseMessage):
    tool_calls = response.tool_calls
    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        history.append(
            ToolMessage(
                content=tools[tool_name].invoke(tool_args), tool_call_id=tool_call["id"]
            )
        )


def process_user_input(user_input: str):

    while True:
        response = chain.invoke({"input": user_input, "history": history})

        if response.tool_calls:
            process_tool_calls(response)
            continue
        else:
            break

    history.append(HumanMessage(content=user_input))
    history.append(AIMessage(content=response.content))

    return response.content


while True:
    user_input = input("You: ")
    if user_input == "exit":
        break

    print(f"AI: {process_user_input(user_input)}")

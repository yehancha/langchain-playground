from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode

load_dotenv()

# Initialize LLM
llm = ChatOllama(model="qwen3:0.6b", temperature=0.4)

# Initialize Tavily search tool
tavily_tool = TavilySearchResults(max_results=5)

# Temperature comparison tool
@tool
def compare_temperatures(location1: str, temperature1: float, location2: str, temperature2: float) -> str:
    """Compare temperatures between two locations. Use this after searching for temperatures of both locations.
    
    Args:
        location1: Name of the first location
        temperature1: Temperature of the first location (in the same unit as temperature2)
        location2: Name of the second location
        temperature2: Temperature of the second location (in the same unit as temperature1)
    
    Returns:
        A string describing which location is colder/warmer and by how much.
    """
    diff = abs(temperature1 - temperature2)
    
    if temperature1 < temperature2:
        return f"{location1} is colder than {location2} by {diff:.1f} degrees. {location1}: {temperature1}°, {location2}: {temperature2}°"
    elif temperature1 > temperature2:
        return f"{location2} is colder than {location1} by {diff:.1f} degrees. {location1}: {temperature1}°, {location2}: {temperature2}°"
    else:
        return f"{location1} and {location2} have the same temperature: {temperature1}°"

tools = [tavily_tool, compare_temperatures]

# Create ToolNode for executing tools
tool_node = ToolNode(tools)

# System prompt to guide agent behavior
SYSTEM_PROMPT = """You are a helpful AI assistant. You are talking to a user. When the user shares information about themselves (like saying "I am Yehan"), remember that this information is about THE USER, not about you. When answering questions about the user, refer to them using "you" or "your" (for example, say "Your name is Yehan" not "I am Yehan"). Remember information shared in the conversation and use it to answer questions. Only use the web search tool when you need current, real-time information that isn't available in our conversation history."""

# Create agent node that calls LLM with tools bound
def agent_node(state: MessagesState):
    """Agent node that calls the LLM with tools bound."""
    system_message = SystemMessage(content=SYSTEM_PROMPT)
    response = llm.bind_tools(tools).invoke([system_message] + state["messages"])
    return {"messages": [response]}

# Conditional routing function
def should_continue(state: MessagesState) -> str:
    """Determine whether to continue to tools or end."""
    messages = state["messages"]
    last_message = messages[-1]
    
    # If the last message is an AIMessage with tool calls, route to tools
    if isinstance(last_message, AIMessage) and hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    # Otherwise, end the conversation
    return "end"

# Build the graph
graph_builder = StateGraph(MessagesState)
graph_builder.add_node("agent", agent_node)
graph_builder.add_node("tools", tool_node)

# Add edges
graph_builder.add_edge(START, "agent")
graph_builder.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        "end": END,
    }
)
graph_builder.add_edge("tools", "agent")  # Loop back to agent after tools

# Compile the graph
graph = graph_builder.compile()

def main():
    """Main interactive loop for the agent."""
    print("LangGraph Web Search Agent")
    print("Type 'exit' to quit\n")
    
    # Maintain conversation history
    messages = []
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if user_input.lower() == "exit":
                print("Goodbye!")
                break
            
            if not user_input:
                continue
            
            # Add user message to history
            messages.append(HumanMessage(content=user_input))
            
            # Invoke the graph with conversation history
            result = graph.invoke({"messages": messages})
            
            # Update messages with the full conversation
            messages = result["messages"]
            
            # Get the last message (should be the AI response)
            last_message = messages[-1]
            
            # Only print content if there are no tool calls
            # (tool calls should execute and loop back to agent)
            if not (isinstance(last_message, AIMessage) and hasattr(last_message, "tool_calls") and last_message.tool_calls):
                print(f"\nAssistant: {last_message.content}\n")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")
            print("Please check your TAVILY_API_KEY in .env file if you're getting API errors.\n")

if __name__ == "__main__":
    main()

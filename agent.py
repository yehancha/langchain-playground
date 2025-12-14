from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode

load_dotenv()

# Initialize LLM
llm = ChatOllama(model="qwen3:0.6b", temperature=0.4)

# Initialize Tavily search tool
tavily_tool = TavilySearchResults(max_results=5)
tools = [tavily_tool]

# Create ToolNode for executing tools
tool_node = ToolNode(tools)

# Create agent node that calls LLM with tools bound
def agent_node(state: MessagesState):
    """Agent node that calls the LLM with tools bound."""
    response = llm.bind_tools(tools).invoke(state["messages"])
    return {"messages": [response]}

# Conditional routing function
def should_continue(state: MessagesState) -> str:
    """Determine whether to continue to tools or end."""
    messages = state["messages"]
    last_message = messages[-1]
    
    # If the last message has tool calls, route to tools
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
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
            print(f"\nAssistant: {last_message.content}\n")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")
            print("Please check your TAVILY_API_KEY in .env file if you're getting API errors.\n")

if __name__ == "__main__":
    main()

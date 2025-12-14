from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# Initialize LLM
llm = ChatOllama(model="qwen3:0.6b", temperature=0.4)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.4)

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

# Create a tool name to tool mapping for easy lookup
tool_map = {tool.name: tool for tool in tools}

# System prompt to guide agent behavior
SYSTEM_PROMPT = """You are a helpful AI assistant. You are talking to a user. When the user shares information about themselves (like saying "I am Yehan"), remember that this information is about THE USER, not about you. When answering questions about the user, refer to them using "you" or "your" (for example, say "Your name is Yehan" not "I am Yehan"). Remember information shared in the conversation and use it to answer questions. Only use the web search tool when you need current, real-time information that isn't available in our conversation history."""

def execute_tools(tool_calls):
    """Execute tool calls and return ToolMessage objects.
    
    Args:
        tool_calls: List of tool call objects from AIMessage
        
    Returns:
        List of ToolMessage objects with tool execution results
    """
    tool_messages = []
    for tool_call in tool_calls:
        # Handle both dict and object tool calls
        if isinstance(tool_call, dict):
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_call_id = tool_call["id"]
        else:
            # Assume it's an object with attributes
            tool_name = tool_call.name
            tool_args = tool_call.args
            tool_call_id = tool_call.id
        
        # Get the tool from the mapping
        tool = tool_map.get(tool_name)
        if tool is None:
            error_msg = f"Tool {tool_name} not found"
            tool_messages.append(
                ToolMessage(
                    content=error_msg,
                    tool_call_id=tool_call_id,
                    name=tool_name
                )
            )
            continue
        
        # Execute the tool
        try:
            result = tool.invoke(tool_args)
            tool_messages.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call_id,
                    name=tool_name
                )
            )
        except Exception as e:
            error_msg = f"Error executing tool {tool_name}: {str(e)}"
            tool_messages.append(
                ToolMessage(
                    content=error_msg,
                    tool_call_id=tool_call_id,
                    name=tool_name
                )
            )
    
    return tool_messages

def run_agent(messages):
    """Run the agent loop: call LLM, execute tools if needed, loop until final response.
    
    Args:
        messages: List of messages (conversation history)
        
    Returns:
        Updated list of messages with agent response
    """
    system_message = SystemMessage(content=SYSTEM_PROMPT)
    
    while True:
        # Call LLM with tools bound
        response = llm.bind_tools(tools).invoke([system_message] + messages)
        messages.append(response)
        
        # Check if the response contains tool calls
        if hasattr(response, "tool_calls") and response.tool_calls:
            # Execute tools
            tool_messages = execute_tools(response.tool_calls)
            messages.extend(tool_messages)
            # Loop back to call LLM again with tool results
        else:
            # No tool calls, return the final response
            break
    
    return messages

def main():
    """Main interactive loop for the agent."""
    print("LangChain Web Search Agent")
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
            
            # Run the agent loop (calls LLM, executes tools, loops until final response)
            messages = run_agent(messages)
            
            # Get the last message (should be the AI response)
            last_message = messages[-1]
            
            # Print the final response
            if isinstance(last_message, AIMessage):
                print(f"\nAssistant: {last_message.content}\n")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")
            print("Please check your TAVILY_API_KEY in .env file if you're getting API errors.\n")

if __name__ == "__main__":
    main()

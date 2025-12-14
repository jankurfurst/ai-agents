"""
Universal LangChain Agent with MCP Server Integration
Uses langchain-mcp-adapters for connecting to custom MCP servers
"""

import asyncio
import os
from dotenv import load_dotenv

from langchain_ollama import ChatOllama
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# Load environment variables
load_dotenv()

# ============================================
# CONFIGURATION
# ============================================

# MCP Server Configuration
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8002/mcp")
MCP_SERVER_NAME = os.getenv("MCP_SERVER_NAME", "mcp-server")
MCP_TRANSPORT = os.getenv("MCP_TRANSPORT", "streamable_http")  # streamable_http | sse | stdio

# LLM Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")  # ollama | openai
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1:8b")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# System Prompt
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", """You are a helpful AI assistant with access to various tools.
Use the available tools to help answer user questions accurately.
Always explain your reasoning and provide clear, concise answers.""")


def get_mcp_server_config() -> dict:
    """
    Build MCP server configuration based on transport type.

    Returns:
        dict: Configuration dictionary for MultiServerMCPClient
    """
    if MCP_TRANSPORT == "streamable_http":
        return {
            MCP_SERVER_NAME: {
                "transport": "streamable_http",
                "url": MCP_SERVER_URL,
            }
        }
    elif MCP_TRANSPORT == "sse":
        return {
            MCP_SERVER_NAME: {
                "transport": "sse",
                "url": MCP_SERVER_URL,
            }
        }
    elif MCP_TRANSPORT == "stdio":
        # For stdio, we need command and args
        command = os.getenv("MCP_STDIO_COMMAND", "python")
        args = os.getenv("MCP_STDIO_ARGS", "server.py").split()
        return {
            MCP_SERVER_NAME: {
                "transport": "stdio",
                "command": command,
                "args": args,
            }
        }
    else:
        raise ValueError(f"Unsupported MCP transport: {MCP_TRANSPORT}")


def create_llm():
    """
    Create and configure the LLM instance.

    Returns:
        ChatOllama: Configured LLM instance
    """
    
    return ChatOllama(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        base_url=OLLAMA_BASE_URL,
    )


def log_stream_event(node_name: str, event: dict) -> None:
    """
    Log streaming event details.

    Args:
        node_name: Name of the current node in the graph
        event: Event dictionary from the stream
    """
    print(f"\n{'='*60}")
    print(f"NODE: {node_name}")
    print(f"{'='*60}")

    for key, value in event.items():
        if key == "messages":
            for msg in value:
                msg_type = type(msg).__name__
                print(f"\n  MESSAGE TYPE: {msg_type}")

                if isinstance(msg, AIMessage):
                    if msg.content:
                        print(f"  CONTENT: {msg.content}")
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        print(f"  TOOL CALLS:")
                        for tc in msg.tool_calls:
                            print(f"    - Name: {tc.get('name', 'unknown')}")
                            print(f"      Args: {tc.get('args', {})}")
                            print(f"      ID: {tc.get('id', 'unknown')}")

                elif isinstance(msg, ToolMessage):
                    print(f"  TOOL NAME: {msg.name}")
                    print(f"  TOOL CALL ID: {msg.tool_call_id}")
                    content_preview = str(msg.content)[:500]
                    if len(str(msg.content)) > 500:
                        content_preview += "..."
                    print(f"  RESULT: {content_preview}")

                elif isinstance(msg, HumanMessage):
                    print(f"  CONTENT: {msg.content}")

                else:
                    print(f"  CONTENT: {getattr(msg, 'content', str(msg))}")


async def create_mcp_agent(tools: list):
    """
    Create a LangChain agent with MCP tools.

    Args:
        tools: List of MCP tools

    Returns:
        Compiled agent ready to process messages
    """
    print(f"\n{'='*60}")
    print("AVAILABLE MCP TOOLS")
    print(f"{'='*60}")
    for tool in tools:
        print(f"  - {tool.name}: {tool.description}")
    print()

    # Create LLM
    llm = create_llm()

    # Create ReAct agent with MCP tools
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
    )

    return agent


async def run_agent(
    agent,
    user_message: str,
    stream: bool = True,
    conversation_history: list | None = None
) -> str:
    """
    Run the agent with a user message.

    Args:
        agent: Compiled LangChain agent
        user_message: User's input message
        stream: Whether to stream the response with logging
        conversation_history: Optional list of previous messages

    Returns:
        str: Agent's final response
    """
    # Build messages
    messages = conversation_history or []
    messages.append(HumanMessage(content=user_message))

    input_data = {"messages": messages}

    if stream:
        print(f"\n{'#'*60}")
        print(f"USER: {user_message}")
        print(f"{'#'*60}")

        final_response = ""

        # Stream through the agent execution
        async for event in agent.astream(input_data, stream_mode="updates"):
            for node_name, node_output in event.items():
                log_stream_event(node_name, node_output)

                # Extract final response from agent node
                if node_name == "model" and "messages" in node_output:
                    for msg in node_output["messages"]:
                        if isinstance(msg, AIMessage) and msg.content:
                            final_response = msg.content

        print(f"\n{'#'*60}")
        print("FINAL RESPONSE:")
        print(f"{'#'*60}")
        print(final_response)

        return final_response
    else:
        # Non-streaming execution
        result = await agent.ainvoke(input_data)

        # Extract final message
        if result.get("messages"):
            last_message = result["messages"][-1]
            if isinstance(last_message, AIMessage):
                return last_message.content

        return str(result)


async def main():
    """Main function demonstrating LangChain + MCP integration."""

    print("="*60)
    print("LangChain Agent with MCP Server Integration")
    print("="*60)
    print(f"MCP Server: {MCP_SERVER_URL}")
    print(f"Transport: {MCP_TRANSPORT}")
    print(f"LLM Provider: {LLM_PROVIDER}")
    print(f"LLM Model: {LLM_MODEL}")
    print("="*60)

    # Get MCP server configuration
    server_config = get_mcp_server_config()

    # Create MCP client and get tools
    client = MultiServerMCPClient(server_config)
    tools = await client.get_tools()

    # Create agent with MCP tools
    agent = await create_mcp_agent(tools)

    # Example conversation
    conversation_history = []

    # Run example queries
    queries = [
        "What tools do you have available?",
        # Add more example queries based on your MCP server
    ]

    for query in queries:
        response = await run_agent(
            agent,
            query,
            stream=True,
            conversation_history=conversation_history
        )

        # Update conversation history
        conversation_history.append(HumanMessage(content=query))
        conversation_history.append(AIMessage(content=response))

        print("\n" + "-"*60 + "\n")


async def interactive_mode():
    """
    Run the agent in interactive mode for user input.
    """
    print("="*60)
    print("Interactive Mode - LangChain Agent with MCP")
    print("Type 'quit' or 'exit' to stop")
    print("="*60)
    print(f"MCP Server: {MCP_SERVER_URL}")
    print(f"LLM Provider: {LLM_PROVIDER}")
    print(f"LLM Model: {LLM_MODEL}")
    print("="*60)

    server_config = get_mcp_server_config()

    # Create MCP client and get tools
    client = MultiServerMCPClient(server_config)
    tools = await client.get_tools()

    # Create agent with MCP tools
    agent = await create_mcp_agent(tools)
    conversation_history = []

    while True:
        try:
            user_input = input("\nYou: ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break

            if not user_input:
                continue

            response = await run_agent(
                agent,
                user_input,
                stream=True,
                conversation_history=conversation_history
            )

            conversation_history.append(HumanMessage(content=user_input))
            conversation_history.append(AIMessage(content=response))

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        asyncio.run(interactive_mode())
    else:
        asyncio.run(main())

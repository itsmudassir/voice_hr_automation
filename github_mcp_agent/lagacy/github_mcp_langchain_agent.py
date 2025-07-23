#!/usr/bin/env python3
"""
GitHub MCP + LangChain + OpenAI Agent
Conversational interface to GitHub using MCP tools
"""

import asyncio
import os
import json
from dotenv import load_dotenv
from typing import List, Dict, Any

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import Tool, StructuredTool

# MCP imports
from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession

load_dotenv()


class GitHubMCPAgent:
    """GitHub MCP Agent using LangChain and OpenAI"""
    
    def __init__(self):
        self.github_pat = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if not self.github_pat:
            raise ValueError("Missing GITHUB_PERSONAL_ACCESS_TOKEN in .env")
        if not self.openai_api_key:
            raise ValueError("Missing OPENAI_API_KEY in .env")
        
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0,
            openai_api_key=self.openai_api_key
        )
        
        self.tools = []
        self.agent = None
        self.session = None
    
    async def initialize(self):
        """Initialize MCP session and load tools"""
        print("🔧 Initializing GitHub MCP Agent...")
        
        # Create MCP connection context managers
        self.mcp_context = streamablehttp_client(
            "https://api.githubcopilot.com/mcp/",
            headers={"Authorization": f"Bearer {self.github_pat}"}
        )
        
        # Enter the context
        self.read, self.write, _ = await self.mcp_context.__aenter__()
        
        # Create session
        self.session = ClientSession(self.read, self.write)
        await self.session.__aenter__()
        await self.session.initialize()
        
        # Load tools
        await self._load_tools()
        
        # Create agent
        self._create_agent()
        
        print(f"✅ Agent initialized with {len(self.tools)} GitHub tools")
    
    async def _load_tools(self):
        """Convert MCP tools to LangChain tools"""
        tools_response = await self.session.list_tools()
        
        # Convert each MCP tool to a LangChain tool
        for mcp_tool in tools_response.tools:
            # Create async function for the tool
            async def tool_func(session=self.session, tool_name=mcp_tool.name, **kwargs):
                try:
                    result = await session.call_tool(tool_name, kwargs)
                    if result.content:
                        return json.loads(result.content[0].text)
                    return {"error": "No content returned"}
                except Exception as e:
                    return {"error": str(e)}
            
            # Create LangChain tool
            langchain_tool = Tool(
                name=mcp_tool.name,
                description=mcp_tool.description if hasattr(mcp_tool, 'description') else f"GitHub tool: {mcp_tool.name}",
                func=lambda **kwargs: asyncio.run(tool_func(**kwargs)),
                coroutine=tool_func
            )
            
            self.tools.append(langchain_tool)
    
    def _create_agent(self):
        """Create the LangChain agent"""
        # Create prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful GitHub assistant that can perform various GitHub operations.
            You have access to many GitHub tools including searching repositories, creating issues, 
            managing pull requests, and more. Always be helpful and explain what you're doing.
            
            When listing repositories or showing results, format them nicely for the user.
            """),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # Create agent
        agent = create_openai_functions_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
        
        # Create executor
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True
        )
    
    async def chat(self, message: str, chat_history: List = None) -> str:
        """Have a conversation with the agent"""
        if chat_history is None:
            chat_history = []
        
        try:
            response = await self.agent_executor.ainvoke({
                "input": message,
                "chat_history": chat_history
            })
            return response["output"]
        except Exception as e:
            return f"Error: {str(e)}"
    
    async def cleanup(self):
        """Clean up MCP connections"""
        if self.session:
            await self.session.__aexit__(None, None, None)
        if hasattr(self, 'mcp_context'):
            await self.mcp_context.__aexit__(None, None, None)


async def main():
    """Main demo function"""
    print("🚀 GitHub MCP + LangChain + OpenAI Agent")
    print("=" * 50)
    
    # Initialize agent
    agent = GitHubMCPAgent()
    await agent.initialize()
    
    # Example conversations
    chat_history = []
    
    # Example 1: Get user info
    print("\n👤 Getting user information...")
    response = await agent.chat("Who am I on GitHub? Tell me about my profile.", chat_history)
    print(f"\n{response}")
    chat_history.extend([
        HumanMessage(content="Who am I on GitHub? Tell me about my profile."),
        AIMessage(content=response)
    ])
    
    # Example 2: Search repositories
    print("\n\n🔍 Searching repositories...")
    response = await agent.chat(
        "Find the top 5 most popular Python machine learning repositories and list them with their stars.",
        chat_history
    )
    print(f"\n{response}")
    chat_history.extend([
        HumanMessage(content="Find the top 5 most popular Python machine learning repositories and list them with their stars."),
        AIMessage(content=response)
    ])
    
    # Example 3: User's repositories
    print("\n\n📂 Listing user repositories...")
    response = await agent.chat(
        "List my own repositories. Show both public and private ones with their descriptions.",
        chat_history
    )
    print(f"\n{response}")
    
    # Interactive mode
    print("\n\n💬 Interactive Mode")
    print("=" * 50)
    print("You can now chat with the GitHub agent. Type 'exit' to quit.")
    
    while True:
        user_input = input("\n> ")
        if user_input.lower() in ['exit', 'quit']:
            break
        
        response = await agent.chat(user_input, chat_history)
        print(f"\n{response}")
        
        chat_history.extend([
            HumanMessage(content=user_input),
            AIMessage(content=response)
        ])
    
    # Cleanup
    await agent.cleanup()
    print("\n✨ Goodbye!")


if __name__ == "__main__":
    asyncio.run(main())
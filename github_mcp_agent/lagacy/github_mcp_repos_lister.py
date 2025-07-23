#!/usr/bin/env python3
"""
GitHub Repository Lister using MCP + LangChain + OpenAI
Simple focused script to list and analyze GitHub repositories
"""

import asyncio
import os
import json
from dotenv import load_dotenv
from typing import Dict, Any

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_core.tools import Tool
from langchain.agents import initialize_agent, AgentType

# MCP imports
from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession

load_dotenv()


async def create_github_tools(session: ClientSession) -> list:
    """Create LangChain tools from MCP session"""
    tools = []
    
    # Tool to search repositories
    async def search_repos(**kwargs):
        try:
            result = await session.call_tool("search_repositories", kwargs)
            if result.content:
                return json.loads(result.content[0].text)
            return {"error": "No content returned"}
        except Exception as e:
            return {"error": str(e)}
    
    tools.append(Tool(
        name="search_repositories",
        description="Search GitHub repositories. Use query parameter with GitHub search syntax.",
        func=lambda **kwargs: asyncio.run(search_repos(**kwargs)),
        coroutine=search_repos
    ))
    
    # Tool to get user info
    async def get_user_info(**kwargs):
        try:
            result = await session.call_tool("get_me", kwargs)
            if result.content:
                return json.loads(result.content[0].text)
            return {"error": "No content returned"}
        except Exception as e:
            return {"error": str(e)}
    
    tools.append(Tool(
        name="get_me",
        description="Get authenticated GitHub user information",
        func=lambda **kwargs: asyncio.run(get_user_info(**kwargs)),
        coroutine=get_user_info
    ))
    
    # Tool to list issues
    async def list_issues(**kwargs):
        try:
            result = await session.call_tool("list_issues", kwargs)
            if result.content:
                return json.loads(result.content[0].text)
            return {"error": "No content returned"}
        except Exception as e:
            return {"error": str(e)}
    
    tools.append(Tool(
        name="list_issues",
        description="List issues from a GitHub repository. Requires owner and repo parameters.",
        func=lambda **kwargs: asyncio.run(list_issues(**kwargs)),
        coroutine=list_issues
    ))
    
    return tools


async def list_my_repos():
    """List repositories for the authenticated user"""
    GITHUB_PAT = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    if not GITHUB_PAT or not OPENAI_API_KEY:
        print("❌ Missing GITHUB_PERSONAL_ACCESS_TOKEN or OPENAI_API_KEY")
        return
    
    print("🚀 GitHub Repository Lister with AI")
    print("=" * 50)
    
    # Create MCP connection
    async with streamablehttp_client(
        "https://api.githubcopilot.com/mcp/",
        headers={"Authorization": f"Bearer {GITHUB_PAT}"}
    ) as (read, write, _):
        
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # Create tools
            tools = await create_github_tools(session)
            
            # Create LLM
            llm = ChatOpenAI(
                model="gpt-4",
                temperature=0,
                openai_api_key=OPENAI_API_KEY
            )
            
            # Create agent
            agent = initialize_agent(
                tools=tools,
                llm=llm,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True
            )
            
            print("\n📋 Listing your repositories...\n")
            
            # Query 1: Get user info and list repos
            result = await agent.ainvoke({
                "input": "First get my GitHub username, then search for all repositories owned by me. List them with their descriptions and star counts."
            })
            
            print("\n" + "=" * 50)
            print("📊 Your Repositories:")
            print("=" * 50)
            print(result["output"])
            
            # Query 2: Analyze repos by language
            print("\n\n🔍 Analyzing repositories by language...\n")
            
            result2 = await agent.ainvoke({
                "input": "Based on the repositories you found, group them by primary programming language and show the count for each language."
            })
            
            print("\n" + "=" * 50)
            print("📈 Language Distribution:")
            print("=" * 50)
            print(result2["output"])
            
            # Query 3: Find most popular repos
            print("\n\n⭐ Finding most popular repositories...\n")
            
            result3 = await agent.ainvoke({
                "input": "From my repositories, identify the top 5 most starred or most interesting projects and explain what makes them notable."
            })
            
            print("\n" + "=" * 50)
            print("🌟 Notable Projects:")
            print("=" * 50)
            print(result3["output"])


async def search_repos_by_topic(topic: str):
    """Search for repositories by topic"""
    GITHUB_PAT = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    if not GITHUB_PAT or not OPENAI_API_KEY:
        print("❌ Missing required environment variables")
        return
    
    print(f"\n🔎 Searching for {topic} repositories...")
    print("=" * 50)
    
    async with streamablehttp_client(
        "https://api.githubcopilot.com/mcp/",
        headers={"Authorization": f"Bearer {GITHUB_PAT}"}
    ) as (read, write, _):
        
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            tools = await create_github_tools(session)
            
            llm = ChatOpenAI(
                model="gpt-4",
                temperature=0,
                openai_api_key=OPENAI_API_KEY
            )
            
            agent = initialize_agent(
                tools=tools,
                llm=llm,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=False  # Less verbose for cleaner output
            )
            
            result = await agent.ainvoke({
                "input": f"Search for the top 10 most popular {topic} repositories on GitHub. "
                         f"List them with their full names, star counts, and brief descriptions. "
                         f"Format the output as a numbered list."
            })
            
            print(result["output"])


async def main():
    """Main function with menu"""
    print("🐙 GitHub MCP + AI Repository Explorer")
    print("=" * 50)
    print("1. List my repositories")
    print("2. Search repositories by topic")
    print("3. Exit")
    
    choice = input("\nSelect an option (1-3): ")
    
    if choice == "1":
        await list_my_repos()
    elif choice == "2":
        topic = input("Enter a topic to search (e.g., 'machine learning', 'web development'): ")
        await search_repos_by_topic(topic)
    elif choice == "3":
        print("Goodbye! 👋")
    else:
        print("Invalid choice")


if __name__ == "__main__":
    asyncio.run(main())
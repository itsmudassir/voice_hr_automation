#!/usr/bin/env python3
"""
Fixed GitHub MCP + LangChain Integration
Properly handles tool function signatures
"""

import asyncio
import os
import json
from dotenv import load_dotenv
from typing import Dict, Any

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_core.tools import Tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate

# MCP imports
from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession

load_dotenv()


class GitHubMCPTools:
    """GitHub MCP Tools wrapper for LangChain"""
    
    def __init__(self, session: ClientSession):
        self.session = session
    
    async def search_repositories(self, query: str, perPage: int = 10) -> Dict[str, Any]:
        """Search GitHub repositories"""
        try:
            result = await self.session.call_tool("search_repositories", {
                "query": query,
                "perPage": perPage
            })
            if result.content:
                return json.loads(result.content[0].text)
            return {"error": "No content returned"}
        except Exception as e:
            return {"error": str(e)}
    
    async def get_me(self) -> Dict[str, Any]:
        """Get authenticated user info"""
        try:
            result = await self.session.call_tool("get_me", {})
            if result.content:
                return json.loads(result.content[0].text)
            return {"error": "No content returned"}
        except Exception as e:
            return {"error": str(e)}
    
    async def list_issues(self, owner: str, repo: str) -> Dict[str, Any]:
        """List issues for a repository"""
        try:
            result = await self.session.call_tool("list_issues", {
                "owner": owner,
                "repo": repo
            })
            if result.content:
                return json.loads(result.content[0].text)
            return {"error": "No content returned"}
        except Exception as e:
            return {"error": str(e)}


async def main():
    """Main function demonstrating GitHub MCP with LangChain"""
    
    GITHUB_PAT = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    if not GITHUB_PAT or not OPENAI_API_KEY:
        print("❌ Missing GITHUB_PERSONAL_ACCESS_TOKEN or OPENAI_API_KEY")
        return
    
    print("🚀 GitHub MCP + LangChain (Fixed)")
    print("=" * 50)
    
    # Create MCP connection
    async with streamablehttp_client(
        "https://api.githubcopilot.com/mcp/",
        headers={"Authorization": f"Bearer {GITHUB_PAT}"}
    ) as (read, write, _):
        
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("✅ MCP session initialized")
            
            # Create tools wrapper
            github_tools = GitHubMCPTools(session)
            
            # Create LangChain tools
            tools = [
                Tool(
                    name="search_repositories",
                    description="Search GitHub repositories. Input should be a JSON string with 'query' field.",
                    func=lambda input_str: asyncio.run(
                        github_tools.search_repositories(
                            json.loads(input_str)["query"]
                        )
                    )
                ),
                Tool(
                    name="get_me",
                    description="Get authenticated GitHub user info. No input required.",
                    func=lambda _: asyncio.run(github_tools.get_me())
                ),
                Tool(
                    name="list_issues",
                    description="List issues for a repository. Input should be JSON with 'owner' and 'repo' fields.",
                    func=lambda input_str: asyncio.run(
                        github_tools.list_issues(
                            **json.loads(input_str)
                        )
                    )
                )
            ]
            
            # Create LLM
            llm = ChatOpenAI(
                model="gpt-4",
                temperature=0,
                openai_api_key=OPENAI_API_KEY
            )
            
            # Create prompt
            prompt = PromptTemplate.from_template("""
You are a helpful GitHub assistant. Use the available tools to help answer questions about GitHub repositories.

Available tools:
{tools}

Tool names: {tool_names}

Question: {input}

{agent_scratchpad}
""")
            
            # Create agent
            agent = create_react_agent(
                llm=llm,
                tools=tools,
                prompt=prompt
            )
            
            agent_executor = AgentExecutor(
                agent=agent,
                tools=tools,
                verbose=True,
                handle_parsing_errors=True
            )
            
            # Example 1: Get user info and repos
            print("\n📋 Getting your GitHub profile and repositories...\n")
            
            result = await agent_executor.ainvoke({
                "input": "Get my GitHub username and then list all my repositories with their visibility status."
            })
            
            print("\n" + "=" * 50)
            print("Result:")
            print("=" * 50)
            print(result["output"])
            
            # Example 2: Search for specific repos
            print("\n\n🔍 Searching for machine learning repositories...\n")
            
            result2 = await agent_executor.ainvoke({
                "input": "Search for the top 5 Python machine learning repositories and show their star counts."
            })
            
            print("\n" + "=" * 50)
            print("ML Repositories:")
            print("=" * 50)
            print(result2["output"])
            
            # Interactive mode
            print("\n\n💬 Interactive Mode - Chat with GitHub")
            print("Type 'exit' to quit")
            print("=" * 50)
            
            while True:
                user_input = input("\n> ")
                if user_input.lower() in ['exit', 'quit']:
                    break
                
                try:
                    result = await agent_executor.ainvoke({"input": user_input})
                    print("\n" + result["output"])
                except Exception as e:
                    print(f"\n❌ Error: {e}")
            
            print("\n✨ Goodbye!")


if __name__ == "__main__":
    asyncio.run(main())
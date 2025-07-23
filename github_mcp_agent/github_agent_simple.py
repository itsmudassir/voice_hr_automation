#!/usr/bin/env python3
"""
Simple GitHub Agent - Direct MCP Usage
Uses all MCP capabilities without restrictions
"""

import asyncio
import os
import json
from dotenv import load_dotenv
from typing import Dict, Any

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool

# MCP imports
from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession

# Load environment variables
load_dotenv()


class GitHubMCPToolkit:
    """Direct GitHub MCP Toolkit"""
    
    def __init__(self):
        self.session = None
        self.github_pat = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")
        if not self.github_pat:
            raise ValueError("Missing GITHUB_PERSONAL_ACCESS_TOKEN")
    
    async def initialize(self):
        """Initialize MCP session"""
        self.mcp_context = streamablehttp_client(
            "https://api.githubcopilot.com/mcp/",
            headers={"Authorization": f"Bearer {self.github_pat}"}
        )
        self.read, self.write, _ = await self.mcp_context.__aenter__()
        self.session = ClientSession(self.read, self.write)
        await self.session.__aenter__()
        await self.session.initialize()
        
        # List available tools
        tools_response = await self.session.list_tools()
        self.available_tools = [t.name for t in tools_response.tools]
        print(f"✅ Initialized with {len(self.available_tools)} MCP tools")
        return self
    
    async def cleanup(self):
        """Clean up MCP session"""
        if self.session:
            await self.session.__aexit__(None, None, None)
        if hasattr(self, 'mcp_context'):
            await self.mcp_context.__aexit__(None, None, None)
    
    async def call_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Call any GitHub MCP tool"""
        try:
            result = await self.session.call_tool(tool_name, params)
            if result.content:
                return json.loads(result.content[0].text)
            return {"error": "No content returned"}
        except Exception as e:
            return {"error": f"MCP error: {str(e)}"}


# Global toolkit instance
toolkit = None


@tool
async def search_repositories(query: str) -> str:
    """
    Search GitHub repositories with full query capabilities.
    
    Args:
        query: Full GitHub search query (e.g., "user:itsmudassir elasticsearch", "language:python ML")
    """
    result = await toolkit.call_tool("search_repositories", {
        "query": query,
        "perPage": 30
    })
    
    if "error" not in result and "items" in result:
        repos = result["items"]
        output = f"**Search Results for: {query}**\n"
        output += f"Found {result.get('total_count', len(repos))} repositories\n\n"
        
        for i, repo in enumerate(repos[:15], 1):
            output += f"{i}. **{repo['full_name']}**\n"
            output += f"   📝 {repo.get('description', 'No description')[:100]}...\n"
            output += f"   ⭐ {repo.get('stargazers_count', 0)} | 🔤 {repo.get('language', 'Unknown')}\n\n"
        
        return output
    
    return f"No results found for query: {query}"


@tool
async def search_code(query: str) -> str:
    """
    Search code across GitHub.
    
    Args:
        query: Code search query (e.g., "user:itsmudassir elasticsearch", "redis connection")
    """
    result = await toolkit.call_tool("search_code", {
        "q": query,
        "perPage": 20
    })
    
    if "error" not in result and "items" in result:
        items = result["items"]
        output = f"**Code Search Results for: {query}**\n"
        output += f"Found {result.get('total_count', len(items))} code matches\n\n"
        
        for i, item in enumerate(items[:10], 1):
            repo_name = item.get('repository', {}).get('full_name', 'Unknown')
            output += f"{i}. **{item.get('name')}** in {repo_name}\n"
            output += f"   📁 Path: {item.get('path')}\n"
            output += f"   🔗 {item.get('html_url', '')}\n\n"
        
        return output
    
    return f"No code results found for: {query}"


@tool
async def get_file_contents(owner: str, repo: str, path: str) -> str:
    """
    Get contents of a specific file.
    
    Args:
        owner: Repository owner
        repo: Repository name
        path: File path
    """
    result = await toolkit.call_tool("get_file_contents", {
        "owner": owner,
        "repo": repo,
        "path": path
    })
    
    if "error" not in result:
        if isinstance(result, dict) and 'content' in result:
            # It's a file
            return f"**File: {path}**\n```\n{result.get('content', 'No content')}```"
        elif isinstance(result, list):
            # It's a directory
            output = f"**Directory: {path}**\n"
            for item in result[:20]:
                output += f"- {item.get('type', 'unknown')}: {item.get('name', 'Unknown')}\n"
            return output
    
    return f"Could not get contents for {owner}/{repo}/{path}"


@tool
async def list_issues(owner: str, repo: str, state: str = "open") -> str:
    """
    List issues for a repository.
    
    Args:
        owner: Repository owner
        repo: Repository name  
        state: Issue state (open, closed, all)
    """
    result = await toolkit.call_tool("list_issues", {
        "owner": owner,
        "repo": repo,
        "state": state,
        "perPage": 20
    })
    
    if "error" not in result and isinstance(result, list):
        output = f"**Issues in {owner}/{repo} ({state})**\n\n"
        
        for i, issue in enumerate(result[:10], 1):
            output += f"{i}. #{issue.get('number')} - {issue.get('title')}\n"
            output += f"   State: {issue.get('state')} | Comments: {issue.get('comments', 0)}\n\n"
        
        return output
    
    return f"No issues found for {owner}/{repo}"


@tool
async def get_user_info(username: str = None) -> str:
    """
    Get GitHub user information.
    
    Args:
        username: GitHub username (if None, gets authenticated user)
    """
    if username:
        # Use search to get user info
        result = await toolkit.call_tool("search_repositories", {
            "query": f"user:{username}",
            "perPage": 1
        })
        
        if "items" in result and result["items"]:
            owner = result["items"][0]["owner"]
            return f"**User: @{owner['login']}**\nType: {owner['type']}\nURL: {owner['html_url']}"
    else:
        result = await toolkit.call_tool("get_me", {})
        if "error" not in result:
            return f"**Authenticated as: @{result.get('login')}**\nName: {result.get('name', 'N/A')}"
    
    return "User not found"


async def create_github_agent():
    """Create a GitHub agent with all MCP tools"""
    
    global toolkit
    
    # Initialize toolkit
    toolkit = await GitHubMCPToolkit().initialize()
    
    # Create LLM with GPT-4o mini
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Create tools
    tools = [
        search_repositories,
        search_code,
        get_file_contents,
        list_issues,
        get_user_info
    ]
    
    # Create prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful GitHub assistant with access to all GitHub MCP tools.

You can search repositories, code, files, issues, and more across all of GitHub.
Use the full power of GitHub's search syntax.

For example:
- "user:username elasticsearch" to find elasticsearch in a user's repos
- "language:python machine learning" for Python ML projects
- "redis user:username" to search for redis in user's code

Always provide helpful, formatted responses with the information found."""),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad")
    ])
    
    # Create agent
    agent = create_openai_tools_agent(llm, tools, prompt)
    
    # Create executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )
    
    return agent_executor


async def main():
    """Main function"""
    print("🐙 Simple GitHub Agent (All MCP Tools)")
    print("=" * 50)
    print("Using full GitHub search capabilities")
    print("=" * 50)
    
    try:
        agent = await create_github_agent()
        
        print("\n📌 Example queries:")
        print("  • 'search for elasticsearch in itsmudassir repos'")
        print("  • 'find code using redis in user itsmudassir'")
        print("  • 'show Python ML projects'")
        print("  • 'list issues in torvalds/linux'")
        print("\nType 'exit' to quit\n")
        
        while True:
            query = input("🔍 > ").strip()
            
            if query.lower() in ['exit', 'quit']:
                break
            
            if not query:
                continue
            
            print("\n🤔 Searching...\n")
            
            result = await agent.ainvoke({"input": query})
            
            print("\n" + "=" * 50)
            print(result["output"])
            print("=" * 50 + "\n")
    
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    finally:
        if toolkit:
            await toolkit.cleanup()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Quick query mode
        query = " ".join(sys.argv[1:])
        
        async def quick_run():
            try:
                agent = await create_github_agent()
                result = await agent.ainvoke({"input": query})
                print(result["output"])
            finally:
                if toolkit:
                    await toolkit.cleanup()
        
        asyncio.run(quick_run())
    else:
        asyncio.run(main())
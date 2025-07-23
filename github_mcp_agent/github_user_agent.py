#!/usr/bin/env python3
"""
GitHub User Agent - Simple User-Focused Agent
Only restricts to a specific user, but uses all MCP capabilities
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


# Global instances
toolkit = None
current_user = None


@tool
async def search_repositories(query: str) -> str:
    """
    Search repositories for the current user.
    
    Args:
        query: Search terms (e.g., "elastic", "api", "python")
    """
    if not current_user:
        return "No user set. Please specify a user first."
    
    # Always add user constraint
    full_query = f"user:{current_user} {query}"
    
    result = await toolkit.call_tool("search_repositories", {
        "query": full_query,
        "perPage": 30
    })
    
    if "error" not in result and "items" in result:
        repos = result["items"]
        output = f"**Repository Search Results for @{current_user}: '{query}'**\n"
        output += f"Found {result.get('total_count', len(repos))} repositories\n\n"
        
        for i, repo in enumerate(repos[:15], 1):
            output += f"{i}. **{repo['name']}**\n"
            output += f"   📝 {repo.get('description', 'No description')[:100]}...\n"
            output += f"   ⭐ {repo.get('stargazers_count', 0)} | 🔤 {repo.get('language', 'Unknown')}\n"
            output += f"   🔗 {repo.get('html_url', '')}\n\n"
        
        return output
    
    return f"No repositories found for query: {query}"


@tool
async def search_code(query: str) -> str:
    """
    Search code in the current user's repositories.
    
    Args:
        query: Code search query (e.g., "elasticsearch", "redis", "import pandas")
    """
    if not current_user:
        return "No user set. Please specify a user first."
    
    # Always add user constraint
    full_query = f"user:{current_user} {query}"
    
    result = await toolkit.call_tool("search_code", {
        "q": full_query,
        "perPage": 20
    })
    
    if "error" not in result and "items" in result:
        items = result["items"]
        output = f"**Code Search Results for @{current_user}: '{query}'**\n"
        output += f"Found {result.get('total_count', len(items))} code matches\n\n"
        
        for i, item in enumerate(items[:10], 1):
            repo_name = item.get('repository', {}).get('name', 'Unknown')
            output += f"{i}. **{item.get('name')}** in {repo_name}\n"
            output += f"   📁 Path: {item.get('path')}\n"
            output += f"   🔗 {item.get('html_url', '')}\n\n"
        
        return output
    
    return f"No code results found for: {query}"


@tool
async def get_file_contents(repo: str, path: str) -> str:
    """
    Get contents of a file in the current user's repository.
    
    Args:
        repo: Repository name
        path: File path
    """
    if not current_user:
        return "No user set. Please specify a user first."
    
    result = await toolkit.call_tool("get_file_contents", {
        "owner": current_user,
        "repo": repo,
        "path": path
    })
    
    if "error" not in result:
        if isinstance(result, dict) and 'content' in result:
            # It's a file
            return f"**File: {current_user}/{repo}/{path}**\n```\n{result.get('content', 'No content')}```"
        elif isinstance(result, list):
            # It's a directory
            output = f"**Directory: {current_user}/{repo}/{path}**\n"
            for item in result[:20]:
                output += f"- {item.get('type', 'unknown')}: {item.get('name', 'Unknown')}\n"
            return output
    
    return f"Could not get contents for {current_user}/{repo}/{path}"


@tool
async def list_issues(repo: str, state: str = "open") -> str:
    """
    List issues for the current user's repository.
    
    Args:
        repo: Repository name
        state: Issue state (open, closed, all)
    """
    if not current_user:
        return "No user set. Please specify a user first."
    
    result = await toolkit.call_tool("list_issues", {
        "owner": current_user,
        "repo": repo,
        "state": state,
        "perPage": 20
    })
    
    if "error" not in result and isinstance(result, list):
        output = f"**Issues in {current_user}/{repo} ({state})**\n\n"
        
        for i, issue in enumerate(result[:10], 1):
            output += f"{i}. #{issue.get('number')} - {issue.get('title')}\n"
            output += f"   State: {issue.get('state')} | Comments: {issue.get('comments', 0)}\n"
            output += f"   Created: {issue.get('created_at', 'Unknown')[:10]}\n\n"
        
        return output
    
    return f"No issues found or repository doesn't exist: {current_user}/{repo}"


@tool
async def get_user_stats() -> str:
    """Get statistics about the current user's repositories."""
    if not current_user:
        return "No user set. Please specify a user first."
    
    # Get all repos
    result = await toolkit.call_tool("search_repositories", {
        "query": f"user:{current_user}",
        "perPage": 100
    })
    
    if "error" not in result and "items" in result:
        repos = result["items"]
        
        # Calculate stats
        total_stars = sum(r.get('stargazers_count', 0) for r in repos)
        total_forks = sum(r.get('forks_count', 0) for r in repos)
        languages = {}
        
        for repo in repos:
            lang = repo.get('language')
            if lang:
                languages[lang] = languages.get(lang, 0) + 1
        
        output = f"**📊 Statistics for @{current_user}**\n\n"
        output += f"Total Repositories: {len(repos)}\n"
        output += f"Total Stars: ⭐ {total_stars}\n"
        output += f"Total Forks: 🍴 {total_forks}\n"
        output += f"Public Repos: {sum(1 for r in repos if not r.get('private', False))}\n"
        output += f"Private Repos: {sum(1 for r in repos if r.get('private', False))}\n\n"
        
        output += "**Languages Used:**\n"
        for lang, count in sorted(languages.items(), key=lambda x: x[1], reverse=True)[:10]:
            output += f"- {lang}: {count} repos\n"
        
        return output
    
    return f"Could not get statistics for @{current_user}"


async def create_user_agent(username: str):
    """Create a GitHub agent focused on a specific user"""
    
    global toolkit, current_user
    
    # Set current user
    current_user = username
    
    # Initialize toolkit if needed
    if not toolkit:
        toolkit = await GitHubMCPToolkit().initialize()
    
    # Verify user exists
    test_result = await toolkit.call_tool("search_repositories", {
        "query": f"user:{username}",
        "perPage": 1
    })
    
    if "error" in test_result or ("items" in test_result and len(test_result["items"]) == 0):
        raise ValueError(f"User @{username} not found or has no repositories")
    
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
        get_user_stats
    ]
    
    # Create prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are a GitHub assistant focused on user @{username}.

All searches and operations are automatically restricted to this user's repositories.
You have access to:
- Search repositories (just specify search terms like "elastic" or "python")
- Search code (find where specific code/libraries are used)
- Get file contents
- List issues
- Get user statistics

The user constraint is automatically added, so users can just ask naturally:
- "find elasticsearch" (will search in {username}'s repos)
- "search for redis in code" (will search in {username}'s code)
- "show me the README in repo X"

Always provide helpful, formatted responses."""),
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
    print("🐙 GitHub User Agent")
    print("=" * 50)
    
    # Get username
    if len(sys.argv) > 1:
        username = sys.argv[1]
    else:
        username = input("Enter GitHub username: ").strip()
    
    if not username:
        print("❌ No username provided")
        return
    
    print(f"\n🔍 Setting up agent for @{username}...")
    
    try:
        agent = await create_user_agent(username)
        print(f"✅ Agent ready for @{username}")
        
        print("\n📌 Example queries:")
        print("  • 'find elasticsearch' (searches in repos)")
        print("  • 'search for redis in code'")
        print("  • 'show statistics'")
        print("  • 'list issues in repo-name'")
        print("  • 'show README.md in elasticProxyServer'")
        print("\nType 'exit' to quit\n")
        
        while True:
            query = input(f"[{username}] > ").strip()
            
            if query.lower() in ['exit', 'quit']:
                break
            
            if not query:
                continue
            
            print("\n🤔 Processing...\n")
            
            result = await agent.ainvoke({"input": query})
            
            print("\n" + "=" * 50)
            print(result["output"])
            print("=" * 50 + "\n")
    
    except ValueError as e:
        print(f"❌ {e}")
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    finally:
        if toolkit:
            await toolkit.cleanup()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 2:
        # Quick query mode
        username = sys.argv[1]
        query = " ".join(sys.argv[2:])
        
        async def quick_run():
            global current_user
            current_user = username
            
            try:
                if not toolkit:
                    globals()['toolkit'] = await GitHubMCPToolkit().initialize()
                
                agent = await create_user_agent(username)
                result = await agent.ainvoke({"input": query})
                print(result["output"])
            finally:
                if toolkit:
                    await toolkit.cleanup()
        
        asyncio.run(quick_run())
    else:
        asyncio.run(main())
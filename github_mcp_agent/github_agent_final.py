#!/usr/bin/env python3
"""
GitHub Agent Final - Complete Working Version
LangChain + OpenAI + GitHub MCP Integration
Answers any question about GitHub repositories and users
"""

import asyncio
import os
import json
from dotenv import load_dotenv
from typing import Dict, Any, List, Optional
from datetime import datetime

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain.memory import ConversationBufferMemory

# MCP imports
from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession

# Load environment variables
load_dotenv()


class GitHubMCPToolkit:
    """GitHub MCP Toolkit - All GitHub operations via MCP"""
    
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
        """Call a GitHub MCP tool"""
        try:
            result = await self.session.call_tool(tool_name, params)
            if result.content:
                return json.loads(result.content[0].text)
            return {"error": "No content returned"}
        except Exception as e:
            return {"error": str(e)}


# Global toolkit instance
toolkit = None


@tool
async def search_repositories(query: str, sort: str = None, order: str = "desc", per_page: int = 30) -> str:
    """
    Search for GitHub repositories.
    
    Args:
        query: Search query (e.g., "language:python stars:>1000", "user:torvalds")
        sort: Sort by stars, forks, updated, or best-match (default)
        order: Sort order - desc or asc
        per_page: Results per page (max 100)
    
    Returns:
        JSON string with search results
    """
    params = {
        "query": query,
        "perPage": min(per_page, 100)
    }
    if sort:
        params["sort"] = sort
    if order:
        params["order"] = order
    
    result = await toolkit.call_tool("search_repositories", params)
    
    if "error" not in result:
        # Format the results nicely
        if "items" in result:
            repos = result["items"]
            formatted = f"Found {result.get('total_count', len(repos))} repositories\n\n"
            
            for i, repo in enumerate(repos[:10], 1):
                formatted += f"{i}. **{repo['full_name']}**\n"
                formatted += f"   ⭐ Stars: {repo.get('stargazers_count', 0)}\n"
                formatted += f"   🔤 Language: {repo.get('language', 'Unknown')}\n"
                formatted += f"   📝 {repo.get('description', 'No description')[:100]}...\n"
                formatted += f"   🔗 {repo.get('html_url', '')}\n\n"
            
            return formatted
    
    return json.dumps(result, indent=2)


@tool
async def get_user_info(username: str = None) -> str:
    """
    Get GitHub user information.
    
    Args:
        username: GitHub username (if None, gets authenticated user)
    
    Returns:
        User information as formatted string
    """
    if username:
        # Search for user's repos to get user info
        result = await toolkit.call_tool("search_repositories", {
            "query": f"user:{username}",
            "perPage": 1
        })
        
        if "items" in result and result["items"]:
            owner = result["items"][0]["owner"]
            info = f"**User: {owner['login']}**\n"
            info += f"Type: {owner.get('type', 'Unknown')}\n"
            info += f"URL: {owner.get('html_url', '')}\n"
            return info
    else:
        # Get authenticated user
        result = await toolkit.call_tool("get_me", {})
        
        if "error" not in result:
            info = f"**Authenticated User: {result.get('login', 'Unknown')}**\n"
            info += f"Name: {result.get('name', 'N/A')}\n"
            info += f"Email: {result.get('email', 'N/A')}\n"
            info += f"Public Repos: {result.get('public_repos', 0)}\n"
            info += f"Followers: {result.get('followers', 0)}\n"
            info += f"Following: {result.get('following', 0)}\n"
            return info
    
    return json.dumps(result, indent=2)


@tool
async def list_user_repositories(username: str, include_private: bool = True) -> str:
    """
    List all repositories for a specific GitHub user.
    
    Args:
        username: GitHub username
        include_private: Whether to include private repos (only works for authenticated user)
    
    Returns:
        Formatted list of repositories
    """
    result = await toolkit.call_tool("search_repositories", {
        "query": f"user:{username}",
        "perPage": 100
    })
    
    if "error" not in result and "items" in result:
        repos = result["items"]
        
        # Group by visibility
        public_repos = [r for r in repos if not r.get('private', False)]
        private_repos = [r for r in repos if r.get('private', False)]
        
        output = f"**{username}'s Repositories**\n"
        output += f"Total: {len(repos)} repositories\n\n"
        
        if public_repos:
            output += f"**🌍 Public Repositories ({len(public_repos)}):**\n"
            for repo in sorted(public_repos, key=lambda x: x.get('stargazers_count', 0), reverse=True)[:20]:
                output += f"- {repo['name']} ⭐ {repo.get('stargazers_count', 0)} [{repo.get('language', 'Unknown')}]\n"
        
        if private_repos and include_private:
            output += f"\n**🔒 Private Repositories ({len(private_repos)}):**\n"
            for repo in private_repos[:10]:
                output += f"- {repo['name']}\n"
        
        return output
    
    return json.dumps(result, indent=2)


@tool
async def get_repository_details(owner: str, repo: str) -> str:
    """
    Get detailed information about a specific repository.
    
    Args:
        owner: Repository owner (username or organization)
        repo: Repository name
    
    Returns:
        Detailed repository information
    """
    # Search for the specific repo
    result = await toolkit.call_tool("search_repositories", {
        "query": f"repo:{owner}/{repo}",
        "perPage": 1
    })
    
    if "error" not in result and "items" in result and result["items"]:
        repo = result["items"][0]
        
        details = f"**Repository: {repo['full_name']}**\n\n"
        details += f"📝 Description: {repo.get('description', 'No description')}\n"
        details += f"⭐ Stars: {repo.get('stargazers_count', 0)}\n"
        details += f"🍴 Forks: {repo.get('forks_count', 0)}\n"
        details += f"👁️ Watchers: {repo.get('watchers_count', 0)}\n"
        details += f"🔤 Language: {repo.get('language', 'Unknown')}\n"
        details += f"📅 Created: {repo.get('created_at', 'Unknown')}\n"
        details += f"🔄 Updated: {repo.get('updated_at', 'Unknown')}\n"
        details += f"🔗 URL: {repo.get('html_url', '')}\n"
        details += f"🏠 Homepage: {repo.get('homepage', 'N/A')}\n"
        details += f"📋 Topics: {', '.join(repo.get('topics', []))}\n"
        details += f"⚖️ License: {repo.get('license', {}).get('name', 'No license')}\n"
        
        return details
    
    return "Repository not found"


@tool
async def analyze_user_languages(username: str) -> str:
    """
    Analyze programming languages used by a GitHub user.
    
    Args:
        username: GitHub username
    
    Returns:
        Language statistics for the user
    """
    result = await toolkit.call_tool("search_repositories", {
        "query": f"user:{username}",
        "perPage": 100
    })
    
    if "error" not in result and "items" in result:
        repos = result["items"]
        
        # Count languages
        language_stats = {}
        total_stars = 0
        
        for repo in repos:
            lang = repo.get('language')
            if lang:
                if lang not in language_stats:
                    language_stats[lang] = {
                        'count': 0,
                        'stars': 0,
                        'repos': []
                    }
                
                language_stats[lang]['count'] += 1
                language_stats[lang]['stars'] += repo.get('stargazers_count', 0)
                language_stats[lang]['repos'].append(repo['name'])
                total_stars += repo.get('stargazers_count', 0)
        
        # Sort by count
        sorted_langs = sorted(language_stats.items(), key=lambda x: x[1]['count'], reverse=True)
        
        output = f"**Programming Languages Analysis for {username}**\n\n"
        output += f"Total repositories: {len(repos)}\n"
        output += f"Total stars: {total_stars}\n\n"
        
        output += "**Languages by Repository Count:**\n"
        for lang, stats in sorted_langs:
            percentage = (stats['count'] / len(repos)) * 100
            output += f"- **{lang}**: {stats['count']} repos ({percentage:.1f}%), {stats['stars']} stars\n"
            output += f"  Top repos: {', '.join(stats['repos'][:3])}\n"
        
        return output
    
    return "No repositories found for analysis"


@tool
async def search_code(query: str, language: str = None, user: str = None) -> str:
    """
    Search for code across GitHub.
    
    Args:
        query: Code search query
        language: Filter by programming language
        user: Filter by user
    
    Returns:
        Code search results
    """
    search_query = query
    if language:
        search_query += f" language:{language}"
    if user:
        search_query += f" user:{user}"
    
    result = await toolkit.call_tool("search_code", {
        "q": search_query,
        "perPage": 10
    })
    
    if "error" not in result and "items" in result:
        items = result["items"]
        output = f"**Code Search Results for: {query}**\n"
        output += f"Found {result.get('total_count', len(items))} results\n\n"
        
        for i, item in enumerate(items[:10], 1):
            output += f"{i}. **{item.get('name', 'Unknown')}**\n"
            output += f"   Repository: {item.get('repository', {}).get('full_name', 'Unknown')}\n"
            output += f"   Path: {item.get('path', 'Unknown')}\n"
            output += f"   URL: {item.get('html_url', '')}\n\n"
        
        return output
    
    return json.dumps(result, indent=2)


@tool
async def get_trending_repositories(language: str = None, since: str = "daily") -> str:
    """
    Get trending repositories.
    
    Args:
        language: Filter by programming language (e.g., "python", "javascript")
        since: Time range - daily, weekly, or monthly
    
    Returns:
        List of trending repositories
    """
    # Create date filter based on 'since'
    from datetime import datetime, timedelta
    
    date_map = {
        "daily": 1,
        "weekly": 7,
        "monthly": 30
    }
    
    days = date_map.get(since, 1)
    date_filter = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    
    query = f"created:>{date_filter}"
    if language:
        query += f" language:{language}"
    
    result = await toolkit.call_tool("search_repositories", {
        "query": query,
        "sort": "stars",
        "order": "desc",
        "perPage": 20
    })
    
    if "error" not in result and "items" in result:
        repos = result["items"]
        
        output = f"**Trending {language or 'All'} Repositories ({since})**\n\n"
        
        for i, repo in enumerate(repos[:15], 1):
            output += f"{i}. **{repo['full_name']}**\n"
            output += f"   ⭐ Stars: {repo.get('stargazers_count', 0)}\n"
            output += f"   📝 {repo.get('description', 'No description')[:80]}...\n"
            output += f"   🔤 Language: {repo.get('language', 'Unknown')}\n\n"
        
        return output
    
    return "No trending repositories found"


async def create_github_agent():
    """Create the GitHub agent with all tools"""
    
    # Initialize toolkit
    global toolkit
    toolkit = await GitHubMCPToolkit().initialize()
    
    # Create LLM
    llm = ChatOpenAI(
        model="gpt-4",
        temperature=0,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Create tools list
    tools = [
        search_repositories,
        get_user_info,
        list_user_repositories,
        get_repository_details,
        analyze_user_languages,
        search_code,
        get_trending_repositories
    ]
    
    # Create prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful GitHub assistant with access to comprehensive GitHub data through MCP tools.
        
You can:
- Search for repositories and code
- Get user information and analyze their projects
- Find trending repositories
- Analyze programming language usage
- Get detailed repository information

Always provide clear, well-formatted responses. When listing repositories, include relevant details like stars, language, and descriptions.
For user analysis, provide comprehensive statistics and insights."""),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad")
    ])
    
    # Create agent
    agent = create_openai_tools_agent(llm, tools, prompt)
    
    # Create memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    # Create executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5
    )
    
    return agent_executor, toolkit


async def main():
    """Main interactive loop"""
    print("🐙 GitHub Agent - Final Version")
    print("=" * 60)
    print("Ask me anything about GitHub users, repositories, or code!")
    print("=" * 60)
    
    try:
        # Create agent
        print("🔧 Initializing GitHub agent...")
        agent_executor, toolkit = await create_github_agent()
        print("✅ Agent ready!\n")
        
        # Show example queries
        print("📌 Example queries:")
        print("  • 'Show me torvalds repositories'")
        print("  • 'What languages does guido van rossum use?'")
        print("  • 'Find the most popular Python machine learning repos'")
        print("  • 'Show trending JavaScript projects this week'")
        print("  • 'Search for async code in rust'")
        print("  • 'Analyze itsmudassir GitHub profile'")
        print("\nType 'exit' to quit\n")
        
        # Interactive loop
        while True:
            try:
                query = input("🔍 Ask me: ").strip()
                
                if query.lower() in ['exit', 'quit']:
                    print("\n👋 Goodbye!")
                    break
                
                if not query:
                    continue
                
                print("\n🤔 Thinking...\n")
                
                # Run agent
                result = await agent_executor.ainvoke({"input": query})
                
                print("\n" + "=" * 60)
                print("📊 ANSWER:")
                print("=" * 60)
                print(result["output"])
                print("=" * 60 + "\n")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}\n")
    
    finally:
        # Cleanup
        if toolkit:
            await toolkit.cleanup()


# Quick query functions
async def quick_query(query: str):
    """Run a single query and exit"""
    try:
        print(f"🔍 Query: {query}\n")
        
        agent_executor, toolkit = await create_github_agent()
        result = await agent_executor.ainvoke({"input": query})
        
        print("\n📊 ANSWER:")
        print("=" * 60)
        print(result["output"])
        
        await toolkit.cleanup()
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Quick query mode
        query = " ".join(sys.argv[1:])
        asyncio.run(quick_query(query))
    else:
        # Interactive mode
        asyncio.run(main())
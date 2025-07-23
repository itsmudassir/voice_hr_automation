#!/usr/bin/env python3
"""
GitHub User-Focused Agent - Improved Version
Uses GPT-4o mini for efficiency and focuses on specific user profiles
Smart MCP tool usage without reading large files
"""

import asyncio
import os
import json
from dotenv import load_dotenv
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import re

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain.memory import ConversationSummaryBufferMemory

# MCP imports
from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession

# Load environment variables
load_dotenv()


class GitHubUserSession:
    """Manages a focused session for a specific GitHub user"""
    
    def __init__(self, username: str):
        self.username = username
        self.user_data = {}
        self.repos_cache = {}
        self.languages_cache = {}
        self.last_cache_update = None
        self.cache_duration = timedelta(minutes=15)
    
    def should_refresh_cache(self) -> bool:
        """Check if cache should be refreshed"""
        if not self.last_cache_update:
            return True
        return datetime.now() - self.last_cache_update > self.cache_duration
    
    def update_cache_timestamp(self):
        """Update cache timestamp"""
        self.last_cache_update = datetime.now()


class GitHubMCPToolkit:
    """Optimized GitHub MCP Toolkit for user-focused operations"""
    
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
        """Call a GitHub MCP tool with error handling"""
        try:
            result = await self.session.call_tool(tool_name, params)
            if result.content:
                return json.loads(result.content[0].text)
            return {"error": "No content returned"}
        except Exception as e:
            return {"error": f"MCP tool error: {str(e)}"}


# Global instances
toolkit: Optional[GitHubMCPToolkit] = None
user_session: Optional[GitHubUserSession] = None


def validate_username(username: str) -> str:
    """Validate and extract GitHub username from input"""
    # Remove URL parts if provided
    username = username.strip()
    
    # Handle GitHub URLs
    if "github.com/" in username:
        match = re.search(r'github\.com/([^/]+)', username)
        if match:
            username = match.group(1)
    
    # Remove @ if present
    username = username.lstrip('@')
    
    # Validate username format
    if not re.match(r'^[a-zA-Z0-9](?:[a-zA-Z0-9]|-(?=[a-zA-Z0-9])){0,38}$', username):
        raise ValueError(f"Invalid GitHub username format: {username}")
    
    return username


@tool
async def get_user_profile() -> str:
    """
    Get the current user's profile information and statistics.
    No arguments needed - uses the session user.
    """
    if not user_session:
        return "No user session active. Please set a user first."
    
    # Search for user's repos to get profile data
    result = await toolkit.call_tool("search_repositories", {
        "query": f"user:{user_session.username}",
        "perPage": 1
    })
    
    if "error" in result:
        return f"Error fetching user profile: {result['error']}"
    
    if "items" in result and result["items"]:
        owner = result["items"][0]["owner"]
        profile = f"**GitHub Profile: @{owner['login']}**\n\n"
        profile += f"🔗 Profile URL: {owner['html_url']}\n"
        profile += f"🆔 User ID: {owner['id']}\n"
        profile += f"📊 Account Type: {owner['type']}\n"
        
        # Get more stats
        stats_result = await toolkit.call_tool("search_repositories", {
            "query": f"user:{user_session.username}",
            "perPage": 100
        })
        
        if "items" in stats_result:
            repos = stats_result["items"]
            total_stars = sum(r.get('stargazers_count', 0) for r in repos)
            total_forks = sum(r.get('forks_count', 0) for r in repos)
            
            profile += f"\n**📈 Statistics:**\n"
            profile += f"- Total Repositories: {len(repos)}\n"
            profile += f"- Total Stars: ⭐ {total_stars}\n"
            profile += f"- Total Forks: 🍴 {total_forks}\n"
            profile += f"- Public Repos: {sum(1 for r in repos if not r.get('private', False))}\n"
            profile += f"- Private Repos: {sum(1 for r in repos if r.get('private', False))}\n"
        
        user_session.user_data = owner
        return profile
    
    return f"User @{user_session.username} not found or has no repositories."


@tool
async def search_user_repos(query: str, language: str = None, sort_by: str = "updated") -> str:
    """
    Search within the current user's repositories.
    
    Args:
        query: Search terms (e.g., "api", "machine learning")
        language: Filter by programming language (optional)
        sort_by: Sort by 'stars', 'forks', 'updated' (default: updated)
    """
    if not user_session:
        return "No user session active. Please set a user first."
    
    # Build search query
    search_query = f"user:{user_session.username} {query}"
    if language:
        search_query += f" language:{language}"
    
    result = await toolkit.call_tool("search_repositories", {
        "query": search_query,
        "sort": sort_by if sort_by in ["stars", "forks", "updated"] else None,
        "perPage": 20
    })
    
    if "error" in result:
        return f"Search error: {result['error']}"
    
    if "items" in result:
        repos = result["items"]
        if not repos:
            return f"No repositories found matching '{query}' for @{user_session.username}"
        
        output = f"**Search Results for '{query}' in @{user_session.username}'s repos:**\n\n"
        
        for i, repo in enumerate(repos[:10], 1):
            output += f"{i}. **{repo['name']}**\n"
            output += f"   📝 {repo.get('description', 'No description')[:80]}...\n"
            output += f"   🔤 Language: {repo.get('language', 'Unknown')}\n"
            output += f"   ⭐ Stars: {repo.get('stargazers_count', 0)} | "
            output += f"🍴 Forks: {repo.get('forks_count', 0)}\n"
            output += f"   🔄 Updated: {repo.get('updated_at', 'Unknown')[:10]}\n\n"
        
        return output
    
    return "No search results found."


@tool
async def analyze_user_tech_stack() -> str:
    """
    Analyze the technology stack and programming languages used by the current user.
    Provides insights into frameworks, tools, and patterns.
    """
    if not user_session:
        return "No user session active. Please set a user first."
    
    # Get all repos with language info
    result = await toolkit.call_tool("search_repositories", {
        "query": f"user:{user_session.username}",
        "perPage": 100
    })
    
    if "error" in result:
        return f"Error analyzing tech stack: {result['error']}"
    
    if "items" not in result:
        return "No repositories found for analysis."
    
    repos = result["items"]
    
    # Analyze languages
    languages = {}
    topics_set = set()
    repo_types = {
        'web': 0, 'mobile': 0, 'data': 0, 'devops': 0, 
        'ml': 0, 'api': 0, 'cli': 0, 'library': 0
    }
    
    for repo in repos:
        # Count languages
        lang = repo.get('language')
        if lang:
            languages[lang] = languages.get(lang, 0) + 1
        
        # Collect topics
        topics = repo.get('topics', [])
        topics_set.update(topics)
        
        # Categorize repos
        name_lower = repo['name'].lower()
        desc_lower = (repo.get('description', '') or '').lower()
        combined = name_lower + ' ' + desc_lower
        
        if any(term in combined for term in ['web', 'frontend', 'react', 'vue', 'angular']):
            repo_types['web'] += 1
        if any(term in combined for term in ['mobile', 'android', 'ios', 'flutter']):
            repo_types['mobile'] += 1
        if any(term in combined for term in ['data', 'etl', 'pipeline', 'analytics']):
            repo_types['data'] += 1
        if any(term in combined for term in ['docker', 'kubernetes', 'ci/cd', 'deploy']):
            repo_types['devops'] += 1
        if any(term in combined for term in ['ml', 'ai', 'machine learning', 'deep learning']):
            repo_types['ml'] += 1
        if any(term in combined for term in ['api', 'rest', 'graphql', 'microservice']):
            repo_types['api'] += 1
        if any(term in combined for term in ['cli', 'command', 'terminal']):
            repo_types['cli'] += 1
        if any(term in combined for term in ['library', 'framework', 'package', 'sdk']):
            repo_types['library'] += 1
    
    # Build output
    output = f"**🛠️ Technology Stack Analysis for @{user_session.username}**\n\n"
    
    # Languages breakdown
    output += "**Programming Languages:**\n"
    sorted_langs = sorted(languages.items(), key=lambda x: x[1], reverse=True)
    for lang, count in sorted_langs[:10]:
        percentage = (count / len(repos)) * 100
        output += f"- {lang}: {count} repos ({percentage:.1f}%)\n"
    
    # Project types
    output += "\n**Project Categories:**\n"
    for category, count in sorted(repo_types.items(), key=lambda x: x[1], reverse=True):
        if count > 0:
            output += f"- {category.title()}: {count} projects\n"
    
    # Topics/Technologies
    if topics_set:
        output += "\n**Technologies & Topics:**\n"
        output += ", ".join(sorted(topics_set)[:20])
    
    # Insights
    output += "\n\n**💡 Insights:**\n"
    primary_lang = sorted_langs[0][0] if sorted_langs else "Unknown"
    output += f"- Primary language: {primary_lang}\n"
    
    if repo_types['web'] > 5:
        output += "- Strong focus on web development\n"
    if repo_types['ml'] > 3:
        output += "- Active in machine learning/AI\n"
    if repo_types['devops'] > 3:
        output += "- DevOps and infrastructure expertise\n"
    
    return output


@tool
async def get_repo_insights(repo_name: str) -> str:
    """
    Get detailed insights about a specific repository without reading files.
    
    Args:
        repo_name: Name of the repository to analyze
    """
    if not user_session:
        return "No user session active. Please set a user first."
    
    # Search for the specific repo
    result = await toolkit.call_tool("search_repositories", {
        "query": f"user:{user_session.username} repo:{repo_name}",
        "perPage": 1
    })
    
    if "error" in result:
        return f"Error fetching repository: {result['error']}"
    
    if "items" not in result or not result["items"]:
        return f"Repository '{repo_name}' not found for @{user_session.username}"
    
    repo = result["items"][0]
    
    # Build insights
    output = f"**📊 Repository Insights: {repo['full_name']}**\n\n"
    
    # Basic info
    output += f"📝 **Description:** {repo.get('description', 'No description')}\n"
    output += f"🔗 **URL:** {repo['html_url']}\n"
    output += f"🔤 **Primary Language:** {repo.get('language', 'Unknown')}\n"
    output += f"⭐ **Stars:** {repo.get('stargazers_count', 0)}\n"
    output += f"🍴 **Forks:** {repo.get('forks_count', 0)}\n"
    output += f"👁️ **Watchers:** {repo.get('watchers_count', 0)}\n"
    output += f"📏 **Size:** {repo.get('size', 0)} KB\n\n"
    
    # Dates
    created = repo.get('created_at', '')[:10]
    updated = repo.get('updated_at', '')[:10]
    output += f"📅 **Timeline:**\n"
    output += f"- Created: {created}\n"
    output += f"- Last Updated: {updated}\n\n"
    
    # Features
    output += f"**🔧 Features:**\n"
    output += f"- License: {repo.get('license', {}).get('name', 'No license')}\n"
    output += f"- Issues: {'✅ Enabled' if repo.get('has_issues') else '❌ Disabled'}\n"
    output += f"- Wiki: {'✅ Enabled' if repo.get('has_wiki') else '❌ Disabled'}\n"
    output += f"- Projects: {'✅ Enabled' if repo.get('has_projects') else '❌ Disabled'}\n"
    
    # Topics
    topics = repo.get('topics', [])
    if topics:
        output += f"\n**🏷️ Topics:** {', '.join(topics)}\n"
    
    # Activity level
    if updated:
        from datetime import datetime
        last_update = datetime.fromisoformat(updated.replace('Z', '+00:00'))
        days_ago = (datetime.now(last_update.tzinfo) - last_update).days
        
        if days_ago < 30:
            activity = "🟢 Very Active"
        elif days_ago < 90:
            activity = "🟡 Active"
        elif days_ago < 365:
            activity = "🟠 Moderate"
        else:
            activity = "🔴 Inactive"
        
        output += f"\n**📈 Activity Status:** {activity} (updated {days_ago} days ago)\n"
    
    return output


@tool
async def find_similar_projects(keywords: str, max_results: int = 10) -> str:
    """
    Find projects similar to the user's work based on keywords.
    
    Args:
        keywords: Keywords to search for (e.g., "machine learning python")
        max_results: Maximum number of results to return
    """
    if not user_session:
        return "No user session active. Please set a user first."
    
    # Search globally but exclude user's own repos
    result = await toolkit.call_tool("search_repositories", {
        "query": f"{keywords} -user:{user_session.username}",
        "sort": "stars",
        "perPage": min(max_results, 20)
    })
    
    if "error" in result:
        return f"Error searching for similar projects: {result['error']}"
    
    if "items" not in result or not result["items"]:
        return f"No similar projects found for '{keywords}'"
    
    output = f"**🔍 Similar Projects to @{user_session.username}'s work:**\n"
    output += f"*Search: {keywords}*\n\n"
    
    for i, repo in enumerate(result["items"][:max_results], 1):
        output += f"{i}. **{repo['full_name']}** ⭐ {repo.get('stargazers_count', 0)}\n"
        output += f"   {repo.get('description', 'No description')[:100]}...\n"
        output += f"   Language: {repo.get('language', 'Unknown')}\n\n"
    
    return output


@tool
async def get_activity_summary(days: int = 30) -> str:
    """
    Get a summary of the user's recent activity.
    
    Args:
        days: Number of days to look back (default: 30)
    """
    if not user_session:
        return "No user session active. Please set a user first."
    
    # Calculate date filter
    from datetime import datetime, timedelta
    since_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    
    # Get recently updated repos
    result = await toolkit.call_tool("search_repositories", {
        "query": f"user:{user_session.username} pushed:>{since_date}",
        "sort": "updated",
        "perPage": 20
    })
    
    if "error" in result:
        return f"Error fetching activity: {result['error']}"
    
    output = f"**📅 Activity Summary for @{user_session.username} (Last {days} days)**\n\n"
    
    if "items" in result and result["items"]:
        repos = result["items"]
        output += f"**🔄 Recently Active Repositories ({len(repos)}):**\n\n"
        
        for repo in repos[:10]:
            updated = repo.get('updated_at', '')[:10]
            output += f"- **{repo['name']}** (updated {updated})\n"
            output += f"  Language: {repo.get('language', 'Unknown')} | "
            output += f"⭐ {repo.get('stargazers_count', 0)}\n"
        
        # Language activity
        lang_activity = {}
        for repo in repos:
            lang = repo.get('language')
            if lang:
                lang_activity[lang] = lang_activity.get(lang, 0) + 1
        
        if lang_activity:
            output += f"\n**🔤 Active Languages:**\n"
            for lang, count in sorted(lang_activity.items(), key=lambda x: x[1], reverse=True):
                output += f"- {lang}: {count} repos\n"
    else:
        output += "No recent activity found in the specified time period."
    
    return output


async def create_user_focused_agent(username: str):
    """Create an agent focused on a specific GitHub user"""
    
    global toolkit, user_session
    
    # Validate username
    try:
        username = validate_username(username)
    except ValueError as e:
        raise ValueError(f"Invalid username: {e}")
    
    # Initialize toolkit if needed
    if not toolkit:
        toolkit = await GitHubMCPToolkit().initialize()
    
    # Create user session
    user_session = GitHubUserSession(username)
    
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
    
    # Create tools list
    tools = [
        get_user_profile,
        search_user_repos,
        analyze_user_tech_stack,
        get_repo_insights,
        find_similar_projects,
        get_activity_summary
    ]
    
    # Create prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are a GitHub assistant focused exclusively on user @{username}.
        
You can ONLY work with this user's repositories and data. You cannot search or access other users' data.

Available operations:
- Get user profile and statistics
- Search within their repositories
- Analyze their technology stack
- Get insights about specific repos
- Find similar projects (for inspiration)
- Get activity summaries

Always be helpful and provide insights based on the data. Format responses clearly with emojis and sections.
If asked about other users or general GitHub searches, politely remind that you're focused on @{username} only."""),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad")
    ])
    
    # Create agent
    agent = create_openai_tools_agent(llm, tools, prompt)
    
    # Create memory (using summary to save tokens)
    memory = ConversationSummaryBufferMemory(
        llm=llm,
        memory_key="chat_history",
        return_messages=True,
        max_token_limit=2000
    )
    
    # Create executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=3  # Limit iterations to save tokens
    )
    
    return agent_executor, username


async def main():
    """Main interactive loop"""
    
    print("🐙 GitHub User-Focused Agent (GPT-4o mini)")
    print("=" * 60)
    print("This agent works exclusively with a specific GitHub user's data.")
    print("=" * 60)
    
    # Get username
    while True:
        username_input = input("\n🔗 Enter GitHub username or profile URL: ").strip()
        if not username_input:
            print("❌ Please enter a username")
            continue
        
        try:
            username = validate_username(username_input)
            print(f"\n🔍 Setting up agent for @{username}...")
            
            agent_executor, validated_username = await create_user_focused_agent(username)
            username = validated_username  # Update the username variable
            print(f"✅ Agent ready! Now focused on @{username}\n")
            break
            
        except ValueError as e:
            print(f"❌ {e}")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Show available commands
    print("📌 Example queries:")
    print(f"  • 'Show me {username}'s profile'")
    print(f"  • 'What languages does {username} use?'")
    print(f"  • 'Search for API projects'")
    print(f"  • 'Analyze the tech stack'")
    print(f"  • 'Show recent activity'")
    print(f"  • 'Tell me about [repo-name]'")
    print("\nType 'exit' to quit or 'switch' to change user\n")
    
    # Interactive loop
    while True:
        try:
            query = input(f"🔍 [{username}] > ").strip()
            
            if query.lower() in ['exit', 'quit']:
                print("\n👋 Goodbye!")
                break
            
            if query.lower() == 'switch':
                # Switch to a new user
                new_username = input("\n🔗 Enter new GitHub username: ").strip()
                if new_username:
                    try:
                        validated_username = validate_username(new_username)
                        agent_executor, new_validated_username = await create_user_focused_agent(validated_username)
                        username = new_validated_username
                        print(f"✅ Switched to @{username}\n")
                    except Exception as e:
                        print(f"❌ Error: {e}")
                continue
            
            if not query:
                continue
            
            print("\n🤔 Analyzing...\n")
            
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
    
    # Cleanup
    if toolkit:
        await toolkit.cleanup()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Quick mode with username
        username = sys.argv[1]
        
        async def quick_run():
            try:
                agent_executor, validated_username = await create_user_focused_agent(username)
                print(f"🐙 GitHub Agent for @{validated_username}\n")
                
                if len(sys.argv) > 2:
                    # Run specific query
                    query = " ".join(sys.argv[2:])
                    result = await agent_executor.ainvoke({"input": query})
                    print(result["output"])
                else:
                    # Get profile
                    result = await agent_executor.ainvoke({"input": "show profile and statistics"})
                    print(result["output"])
                
            except Exception as e:
                print(f"❌ Error: {e}")
            finally:
                if toolkit:
                    await toolkit.cleanup()
        
        asyncio.run(quick_run())
    else:
        # Interactive mode
        asyncio.run(main())
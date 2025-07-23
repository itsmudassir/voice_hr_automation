#!/usr/bin/env python3
"""
GitHub User Repository Chat
Chat with any GitHub user's repositories
"""

import asyncio
import os
import json
from dotenv import load_dotenv
from typing import Dict, Any, Optional

# MCP imports
from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession

# OpenAI imports
from openai import AsyncOpenAI

load_dotenv()


class GitHubUserChat:
    """Chat interface for any GitHub user's repositories"""
    
    def __init__(self):
        self.github_pat = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if not self.github_pat:
            raise ValueError("Missing GITHUB_PERSONAL_ACCESS_TOKEN")
        if not self.openai_api_key:
            raise ValueError("Missing OPENAI_API_KEY")
        
        self.openai = AsyncOpenAI(api_key=self.openai_api_key)
        self.session = None
        self.current_user = None
        self.user_repos = None
    
    async def connect(self):
        """Connect to GitHub MCP"""
        print("🔌 Connecting to GitHub...")
        
        self.mcp_context = streamablehttp_client(
            "https://api.githubcopilot.com/mcp/",
            headers={"Authorization": f"Bearer {self.github_pat}"}
        )
        
        self.read, self.write, _ = await self.mcp_context.__aenter__()
        self.session = ClientSession(self.read, self.write)
        await self.session.__aenter__()
        await self.session.initialize()
        
        print("✅ Connected to GitHub MCP")
    
    async def set_user(self, username: str):
        """Set the current user to chat about"""
        self.current_user = username
        print(f"\n👤 Loading repositories for user: {username}...")
        
        # Get user's repositories
        result = await self.session.call_tool("search_repositories", {
            "query": f"user:{username}",
            "perPage": 100
        })
        
        if result.content:
            data = json.loads(result.content[0].text)
            if 'items' in data:
                self.user_repos = data['items']
                total = data.get('total_count', len(self.user_repos))
                public_count = sum(1 for r in self.user_repos if not r.get('private'))
                
                print(f"✅ Found {total} repositories for {username}")
                print(f"   📂 {public_count} public, {len(self.user_repos) - public_count} private")
                
                # Show language breakdown
                languages = {}
                for repo in self.user_repos:
                    lang = repo.get('language', 'Unknown')
                    if lang:
                        languages[lang] = languages.get(lang, 0) + 1
                
                if languages:
                    print(f"   🔤 Languages: {', '.join(f'{k} ({v})' for k, v in sorted(languages.items(), key=lambda x: x[1], reverse=True)[:5])}")
                
                return True
            else:
                print(f"❌ No repositories found for user: {username}")
                return False
        else:
            print(f"❌ Failed to fetch repositories for user: {username}")
            return False
    
    async def search_user_repos(self, query: str) -> Dict[str, Any]:
        """Search within the current user's repositories"""
        if not self.current_user:
            return {"error": "No user selected"}
        
        # Search with user constraint
        full_query = f"user:{self.current_user} {query}"
        
        result = await self.session.call_tool("search_repositories", {
            "query": full_query,
            "perPage": 30
        })
        
        if result.content:
            return json.loads(result.content[0].text)
        return {"error": "No results found"}
    
    async def get_repo_details(self, repo_name: str) -> Optional[Dict[str, Any]]:
        """Get details for a specific repository"""
        if not self.user_repos:
            return None
        
        # Find the repo in cached data
        for repo in self.user_repos:
            if repo['name'].lower() == repo_name.lower():
                return repo
        
        # If not found, try to fetch it
        result = await self.session.call_tool("search_repositories", {
            "query": f"user:{self.current_user} repo:{repo_name}",
            "perPage": 1
        })
        
        if result.content:
            data = json.loads(result.content[0].text)
            if data.get('items'):
                return data['items'][0]
        
        return None
    
    async def chat(self, message: str) -> str:
        """Process a chat message about the current user's repos"""
        
        if not self.current_user:
            return "❌ No user selected. Use 'user <username>' to select a GitHub user."
        
        message_lower = message.lower()
        
        # Determine action and get data
        github_data = None
        context = f"Current user: {self.current_user}"
        
        if any(word in message_lower for word in ["all repo", "list repo", "show repo", "all project"]):
            # List all repos
            github_data = {"items": self.user_repos, "total_count": len(self.user_repos)}
            context += f"\nShowing all {len(self.user_repos)} repositories"
            
        elif any(word in message_lower for word in ["popular", "starred", "top", "best"]):
            # Sort by stars
            sorted_repos = sorted(self.user_repos, key=lambda x: x.get('stargazers_count', 0), reverse=True)
            github_data = {"items": sorted_repos[:10], "showing": "top 10 by stars"}
            context += "\nShowing most popular repositories"
            
        elif any(word in message_lower for word in ["recent", "latest", "new"]):
            # Sort by update date
            sorted_repos = sorted(self.user_repos, key=lambda x: x.get('updated_at', ''), reverse=True)
            github_data = {"items": sorted_repos[:10], "showing": "10 most recently updated"}
            context += "\nShowing recently updated repositories"
            
        elif "language:" in message_lower or any(lang in message_lower for lang in ["python", "javascript", "java", "typescript", "go", "rust"]):
            # Filter by language
            lang_match = None
            for lang in ["python", "javascript", "java", "typescript", "go", "rust", "c++", "c#"]:
                if lang in message_lower:
                    lang_match = lang
                    break
            
            if lang_match:
                filtered = [r for r in self.user_repos if r.get('language', '').lower() == lang_match]
                github_data = {"items": filtered, "filter": f"language={lang_match}"}
                context += f"\nFiltered by language: {lang_match}"
            
        elif "search" in message_lower or "find" in message_lower:
            # Search within user's repos
            search_terms = message.replace("search", "").replace("find", "").strip()
            github_data = await self.search_user_repos(search_terms)
            context += f"\nSearching for: {search_terms}"
            
        else:
            # Try to find a specific repo or general query
            words = message.split()
            for word in words:
                repo = await self.get_repo_details(word)
                if repo:
                    github_data = {"items": [repo], "showing": "specific repository"}
                    context += f"\nShowing details for repository: {word}"
                    break
            
            if not github_data:
                # Default to showing all repos with the query context
                github_data = {"items": self.user_repos, "total_count": len(self.user_repos)}
                context += f"\nQuery: {message}"
        
        # Use OpenAI to format response
        system_prompt = f"""You are a helpful GitHub repository assistant.
        {context}
        
        The user asked: "{message}"
        
        Format the GitHub data into a clear, friendly response.
        For repositories, show:
        - Name and description
        - Primary language and stars (if > 0)
        - Last updated (if relevant)
        - Any other relevant details based on the query
        
        Be concise but informative. Group or summarize if there are many results."""
        
        # Prepare data for OpenAI (limit size)
        data_str = json.dumps(github_data, indent=2)
        if len(data_str) > 3000:
            # Truncate items if too long
            if 'items' in github_data and len(github_data['items']) > 10:
                github_data['items'] = github_data['items'][:10]
                github_data['truncated'] = True
            data_str = json.dumps(github_data, indent=2)
        
        response = await self.openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"GitHub data:\n{data_str}"}
            ],
            temperature=0.7,
            max_tokens=600
        )
        
        return response.choices[0].message.content
    
    async def cleanup(self):
        """Clean up connections"""
        if self.session:
            await self.session.__aexit__(None, None, None)
        if hasattr(self, 'mcp_context'):
            await self.mcp_context.__aexit__(None, None, None)


async def main():
    """Main chat interface"""
    print("🐙 GitHub User Repository Chat")
    print("=" * 50)
    print("Chat with any GitHub user's repositories!")
    print("\nCommands:")
    print("  • user <username> - Switch to a different user")
    print("  • help - Show available commands")
    print("  • exit - Quit the chat")
    print("=" * 50)
    
    chat = GitHubUserChat()
    
    try:
        await chat.connect()
        
        # Ask for initial user
        print("\n📝 Enter a GitHub username to start chatting about their repositories")
        print("   (e.g., 'torvalds', 'gvanrossum', 'itsmudassir')")
        
        while True:
            username = input("\nGitHub username: ").strip()
            if username:
                success = await chat.set_user(username)
                if success:
                    break
                else:
                    print("Try another username...")
            else:
                print("Please enter a username")
        
        print(f"\n💬 Now chatting about {chat.current_user}'s repositories!")
        print("Ask me anything about their repos...\n")
        
        # Main chat loop
        while True:
            user_input = input(f"You [{chat.current_user}]: ").strip()
            
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("\nGitHub: Goodbye! 👋")
                break
            
            elif user_input.lower() == 'help':
                print("\n📚 Available queries:")
                print("  • 'show all repos' - List all repositories")
                print("  • 'show popular repos' - Show most starred repositories")
                print("  • 'show recent repos' - Show recently updated repos")
                print("  • 'find python projects' - Search for specific repos")
                print("  • 'show <repo-name>' - Get details about a specific repo")
                print("  • 'python repos' - Filter by programming language")
                print("  • 'user <username>' - Switch to different user\n")
                continue
            
            elif user_input.lower().startswith('user '):
                # Switch user
                new_username = user_input[5:].strip()
                if new_username:
                    success = await chat.set_user(new_username)
                    if success:
                        print(f"\n💬 Now chatting about {chat.current_user}'s repositories!\n")
                continue
            
            elif not user_input:
                continue
            
            # Get and display response
            print(f"\nGitHub: ", end="", flush=True)
            response = await chat.chat(user_input)
            print(response)
            print()
    
    except KeyboardInterrupt:
        print("\n\nGitHub: Goodbye! 👋")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        await chat.cleanup()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Quick mode: python github_user_chat.py <username>
        username = sys.argv[1]
        
        async def quick_chat():
            chat = GitHubUserChat()
            await chat.connect()
            success = await chat.set_user(username)
            
            if success:
                print(f"\n💬 Quick mode: Chatting about {username}'s repos")
                print("Type your questions (or 'exit' to quit):\n")
                
                while True:
                    try:
                        query = input("> ").strip()
                        if query.lower() in ['exit', 'quit']:
                            break
                        if query:
                            response = await chat.chat(query)
                            print(f"\n{response}\n")
                    except KeyboardInterrupt:
                        break
            
            await chat.cleanup()
        
        asyncio.run(quick_chat())
    else:
        asyncio.run(main())
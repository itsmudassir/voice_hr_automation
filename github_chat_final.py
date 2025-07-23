#!/usr/bin/env python3
"""
GitHub User Chat - Final Version
Simple and robust chat interface for any GitHub user's repositories
"""

import asyncio
import os
import json
import sys
from dotenv import load_dotenv
from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession
from openai import OpenAI

load_dotenv()


class GitHubRepoChat:
    def __init__(self):
        self.github_pat = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if not self.github_pat or not self.openai_api_key:
            raise ValueError("Missing required environment variables")
        
        self.openai = OpenAI(api_key=self.openai_api_key)
        self.current_user = None
        self.repos_data = None
    
    async def get_user_repos(self, username):
        """Fetch repositories for a specific user"""
        async with streamablehttp_client(
            "https://api.githubcopilot.com/mcp/",
            headers={"Authorization": f"Bearer {self.github_pat}"}
        ) as (read, write, _):
            
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                # Get repositories
                result = await session.call_tool("search_repositories", {
                    "query": f"user:{username}",
                    "perPage": 100
                })
                
                if result.content:
                    data = json.loads(result.content[0].text)
                    return data
                return None
    
    def process_query(self, query, repos_data):
        """Process a query about repositories using OpenAI"""
        
        if not repos_data or 'items' not in repos_data:
            return "No repository data available."
        
        repos = repos_data['items']
        
        # Create a simplified version of repo data for OpenAI
        simplified_repos = []
        for repo in repos:
            simplified_repos.append({
                'name': repo['name'],
                'description': repo.get('description', 'No description'),
                'language': repo.get('language', 'Unknown'),
                'stars': repo.get('stargazers_count', 0),
                'updated': repo.get('updated_at', 'Unknown'),
                'url': repo.get('html_url', ''),
                'private': repo.get('private', False)
            })
        
        # Create the prompt
        system_prompt = f"""You are a helpful assistant that answers questions about GitHub user {self.current_user}'s repositories.
        You have data about {len(simplified_repos)} repositories.
        
        Answer the user's question based on the repository data provided.
        Be concise and informative. Format lists nicely.
        If asked to list all repos, show them in a clean format with key details."""
        
        # Prepare the message
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Repository data: {json.dumps(simplified_repos[:50], indent=2)}"},  # Limit to 50 repos
            {"role": "user", "content": f"Question: {query}"}
        ]
        
        # Get response from OpenAI
        response = self.openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7,
            max_tokens=800
        )
        
        return response.choices[0].message.content


async def main():
    """Main function"""
    chat = GitHubRepoChat()
    
    print("🐙 GitHub Repository Chat")
    print("=" * 50)
    
    # Get username from command line or ask for it
    if len(sys.argv) > 1:
        username = sys.argv[1]
    else:
        username = input("Enter GitHub username: ").strip()
    
    if not username:
        print("❌ No username provided")
        return
    
    print(f"\n📋 Loading repositories for {username}...")
    
    # Fetch repositories
    try:
        repos_data = await chat.get_user_repos(username)
        
        if repos_data and 'items' in repos_data:
            chat.current_user = username
            chat.repos_data = repos_data
            
            total = repos_data.get('total_count', len(repos_data['items']))
            print(f"✅ Found {total} repositories")
            
            # Count languages
            languages = {}
            for repo in repos_data['items']:
                lang = repo.get('language')
                if lang:
                    languages[lang] = languages.get(lang, 0) + 1
            
            if languages:
                top_langs = sorted(languages.items(), key=lambda x: x[1], reverse=True)[:5]
                print(f"🔤 Top languages: {', '.join(f'{l[0]} ({l[1]})' for l in top_langs)}")
            
            print(f"\n💬 Chat about {username}'s repositories")
            print("Type 'exit' to quit\n")
            
            # Chat loop
            while True:
                query = input("> ").strip()
                
                if query.lower() in ['exit', 'quit']:
                    print("👋 Goodbye!")
                    break
                
                if not query:
                    continue
                
                # Process query
                print("\n" + chat.process_query(query, repos_data))
                print()
        
        else:
            print(f"❌ No repositories found for user: {username}")
    
    except Exception as e:
        print(f"❌ Error: {e}")


# Quick functions for specific queries
async def list_user_repos(username):
    """Quick function to list all repos"""
    chat = GitHubRepoChat()
    
    print(f"📋 Fetching all repositories for {username}...\n")
    
    repos_data = await chat.get_user_repos(username)
    
    if repos_data and 'items' in repos_data:
        repos = repos_data['items']
        
        # Sort by stars
        repos.sort(key=lambda x: x.get('stargazers_count', 0), reverse=True)
        
        print(f"Found {len(repos)} repositories:\n")
        
        # Group by visibility
        public_repos = [r for r in repos if not r.get('private')]
        private_repos = [r for r in repos if r.get('private')]
        
        if public_repos:
            print("🌍 PUBLIC REPOSITORIES:")
            print("-" * 40)
            for i, repo in enumerate(public_repos, 1):
                stars = repo.get('stargazers_count', 0)
                lang = repo.get('language', 'Unknown')
                desc = repo.get('description', 'No description')[:60]
                print(f"{i:3d}. {repo['name']:<30} ⭐ {stars:<6} [{lang}]")
                if desc:
                    print(f"     {desc}...")
        
        if private_repos:
            print("\n🔒 PRIVATE REPOSITORIES:")
            print("-" * 40)
            for i, repo in enumerate(private_repos, 1):
                print(f"{i:3d}. {repo['name']}")
    else:
        print(f"❌ No repositories found for {username}")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "list":
        # Quick list mode: python github_chat_final.py list <username>
        asyncio.run(list_user_repos(sys.argv[2]))
    else:
        # Interactive chat mode
        asyncio.run(main())
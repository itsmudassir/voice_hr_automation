#!/usr/bin/env python3
"""
Simple GitHub Terminal Chat
Direct conversation with GitHub repositories
"""

import asyncio
import os
import json
from dotenv import load_dotenv
from typing import Dict, Any

# MCP imports
from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession

# OpenAI imports
from openai import AsyncOpenAI

load_dotenv()


class SimpleGitHubChat:
    """Simple GitHub chat interface"""
    
    def __init__(self):
        self.github_pat = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if not self.github_pat:
            raise ValueError("Missing GITHUB_PERSONAL_ACCESS_TOKEN")
        if not self.openai_api_key:
            raise ValueError("Missing OPENAI_API_KEY")
        
        self.openai = AsyncOpenAI(api_key=self.openai_api_key)
        self.session = None
        self.username = None
    
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
        
        # Get username
        result = await self.session.call_tool("get_me", {})
        if result.content:
            user_data = json.loads(result.content[0].text)
            self.username = user_data.get('login', 'Unknown')
            print(f"✅ Connected as: {self.username}")
    
    async def search_my_repos(self) -> Dict[str, Any]:
        """Get all user's repositories"""
        if not self.username:
            return {"error": "Not connected"}
        
        result = await self.session.call_tool("search_repositories", {
            "query": f"user:{self.username}",
            "perPage": 100
        })
        
        if result.content:
            return json.loads(result.content[0].text)
        return {"error": "No repositories found"}
    
    async def search_repos(self, query: str) -> Dict[str, Any]:
        """Search for repositories"""
        result = await self.session.call_tool("search_repositories", {
            "query": query,
            "perPage": 20
        })
        
        if result.content:
            return json.loads(result.content[0].text)
        return {"error": "No results found"}
    
    async def get_notifications(self) -> list:
        """Get user notifications"""
        result = await self.session.call_tool("list_notifications", {
            "perPage": 10
        })
        
        if result.content:
            return json.loads(result.content[0].text)
        return []
    
    async def chat(self, message: str) -> str:
        """Process a chat message and return response"""
        
        # Analyze the message to determine action
        message_lower = message.lower()
        
        # Execute appropriate GitHub action
        github_data = None
        action_taken = ""
        
        if "my repo" in message_lower or "list repo" in message_lower:
            github_data = await self.search_my_repos()
            action_taken = "fetching your repositories"
        elif "notification" in message_lower:
            github_data = await self.get_notifications()
            action_taken = "checking your notifications"
        elif "search" in message_lower or "find" in message_lower:
            # Extract search query
            search_terms = message.replace("search", "").replace("find", "").strip()
            github_data = await self.search_repos(search_terms)
            action_taken = f"searching for {search_terms}"
        else:
            # General search based on the message
            github_data = await self.search_repos(message)
            action_taken = "searching GitHub"
        
        # Use OpenAI to format the response
        system_prompt = f"""You are a helpful GitHub assistant. 
        The user asked: "{message}"
        You performed: {action_taken}
        
        Format the GitHub data into a clear, friendly response.
        For repositories, show name, description, stars, and language.
        Keep it concise but informative."""
        
        response = await self.openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"GitHub data: {json.dumps(github_data, indent=2)[:2000]}"}
            ],
            temperature=0.7,
            max_tokens=500
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
    print("🐙 GitHub Terminal Chat")
    print("=" * 50)
    print("Chat naturally about your GitHub repositories!")
    print("Examples: 'show my repos', 'find python AI projects', 'check notifications'")
    print("Type 'exit' to quit\n")
    
    chat = SimpleGitHubChat()
    
    try:
        await chat.connect()
        print("\n💬 Ready to chat!\n")
        
        while True:
            # Get user input
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("\nGitHub: Goodbye! 👋")
                break
            
            if not user_input:
                continue
            
            # Get and display response
            print("\nGitHub: ", end="", flush=True)
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
    asyncio.run(main())
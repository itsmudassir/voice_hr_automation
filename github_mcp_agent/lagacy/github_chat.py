#!/usr/bin/env python3
"""
GitHub Repository Chat - Interactive Terminal Interface
Chat with your GitHub repositories using natural language
"""

import asyncio
import os
import json
from dotenv import load_dotenv
from datetime import datetime
from typing import Dict, Any, List

# MCP imports
from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession

# OpenAI imports
from openai import AsyncOpenAI

load_dotenv()


class GitHubChat:
    """Interactive GitHub chat interface"""
    
    def __init__(self):
        self.github_pat = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if not self.github_pat:
            raise ValueError("Missing GITHUB_PERSONAL_ACCESS_TOKEN")
        if not self.openai_api_key:
            raise ValueError("Missing OPENAI_API_KEY")
        
        self.openai = AsyncOpenAI(api_key=self.openai_api_key)
        self.session = None
        self.user_info = None
        self.chat_history = []
    
    async def initialize(self):
        """Initialize MCP session and get user info"""
        print("🔧 Connecting to GitHub...")
        
        # Create MCP connection
        self.mcp_context = streamablehttp_client(
            "https://api.githubcopilot.com/mcp/",
            headers={"Authorization": f"Bearer {self.github_pat}"}
        )
        
        self.read, self.write, _ = await self.mcp_context.__aenter__()
        self.session = ClientSession(self.read, self.write)
        await self.session.__aenter__()
        await self.session.initialize()
        
        # Get user info
        result = await self.session.call_tool("get_me", {})
        if result.content:
            self.user_info = json.loads(result.content[0].text)
            print(f"✅ Connected as: {self.user_info.get('login', 'Unknown')}")
        
        print("💬 Ready to chat! Type 'help' for commands or 'exit' to quit.\n")
    
    async def cleanup(self):
        """Clean up connections"""
        if self.session:
            await self.session.__aexit__(None, None, None)
        if hasattr(self, 'mcp_context'):
            await self.mcp_context.__aexit__(None, None, None)
    
    async def process_command(self, user_input: str) -> str:
        """Process user input and generate response"""
        
        # Add to chat history
        self.chat_history.append({"role": "user", "content": user_input})
        
        # Determine what tools to use based on the query
        tools_to_use = await self._determine_tools(user_input)
        
        # Execute the tools and get results
        tool_results = await self._execute_tools(tools_to_use)
        
        # Generate response using OpenAI
        response = await self._generate_response(user_input, tool_results)
        
        # Add to chat history
        self.chat_history.append({"role": "assistant", "content": response})
        
        return response
    
    async def _determine_tools(self, query: str) -> List[Dict[str, Any]]:
        """Determine which GitHub tools to use based on the query"""
        
        # Simple keyword-based tool selection
        tools = []
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["my repo", "my project", "list repo", "show repo"]):
            username = self.user_info.get('login') if self.user_info else None
            if username:
                tools.append({
                    "tool": "search_repositories",
                    "params": {"query": f"user:{username}", "perPage": 30}
                })
        
        elif any(word in query_lower for word in ["search", "find", "popular", "trending"]):
            # Extract search terms
            search_query = query
            if "python" in query_lower:
                search_query = "language:python " + search_query
            tools.append({
                "tool": "search_repositories",
                "params": {"query": search_query, "perPage": 10}
            })
        
        elif any(word in query_lower for word in ["issue", "bug", "problem"]):
            # Need to extract repo info from context
            tools.append({
                "tool": "list_issues",
                "params": {"owner": "owner", "repo": "repo"}  # Would need context
            })
        
        elif any(word in query_lower for word in ["notification", "alert", "update"]):
            tools.append({
                "tool": "list_notifications",
                "params": {"perPage": 10}
            })
        
        return tools
    
    async def _execute_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute the selected tools and return results"""
        results = []
        
        for tool_config in tools:
            tool_name = tool_config["tool"]
            params = tool_config["params"]
            
            try:
                result = await self.session.call_tool(tool_name, params)
                if result.content:
                    data = json.loads(result.content[0].text)
                    results.append({
                        "tool": tool_name,
                        "success": True,
                        "data": data
                    })
                else:
                    results.append({
                        "tool": tool_name,
                        "success": False,
                        "error": "No content returned"
                    })
            except Exception as e:
                results.append({
                    "tool": tool_name,
                    "success": False,
                    "error": str(e)
                })
        
        return results
    
    async def _generate_response(self, query: str, tool_results: List[Dict[str, Any]]) -> str:
        """Generate a natural language response using OpenAI"""
        
        # Build context from tool results
        context = "Tool results:\n"
        for result in tool_results:
            if result["success"]:
                context += f"\n{result['tool']} returned: {json.dumps(result['data'], indent=2)[:1000]}...\n"
            else:
                context += f"\n{result['tool']} failed: {result['error']}\n"
        
        # Create messages for OpenAI
        messages = [
            {"role": "system", "content": """You are a helpful GitHub assistant. 
            Based on the tool results provided, answer the user's question in a clear, 
            concise, and friendly manner. Format lists nicely and highlight important information."""},
            {"role": "user", "content": f"User query: {query}\n\n{context}"}
        ]
        
        # Get response from OpenAI
        response = await self.openai.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        
        return response.choices[0].message.content
    
    def show_help(self):
        """Show help message"""
        help_text = """
🔍 GitHub Chat Commands:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📂 Repository Commands:
  • "list my repos" - Show all your repositories
  • "search [topic]" - Search for repositories
  • "find python machine learning" - Search with filters

📊 Information Commands:
  • "my profile" - Show your GitHub profile
  • "notifications" - Check your notifications
  • "trending python" - Find trending repositories

💡 Examples:
  • "Show me my most popular repositories"
  • "Find the best React libraries"
  • "What Python projects am I working on?"

🔧 System Commands:
  • help - Show this help message
  • clear - Clear the screen
  • exit/quit - Exit the chat

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        """
        print(help_text)


async def main():
    """Main chat loop"""
    print("🐙 GitHub Repository Chat")
    print("=" * 60)
    
    # Initialize chat
    chat = GitHubChat()
    
    try:
        await chat.initialize()
        
        # Show initial help
        chat.show_help()
        
        # Main chat loop
        while True:
            try:
                # Get user input
                user_input = input("\n🐙 > ").strip()
                
                # Handle commands
                if user_input.lower() in ['exit', 'quit']:
                    print("\n👋 Goodbye!")
                    break
                elif user_input.lower() == 'help':
                    chat.show_help()
                    continue
                elif user_input.lower() == 'clear':
                    os.system('clear' if os.name == 'posix' else 'cls')
                    continue
                elif not user_input:
                    continue
                
                # Process the command
                print("\n🤔 Thinking...")
                response = await chat.process_command(user_input)
                print(f"\n{response}")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
    
    finally:
        await chat.cleanup()


# Quick start commands for common queries
async def quick_list_repos():
    """Quick command to list user's repositories"""
    chat = GitHubChat()
    await chat.initialize()
    response = await chat.process_command("list all my repositories")
    print(response)
    await chat.cleanup()


async def quick_search(query: str):
    """Quick command to search repositories"""
    chat = GitHubChat()
    await chat.initialize()
    response = await chat.process_command(f"search for {query}")
    print(response)
    await chat.cleanup()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "list":
            asyncio.run(quick_list_repos())
        elif sys.argv[1] == "search" and len(sys.argv) > 2:
            asyncio.run(quick_search(" ".join(sys.argv[2:])))
        else:
            print("Usage: python github_chat.py [list | search <query>]")
    else:
        asyncio.run(main())
#!/usr/bin/env python3
"""
Direct GitHub MCP Usage
Shows how to use GitHub MCP tools directly without LangChain agents
"""

import asyncio
import os
import json
from dotenv import load_dotenv
from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession

load_dotenv()


async def use_github_mcp_directly():
    """Use GitHub MCP tools directly via session"""
    
    GITHUB_PAT = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")
    
    if not GITHUB_PAT:
        print("❌ Missing GITHUB_PERSONAL_ACCESS_TOKEN in .env")
        return
    
    print("🚀 Direct GitHub MCP Usage")
    print("=" * 50)
    
    # Create direct MCP connection
    async with streamablehttp_client(
        "https://api.githubcopilot.com/mcp/",
        headers={"Authorization": f"Bearer {GITHUB_PAT}"}
    ) as (read, write, _):
        
        async with ClientSession(read, write) as session:
            # Initialize session
            print("📡 Initializing MCP session...")
            await session.initialize()
            print("✅ Session initialized!")
            
            # List available tools
            print("\n📋 Available tools:")
            tools_response = await session.list_tools()
            print(f"✅ Found {len(tools_response.tools)} tools")
            
            # Show first few tools
            for tool in tools_response.tools[:5]:
                print(f"   • {tool.name}")
            
            # Example 1: Get user info
            print("\n\n1️⃣ Getting authenticated user info...")
            try:
                result = await session.call_tool("get_me", {})
                if result.content:
                    data = json.loads(result.content[0].text)
                    print(f"✅ Authenticated as: {data.get('login', 'Unknown')}")
                    print(f"   Name: {data.get('name', 'N/A')}")
                    print(f"   Public repos: {data.get('public_repos', 0)}")
                    username = data.get('login')
            except Exception as e:
                print(f"❌ Error: {e}")
                username = None
            
            # Example 2: Search repositories
            print("\n\n2️⃣ Searching for popular Python repositories...")
            try:
                result = await session.call_tool("search_repositories", {
                    "query": "language:python stars:>10000",
                    "perPage": 5
                })
                if result.content:
                    data = json.loads(result.content[0].text)
                    if 'items' in data:
                        print(f"✅ Found {data.get('total_count', 0)} total repositories")
                        print("   Top 5:")
                        for repo in data['items'][:5]:
                            print(f"   • {repo['full_name']} ⭐ {repo['stargazers_count']}")
            except Exception as e:
                print(f"❌ Error: {e}")
            
            # Example 3: List your repositories
            if username:
                print(f"\n\n3️⃣ Listing repositories for {username}...")
                try:
                    result = await session.call_tool("search_repositories", {
                        "query": f"user:{username}",
                        "perPage": 10
                    })
                    if result.content:
                        data = json.loads(result.content[0].text)
                        if 'items' in data:
                            print(f"✅ Found {len(data['items'])} repositories:")
                            for repo in data['items']:
                                visibility = "🔒" if repo.get('private') else "🌍"
                                print(f"   • {visibility} {repo['name']}")
                except Exception as e:
                    print(f"❌ Error: {e}")
            
            # Example 4: Check notifications
            print("\n\n4️⃣ Checking notifications...")
            try:
                result = await session.call_tool("list_notifications", {
                    "perPage": 5
                })
                if result.content:
                    data = json.loads(result.content[0].text)
                    if isinstance(data, list):
                        if len(data) == 0:
                            print("✅ No unread notifications!")
                        else:
                            print(f"✅ You have {len(data)} notifications")
            except Exception as e:
                print(f"❌ Error: {e}")
    
    print("\n\n✨ Direct GitHub MCP usage successful!")
    print("\nKey points:")
    print("  • Use streamablehttp_client for connection")
    print("  • Create ClientSession for tool calls")
    print("  • Call tools with session.call_tool(name, params)")
    print("  • Results come back as JSON in result.content[0].text")


if __name__ == "__main__":
    asyncio.run(use_github_mcp_directly())
#!/usr/bin/env python3
"""
List all repository names for a GitHub user
"""

import asyncio
import os
import json
from dotenv import load_dotenv
from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession

load_dotenv()


async def list_all_repos():
    """List all repositories for the authenticated user"""
    
    GITHUB_PAT = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")
    
    if not GITHUB_PAT:
        print("❌ Missing GITHUB_PERSONAL_ACCESS_TOKEN")
        return
    
    print("🔍 Fetching all repositories for itsmudassir...")
    print("=" * 60)
    
    # Create MCP connection
    async with streamablehttp_client(
        "https://api.githubcopilot.com/mcp/",
        headers={"Authorization": f"Bearer {GITHUB_PAT}"}
    ) as (read, write, _):
        
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # Search for all repos with higher page limit
            result = await session.call_tool("search_repositories", {
                "query": "user:itsmudassir",
                "perPage": 100  # Maximum allowed
            })
            
            if result.content:
                data = json.loads(result.content[0].text)
                
                if 'items' in data:
                    repos = data['items']
                    total_count = data.get('total_count', len(repos))
                    
                    print(f"✅ Found {total_count} repositories")
                    print(f"📋 Showing {len(repos)} repositories:\n")
                    
                    # Group by visibility
                    public_repos = []
                    private_repos = []
                    
                    for repo in repos:
                        if repo.get('private'):
                            private_repos.append(repo['name'])
                        else:
                            public_repos.append(repo['name'])
                    
                    # Print public repos
                    if public_repos:
                        print("🌍 PUBLIC REPOSITORIES:")
                        print("-" * 40)
                        for i, name in enumerate(sorted(public_repos), 1):
                            print(f"{i:3d}. {name}")
                    
                    # Print private repos
                    if private_repos:
                        print(f"\n🔒 PRIVATE REPOSITORIES:")
                        print("-" * 40)
                        for i, name in enumerate(sorted(private_repos), 1):
                            print(f"{i:3d}. {name}")
                    
                    # Summary
                    print(f"\n📊 SUMMARY:")
                    print(f"   Total: {len(repos)} repositories")
                    print(f"   Public: {len(public_repos)}")
                    print(f"   Private: {len(private_repos)}")
                    
                    if total_count > len(repos):
                        print(f"\n⚠️  Note: Showing first {len(repos)} of {total_count} total repositories")
                        print("   (GitHub API limits results per request)")
                else:
                    print("❌ No repositories found")
            else:
                print("❌ Failed to fetch repositories")


if __name__ == "__main__":
    asyncio.run(list_all_repos())
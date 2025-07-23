#!/usr/bin/env python3

import asyncio
from github_user_focused_agent import create_user_focused_agent

async def test():
    agent, username = await create_user_focused_agent("https://github.com/MJunaidAhmad/")
    
    # Search for WordPress skills
    result1 = await agent.ainvoke({"input": "search for wordpress projects"})
    print("=== WORDPRESS SEARCH ===")
    print(result1["output"])
    
    # Check PHP experience  
    result2 = await agent.ainvoke({"input": "search for PHP projects"})
    print("\n=== PHP SEARCH ===")
    print(result2["output"])
    
    # Tech stack analysis
    result3 = await agent.ainvoke({"input": "analyze tech stack"})
    print("\n=== TECH STACK ===")
    print(result3["output"])
    
    # Recent activity
    result4 = await agent.ainvoke({"input": "show recent activity"})
    print("\n=== RECENT ACTIVITY ===")
    print(result4["output"])

asyncio.run(test())
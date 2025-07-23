#!/usr/bin/env python3
"""
Test script for GitHub Evaluation Agent
Demonstrates how to evaluate a candidate for different roles
"""

import asyncio
import sys
from github_evaluation_agent import create_evaluation_agent


async def test_evaluation():
    """Test the evaluation agent with sample queries"""
    
    if len(sys.argv) < 2:
        print("Usage: python test_evaluation_agent.py <github_username_or_url>")
        print("\nExamples:")
        print("  python test_evaluation_agent.py torvalds")
        print("  python test_evaluation_agent.py @torvalds")
        print("  python test_evaluation_agent.py https://github.com/torvalds")
        sys.exit(1)
    
    username = sys.argv[1]
    
    print(f"🔍 Testing GitHub Evaluation Agent for @{username}")
    print("=" * 60)
    
    try:
        # Create agent
        agent = await create_evaluation_agent(username)
        print(f"✅ Agent ready for @{username}\n")
        
        # Test queries
        test_queries = [
            "Evaluate for a Backend Developer role",
            "What are the top technical skills?",
            "Show collaboration metrics",
            "Identify any red flags",
            "Generate a hiring recommendation"
        ]
        
        for query in test_queries:
            print(f"\n📋 Query: {query}")
            print("-" * 40)
            
            result = await agent.ainvoke({"input": query})
            print(result["output"])
            print("=" * 60)
            
            # Small delay between queries
            await asyncio.sleep(1)
    
    except Exception as e:
        print(f"❌ Error: {e}")
    
    finally:
        # Cleanup
        from github_evaluation_agent import toolkit
        if toolkit:
            await toolkit.cleanup()


if __name__ == "__main__":
    asyncio.run(test_evaluation())
#!/usr/bin/env python3
"""
GitHub AI Evaluator - Actually intelligent evaluation
No hardcoded skills bullshit - AI figures out what to look for
"""

import asyncio
import os
import json
import re
from datetime import datetime
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool

# MCP imports
from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession

# Load environment variables
load_dotenv()


class GitHubMCPToolkit:
    """GitHub MCP Toolkit"""
    
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
            return {"error": f"MCP error: {str(e)}"}


# Global instances
toolkit: Optional[GitHubMCPToolkit] = None
current_user: Optional[str] = None
current_role: Optional[str] = None


@tool
async def determine_role_requirements(role: str) -> str:
    """
    AI determines what skills, tools, and technologies are needed for a role.
    No hardcoded bullshit - pure AI reasoning.
    
    Args:
        role: The job role to analyze
    """
    # Use GPT to figure out what skills are actually needed
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    prompt = f"""For the role: {role}

List the ACTUAL technical skills, programming languages, frameworks, and tools that someone in this role would use.

Be specific and realistic. For example:
- ML Engineer: Python, TensorFlow/PyTorch, Pandas, Jupyter, scikit-learn, MLflow
- WordPress Developer: PHP, WordPress, MySQL, JavaScript, CSS, WooCommerce
- DevOps: Docker, Kubernetes, Terraform, AWS/GCP, CI/CD, Python/Go
- iOS Developer: Swift, SwiftUI, Xcode, Core Data, CocoaPods

Format as:
LANGUAGES: [list]
FRAMEWORKS: [list]
TOOLS: [list]
KEYWORDS: [list of terms to search for]"""

    response = await llm.ainvoke(prompt)
    return response.content


@tool
async def smart_skill_search(skill_category: str, search_terms: str) -> str:
    """
    Search for evidence of skills using intelligent queries.
    
    Args:
        skill_category: Category like "machine learning", "web development", etc.
        search_terms: Specific terms to search for
    """
    if not current_user:
        return "No user set"
    
    # Search repos
    terms = search_terms.replace(", ", " OR ")
    repo_result = await toolkit.call_tool("search_repositories", {
        "query": f"user:{current_user} {terms}",
        "perPage": 20
    })
    
    # Search code
    primary_term = search_terms.split(",")[0].strip()
    code_result = await toolkit.call_tool("search_code", {
        "q": f"user:{current_user} {primary_term}",
        "perPage": 10
    })
    
    # Format results
    output = f"**{skill_category} Analysis**\n\n"
    
    repo_count = 0
    relevant_repos = []
    
    if repo_result and "items" in repo_result:
        repos = repo_result["items"]
        repo_count = len(repos)
        
        # Get relevant repos with details
        for repo in repos[:5]:
            relevant_repos.append({
                "name": repo["name"],
                "description": repo.get("description", ""),
                "language": repo.get("language", "Unknown"),
                "stars": repo.get("stargazers_count", 0),
                "updated": repo.get("updated_at", "")[:10]
            })
    
    code_files = 0
    if code_result and "items" in code_result:
        code_files = len(code_result["items"])
    
    output += f"**Repositories Found:** {repo_count}\n"
    if relevant_repos:
        output += "**Top Relevant Projects:**\n"
        for r in relevant_repos:
            output += f"- {r['name']} ({r['language']})\n"
            if r['description']:
                output += f"  {r['description'][:80]}...\n"
            output += f"  ⭐ {r['stars']} | Updated: {r['updated']}\n"
    
    output += f"\n**Code Files Found:** {code_files}\n"
    
    # Determine proficiency
    if repo_count >= 5 and code_files >= 5:
        output += "\n**Proficiency Level:** STRONG"
    elif repo_count >= 2 or code_files >= 2:
        output += "\n**Proficiency Level:** MODERATE"
    elif repo_count >= 1 or code_files >= 1:
        output += "\n**Proficiency Level:** BASIC"
    else:
        output += "\n**Proficiency Level:** NONE"
    
    return output


@tool
async def analyze_github_profile_holistically() -> str:
    """
    Get a complete picture of the user's GitHub profile.
    What they actually work on, not what we expect them to work on.
    """
    if not current_user:
        return "No user set"
    
    # Get all repos
    all_repos = await toolkit.call_tool("search_repositories", {
        "query": f"user:{current_user}",
        "sort": "updated",
        "perPage": 100
    })
    
    if not all_repos or "items" not in all_repos:
        return "Unable to analyze profile"
    
    repos = all_repos["items"]
    
    # Analyze what they ACTUALLY work on
    languages = {}
    topics_set = set()
    recent_projects = []
    total_stars = 0
    
    for repo in repos:
        # Language distribution
        lang = repo.get("language")
        if lang:
            languages[lang] = languages.get(lang, 0) + 1
        
        # Topics
        if repo.get("topics"):
            topics_set.update(repo["topics"])
        
        # Stars
        total_stars += repo.get("stargazers_count", 0)
        
        # Recent projects
        if repo.get("updated_at"):
            try:
                updated = datetime.fromisoformat(repo["updated_at"].replace("Z", "+00:00"))
                if (datetime.now(updated.tzinfo) - updated).days < 180:
                    recent_projects.append(repo)
            except:
                pass
    
    output = f"**Actual GitHub Profile Analysis for @{current_user}**\n\n"
    
    output += f"**Profile Stats:**\n"
    output += f"- Total Repositories: {len(repos)}\n"
    output += f"- Total Stars: ⭐ {total_stars}\n"
    output += f"- Recently Active Projects: {len(recent_projects)}\n\n"
    
    output += f"**What They Actually Work With:**\n"
    for lang, count in sorted(languages.items(), key=lambda x: x[1], reverse=True)[:10]:
        percentage = (count / len(repos) * 100) if repos else 0
        output += f"- {lang}: {count} projects ({percentage:.1f}%)\n"
    
    if topics_set:
        output += f"\n**Technologies & Topics They Use:**\n"
        output += ", ".join(sorted(topics_set)[:20])
    
    output += f"\n\n**Recent Focus (Last 6 months):**\n"
    for proj in recent_projects[:5]:
        output += f"- {proj['name']}"
        if proj.get('language'):
            output += f" ({proj['language']})"
        if proj.get('description'):
            output += f": {proj['description'][:60]}..."
        output += "\n"
    
    return output


@tool
async def generate_intelligent_evaluation(role: str) -> str:
    """
    Generate an evaluation based on what the person actually does vs what the role needs.
    No stupid assumptions.
    """
    if not current_user:
        return "No user set"
    
    output = f"**Intelligent Evaluation: @{current_user} for {role}**\n"
    output += "=" * 60 + "\n\n"
    
    # This is just a framework - the actual evaluation happens through
    # the agent combining results from other tools
    output += "Analyzing based on:\n"
    output += "1. What skills this role ACTUALLY needs (not hardcoded assumptions)\n"
    output += "2. What the candidate ACTUALLY works on (not what we expect)\n"
    output += "3. How their real experience matches the real requirements\n"
    
    return output


async def create_intelligent_evaluator(username: str):
    """Create an actually intelligent evaluation agent"""
    
    global toolkit, current_user
    
    # Extract username from URL
    username = username.strip()
    if "github.com/" in username:
        match = re.search(r'github\.com/([^/]+)', username)
        if match:
            username = match.group(1)
    username = username.lstrip('@')
    
    # Validate username
    if not re.match(r'^[a-zA-Z0-9](?:[a-zA-Z0-9]|-(?=[a-zA-Z0-9])){0,38}$', username):
        raise ValueError(f"Invalid GitHub username: {username}")
    
    current_user = username
    
    # Initialize toolkit
    if not toolkit:
        toolkit = await GitHubMCPToolkit().initialize()
    
    # Verify user exists
    test_result = await toolkit.call_tool("search_repositories", {
        "query": f"user:{username}",
        "perPage": 1
    })
    
    if "error" in test_result or test_result.get("total_count", 0) == 0:
        raise ValueError(f"User @{username} not found or has no public repositories")
    
    # Create LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Tools
    tools = [
        determine_role_requirements,
        smart_skill_search,
        analyze_github_profile_holistically,
        generate_intelligent_evaluation
    ]
    
    # Prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are an intelligent GitHub profile evaluator.

Current user: @{username}

Your approach:
1. First, use determine_role_requirements to understand what the role ACTUALLY needs
2. Then, use analyze_github_profile_holistically to see what the person ACTUALLY does
3. Use smart_skill_search to find specific evidence for relevant skills
4. Generate an intelligent evaluation based on real matches, not assumptions

DO NOT assume an ML engineer needs frontend skills.
DO NOT assume a WordPress developer needs Kubernetes.
DO NOT use hardcoded skill lists.

Think intelligently about what matters for each role."""),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad")
    ])
    
    # Create agent
    agent = create_openai_tools_agent(llm, tools, prompt)
    
    # Create executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        handle_parsing_errors=True,
        max_iterations=8
    )
    
    return agent_executor


async def main():
    """Main function"""
    import sys
    
    print("🧠 GitHub AI Evaluator - Actually Intelligent")
    print("=" * 60)
    
    if len(sys.argv) < 2:
        print("Usage: python github_ai_evaluator.py <username/url> [role]")
        print("\nExamples:")
        print('  python github_ai_evaluator.py torvalds')
        print('  python github_ai_evaluator.py https://github.com/AizazSharif "ML Engineer"')
        return
    
    username = sys.argv[1]
    
    try:
        agent = await create_intelligent_evaluator(username)
        print(f"✅ Ready to evaluate @{current_user}\n")
        
        if len(sys.argv) > 2:
            # Direct evaluation
            role = " ".join(sys.argv[2:])
            print(f"📋 Evaluating for: {role}\n")
            
            result = await agent.ainvoke({
                "input": f"Evaluate this person for the role: {role}. First figure out what skills this role needs, then see what they actually have."
            })
            
            print(result["output"])
        else:
            # Interactive mode
            print("Enter role to evaluate (or 'analyze' for general analysis)")
            print("Type 'exit' to quit\n")
            
            while True:
                query = input(f"[{current_user}] > ").strip()
                
                if query.lower() in ['exit', 'quit']:
                    break
                
                if query.lower() == 'analyze':
                    query = "Analyze this person's GitHub profile to understand what they actually work on"
                elif not query.startswith("evaluate"):
                    query = f"Evaluate this person for: {query}"
                
                if not query:
                    continue
                
                print("\n🤔 Thinking intelligently...\n")
                
                result = await agent.ainvoke({"input": query})
                
                print("\n" + "=" * 60)
                print(result["output"])
                print("=" * 60 + "\n")
    
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        if toolkit:
            await toolkit.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
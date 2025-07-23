#!/usr/bin/env python3
"""
GitHub Evaluation Agent V2 - LLM-Aided Evaluation
A more flexible, LLM-driven approach to candidate evaluation
"""

import asyncio
import os
import json
import re
from datetime import datetime
from typing import Dict, Any, Optional
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
    """GitHub MCP Toolkit for evaluation"""
    
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


@tool
async def analyze_github_profile(aspect: str = "general") -> str:
    """
    Analyze a specific aspect of the GitHub profile using intelligent search.
    
    Args:
        aspect: What to analyze (e.g., "wordpress skills", "backend experience", 
                "recent activity", "collaboration", "project quality")
    """
    if not current_user:
        return "No user set for evaluation"
    
    aspect_lower = aspect.lower()
    
    # Intelligently determine what to search for based on aspect
    if "wordpress" in aspect_lower or "wp" in aspect_lower or "cms" in aspect_lower:
        # Search for WordPress-related repositories and code
        repo_result = await toolkit.call_tool("search_repositories", {
            "query": f"user:{current_user} wordpress OR wp OR cms OR php",
            "perPage": 20
        })
        
        code_result = await toolkit.call_tool("search_code", {
            "q": f"user:{current_user} wordpress OR functions.php OR wp-content OR add_action",
            "perPage": 10
        })
        
    elif any(term in aspect_lower for term in ["backend", "api", "server"]):
        repo_result = await toolkit.call_tool("search_repositories", {
            "query": f"user:{current_user} api OR backend OR server OR rest",
            "perPage": 20
        })
        
        code_result = await toolkit.call_tool("search_code", {
            "q": f"user:{current_user} api OR server OR database OR backend",
            "perPage": 10
        })
        
    elif any(term in aspect_lower for term in ["frontend", "ui", "react", "vue"]):
        repo_result = await toolkit.call_tool("search_repositories", {
            "query": f"user:{current_user} frontend OR react OR vue OR ui",
            "perPage": 20
        })
        
        code_result = await toolkit.call_tool("search_code", {
            "q": f"user:{current_user} react OR vue OR component OR jsx",
            "perPage": 10
        })
        
    elif "activity" in aspect_lower or "recent" in aspect_lower:
        # Get recent activity
        from datetime import datetime, timedelta
        since_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
        
        repo_result = await toolkit.call_tool("search_repositories", {
            "query": f"user:{current_user} pushed:>{since_date}",
            "sort": "updated",
            "perPage": 20
        })
        
        code_result = None
        
    elif "collaborat" in aspect_lower or "team" in aspect_lower:
        # Search for PRs and issues
        pr_result = await toolkit.call_tool("search_pull_requests", {
            "query": f"author:{current_user}",
            "perPage": 20
        })
        
        issue_result = await toolkit.call_tool("search_issues", {
            "query": f"author:{current_user}",
            "perPage": 20
        })
        
        output = f"**Collaboration Analysis for @{current_user}**\n\n"
        
        if pr_result and "items" in pr_result:
            output += f"Pull Requests: {len(pr_result['items'])} found\n"
            merged = sum(1 for pr in pr_result['items'] if pr.get('merged_at'))
            output += f"Merged PRs: {merged}\n\n"
        
        if issue_result and "items" in issue_result:
            output += f"Issues created: {len(issue_result['items'])}\n"
        
        return output
        
    else:
        # General profile analysis
        repo_result = await toolkit.call_tool("search_repositories", {
            "query": f"user:{current_user}",
            "sort": "stars",
            "perPage": 30
        })
        
        code_result = None
    
    # Format results
    output = f"**Analysis: {aspect} for @{current_user}**\n\n"
    
    if repo_result and "items" in repo_result:
        repos = repo_result["items"]
        total_stars = sum(r.get('stargazers_count', 0) for r in repos)
        
        output += f"**Repository Summary:**\n"
        output += f"- Total repositories found: {len(repos)}\n"
        output += f"- Total stars: ⭐ {total_stars}\n"
        
        # Language distribution
        languages = {}
        for repo in repos:
            lang = repo.get('language')
            if lang:
                languages[lang] = languages.get(lang, 0) + 1
        
        if languages:
            output += f"\n**Languages:**\n"
            for lang, count in sorted(languages.items(), key=lambda x: x[1], reverse=True)[:5]:
                output += f"- {lang}: {count} repos\n"
        
        # Top repositories
        output += f"\n**Relevant Repositories:**\n"
        for repo in repos[:5]:
            output += f"- **{repo['name']}** ({repo.get('language', 'Unknown')})\n"
            if repo.get('description'):
                output += f"  {repo['description'][:80]}...\n"
            output += f"  ⭐ {repo.get('stargazers_count', 0)} | "
            output += f"🍴 {repo.get('forks_count', 0)}\n"
    
    if code_result and "items" in code_result:
        output += f"\n**Code Analysis:**\n"
        output += f"Found {len(code_result['items'])} relevant code files\n"
        
        # Show a few examples
        for item in code_result["items"][:3]:
            output += f"- {item.get('name')} in {item.get('repository', {}).get('name', 'Unknown')}\n"
    
    return output


@tool
async def evaluate_candidate_fit(role: str, requirements: str = "") -> str:
    """
    Evaluate how well the candidate fits a specific role based on their GitHub profile.
    
    Args:
        role: The job role (e.g., "WordPress Developer", "Backend Engineer")
        requirements: Additional specific requirements to check
    """
    if not current_user:
        return "No user set for evaluation"
    
    # Get overall profile stats
    profile_result = await toolkit.call_tool("search_repositories", {
        "query": f"user:{current_user}",
        "perPage": 100
    })
    
    if not profile_result or "items" not in profile_result:
        return f"Unable to fetch profile data for @{current_user}"
    
    repos = profile_result["items"]
    
    # Calculate basic metrics
    total_repos = len(repos)
    total_stars = sum(r.get('stargazers_count', 0) for r in repos)
    total_forks = sum(r.get('forks_count', 0) for r in repos)
    
    # Recent activity
    from datetime import datetime, timedelta
    recent_repos = 0
    for repo in repos:
        if repo.get('updated_at'):
            updated = datetime.fromisoformat(repo['updated_at'].replace('Z', '+00:00'))
            if datetime.now(updated.tzinfo) - updated < timedelta(days=90):
                recent_repos += 1
    
    # Language analysis
    languages = {}
    for repo in repos:
        lang = repo.get('language')
        if lang:
            languages[lang] = languages.get(lang, 0) + 1
    
    # Build evaluation
    output = f"**Candidate Evaluation: @{current_user} for {role}**\n"
    output += "=" * 50 + "\n\n"
    
    output += f"**Profile Overview:**\n"
    output += f"- Total Repositories: {total_repos}\n"
    output += f"- Total Stars: ⭐ {total_stars}\n"
    output += f"- Total Forks: 🍴 {total_forks}\n"
    output += f"- Recent Activity: {recent_repos} repos updated in last 90 days\n\n"
    
    output += f"**Technical Stack:**\n"
    for lang, count in sorted(languages.items(), key=lambda x: x[1], reverse=True)[:8]:
        percentage = (count / total_repos) * 100
        output += f"- {lang}: {count} repos ({percentage:.1f}%)\n"
    
    # Role-specific analysis
    output += f"\n**Role-Specific Analysis for {role}:**\n"
    
    role_lower = role.lower()
    if "wordpress" in role_lower:
        php_count = languages.get('PHP', 0)
        js_count = languages.get('JavaScript', 0)
        html_count = languages.get('HTML', 0)
        
        if php_count > 0:
            output += f"✅ PHP experience: {php_count} repositories\n"
        else:
            output += f"❌ No PHP repositories found\n"
        
        if js_count > 0:
            output += f"✅ JavaScript experience: {js_count} repositories\n"
        
        # Search for WordPress-specific code
        wp_search = await toolkit.call_tool("search_code", {
            "q": f"user:{current_user} wordpress OR wp-content OR functions.php",
            "perPage": 5
        })
        
        if wp_search and "items" in wp_search and len(wp_search["items"]) > 0:
            output += f"✅ WordPress-specific code found: {len(wp_search['items'])} files\n"
        else:
            output += f"⚠️  No WordPress-specific code found in public repos\n"
    
    # Activity assessment
    output += f"\n**Activity Assessment:**\n"
    activity_rate = (recent_repos / total_repos * 100) if total_repos > 0 else 0
    
    if activity_rate > 50:
        output += f"🟢 Highly active: {activity_rate:.1f}% repos recently updated\n"
    elif activity_rate > 25:
        output += f"🟡 Moderately active: {activity_rate:.1f}% repos recently updated\n"
    else:
        output += f"🔴 Low activity: {activity_rate:.1f}% repos recently updated\n"
    
    # Overall recommendation
    output += f"\n**Summary:**\n"
    
    if total_stars > 100:
        output += "- Strong community engagement (100+ stars)\n"
    if total_repos > 20:
        output += "- Extensive portfolio of projects\n"
    if activity_rate > 30:
        output += "- Active development and maintenance\n"
    
    return output


@tool
async def generate_evaluation_report() -> str:
    """
    Generate a comprehensive evaluation report for the current candidate.
    """
    if not current_user:
        return "No user set for evaluation"
    
    output = f"**COMPREHENSIVE EVALUATION REPORT**\n"
    output += f"**Candidate: @{current_user}**\n"
    output += f"**Date: {datetime.now().strftime('%Y-%m-%d')}**\n"
    output += "=" * 60 + "\n\n"
    
    # Get all repositories
    all_repos = await toolkit.call_tool("search_repositories", {
        "query": f"user:{current_user}",
        "perPage": 100
    })
    
    if not all_repos or "items" not in all_repos:
        return "Unable to generate report - no data available"
    
    repos = all_repos["items"]
    
    # Calculate comprehensive metrics
    total_stars = sum(r.get('stargazers_count', 0) for r in repos)
    total_forks = sum(r.get('forks_count', 0) for r in repos)
    
    # Top projects
    top_repos = sorted(repos, key=lambda x: x.get('stargazers_count', 0), reverse=True)[:5]
    
    output += "**TOP PROJECTS:**\n"
    for repo in top_repos:
        output += f"• {repo['name']} - ⭐ {repo.get('stargazers_count', 0)}\n"
        if repo.get('description'):
            output += f"  {repo['description'][:100]}\n"
    
    # Recent activity
    from datetime import datetime, timedelta
    recent_date = datetime.now() - timedelta(days=90)
    recent_active = 0
    
    for repo in repos:
        if repo.get('updated_at'):
            updated = datetime.fromisoformat(repo['updated_at'].replace('Z', '+00:00'))
            if updated.replace(tzinfo=None) > recent_date:
                recent_active += 1
    
    output += f"\n**ACTIVITY METRICS:**\n"
    output += f"• Total Repositories: {len(repos)}\n"
    output += f"• Recently Active: {recent_active} (last 90 days)\n"
    output += f"• Total Stars: ⭐ {total_stars}\n"
    output += f"• Total Forks: 🍴 {total_forks}\n"
    
    # Language expertise
    languages = {}
    for repo in repos:
        lang = repo.get('language')
        if lang:
            languages[lang] = languages.get(lang, 0) + 1
    
    output += f"\n**TECHNICAL EXPERTISE:**\n"
    for lang, count in sorted(languages.items(), key=lambda x: x[1], reverse=True)[:10]:
        output += f"• {lang}: {count} projects\n"
    
    # Generate scores
    tech_score = min(100, (total_stars // 10) + (len(repos) // 2))
    activity_score = min(100, (recent_active / len(repos) * 100) if repos else 0)
    
    output += f"\n**EVALUATION SCORES:**\n"
    output += f"• Technical Impact: {tech_score}/100\n"
    output += f"• Activity Level: {int(activity_score)}/100\n"
    output += f"• Portfolio Size: {len(repos)} projects\n"
    
    # Recommendations
    output += f"\n**RECOMMENDATIONS:**\n"
    
    if tech_score > 70:
        output += "✅ Strong technical portfolio with community recognition\n"
    elif tech_score > 40:
        output += "🟡 Solid technical foundation, growing impact\n"
    else:
        output += "🔶 Building technical presence\n"
    
    if activity_score > 50:
        output += "✅ Highly active developer\n"
    elif activity_score > 25:
        output += "🟡 Moderate activity level\n"
    else:
        output += "🔶 Low recent activity\n"
    
    return output


async def create_evaluation_agent(username: str):
    """Create an evaluation agent for a specific GitHub user"""
    
    global toolkit, current_user
    
    # Validate and extract username from URL if needed
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
    
    # Set current user
    current_user = username
    
    # Initialize toolkit if needed
    if not toolkit:
        toolkit = await GitHubMCPToolkit().initialize()
    
    # Verify user exists
    test_result = await toolkit.call_tool("search_repositories", {
        "query": f"user:{username}",
        "perPage": 1
    })
    
    if "error" in test_result:
        raise ValueError(f"Error accessing GitHub: {test_result['error']}")
    
    if test_result.get("total_count", 0) == 0:
        raise ValueError(f"User @{username} not found or has no public repositories")
    
    # Create LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Create tools
    tools = [
        analyze_github_profile,
        evaluate_candidate_fit,
        generate_evaluation_report
    ]
    
    # Create prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are an expert technical recruiter evaluating GitHub profile @{username}.

Your goal is to provide insightful, data-driven evaluations of candidates based on their GitHub activity.

Available tools:
1. analyze_github_profile - Analyze specific aspects (e.g., "wordpress skills", "backend experience", "recent activity")
2. evaluate_candidate_fit - Evaluate fit for a specific role
3. generate_evaluation_report - Generate a comprehensive report

When asked to evaluate for a role:
1. First analyze relevant technical skills
2. Check recent activity and collaboration
3. Evaluate overall fit for the role
4. Provide specific, actionable insights

Be concise but thorough. Focus on evidence from actual repositories and code.

When evaluating for a role, structure your response as:
1. **Quick Summary** (2-3 lines)
2. **Key Findings** (bullet points)
3. **Recommendation** (STRONG_HIRE / INTERVIEW_RECOMMENDED / NO_HIRE)
4. **Next Steps** (if applicable)

Keep the total response under 500 words."""),
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
        max_iterations=5
    )
    
    return agent_executor


async def main():
    """Main function"""
    import sys
    
    print("🔍 GitHub Evaluation Agent V2 (LLM-Aided)")
    print("=" * 60)
    
    if len(sys.argv) < 2:
        print("Usage: python github_evaluation_agent_v2.py <username_or_url> [query]")
        print("\nExamples:")
        print('  python github_evaluation_agent_v2.py torvalds')
        print('  python github_evaluation_agent_v2.py https://github.com/torvalds')
        print('  python github_evaluation_agent_v2.py torvalds "evaluate for backend role"')
        return
    
    username = sys.argv[1]
    
    try:
        agent = await create_evaluation_agent(username)
        print(f"✅ Agent ready for @{current_user}\n")
        
        if len(sys.argv) > 2:
            # Direct query mode
            query = " ".join(sys.argv[2:])
            print(f"📋 Query: {query}\n")
            result = await agent.ainvoke({"input": query})
            print(result["output"])
        else:
            # Interactive mode
            print("📌 Example queries:")
            print("  • 'Evaluate for WordPress Developer role'")
            print("  • 'Analyze backend development skills'")
            print("  • 'Check recent activity and collaboration'")
            print("  • 'Generate comprehensive evaluation report'")
            print("\nType 'exit' to quit\n")
            
            while True:
                query = input(f"[{current_user}] > ").strip()
                
                if query.lower() in ['exit', 'quit']:
                    break
                
                if not query:
                    continue
                
                print("\n🤔 Evaluating...\n")
                
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
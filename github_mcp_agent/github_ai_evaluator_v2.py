#!/usr/bin/env python3
"""
GitHub AI Evaluator V2 - Intelligent evaluation with structured output
AI figures out what skills are needed, then provides structured scoring
"""

import asyncio
import os
import json
import re
from datetime import datetime, timedelta
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
evaluation_data: Dict[str, Any] = {}


@tool
async def determine_role_requirements(role: str) -> str:
    """
    AI determines what skills, tools, and technologies are needed for a role.
    No hardcoded bullshit - pure AI reasoning.
    
    Args:
        role: The job role to analyze
    """
    global evaluation_data
    
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
    
    # Store for later use
    evaluation_data['role_requirements'] = response.content
    
    return response.content


@tool
async def smart_skill_search(skill_category: str, search_terms: str) -> str:
    """
    Search for evidence of skills using intelligent queries.
    
    Args:
        skill_category: Category like "machine learning", "web development", etc.
        search_terms: Specific terms to search for
    """
    global evaluation_data
    
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
    proficiency = "NONE"
    if repo_count >= 5 and code_files >= 5:
        proficiency = "STRONG"
    elif repo_count >= 2 or code_files >= 2:
        proficiency = "MODERATE"
    elif repo_count >= 1 or code_files >= 1:
        proficiency = "BASIC"
    
    output += f"\n**Proficiency Level:** {proficiency}"
    
    # Store skill data
    if 'skills_assessed' not in evaluation_data:
        evaluation_data['skills_assessed'] = {}
    
    evaluation_data['skills_assessed'][skill_category] = {
        'proficiency': proficiency,
        'repo_count': repo_count,
        'repos': [r['name'] for r in relevant_repos[:3]],
        'code_files': code_files
    }
    
    return output


@tool
async def analyze_github_profile_holistically() -> str:
    """
    Get a complete picture of the user's GitHub profile.
    What they actually work on, not what we expect them to work on.
    """
    global evaluation_data
    
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
                if (datetime.now(updated.tzinfo) - updated).days < 90:
                    recent_projects.append(repo)
            except:
                pass
    
    # Get collaboration data
    pr_result = await toolkit.call_tool("search_pull_requests", {
        "query": f"author:{current_user}",
        "perPage": 50
    })
    
    total_prs = 0
    merged_prs = 0
    
    if pr_result and "items" in pr_result:
        prs = pr_result["items"]
        total_prs = len(prs)
        merged_prs = sum(1 for pr in prs if pr.get('merged_at'))
    
    # Store in evaluation data
    evaluation_data['profile_stats'] = {
        'total_repos': len(repos),
        'total_stars': total_stars,
        'recent_activity': len(recent_projects),
        'languages': languages,
        'top_languages': sorted(languages.items(), key=lambda x: x[1], reverse=True)[:5],
        'total_prs': total_prs,
        'merged_prs': merged_prs
    }
    
    output = f"**Actual GitHub Profile Analysis for @{current_user}**\n\n"
    
    output += f"**Profile Stats:**\n"
    output += f"- Total Repositories: {len(repos)}\n"
    output += f"- Total Stars: ⭐ {total_stars}\n"
    output += f"- Recently Active Projects: {len(recent_projects)}\n"
    output += f"- Pull Requests: {total_prs} (Merged: {merged_prs})\n\n"
    
    output += f"**What They Actually Work With:**\n"
    for lang, count in sorted(languages.items(), key=lambda x: x[1], reverse=True)[:10]:
        percentage = (count / len(repos) * 100) if repos else 0
        output += f"- {lang}: {count} projects ({percentage:.1f}%)\n"
    
    if topics_set:
        output += f"\n**Technologies & Topics They Use:**\n"
        output += ", ".join(sorted(topics_set)[:20])
    
    output += f"\n\n**Recent Focus (Last 90 days):**\n"
    for proj in recent_projects[:5]:
        output += f"- {proj['name']}"
        if proj.get('language'):
            output += f" ({proj['language']})"
        if proj.get('description'):
            output += f": {proj['description'][:60]}..."
        output += "\n"
    
    return output


@tool
async def generate_evaluation_summary() -> str:
    """
    Generate the structured evaluation summary with scores.
    """
    global evaluation_data
    
    if not current_user or not current_role:
        return "Missing user or role information"
    
    # Calculate scores based on collected data
    stats = evaluation_data.get('profile_stats', {})
    skills = evaluation_data.get('skills_assessed', {})
    
    # Technical score (based on skill proficiency)
    tech_score = 0
    skill_count = len(skills)
    if skill_count > 0:
        for skill_data in skills.values():
            if skill_data['proficiency'] == 'STRONG':
                tech_score += 25
            elif skill_data['proficiency'] == 'MODERATE':
                tech_score += 15
            elif skill_data['proficiency'] == 'BASIC':
                tech_score += 5
    
    # Bonus for stars
    tech_score += min(20, stats.get('total_stars', 0) // 10)
    tech_score = min(100, tech_score)
    
    # Activity score
    total_repos = stats.get('total_repos', 0)
    recent_activity = stats.get('recent_activity', 0)
    activity_score = 0
    if total_repos > 0:
        activity_score = min(100, int((recent_activity / total_repos) * 100))
    
    # Collaboration score
    total_prs = stats.get('total_prs', 0)
    merged_prs = stats.get('merged_prs', 0)
    collab_score = min(100, (total_prs // 5) * 10 + (merged_prs // 3) * 10)
    
    # Overall score
    overall_score = int((tech_score * 0.6) + (activity_score * 0.2) + (collab_score * 0.2))
    
    # Recommendation
    if overall_score >= 70:
        recommendation = "INTERVIEW_RECOMMENDED"
        if overall_score >= 85:
            recommendation = "STRONG_HIRE"
    else:
        recommendation = "NO_HIRE"
    
    # Build output
    output = "\n" + "=" * 60 + "\n"
    output += "📋 EVALUATION SUMMARY\n"
    output += "=" * 60 + "\n"
    output += f"Candidate: @{current_user}\n"
    output += f"Role: {current_role}\n\n"
    
    output += "📊 SCORES:\n"
    output += f"• Technical: {tech_score}/100\n"
    output += f"• Activity: {activity_score}/100\n"
    output += f"• Collaboration: {collab_score}/100\n"
    output += f"• Overall: {overall_score}/100\n\n"
    
    output += f"🎯 RECOMMENDATION: {recommendation}\n\n"
    
    output += "🛠️ SKILLS:\n"
    for skill_name, skill_data in skills.items():
        emoji = "🟢" if skill_data['proficiency'] == 'STRONG' else "🟡" if skill_data['proficiency'] == 'MODERATE' else "🔴"
        output += f"{emoji} {skill_name}: {skill_data['proficiency']}\n"
        if skill_data['repos']:
            output += f"   Repos: {', '.join(skill_data['repos'])}\n"
    
    output += f"\n📈 ACTIVITY:\n"
    output += f"• Total Repositories: {stats.get('total_repos', 0)}\n"
    output += f"• Total Stars: {stats.get('total_stars', 0)}\n"
    output += f"• Recent Activity: {stats.get('recent_activity', 0)} repos\n"
    
    top_langs = stats.get('top_languages', [])
    if top_langs:
        lang_list = [f"{lang[0]}" for lang in top_langs]
        output += f"• Languages: {', '.join(lang_list)}\n"
    
    # Red flags
    red_flags = []
    if recent_activity < 2:
        red_flags.append("Limited recent activity")
    if tech_score < 30:
        red_flags.append("Limited evidence of required skills")
    if collab_score < 20:
        red_flags.append("Low collaboration activity")
    
    if red_flags:
        output += f"\n⚠️ RED FLAGS:\n"
        for flag in red_flags:
            output += f"• {flag}\n"
    
    output += "=" * 60
    
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
        generate_evaluation_summary
    ]
    
    # Prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are an intelligent GitHub profile evaluator.

Current user: @{username}

Your evaluation process:
1. Use determine_role_requirements to understand what the role ACTUALLY needs
2. Use analyze_github_profile_holistically to see what the person ACTUALLY does
3. Use smart_skill_search for EACH main skill the role requires (based on step 1)
4. End with generate_evaluation_summary to create the structured output

Important:
- Evaluate based on what the role actually needs, not hardcoded assumptions
- Search for skills that make sense for the role
- The summary will be auto-generated based on your findings"""),
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
        max_iterations=10
    )
    
    return agent_executor


async def main():
    """Main function"""
    global current_role, evaluation_data
    
    import sys
    
    print("🧠 GitHub AI Evaluator V2 - Intelligent + Structured")
    print("=" * 60)
    
    if len(sys.argv) < 3:
        print("Usage: python github_ai_evaluator_v2.py <username/url> <role>")
        print("\nExamples:")
        print('  python github_ai_evaluator_v2.py torvalds "Linux Kernel Developer"')
        print('  python github_ai_evaluator_v2.py https://github.com/AizazSharif "ML Engineer"')
        return
    
    username = sys.argv[1]
    role = " ".join(sys.argv[2:])
    current_role = role
    
    # Reset evaluation data
    evaluation_data = {}
    
    try:
        agent = await create_intelligent_evaluator(username)
        print(f"✅ Ready to evaluate @{current_user} for {role}\n")
        
        # Run evaluation
        result = await agent.ainvoke({
            "input": f"""Evaluate this person for the role: {role}

Follow these steps:
1. First determine what skills this role needs
2. Analyze their GitHub profile holistically
3. Search for evidence of the key skills needed for this role
4. Generate the evaluation summary

Be thorough but focused on what actually matters for this specific role."""
        })
        
        print(result["output"])
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_username = ''.join(c for c in current_user if c.isalnum() or c in '-_')
        filename = f"ai_evaluation_{safe_username}_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump({
                'username': current_user,
                'role': role,
                'evaluation_data': evaluation_data,
                'timestamp': timestamp
            }, f, indent=2)
        
        print(f"\n💾 Results saved to: {filename}")
    
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        if toolkit:
            await toolkit.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
#!/usr/bin/env python3
"""
GitHub Evaluation Agent - Fixed Version
Maintains the sophisticated scoring system while preventing crashes
"""

import asyncio
import os
import json
import re
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
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


class EvidenceQuality(Enum):
    """Quality levels for skill evidence"""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    NONE = "NONE"


class HiringRecommendation(Enum):
    """Hiring recommendation levels"""
    STRONG_HIRE = "STRONG_HIRE"
    INTERVIEW_RECOMMENDED = "INTERVIEW_RECOMMENDED"
    NO_HIRE = "NO_HIRE"


@dataclass
class SkillEvidence:
    """Evidence for a specific skill"""
    skill_name: str
    evidence_quality: str
    repositories: List[str] = field(default_factory=list)
    code_examples: List[str] = field(default_factory=list)
    complexity_score: int = 0  # 0-5
    years_of_experience: int = 0


@dataclass
class CandidateProfile:
    """Complete candidate profile"""
    username: str
    job_role: str
    experience_level: str
    profile_stats: Dict[str, Any]
    repositories_analyzed: int
    languages_used: List[str]
    primary_languages: List[str] = field(default_factory=list)


@dataclass
class EvaluationResult:
    """Final evaluation result"""
    candidate_profile: CandidateProfile
    skill_assessments: List[SkillEvidence]
    technical_score: int  # 0-100
    activity_score: int  # 0-100
    collaboration_score: int  # 0-100
    overall_scores: Dict[str, int]
    recommendation: str
    confidence: str
    interview_focus: List[str]
    red_flags: List[str]
    timestamp: str


class SkillMapper:
    """Maps job roles to required skills"""
    
    ROLE_SKILLS = {
        "wordpress": {
            "core_skills": [
                "PHP Development", "WordPress Core", "JavaScript/jQuery", "MySQL"
            ],
            "skills": {
                "PHP Development": ["php", "wordpress", "wp-content", "functions.php"],
                "WordPress Core": ["wordpress", "wp-admin", "wp-includes", "wp-config"],
                "JavaScript/jQuery": ["javascript", "jquery", "ajax", "wp-ajax"],
                "MySQL": ["mysql", "database", "wpdb", "sql"]
            },
            "weights": {"technical": 0.7, "collaboration": 0.15, "activity": 0.15}
        },
        "backend": {
            "core_skills": [
                "Server Languages", "Database", "API Design", "Testing"
            ],
            "skills": {
                "Server Languages": ["python", "java", "go", "nodejs", "ruby"],
                "Database": ["postgresql", "mysql", "mongodb", "redis"],
                "API Design": ["rest", "graphql", "api", "swagger"],
                "Testing": ["test", "pytest", "jest", "unit"]
            },
            "weights": {"technical": 0.7, "collaboration": 0.2, "activity": 0.1}
        },
        "frontend": {
            "core_skills": [
                "JavaScript Frameworks", "CSS/Styling", "Build Tools", "Testing"
            ],
            "skills": {
                "JavaScript Frameworks": ["react", "vue", "angular", "svelte"],
                "CSS/Styling": ["css", "sass", "styled-components", "tailwind"],
                "Build Tools": ["webpack", "vite", "rollup", "parcel"],
                "Testing": ["jest", "cypress", "testing-library", "vitest"]
            },
            "weights": {"technical": 0.65, "collaboration": 0.2, "activity": 0.15}
        },
        "fullstack": {
            "core_skills": [
                "Frontend", "Backend", "Database", "DevOps"
            ],
            "skills": {
                "Frontend": ["react", "vue", "javascript", "css"],
                "Backend": ["nodejs", "python", "api", "server"],
                "Database": ["postgresql", "mongodb", "mysql", "redis"],
                "DevOps": ["docker", "kubernetes", "ci/cd", "aws"]
            },
            "weights": {"technical": 0.6, "collaboration": 0.25, "activity": 0.15}
        }
    }
    
    @classmethod
    def get_skills_for_role(cls, job_role: str) -> Dict:
        """Get skills configuration for a job role"""
        role_lower = job_role.lower()
        
        # Check for direct match first
        for key in cls.ROLE_SKILLS:
            if key in role_lower:
                return cls.ROLE_SKILLS[key]
        
        # Default to fullstack
        return cls.ROLE_SKILLS["fullstack"]


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


@tool
async def quick_skill_check(skill_name: str, keywords: str) -> str:
    """
    Quick check for a specific skill using targeted search.
    
    Args:
        skill_name: Name of the skill to check
        keywords: Comma-separated keywords to search
    """
    if not current_user:
        return f"No user set. Skill: {skill_name} - No evidence"
    
    # Search repositories
    repo_result = await toolkit.call_tool("search_repositories", {
        "query": f"user:{current_user} {keywords.replace(',', ' OR ')}",
        "perPage": 10
    })
    
    # Count results
    repo_count = 0
    repo_names = []
    
    if repo_result and "items" in repo_result:
        repos = repo_result["items"]
        repo_count = len(repos)
        repo_names = [r["name"] for r in repos[:3]]
    
    # Quick search for code
    code_result = await toolkit.call_tool("search_code", {
        "q": f"user:{current_user} {keywords.split(',')[0]}",
        "perPage": 5
    })
    
    code_count = 0
    if code_result and "items" in code_result:
        code_count = len(code_result["items"])
    
    # Format result
    output = f"**{skill_name}**\n"
    output += f"Repos: {repo_count} found"
    if repo_names:
        output += f" ({', '.join(repo_names)})"
    output += f"\nCode: {code_count} files found"
    
    # Determine quality
    if repo_count >= 3 and code_count >= 3:
        output += "\nQuality: HIGH"
    elif repo_count >= 1 or code_count >= 1:
        output += "\nQuality: MEDIUM"
    else:
        output += "\nQuality: NONE"
    
    return output


@tool
async def get_profile_stats() -> str:
    """Get basic profile statistics quickly"""
    if not current_user:
        return "No user set"
    
    result = await toolkit.call_tool("search_repositories", {
        "query": f"user:{current_user}",
        "perPage": 100
    })
    
    if not result or "items" not in result:
        return "Unable to fetch profile stats"
    
    repos = result["items"]
    total_repos = len(repos)
    total_stars = sum(r.get('stargazers_count', 0) for r in repos)
    
    # Recent activity
    recent_count = 0
    languages = {}
    
    for repo in repos:
        # Count languages
        lang = repo.get('language')
        if lang:
            languages[lang] = languages.get(lang, 0) + 1
        
        # Check if recently updated
        if repo.get('updated_at'):
            try:
                updated = datetime.fromisoformat(repo['updated_at'].replace('Z', '+00:00'))
                if datetime.now(updated.tzinfo) - updated < timedelta(days=90):
                    recent_count += 1
            except:
                pass
    
    output = f"**Profile Stats for @{current_user}**\n"
    output += f"Total Repos: {total_repos}\n"
    output += f"Total Stars: {total_stars}\n"
    output += f"Recent Activity: {recent_count} repos (90 days)\n"
    output += f"Top Languages: {', '.join(list(languages.keys())[:5])}"
    
    return output


@tool  
async def check_collaboration() -> str:
    """Quick collaboration check"""
    if not current_user:
        return "No user set"
    
    # Check PRs
    pr_result = await toolkit.call_tool("search_pull_requests", {
        "query": f"author:{current_user}",
        "perPage": 30
    })
    
    total_prs = 0
    merged_prs = 0
    
    if pr_result and "items" in pr_result:
        prs = pr_result["items"]
        total_prs = len(prs)
        merged_prs = sum(1 for pr in prs if pr.get('merged_at'))
    
    # Check issues
    issue_result = await toolkit.call_tool("search_issues", {
        "query": f"author:{current_user}",
        "perPage": 20
    })
    
    total_issues = 0
    if issue_result and "items" in issue_result:
        total_issues = len(issue_result["items"])
    
    output = f"**Collaboration Metrics**\n"
    output += f"PRs Created: {total_prs}\n"
    output += f"PRs Merged: {merged_prs}\n"
    output += f"Issues Created: {total_issues}\n"
    
    # Calculate score
    score = min(10, (total_prs // 5) + (merged_prs // 3) + (total_issues // 10))
    output += f"Collaboration Score: {score}/10"
    
    return output


async def create_evaluation_agent(username: str, job_role: str, experience_level: str = "mid"):
    """Create evaluation agent"""
    
    global toolkit, current_user
    
    # Extract username from URL if needed
    username = username.strip()
    if "github.com/" in username:
        match = re.search(r'github\.com/([^/]+)', username)
        if match:
            username = match.group(1)
    username = username.lstrip('@')
    
    # Set current user
    current_user = username
    
    # Get skills config
    skills_config = SkillMapper.get_skills_for_role(job_role)
    
    # Initialize toolkit
    if not toolkit:
        toolkit = await GitHubMCPToolkit().initialize()
    
    # Verify user exists
    test_result = await toolkit.call_tool("search_repositories", {
        "query": f"user:{username}",
        "perPage": 1
    })
    
    if "error" in test_result or test_result.get("total_count", 0) == 0:
        raise ValueError(f"User @{username} not found or has no repositories")
    
    # Create LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Create tools
    tools = [
        quick_skill_check,
        get_profile_stats,
        check_collaboration
    ]
    
    # Create prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are evaluating @{username} for {job_role} ({experience_level}).

Skills to check: {', '.join(skills_config.get('core_skills', [])[:4])}

Use the tools efficiently:
1. quick_skill_check - Check each skill
2. get_profile_stats - Get overall metrics
3. check_collaboration - Get teamwork data

Be concise and factual."""),
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
        max_iterations=6
    )
    
    return agent_executor, skills_config


async def evaluate_candidate_fast(username: str, job_role: str, experience_level: str = "mid"):
    """Fast evaluation with scoring"""
    
    # Extract username from URL if needed
    original_username = username
    username = username.strip()
    if "github.com/" in username:
        match = re.search(r'github\.com/([^/]+)', username)
        if match:
            username = match.group(1)
    username = username.lstrip('@')
    
    print(f"\n🔍 Evaluating: @{username} for {job_role}")
    print("=" * 60)
    
    # Create agent
    agent, skills_config = await create_evaluation_agent(username, job_role, experience_level)
    
    # Get profile stats first
    print("📊 Getting profile statistics...")
    stats_result = await agent.ainvoke({"input": "get profile statistics"})
    stats_output = stats_result.get("output", "")
    
    # Parse basic stats
    total_repos = 0
    total_stars = 0
    recent_repos = 0
    languages = []
    
    for line in stats_output.split('\n'):
        if "Total Repos:" in line:
            total_repos = int(re.findall(r'\d+', line)[0]) if re.findall(r'\d+', line) else 0
        elif "Total Stars:" in line:
            total_stars = int(re.findall(r'\d+', line)[0]) if re.findall(r'\d+', line) else 0
        elif "Recent Activity:" in line:
            recent_repos = int(re.findall(r'\d+', line)[0]) if re.findall(r'\d+', line) else 0
        elif "Top Languages:" in line:
            lang_text = line.split(":", 1)[1].strip()
            languages = [l.strip() for l in lang_text.split(',') if l.strip()]
    
    # Check skills
    print("\n🛠️ Checking technical skills...")
    skill_results = []
    skills_to_check = list(skills_config.get("skills", {}).items())[:4]
    
    for skill_name, keywords in skills_to_check:
        result = await agent.ainvoke({
            "input": f"check skill '{skill_name}' with keywords '{','.join(keywords[:3])}'"
        })
        skill_output = result.get("output", "")
        
        # Parse quality from output
        quality = "NONE"
        repos = []
        
        if "Quality: HIGH" in skill_output:
            quality = "HIGH"
        elif "Quality: MEDIUM" in skill_output:
            quality = "MEDIUM"
        
        # Extract repo names if any
        if "(" in skill_output and ")" in skill_output:
            repo_text = skill_output[skill_output.find("(")+1:skill_output.find(")")]
            repos = [r.strip() for r in repo_text.split(',')][:3]
        
        skill_results.append({
            "name": skill_name,
            "quality": quality,
            "repos": repos
        })
    
    # Check collaboration
    print("\n🤝 Checking collaboration...")
    collab_result = await agent.ainvoke({"input": "check collaboration metrics"})
    collab_output = collab_result.get("output", "")
    
    # Parse collaboration score
    collab_score = 5  # default
    if "Collaboration Score:" in collab_output:
        score_match = re.search(r'Collaboration Score: (\d+)/10', collab_output)
        if score_match:
            collab_score = int(score_match.group(1))
    
    # Calculate scores
    tech_score = sum(
        30 if s["quality"] == "HIGH" else 15 if s["quality"] == "MEDIUM" else 0
        for s in skill_results
    )
    tech_score = min(100, tech_score + (min(total_stars, 100) // 10))
    
    activity_score = min(100, (recent_repos / max(total_repos, 1)) * 100)
    collaboration_score = collab_score * 10
    
    # Overall score
    weights = skills_config.get("weights", {"technical": 0.6, "activity": 0.2, "collaboration": 0.2})
    overall = int(
        tech_score * weights["technical"] +
        activity_score * weights["activity"] +
        collaboration_score * weights["collaboration"]
    )
    
    # Recommendation
    if overall >= 70:
        recommendation = "INTERVIEW_RECOMMENDED"
        if overall >= 85:
            recommendation = "STRONG_HIRE"
    else:
        recommendation = "NO_HIRE"
    
    # Print results
    print("\n" + "="*60)
    print("📋 EVALUATION SUMMARY")
    print("="*60)
    print(f"Candidate: @{username}")
    print(f"Role: {job_role} ({experience_level})")
    print(f"\n📊 SCORES:")
    print(f"• Technical: {tech_score}/100")
    print(f"• Activity: {int(activity_score)}/100")
    print(f"• Collaboration: {collaboration_score}/100")
    print(f"• Overall: {overall}/100")
    print(f"\n🎯 RECOMMENDATION: {recommendation}")
    
    print(f"\n🛠️ SKILLS:")
    for skill in skill_results:
        emoji = "🟢" if skill["quality"] == "HIGH" else "🟡" if skill["quality"] == "MEDIUM" else "🔴"
        print(f"{emoji} {skill['name']}: {skill['quality']}")
        if skill["repos"]:
            print(f"   Repos: {', '.join(skill['repos'])}")
    
    print(f"\n📈 ACTIVITY:")
    print(f"• Total Repositories: {total_repos}")
    print(f"• Total Stars: {total_stars}")
    print(f"• Recent Activity: {recent_repos} repos")
    print(f"• Languages: {', '.join(languages[:5])}")
    
    # Red flags
    red_flags = []
    if recent_repos < 2:
        red_flags.append("Very low recent activity")
    if sum(1 for s in skill_results if s["quality"] != "NONE") < 2:
        red_flags.append("Limited evidence of required skills")
    
    if red_flags:
        print(f"\n⚠️ RED FLAGS:")
        for flag in red_flags:
            print(f"• {flag}")
    
    print("="*60)
    
    return {
        "username": username,
        "role": job_role,
        "scores": {
            "technical": tech_score,
            "activity": int(activity_score),
            "collaboration": collaboration_score,
            "overall": overall
        },
        "recommendation": recommendation,
        "skills": skill_results,
        "stats": {
            "repos": total_repos,
            "stars": total_stars,
            "recent": recent_repos,
            "languages": languages
        }
    }


async def main():
    """Main function"""
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python github_evaluation_agent_fixed.py <username/url> <role> [level]")
        print("\nExamples:")
        print('  python github_evaluation_agent_fixed.py torvalds "backend developer"')
        print('  python github_evaluation_agent_fixed.py https://github.com/torvalds "backend" senior')
        return
    
    username = sys.argv[1]
    job_role = sys.argv[2]
    experience_level = sys.argv[3] if len(sys.argv) > 3 else "mid"
    
    try:
        result = await evaluate_candidate_fast(username, job_role, experience_level)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_username = ''.join(c for c in result["username"] if c.isalnum() or c in '-_')
        filename = f"evaluation_{safe_username}_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"\n💾 Results saved to: {filename}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        if toolkit:
            await toolkit.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
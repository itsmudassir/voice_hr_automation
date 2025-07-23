#!/usr/bin/env python3
"""
GitHub Evaluation Agent - Advanced User Analysis with MCP
Combines MCP tools with sophisticated scoring and evaluation
Fixed to analyze a specific user's GitHub profile
"""

import asyncio
import os
import json
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool

# MCP imports
from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession

# Load environment variables
from dotenv import load_dotenv
load_dotenv()


@dataclass
class SkillEvidence:
    """Structure for skill evidence data"""
    skill_name: str
    repositories: List[str]
    evidence_quality: str  # HIGH/MEDIUM/LOW/NONE
    complexity_score: int  # 1-5
    specific_examples: List[str]


@dataclass
class CandidateProfile:
    """Structure for candidate profile data"""
    username: str
    job_role: str
    experience_level: str
    profile_stats: Dict
    repositories_analyzed: int
    total_commits: int
    languages_used: List[str]
    collaboration_score: int


@dataclass
class EvaluationResult:
    """Structure for final evaluation result"""
    candidate_profile: CandidateProfile
    skill_assessments: List[SkillEvidence]
    overall_scores: Dict[str, int]
    recommendation: str
    confidence: str
    interview_focus: List[str]
    red_flags: List[str]
    timestamp: str


class SkillMapper:
    """Dynamic skill mapping for different job roles"""
    
    ROLE_SKILLS = {
        "devops": {
            "core_skills": [
                "Infrastructure as Code", "CI/CD Pipelines", "Containerization", 
                "Cloud Platforms", "Monitoring & Observability", "Configuration Management",
                "Security & Compliance", "Scripting & Automation"
            ],
            "search_terms": {
                "Infrastructure as Code": ["terraform", "ansible", "cloudformation", "pulumi", "iac"],
                "CI/CD Pipelines": ["jenkins", "github-actions", "gitlab-ci", "azure-devops", "cicd"],
                "Containerization": ["docker", "kubernetes", "container", "helm", "k8s"],
                "Cloud Platforms": ["aws", "azure", "gcp", "cloud", "serverless"],
                "Monitoring & Observability": ["prometheus", "grafana", "elk", "monitoring", "logging"],
                "Configuration Management": ["ansible", "puppet", "chef", "salt"],
                "Security & Compliance": ["security", "compliance", "scanning", "vulnerability"],
                "Scripting & Automation": ["bash", "python", "automation", "script"]
            },
            "weights": {"technical": 0.7, "collaboration": 0.2, "activity": 0.1}
        },
        "fullstack": {
            "core_skills": [
                "Frontend Frameworks", "Backend Development", "Database Management",
                "API Development", "Testing", "DevOps", "UI/UX", "Performance"
            ],
            "search_terms": {
                "Frontend Frameworks": ["react", "vue", "angular", "svelte", "nextjs"],
                "Backend Development": ["nodejs", "python", "java", "go", "ruby", "express"],
                "Database Management": ["postgresql", "mysql", "mongodb", "redis", "sql"],
                "API Development": ["rest", "graphql", "api", "microservices", "fastapi"],
                "Testing": ["jest", "pytest", "junit", "cypress", "testing"],
                "DevOps": ["docker", "kubernetes", "aws", "deployment"],
                "UI/UX": ["design", "ui", "ux", "figma", "responsive"],
                "Performance": ["optimization", "performance", "caching", "cdn"]
            },
            "weights": {"technical": 0.6, "collaboration": 0.25, "activity": 0.15}
        },
        "frontend": {
            "core_skills": [
                "JavaScript Frameworks", "CSS & Styling", "Build Tools", "Testing",
                "Performance Optimization", "Accessibility", "UI/UX", "Mobile Development"
            ],
            "search_terms": {
                "JavaScript Frameworks": ["react", "vue", "angular", "svelte", "typescript"],
                "CSS & Styling": ["css", "sass", "tailwind", "styled-components", "design"],
                "Build Tools": ["webpack", "vite", "rollup", "babel", "esbuild"],
                "Testing": ["jest", "cypress", "testing-library", "playwright"],
                "Performance Optimization": ["performance", "optimization", "lighthouse", "core-web-vitals"],
                "Accessibility": ["a11y", "accessibility", "wcag", "aria"],
                "UI/UX": ["design", "figma", "sketch", "prototype"],
                "Mobile Development": ["react-native", "flutter", "mobile", "responsive"]
            },
            "weights": {"technical": 0.65, "collaboration": 0.2, "activity": 0.15}
        },
        "backend": {
            "core_skills": [
                "Server-Side Languages", "Database Design", "API Architecture", "Security",
                "Performance & Scaling", "Message Queues", "Caching", "Microservices"
            ],
            "search_terms": {
                "Server-Side Languages": ["python", "java", "nodejs", "go", "rust", "php"],
                "Database Design": ["postgresql", "mysql", "mongodb", "redis", "database"],
                "API Architecture": ["rest", "graphql", "grpc", "api", "microservices"],
                "Security": ["authentication", "authorization", "security", "jwt", "oauth"],
                "Performance & Scaling": ["scaling", "performance", "optimization", "load-balancing"],
                "Message Queues": ["rabbitmq", "kafka", "redis", "pubsub", "queue"],
                "Caching": ["redis", "memcached", "cache", "cdn"],
                "Microservices": ["microservice", "docker", "kubernetes", "service-mesh"]
            },
            "weights": {"technical": 0.7, "collaboration": 0.2, "activity": 0.1}
        },
        "data": {
            "core_skills": [
                "Data Engineering", "Machine Learning", "Data Analysis", "Big Data",
                "ETL/ELT", "Data Visualization", "Cloud Data", "Statistics"
            ],
            "search_terms": {
                "Data Engineering": ["spark", "airflow", "etl", "pipeline", "data-engineering"],
                "Machine Learning": ["ml", "tensorflow", "pytorch", "scikit-learn", "model"],
                "Data Analysis": ["pandas", "numpy", "analysis", "jupyter", "notebook"],
                "Big Data": ["hadoop", "spark", "hive", "presto", "bigdata"],
                "ETL/ELT": ["etl", "elt", "airflow", "dbt", "pipeline"],
                "Data Visualization": ["tableau", "powerbi", "plotly", "dashboard", "visualization"],
                "Cloud Data": ["snowflake", "databricks", "bigquery", "redshift"],
                "Statistics": ["statistics", "probability", "hypothesis", "regression"]
            },
            "weights": {"technical": 0.75, "collaboration": 0.15, "activity": 0.1}
        },
        "wordpress": {
            "core_skills": [
                "PHP Development", "WordPress Core", "MySQL/Database", "JavaScript/jQuery",
                "Theme Development", "Plugin Development", "WooCommerce", "REST API"
            ],
            "skills": {
                "PHP Development": ["php", "wordpress", "wp-content", "functions.php"],
                "WordPress Core": ["wordpress", "wp-admin", "wp-includes", "wp-config"],
                "MySQL/Database": ["mysql", "database", "wpdb", "sql", "query"],
                "JavaScript/jQuery": ["javascript", "jquery", "ajax", "wp-ajax"],
                "Theme Development": ["style.css", "functions.php", "template", "customizer"],
                "Plugin Development": ["add_action", "add_filter", "hooks", "shortcode"],
                "WooCommerce": ["woocommerce", "product", "cart", "checkout"],
                "REST API": ["rest-api", "wp-json", "api", "endpoints"]
            },
            "weights": {"technical": 0.7, "collaboration": 0.15, "activity": 0.15}
        }
    }
    
    @classmethod
    def get_skills_for_role(cls, job_role: str) -> Dict:
        """Get skills configuration for a job role"""
        normalized_role = cls._normalize_role(job_role)
        return cls.ROLE_SKILLS.get(normalized_role, cls.ROLE_SKILLS["fullstack"])
    
    @staticmethod
    def _normalize_role(job_role: str) -> str:
        """Normalize job role name to match skill mappings"""
        role_lower = job_role.lower()
        if any(term in role_lower for term in ["devops", "sre", "infrastructure", "platform"]):
            return "devops"
        elif any(term in role_lower for term in ["fullstack", "full-stack", "full stack"]):
            return "fullstack"
        elif any(term in role_lower for term in ["frontend", "front-end", "ui", "react developer"]):
            return "frontend"
        elif any(term in role_lower for term in ["backend", "back-end", "api", "server"]):
            return "backend"
        elif any(term in role_lower for term in ["data", "ml", "machine learning", "analytics"]):
            return "data"
        elif any(term in role_lower for term in ["wordpress", "wp", "cms"]):
            return "wordpress"
        return "fullstack"


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
    
    async def call_tool(self, tool_name: str, params: Dict) -> Dict:
        """Call any GitHub MCP tool"""
        try:
            result = await self.session.call_tool(tool_name, params)
            if result.content:
                return json.loads(result.content[0].text)
            return {"error": "No content returned"}
        except Exception as e:
            return {"error": f"MCP error: {str(e)}"}


# Global instances
toolkit = None
current_user = None
skills_config = None


@tool
async def analyze_repositories_for_skill(skill_name: str, search_terms: str) -> str:
    """
    Search and analyze repositories for a specific skill.
    
    Args:
        skill_name: Name of the skill being evaluated
        search_terms: Comma-separated search terms (e.g., "docker,kubernetes,k8s")
    """
    if not current_user:
        return "No user set. Please specify a user first."
    
    terms = [term.strip() for term in search_terms.split(',')]
    all_results = []
    
    # Search for each term
    for term in terms[:5]:  # Limit to 5 terms
        query = f"user:{current_user} {term}"
        
        # Search in repositories
        repo_result = await toolkit.call_tool("search_repositories", {
            "query": query,
            "perPage": 10
        })
        
        if "items" in repo_result:
            for repo in repo_result["items"]:
                repo_info = {
                    "name": repo["name"],
                    "description": repo.get("description", ""),
                    "language": repo.get("language", "Unknown"),
                    "stars": repo.get("stargazers_count", 0),
                    "updated": repo.get("updated_at", ""),
                    "topics": repo.get("topics", []),
                    "skill_match": skill_name,
                    "search_term": term
                }
                all_results.append(repo_info)
    
    # Also search in code
    code_query = f"user:{current_user} " + " OR ".join(terms[:3])
    code_result = await toolkit.call_tool("search_code", {
        "q": code_query,
        "perPage": 15
    })
    
    code_matches = 0
    if "items" in code_result:
        code_matches = len(code_result["items"])
    
    # Format results
    output = f"**Skill Analysis: {skill_name}**\n"
    output += f"Search terms: {', '.join(terms)}\n\n"
    
    if all_results:
        # Remove duplicates
        unique_repos = {}
        for repo in all_results:
            if repo["name"] not in unique_repos:
                unique_repos[repo["name"]] = repo
        
        output += f"Found {len(unique_repos)} relevant repositories:\n"
        for repo in list(unique_repos.values())[:10]:
            output += f"\n- **{repo['name']}** ({repo['language']})\n"
            if repo['description']:
                output += f"  {repo['description'][:100]}...\n"
            output += f"  ⭐ {repo['stars']} | Updated: {repo['updated'][:10]}\n"
            if repo['topics']:
                output += f"  Topics: {', '.join(repo['topics'][:5])}\n"
    
    output += f"\nCode matches: {code_matches} files"
    
    return output


@tool
async def get_user_activity_metrics() -> str:
    """Get detailed activity metrics for the current user."""
    if not current_user:
        return "No user set. Please specify a user first."
    
    # Get all user repositories
    all_repos = []
    result = await toolkit.call_tool("search_repositories", {
        "query": f"user:{current_user}",
        "perPage": 100
    })
    
    if "items" in result:
        all_repos = result["items"]
    
    # Calculate metrics
    total_stars = sum(r.get("stargazers_count", 0) for r in all_repos)
    total_forks = sum(r.get("forks_count", 0) for r in all_repos)
    
    # Language distribution
    languages = {}
    for repo in all_repos:
        lang = repo.get("language")
        if lang:
            languages[lang] = languages.get(lang, 0) + 1
    
    # Recent activity (repos updated in last year)
    recent_repos = 0
    very_recent_repos = 0
    now = datetime.now()
    
    for repo in all_repos:
        updated = repo.get("updated_at", "")
        if updated:
            try:
                update_date = datetime.fromisoformat(updated.replace('Z', '+00:00'))
                days_ago = (now - update_date.replace(tzinfo=None)).days
                if days_ago < 365:
                    recent_repos += 1
                if days_ago < 90:
                    very_recent_repos += 1
            except:
                pass
    
    # Build output
    output = f"**Activity Metrics for @{current_user}**\n\n"
    output += f"**Repository Statistics:**\n"
    output += f"- Total Repositories: {len(all_repos)}\n"
    output += f"- Total Stars: ⭐ {total_stars}\n"
    output += f"- Total Forks: 🍴 {total_forks}\n"
    output += f"- Public Repos: {sum(1 for r in all_repos if not r.get('private', False))}\n"
    output += f"- Private Repos: {sum(1 for r in all_repos if r.get('private', False))}\n\n"
    
    output += f"**Activity Levels:**\n"
    output += f"- Repos updated in last year: {recent_repos}\n"
    output += f"- Repos updated in last 90 days: {very_recent_repos}\n"
    output += f"- Activity rate: {(very_recent_repos/max(1, len(all_repos))*100):.1f}% recently active\n\n"
    
    output += f"**Language Distribution:**\n"
    for lang, count in sorted(languages.items(), key=lambda x: x[1], reverse=True)[:10]:
        percentage = (count / len(all_repos)) * 100
        output += f"- {lang}: {count} repos ({percentage:.1f}%)\n"
    
    return output


@tool
async def analyze_collaboration_patterns() -> str:
    """Analyze collaboration and contribution patterns."""
    if not current_user:
        return "No user set. Please specify a user first."
    
    # Search for pull requests
    pr_query = f"author:{current_user} type:pr"
    pr_result = await toolkit.call_tool("search_issues", {
        "query": pr_query,
        "perPage": 100
    })
    
    # Search for issues
    issue_query = f"author:{current_user} type:issue"
    issue_result = await toolkit.call_tool("search_issues", {
        "query": issue_query,
        "perPage": 50
    })
    
    # Analyze PR data
    total_prs = 0
    merged_prs = 0
    external_prs = 0
    
    if "items" in pr_result:
        total_prs = len(pr_result["items"])
        for pr in pr_result["items"]:
            if pr.get("pull_request", {}).get("merged_at"):
                merged_prs += 1
            # Check if PR is to external repo
            repo_url = pr.get("repository_url", "")
            if current_user not in repo_url:
                external_prs += 1
    
    # Analyze issue data
    total_issues = 0
    if "items" in issue_result:
        total_issues = len(issue_result["items"])
    
    # Search for code reviews (comments on others' PRs)
    review_query = f"commenter:{current_user} type:pr"
    review_result = await toolkit.call_tool("search_issues", {
        "query": review_query,
        "perPage": 50
    })
    
    reviews = 0
    if "items" in review_result:
        reviews = len(review_result["items"])
    
    # Build output
    output = f"**Collaboration Analysis for @{current_user}**\n\n"
    
    output += f"**Pull Request Activity:**\n"
    output += f"- Total PRs created: {total_prs}\n"
    output += f"- Merged PRs: {merged_prs}\n"
    output += f"- Merge rate: {(merged_prs/max(1, total_prs)*100):.1f}%\n"
    output += f"- External contributions: {external_prs}\n\n"
    
    output += f"**Issue Participation:**\n"
    output += f"- Issues created: {total_issues}\n"
    output += f"- PR reviews/comments: {reviews}\n\n"
    
    output += f"**Collaboration Score:**\n"
    collab_score = min(10, (total_prs + reviews * 2 + external_prs * 3) // 10)
    output += f"- Score: {collab_score}/10\n"
    
    if collab_score >= 7:
        output += "- Level: Highly Collaborative\n"
    elif collab_score >= 4:
        output += "- Level: Moderately Collaborative\n"
    else:
        output += "- Level: Limited Collaboration Evidence\n"
    
    return output


@tool
async def evaluate_repository_quality(repo_name: str) -> str:
    """
    Evaluate the quality of a specific repository.
    
    Args:
        repo_name: Name of the repository to evaluate
    """
    if not current_user:
        return "No user set. Please specify a user first."
    
    # Get repository details
    search_result = await toolkit.call_tool("search_repositories", {
        "query": f"user:{current_user} repo:{repo_name}",
        "perPage": 1
    })
    
    if "items" not in search_result or not search_result["items"]:
        return f"Repository {repo_name} not found for user {current_user}"
    
    repo = search_result["items"][0]
    
    # Get file structure to check for quality indicators
    files_result = await toolkit.call_tool("get_file_contents", {
        "owner": current_user,
        "repo": repo_name,
        "path": "/"
    })
    
    # Quality indicators
    quality_indicators = {
        "has_readme": False,
        "has_license": False,
        "has_tests": False,
        "has_ci": False,
        "has_docker": False,
        "has_docs": False,
        "has_gitignore": False
    }
    
    if isinstance(files_result, list):
        file_names = [f.get("name", "").lower() for f in files_result]
        quality_indicators["has_readme"] = "readme.md" in file_names or "readme" in file_names
        quality_indicators["has_license"] = "license" in file_names
        quality_indicators["has_docker"] = "dockerfile" in file_names or "docker-compose.yml" in file_names
        quality_indicators["has_gitignore"] = ".gitignore" in file_names
        quality_indicators["has_tests"] = any(name in file_names for name in ["tests", "test", "__tests__", "spec"])
        quality_indicators["has_docs"] = "docs" in file_names or "documentation" in file_names
        quality_indicators["has_ci"] = ".github" in file_names or ".gitlab-ci.yml" in file_names
    
    # Calculate quality score
    quality_score = sum(1 for v in quality_indicators.values() if v)
    
    # Build output
    output = f"**Repository Quality Analysis: {repo_name}**\n\n"
    output += f"**Basic Info:**\n"
    output += f"- Language: {repo.get('language', 'Unknown')}\n"
    output += f"- Stars: ⭐ {repo.get('stargazers_count', 0)}\n"
    output += f"- Forks: 🍴 {repo.get('forks_count', 0)}\n"
    output += f"- Size: {repo.get('size', 0)} KB\n"
    output += f"- Last Updated: {repo.get('updated_at', 'Unknown')[:10]}\n\n"
    
    output += f"**Quality Indicators ({quality_score}/7):**\n"
    output += f"- README: {'✅' if quality_indicators['has_readme'] else '❌'}\n"
    output += f"- License: {'✅' if quality_indicators['has_license'] else '❌'}\n"
    output += f"- Tests: {'✅' if quality_indicators['has_tests'] else '❌'}\n"
    output += f"- CI/CD: {'✅' if quality_indicators['has_ci'] else '❌'}\n"
    output += f"- Docker: {'✅' if quality_indicators['has_docker'] else '❌'}\n"
    output += f"- Documentation: {'✅' if quality_indicators['has_docs'] else '❌'}\n"
    output += f"- .gitignore: {'✅' if quality_indicators['has_gitignore'] else '❌'}\n\n"
    
    if quality_score >= 6:
        output += "**Assessment:** Excellent repository structure and practices\n"
    elif quality_score >= 4:
        output += "**Assessment:** Good repository with room for improvement\n"
    else:
        output += "**Assessment:** Basic repository, missing key quality indicators\n"
    
    return output


class ScoringEngine:
    """Calculate scores and generate recommendations"""
    
    def __init__(self, skills_config: Dict):
        self.skills_config = skills_config
        self.weights = skills_config.get("weights", {"technical": 0.6, "collaboration": 0.25, "activity": 0.15})
    
    def analyze_skill_evidence(self, skill_data: str) -> SkillEvidence:
        """Parse skill analysis data into SkillEvidence"""
        # Extract repository names and evidence
        repos = []
        examples = []
        
        lines = skill_data.split('\n')
        for i, line in enumerate(lines):
            if line.startswith('- **') and '**' in line[4:]:
                repo_name = line[4:line.index('**', 4)]
                repos.append(repo_name)
                # Get description from next line if available
                if i + 1 < len(lines) and lines[i + 1].strip():
                    examples.append(f"{repo_name}: {lines[i + 1].strip()[:100]}")
        
        # Extract skill name
        skill_name = "Unknown"
        if "Skill Analysis:" in skill_data:
            skill_line = [line for line in lines if "Skill Analysis:" in line]
            if skill_line:
                skill_name = skill_line[0].split("Skill Analysis:")[-1].strip().rstrip('**').lstrip('**')
        
        # Calculate evidence quality
        if len(repos) >= 3:
            evidence_quality = "HIGH"
            complexity_score = min(5, len(repos))
        elif len(repos) >= 1:
            evidence_quality = "MEDIUM"
            complexity_score = min(3, len(repos) + 1)
        else:
            evidence_quality = "NONE"
            complexity_score = 0
        
        return SkillEvidence(
            skill_name=skill_name,
            repositories=repos[:10],
            evidence_quality=evidence_quality,
            complexity_score=complexity_score,
            specific_examples=examples[:5]
        )
    
    def parse_activity_metrics(self, activity_data: str) -> Tuple[Dict, List[str]]:
        """Parse activity metrics from tool output"""
        metrics = {}
        languages = []
        
        lines = activity_data.split('\n')
        for line in lines:
            if "Total Repositories:" in line:
                try:
                    metrics["total_repos"] = int(re.search(r'\d+', line).group())
                except:
                    pass
            elif "Total Stars:" in line:
                try:
                    metrics["total_stars"] = int(re.search(r'\d+', line).group())
                except:
                    pass
            elif "Repos updated in last 90 days:" in line:
                try:
                    metrics["recent_repos"] = int(re.search(r'\d+', line).group())
                except:
                    pass
            elif re.match(r'- \w+: \d+ repos', line):
                lang = line.split(':')[0].strip('- ')
                languages.append(lang)
        
        return metrics, languages
    
    def parse_collaboration_data(self, collab_data: str) -> Dict:
        """Parse collaboration data from tool output"""
        collab = {}
        
        lines = collab_data.split('\n')
        for line in lines:
            if "Total PRs created:" in line:
                try:
                    collab["total_prs"] = int(re.search(r'\d+', line).group())
                except:
                    pass
            elif "Merged PRs:" in line:
                try:
                    collab["merged_prs"] = int(re.search(r'\d+', line).group())
                except:
                    pass
            elif "External contributions:" in line:
                try:
                    collab["external_prs"] = int(re.search(r'\d+', line).group())
                except:
                    pass
            elif "Score:" in line and "/10" in line:
                try:
                    collab["collab_score"] = int(re.search(r'(\d+)/10', line).group(1))
                except:
                    pass
        
        return collab
    
    def calculate_overall_scores(self, skill_assessments: List[SkillEvidence], 
                               activity_metrics: Dict, collaboration_data: Dict) -> Dict[str, int]:
        """Calculate overall candidate scores"""
        
        # Technical score
        if skill_assessments:
            technical_score = sum(skill.complexity_score for skill in skill_assessments) / len(skill_assessments)
            technical_score = int((technical_score / 5) * 10)
        else:
            technical_score = 0
        
        # Activity score
        total_repos = activity_metrics.get("total_repos", 0)
        recent_repos = activity_metrics.get("recent_repos", 0)
        activity_rate = (recent_repos / max(1, total_repos)) * 100 if total_repos > 0 else 0
        activity_score = min(10, int(activity_rate / 10))
        
        # Collaboration score
        collaboration_score = collaboration_data.get("collab_score", 0)
        
        # Weighted overall score
        overall_score = int(
            technical_score * self.weights["technical"] +
            activity_score * self.weights["activity"] +
            collaboration_score * self.weights["collaboration"]
        )
        
        return {
            "technical": technical_score,
            "activity": activity_score,
            "collaboration": collaboration_score,
            "overall": overall_score
        }
    
    def generate_recommendation(self, overall_scores: Dict[str, int], 
                              experience_level: str) -> Tuple[str, str]:
        """Generate hire recommendation and confidence level"""
        
        overall_score = overall_scores["overall"]
        technical_score = overall_scores["technical"]
        
        # Adjust thresholds based on experience level
        if experience_level.lower() == "senior":
            hire_threshold = 7
            interview_threshold = 5
        elif experience_level.lower() == "junior":
            hire_threshold = 5
            interview_threshold = 3
        else:  # mid level
            hire_threshold = 6
            interview_threshold = 4
        
        # Generate recommendation
        if overall_score >= hire_threshold and technical_score >= hire_threshold:
            recommendation = "STRONG_HIRE"
            confidence = "HIGH" if overall_score >= hire_threshold + 1 else "MEDIUM"
        elif overall_score >= interview_threshold:
            recommendation = "INTERVIEW_RECOMMENDED"
            confidence = "MEDIUM"
        else:
            recommendation = "NO_HIRE"
            confidence = "HIGH" if overall_score <= interview_threshold - 2 else "MEDIUM"
        
        return recommendation, confidence
    
    def generate_interview_focus(self, skill_assessments: List[SkillEvidence]) -> List[str]:
        """Generate interview focus areas"""
        focus_areas = []
        
        # Skills needing verification
        medium_skills = [s for s in skill_assessments if s.evidence_quality == "MEDIUM"]
        if medium_skills:
            focus_areas.append(f"Verify {medium_skills[0].skill_name} expertise with practical scenarios")
        
        # Missing skills
        missing_skills = [s for s in skill_assessments if s.evidence_quality == "NONE"]
        if missing_skills:
            focus_areas.append(f"Assess {missing_skills[0].skill_name} knowledge and learning approach")
        
        # Strong skills for architecture
        strong_skills = [s for s in skill_assessments if s.evidence_quality == "HIGH"]
        if strong_skills:
            focus_areas.append(f"Deep dive into {strong_skills[0].skill_name} architecture decisions")
        
        # General areas
        focus_areas.extend([
            "Problem-solving methodology and debugging approach",
            "Team collaboration and code review practices",
            "Experience with production issues and incident response"
        ])
        
        return focus_areas[:5]
    
    def identify_red_flags(self, activity_metrics: Dict, skill_assessments: List[SkillEvidence], 
                          collaboration_data: Dict) -> List[str]:
        """Identify potential red flags"""
        red_flags = []
        
        # Low activity
        if activity_metrics.get("recent_repos", 0) < 2:
            red_flags.append("Very low recent activity (< 2 repos updated in 90 days)")
        
        # Missing critical skills
        none_skills = [s for s in skill_assessments if s.evidence_quality == "NONE"]
        if len(none_skills) > len(skill_assessments) * 0.5:
            red_flags.append("Missing evidence for majority of required skills")
        
        # Low collaboration
        if collaboration_data.get("collab_score", 0) < 3:
            red_flags.append("Limited collaboration evidence")
        
        # Low merge rate
        total_prs = collaboration_data.get("total_prs", 0)
        merged_prs = collaboration_data.get("merged_prs", 0)
        if total_prs > 5 and merged_prs / total_prs < 0.3:
            red_flags.append("Low PR merge rate (< 30%)")
        
        return red_flags


async def create_evaluation_agent(username: str, job_role: str, experience_level: str = "mid"):
    """Create an evaluation agent for a specific user and role"""
    
    global toolkit, current_user, skills_config
    
    # Validate and extract username from URL if needed
    username = username.strip()
    
    # Handle GitHub URLs
    if "github.com/" in username:
        import re
        match = re.search(r'github\.com/([^/]+)', username)
        if match:
            username = match.group(1)
    
    # Remove @ if present
    username = username.lstrip('@')
    
    # Validate username format
    import re
    if not re.match(r'^[a-zA-Z0-9](?:[a-zA-Z0-9]|-(?=[a-zA-Z0-9])){0,38}$', username):
        raise ValueError(f"Invalid GitHub username format: {username}")
    
    # Set current user
    current_user = username
    
    # Get skills configuration
    skills_config = SkillMapper.get_skills_for_role(job_role)
    
    # Initialize toolkit
    if not toolkit:
        toolkit = await GitHubMCPToolkit().initialize()
    
    # Verify user exists
    test_result = await toolkit.call_tool("search_repositories", {
        "query": f"user:{username}",
        "perPage": 1
    })
    
    if "error" in test_result:
        raise ValueError(f"Error accessing GitHub: {test_result['error']}")
    
    # Create LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Create tools
    tools = [
        analyze_repositories_for_skill,
        get_user_activity_metrics,
        analyze_collaboration_patterns,
        evaluate_repository_quality
    ]
    
    # Create prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are a GitHub profile evaluator for {job_role} candidates.
        
Current evaluation context:
- Username: @{username}
- Role: {job_role}
- Experience Level: {experience_level}
- Skills to evaluate: {', '.join(skills_config['core_skills'])}

Use the tools to gather comprehensive data about the candidate's GitHub profile.
Focus on finding evidence of technical skills, collaboration, and recent activity."""),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad")
    ])
    
    # Create agent
    agent = create_openai_tools_agent(llm, tools, prompt)
    
    # Create executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,  # Reduce output verbosity
        handle_parsing_errors=True,
        max_iterations=min(len(skills_config['core_skills']) + 3, 15)  # Limit iterations
    )
    
    return agent_executor


async def evaluate_candidate(username: str, job_role: str, experience_level: str = "mid") -> EvaluationResult:
    """Execute complete candidate evaluation"""
    
    # Extract username if URL provided
    original_username = username
    username = username.strip()
    
    # Handle GitHub URLs
    if "github.com/" in username:
        import re
        match = re.search(r'github\.com/([^/]+)', username)
        if match:
            username = match.group(1)
    
    # Remove @ if present
    username = username.lstrip('@')
    
    print(f"\n🔍 Evaluating GitHub Profile: @{username}")
    print(f"   Role: {job_role} ({experience_level})")
    print("=" * 60)
    
    # Create evaluation agent
    agent = await create_evaluation_agent(username, job_role, experience_level)
    
    # Initialize scoring engine
    scoring_engine = ScoringEngine(skills_config)
    
    # Step 1: Analyze each skill
    print("\n📊 Analyzing Technical Skills...")
    skill_assessments = []
    
    for skill_name, search_terms in skills_config.get("search_terms", {}).items():
        print(f"   - Evaluating {skill_name}...")
        
        query = f"analyze repositories for skill '{skill_name}' with search terms: {','.join(search_terms[:5])}"
        result = await agent.ainvoke({"input": query})
        
        skill_data = result.get("output", "")
        skill_evidence = scoring_engine.analyze_skill_evidence(skill_data)
        skill_evidence.skill_name = skill_name  # Ensure correct skill name
        skill_assessments.append(skill_evidence)
    
    # Step 2: Get activity metrics
    print("\n📈 Analyzing Activity Patterns...")
    activity_result = await agent.ainvoke({"input": "get user activity metrics"})
    activity_data = activity_result.get("output", "")
    activity_metrics, languages = scoring_engine.parse_activity_metrics(activity_data)
    
    # Step 3: Get collaboration data
    print("\n🤝 Analyzing Collaboration...")
    collab_result = await agent.ainvoke({"input": "analyze collaboration patterns"})
    collab_data = collab_result.get("output", "")
    collaboration_metrics = scoring_engine.parse_collaboration_data(collab_data)
    
    # Step 4: Calculate scores
    print("\n🎯 Calculating Scores...")
    overall_scores = scoring_engine.calculate_overall_scores(
        skill_assessments, activity_metrics, collaboration_metrics
    )
    
    # Step 5: Generate recommendation
    recommendation, confidence = scoring_engine.generate_recommendation(
        overall_scores, experience_level
    )
    
    # Step 6: Generate interview focus and red flags
    interview_focus = scoring_engine.generate_interview_focus(skill_assessments)
    red_flags = scoring_engine.identify_red_flags(
        activity_metrics, skill_assessments, collaboration_metrics
    )
    
    # Create candidate profile
    candidate_profile = CandidateProfile(
        username=username,
        job_role=job_role,
        experience_level=experience_level,
        profile_stats={
            "total_repos": activity_metrics.get("total_repos", 0),
            "total_stars": activity_metrics.get("total_stars", 0),
            "recent_activity": activity_metrics.get("recent_repos", 0)
        },
        repositories_analyzed=sum(len(s.repositories) for s in skill_assessments),
        total_commits=0,  # Would need additional API calls
        languages_used=languages[:10],
        collaboration_score=overall_scores["collaboration"]
    )
    
    # Create evaluation result
    evaluation_result = EvaluationResult(
        candidate_profile=candidate_profile,
        skill_assessments=skill_assessments,
        overall_scores=overall_scores,
        recommendation=recommendation,
        confidence=confidence,
        interview_focus=interview_focus,
        red_flags=red_flags,
        timestamp=datetime.now().isoformat()
    )
    
    return evaluation_result


def print_evaluation_summary(result: EvaluationResult):
    """Print formatted evaluation summary"""
    print(f"\n{'='*60}")
    print(f"GITHUB PROFILE EVALUATION REPORT")
    print(f"{'='*60}")
    print(f"Candidate: @{result.candidate_profile.username}")
    print(f"Position: {result.candidate_profile.job_role} ({result.candidate_profile.experience_level})")
    print(f"Evaluated: {result.timestamp[:19]}")
    
    print(f"\n📊 RECOMMENDATION: {result.recommendation}")
    print(f"   Confidence: {result.confidence}")
    
    print(f"\n📈 OVERALL SCORES:")
    for category, score in result.overall_scores.items():
        bar = "█" * score + "░" * (10 - score)
        print(f"   {category.title():15} [{bar}] {score}/10")
    
    print(f"\n🛠️ TECHNICAL COMPETENCIES:")
    for skill in result.skill_assessments:
        quality_emoji = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🟠", "NONE": "🔴"}.get(skill.evidence_quality, "⚪")
        print(f"   {quality_emoji} {skill.skill_name}: {skill.evidence_quality} ({skill.complexity_score}/5)")
        if skill.repositories:
            print(f"      Repos: {', '.join(skill.repositories[:3])}")
    
    print(f"\n📋 PROFILE STATISTICS:")
    stats = result.candidate_profile.profile_stats
    print(f"   Total Repositories: {stats.get('total_repos', 0)}")
    print(f"   Total Stars: ⭐ {stats.get('total_stars', 0)}")
    print(f"   Recent Activity: {stats.get('recent_activity', 0)} repos in last 90 days")
    print(f"   Languages: {', '.join(result.candidate_profile.languages_used[:5])}")
    
    if result.interview_focus:
        print(f"\n🎯 INTERVIEW FOCUS AREAS:")
        for i, focus in enumerate(result.interview_focus, 1):
            print(f"   {i}. {focus}")
    
    if result.red_flags:
        print(f"\n⚠️  RED FLAGS:")
        for flag in result.red_flags:
            print(f"   • {flag}")
    
    print(f"\n{'='*60}")


def save_evaluation(result: EvaluationResult, filename: Optional[str] = None):
    """Save evaluation to JSON file"""
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Sanitize username for filename (remove special characters)
        username = result.candidate_profile.username
        safe_username = ''.join(c for c in username if c.isalnum() or c in '-_')
        filename = f"evaluation_{safe_username}_{timestamp}.json"
    
    evaluation_dict = asdict(result)
    
    with open(filename, 'w') as f:
        json.dump(evaluation_dict, f, indent=2)
    
    print(f"\n💾 Evaluation saved to: {filename}")


async def main():
    """Main function for testing"""
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python github_evaluation_agent.py <username> <job_role> [experience_level]")
        print("\nExamples:")
        print('  python github_evaluation_agent.py torvalds "Senior Backend Engineer" senior')
        print('  python github_evaluation_agent.py itsmudassir "DevOps Engineer" mid')
        print('  python github_evaluation_agent.py octocat "Full Stack Developer" junior')
        return
    
    username = sys.argv[1]
    job_role = sys.argv[2]
    experience_level = sys.argv[3] if len(sys.argv) > 3 else "mid"
    
    try:
        # Run evaluation
        result = await evaluate_candidate(username, job_role, experience_level)
        
        # Print summary
        print_evaluation_summary(result)
        
        # Save detailed results
        save_evaluation(result)
        
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
    finally:
        if toolkit:
            await toolkit.cleanup()


async def create_simple_evaluation_agent(username: str):
    """Simplified wrapper for interactive mode - creates agent for general evaluation"""
    # Default to Full Stack Developer for general evaluation
    return await create_evaluation_agent(username, "Full Stack Developer", "mid")


if __name__ == "__main__":
    import sys
    
    # Support both original and simplified usage
    if len(sys.argv) == 2:
        # Simplified interactive mode with just username
        username = sys.argv[1]
        
        async def interactive_main():
            """Interactive evaluation mode"""
            print(f"🔍 GitHub Evaluation Agent")
            print("=" * 60)
            
            try:
                # Create agent with general evaluation profile
                agent = await create_simple_evaluation_agent(username)
                print(f"✅ Agent ready for @{username}")
                print("\n📌 Example queries:")
                print("  • 'Evaluate for Backend Developer role'")
                print("  • 'Assess DevOps skills'")
                print("  • 'Show technical strengths'")
                print("  • 'Identify red flags'")
                print("  • 'Generate hiring recommendation'")
                print("\nType 'exit' to quit\n")
                
                while True:
                    query = input(f"[{username}] > ").strip()
                    
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
        
        asyncio.run(interactive_main())
    else:
        # Original mode with full parameters
        asyncio.run(main())
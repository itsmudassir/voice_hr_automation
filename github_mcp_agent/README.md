# GitHub MCP Agent

A powerful GitHub agent that uses MCP (Model Context Protocol), LangChain, and OpenAI to answer questions about GitHub repositories and users.

## 🚀 New: Evaluation Agents for HR Automation

### V2 Agent (Recommended) - `github_evaluation_agent_v2.py`
A simpler, more reliable LLM-aided evaluation agent:
- **Flexible evaluation** - Works with any role type
- **Smart analysis** - LLM determines what to search for
- **Concise output** - Structured recommendations
- **No crashes** - Handles edge cases gracefully

### V1 Agent - `github_evaluation_agent.py`
The original evaluation agent provides detailed analysis:
- **Role-based evaluation** - Backend, Frontend, DevOps, Full Stack, Data roles
- **Comprehensive scoring** - Technical, activity, and collaboration metrics
- **Evidence-based analysis** - Links to actual code and contributions
- **Hiring recommendations** - STRONG_HIRE, INTERVIEW_RECOMMENDED, NO_HIRE
- **Red flag detection** - Identifies potential concerns
- **Interview focus areas** - Suggests what to explore in interviews

See [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md) for detailed documentation.

## 🌟 User-Focused Agent

The improved `github_user_focused_agent.py` provides:
- **User-specific sessions** - Works exclusively with one GitHub user at a time
- **GPT-4o mini** - Faster and more cost-effective
- **Smart MCP usage** - Avoids reading large files, uses search efficiently
- **Session management** - Switch between users easily
- **Optimized tools** - Focused on insights without wasting tokens

## Features

- 🔍 **Search repositories** - By language, user, stars, topics
- 👤 **Analyze user profiles** - Language statistics, project analysis
- 📈 **Find trending repos** - Daily, weekly, monthly trends
- 📊 **Repository details** - Full information about any repository
- 💻 **Code search** - Search code across all of GitHub
- 🤖 **Natural conversation** - Ask questions in natural language

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Environment Variables

Create a `.env` file with:

```env
GITHUB_PERSONAL_ACCESS_TOKEN=your_github_pat_here
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. GitHub Token Setup

1. Go to https://github.com/settings/tokens
2. Create a new token with `repo` scope
3. Copy the token to your `.env` file

## Usage

### Evaluation Agent V2 (Recommended) 🎯

The V2 agent is more reliable and flexible:

```bash
# Interactive mode
python github_evaluation_agent_v2.py torvalds

# Direct evaluation with URL
python github_evaluation_agent_v2.py https://github.com/Vaibhavs10

# Quick evaluation for any role
python github_evaluation_agent_v2.py torvalds "evaluate for wordpress developer"
python github_evaluation_agent_v2.py @octocat "evaluate for devops role"
```

### Evaluation Agent V1 (Original) 📊

The V1 agent provides detailed scoring:

```bash
# Full evaluation mode
python github_evaluation_agent.py torvalds "Backend Developer" senior

# Test script
python test_evaluation_agent.py torvalds
```

**Example evaluation queries:**
- "Evaluate for a DevOps Engineer position"
- "Assess frontend development skills"
- "Generate a full hiring report"
- "What are the red flags for this candidate?"
- "Show collaboration and teamwork evidence"

### User-Focused Agent (Recommended) 🌟

The new user-focused agent provides a better experience:

```bash
# Interactive mode - prompts for username
python github_user_focused_agent.py

# Quick mode with username
python github_user_focused_agent.py torvalds

# Quick mode with query
python github_user_focused_agent.py torvalds "show tech stack"
```

**Features:**
- Asks for a specific GitHub username first
- Works only with that user's data (no general searches)
- Uses GPT-4o mini for efficiency
- Smart MCP tool usage
- Session management (switch users with 'switch' command)

**Example queries:**
- "Show profile and stats"
- "What's the tech stack?"
- "Search for web projects"
- "Show recent activity"
- "Tell me about linux repo"
- "Find similar projects to learn from"

### Original Agent (Full GitHub Access)

```bash
python github_agent_final.py
```

Then ask questions like:
- "Show me torvalds repositories"
- "What languages does guido van rossum use?"
- "Find the most popular Python machine learning repos"
- "Show trending JavaScript projects this week"
- "Analyze itsmudassir GitHub profile"

## Available Tools

The agent has access to these GitHub operations:

1. **search_repositories** - Search with GitHub query syntax
2. **get_user_info** - Get user profile information
3. **list_user_repositories** - List all repos for a user
4. **get_repository_details** - Detailed repo information
5. **analyze_user_languages** - Language usage statistics
6. **search_code** - Search code across GitHub
7. **get_trending_repositories** - Find trending projects

## Examples

### Find Popular Projects
```
Q: Find the most popular Python web frameworks
A: Lists Django, Flask, FastAPI with stars, descriptions, and links
```

### Analyze a User
```
Q: What kind of projects does torvalds work on?
A: Shows Linus Torvalds' repositories, mainly C projects including Linux kernel
```

### Search Code
```
Q: Find examples of async/await in Rust
A: Returns code snippets and files using async/await in Rust projects
```

### Trending Analysis
```
Q: What are the hottest AI projects this week?
A: Lists trending AI/ML repositories with descriptions and star counts
```

## Architecture

- **MCP (Model Context Protocol)** - Direct connection to GitHub's API
- **LangChain** - Agent framework for tool orchestration
- **OpenAI GPT-4** - Natural language understanding and response generation
- **Async Python** - Efficient concurrent API calls

## Troubleshooting

### "Missing environment variables"
- Ensure `.env` file exists with both tokens
- Check token validity

### "Connection errors"
- Verify internet connection
- Check if GitHub API is accessible
- Ensure token has correct permissions

### "No results found"
- Try different search terms
- Check if user/repo exists
- Some repos may be private

## License

MIT License - Feel free to use and modify!
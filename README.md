# Legacy Code Challenge System

A system for creating automated coding challenges from GitHub repositories, with an interactive web interface for students.

## 🎯 Overview

This system:
1. **Clones** a GitHub repository
2. **Analyzes** the code to find suitable functions
3. **Injects bugs** using AI (GPT-4) with varying difficulty levels
4. **Generates test cases** automatically
5. **Provides a web interface** for students to fix the bugs

## 📦 Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Set up your OpenAI API key
echo "OPENAI_API_KEY=your-key-here" > .env
```

## 🚀 Quick Start

### Step 1: Generate a Challenge

```bash
# Basic usage (Level 1 - Messy Code)
python main.py https://github.com/username/repo.git

# Specify difficulty level
python main.py https://github.com/username/repo.git --level 2

# Difficulty levels:
# 1 = Messy Code (obfuscated variables, misleading comments)
# 2 = Spaghetti Logic (nested conditions, swapped logic)
# 3 = Sensitive Code (wrong numeric constants)
```

This will create a workspace in `workspaces/repo_name/` with:
- The sabotaged code
- `challenge_run.py` - Test runner
- `STUDENT_README.md` - Instructions for students

### Step 2: Launch the Student Interface

```bash
# Start the web interface
python student_interface.py

# Or specify a different workspace
python student_interface.py workspaces/Ugly_Legacy_code

# Options:
#   --share    Create a public share link (for remote students)
#   --port     Specify port (default: 7860)
```

Then open http://localhost:7860 in your browser!

## 🎓 Student Interface Features

### 1. **Challenge Tab**
- Read the full challenge instructions
- See the bug report with expected vs actual output
- Review constraints and evaluation criteria

### 2. **Code Editor Tab**
- Edit the buggy code directly in the browser
- Submit your solution with an explanation
- Get instant feedback and scoring
- Multiple test cases verify the fix

### 3. **AI Helper Chat**
- Ask questions about debugging strategies
- Get help understanding complex code
- Learn debugging techniques
- **Note:** The AI won't give you the answer directly!

## 🎯 Scoring System

- **50 points**: All unit tests pass
- **30 points**: Fix is minimal (1-3 lines)
- **20 points**: Clear explanation provided

**Total: 100 points**

## 📁 Project Structure

```
LegacyInterview2/
├── main.py                    # Challenge generator
├── student_interface.py       # Web GUI for students
├── requirements.txt           # Dependencies
├── .env                       # API keys (create this!)
├── architect/                 # Core system modules
│   ├── graph.py              # LangGraph workflow
│   ├── repo_cloner.py        # GitHub cloning
│   ├── file_mapper.py        # File selection (randomized)
│   ├── saboteur.py           # Bug injection with GPT
│   ├── challenge_deployer.py # Test case generation
│   └── readme_generator.py   # Student instructions
└── workspaces/               # Generated challenges
    └── repo_name/
        ├── file1.py          # Sabotaged code
        ├── challenge_run.py  # Test runner
        ├── STUDENT_README.md # Instructions
        └── submissions/      # Student submissions
```

## 🔧 How It Works

### Challenge Generation Pipeline

1. **Clone Repo** → Downloads the target repository
2. **Map Files** → Scores and selects the best file for sabotage
3. **Sabotage** → GPT-4 injects bugs and obfuscates code
4. **Deploy** → Creates test cases using GPT-4o-mini
5. **Generate README** → Creates student instructions

### Randomization

The system is highly randomized to ensure each run produces different challenges:
- Random seed based on `time.time() + process_id`
- Random selection from top 2-5 candidate files/functions
- Random temperature for GPT (0.15-0.4)
- Random number of test cases (2-4 additional tests)
- Random number of bugs per difficulty level

### AI Helper Safety

The AI helper is constrained to:
- ✅ Explain concepts and debugging strategies
- ✅ Help understand code structure
- ✅ Guide with questions
- ❌ Never reveal the exact bug location
- ❌ Never show the corrected code
- ❌ Never give away the solution

## 🎨 Example Usage

```bash
# Generate a challenge from a public repo
python main.py https://github.com/refaelz1/Ugly_Legacy_code.git --level 1

# Start the student interface
python student_interface.py workspaces/Ugly_Legacy_code

# Students can now:
# 1. Read the challenge at http://localhost:7860
# 2. Edit the code in the browser
# 3. Chat with the AI helper for guidance
# 4. Submit and get instant scoring
```

## 🧪 Testing

Students can test their fixes at any time:

```bash
cd workspaces/Ugly_Legacy_code
python challenge_run.py
```

This runs all test cases and shows which ones pass/fail.

## 🎓 Educational Use Cases

- **Coding Interviews**: Test debugging skills under pressure
- **Code Review Training**: Learn to read and understand messy code
- **Legacy Code Workshops**: Practice dealing with real-world scenarios
- **Programming Courses**: Automated homework generation
- **Self-Study**: Practice debugging with AI guidance

## 🔒 Security Notes

- The system executes student code in the same environment
- For production use, consider sandboxing (Docker containers, etc.)
- The AI helper is instructed not to give solutions, but creative students might find workarounds
- Submissions are logged locally in `submissions/` folder

## 🛠️ Configuration

### Environment Variables

```bash
# .env file
OPENAI_API_KEY=sk-...          # Required for GPT-4/GPT-4o-mini

# Optional (for Intel corporate network)
# http_proxy=http://proxy-iil.intel.com:912
# https_proxy=http://proxy-iil.intel.com:912
```

### Customization

- **Difficulty levels**: Modify `_LEVEL_INSTRUCTIONS` in `saboteur.py`
- **Scoring weights**: Adjust in `ChallengeGrader.run_tests()`
- **AI helper constraints**: Edit system prompt in `AIHelper.get_system_prompt()`
- **Test case count**: Change `random.randint(2, 4)` in `challenge_deployer.py`

## 📝 Tips for Instructors

1. **Pre-generate challenges** for each student to ensure variety
2. **Review submissions** in the `submissions/` folder
3. **Customize scoring** based on your course requirements
4. **Monitor AI helper** conversations to identify struggling students
5. **Use different repositories** for different difficulty levels

## 🐛 Troubleshooting

**Issue**: "ModuleNotFoundError" when running tests
- **Solution**: Make sure the repository has no external dependencies, or add them to the execution environment

**Issue**: AI helper gives away too much
- **Solution**: Strengthen the system prompt constraints in `AIHelper.get_system_prompt()`

**Issue**: No suitable file found
- **Solution**: The repo might not have functions that meet the criteria (20+ lines, uses primitives, etc.)

**Issue**: All test cases fail after generation
- **Solution**: GPT might have created invalid test cases. Run again (randomization will produce different results)

## 📄 License

This is an educational project. Use responsibly and ensure you have rights to the repositories you're sabotaging!

## 🙏 Credits

Built with:
- LangChain + LangGraph for AI orchestration
- OpenAI GPT-4 and GPT-4o-mini for code analysis and generation
- Gradio for the web interface
- GitPython for repository management

---

**Happy Debugging! 🐛➜✨**

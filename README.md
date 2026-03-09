# Legacy Code Challenge System 

An AI-powered system that transforms any public GitHub repository into a live debugging challenge. Generate complex code challenges with configurable nested function calls, inject intelligent bugs, and provide students with a Gradio web interface for solving them.

##  Overview

The system takes a GitHub repository and:
1. **Clones** the repository
2. **Analyzes** code structure using AST to find functions at specific nesting depths
3. **Injects bugs** using GPT-4 with configurable call-chain complexity
4. **Inflates functions** to hide bugs in deeply nested call chains
5. **Optionally obfuscates** code with renaming and restructuring
6. **Generates test cases** (5 public + 5 secret tests)
7. **Creates student interface** with AI hint system

##  Installation

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-...
```

*(For Intel corporate network, uncomment proxy lines in the code)*

---

##  Quick Start

### Generate a Challenge

```bash
# Basic usage (default: nesting level 3)
python main.py https://github.com/username/repo.git

# Specify nesting level (1-6) - higher = deeper call chains
python main.py https://github.com/mahmoud/boltons.git --nesting-level 4

# Enable refactoring/obfuscation
python main.py https://github.com/mahmoud/boltons.git --nesting-level 5 --refactoring

# Inject multiple bugs
python main.py https://github.com/mahmoud/boltons.git --num-bugs 2

# Enable debug mode to see call chain visualization
python main.py https://github.com/mahmoud/boltons.git --nesting-level 4 --debug
```

**Command-line parameters:**
- `--nesting-level N` (1-6): Target call-chain depth for bug placement (default: 3)
- `--num-bugs N`: Number of bugs to inject (default: 1)
- `--refactoring`: Apply obfuscation and code restructuring (default: off)
- `--debug`: Show call chain visualization and detailed logging (default: off)

### Launch Student Interface

```bash
python student_interface.py workspaces/repo_name
```

Opens at `http://localhost:7860` (use `--port 7861` to change)

---

##  Key Features

###  Nesting Level System (1-6)

The system finds or creates functions at specific call depths:

- **Level 1-2**: Surface functions calling helpers directly
- **Level 3-4**: Multi-layer call chains with intermediate processors
- **Level 5-6**: Deep hierarchies with complex dependencies

**Dynamic Augmentation**: If no deep chain exists, the system automatically creates intermediate wrapper functions (e.g., `_intermediate_processor_1`, `_intermediate_processor_2`) to reach the target depth!

###  Call Chain Visualization (Debug Mode)

When `--debug` is enabled, see the exact call path:

```
Bug Location: unique_iter (Depth: 4)
Call Path:
> unique

  > _intermediate_processor_3
  
    > _intermediate_processor_2
    
      > _intermediate_processor_1
      
        > unique_iter   BUG HERE
```

###  Intelligent Bug Injection

- GPT-4 analyzes function semantics and creates subtle, realistic bugs
- Generates diverse test cases (5 public for development + 5 secret for validation)
- Ensures bugs are detectable but not obvious
- Multiple bug types: off-by-one, wrong operators, inverted conditions, etc.

###  Code Obfuscation (Optional)

With `--refactoring` flag:
- Renames variables to generic names (`var1`, `temp_x`)
- Adds misleading comments
- Restructures control flow
- Makes debugging significantly harder

###  AI Hint System

Progressive hint disclosure based on usage:
- **0-2 hints**: General debugging strategies
- **3-4 hints**: May suggest bug categories (off-by-one, logic error)
- **5+ hints**: May point to specific helper functions
- **Never reveals**: Exact line numbers or corrected code

---

##  Student Interface

### Challenge Setup Page
- Beautiful landing page with challenge title
- Shows challenge parameters (nesting level, refactoring status, number of bugs)
- Start Challenge button to enter main interface

### Main Challenge Page

** Challenge Instructions**
- Mission briefing with bug description
- Function signature and expected behavior
- Test case examples

** Challenge Parameters**
- Nesting Level: Shows call-chain depth
- Refactoring: Enabled/Disabled
- Debug Mode: On/Off
- Number of Bugs: 1-5

** Hint System**
- Chat interface with AI assistant
- Ask questions about the bug
- Progressive hint disclosure
- Hints affect final score (-5 points per hint)

** Code Editor**
- Syntax-highlighted Python editor
- Edit the buggy code directly
- Save changes with instant feedback

** Testing & Submission**
- Run Tests button to execute test suite
- Real-time test results display
- Pass/fail indicators for each test case

** Scoring**
- Test score (0-100 based on passing tests)
- Hint penalty calculation
- Final score display with breakdown

---

##  Project Structure

```
LegacyInterview2/
 main.py                    # Challenge generator (CLI)
 student_interface.py       # Gradio web interface for students
 requirements.txt           # Python dependencies
 .env                       # API keys (create this!)

 architect/                 # Core challenge generation
    graph.py              # LangGraph workflow orchestration
    state.py              # ArchitectState TypedDict definition
    repo_cloner.py        # GitHub repository cloning
    file_mapper.py        # AST analysis & file selection
    saboteur.py           # Bug injection & function inflation
    challenge_deployer.py # Test case generation
    readme_generator.py   # Student instruction generation
    nodes.py              # Individual workflow nodes

 workspaces/               # Generated challenges (auto-created)
     repo_name/
         repo/             # Cloned repository with sabotaged files
         challenge_run.py          # Public test runner (5 tests)
         challenge_run_secret.py   # Secret test runner (5 tests)
         STUDENT_README.md         # Challenge instructions
         detailed_explanation.txt  # Instructor reference
         challenge_state.json      # Metadata for grading
```

---

##  How It Works

### Challenge Generation Pipeline

1. **Clone Repository**
   - Downloads GitHub repo using GitPython
   - Preserves full directory structure

2. **Map Files** 
   - Builds call graphs for all Python files using AST
   - Calculates maximum nesting depth for each file
   - Scores files based on complexity and depth match
   - Weighted random selection (prefers files with target depth)

3. **Select Functions**
   - Finds surface functions with deep call chains
   - 60% probability: use existing deep chains
   - 40% probability: augment shallow chains by creating intermediate functions

4. **Inject Bugs**
   - GPT-4 analyzes target function semantics
   - Creates subtle, realistic bugs (off-by-one, wrong operators, etc.)
   - Generates 10 diverse test cases (5 public + 5 secret)
   - Verifies bug is detectable with test cases

5. **Inflate Hierarchy**
   - Expands ALL functions in call chain to 50+ lines
   - Adds dummy variables, redundant checks, extra calculations
   - Makes bugs much harder to spot

6. **Obfuscate (Optional)**
   - Renames variables to generic names
   - Adds misleading comments
   - Restructures control flow

7. **Deploy Challenge**
   - Writes sabotaged code back to files
   - Creates challenge_run.py with public tests
   - Creates challenge_run_secret.py with secret tests
   - Generates STUDENT_README.md with instructions
   - Creates detailed_explanation.txt for instructors

### Test Case Strategy

**Public Tests (5 cases)**
- Simpler test cases shown during development
- 2-3 should pass with buggy code (for encouragement)
- 2-3 should fail (to guide toward bug)
- Help students understand expected behavior

**Secret Tests (5 cases)**
- Harder edge cases for final validation
- All should fail with buggy code
- More comprehensive coverage
- Only revealed during final submission

**Key principle**: Passing public tests  Passing secret tests!

---

##  Example Workflows

### For Instructors

```bash
# Generate easy challenge (shallow nesting, no obfuscation)
python main.py https://github.com/mahmoud/boltons.git --nesting-level 2

# Generate hard challenge (deep nesting + obfuscation)
python main.py https://github.com/mahmoud/boltons.git --nesting-level 5 --refactoring

# Generate with debug info to verify call chain
python main.py https://github.com/mahmoud/boltons.git --nesting-level 4 --debug

# Review the generated challenge
cat workspaces/boltons/STUDENT_README.md
cat workspaces/boltons/detailed_explanation.txt

# Test the challenge yourself
cd workspaces/boltons
python challenge_run.py
python challenge_run_secret.py
```

### For Students

```bash
# Launch the interface
python student_interface.py workspaces/boltons

# Then in the browser at http://localhost:7860:
# 1. Read the challenge instructions
# 2. Ask AI for hints if stuck
# 3. Edit the code in the editor
# 4. Save changes
# 5. Run tests to see results
# 6. Submit when all tests pass!
```

---

##  Scoring System

**Test Score (0-100)**
- Proportional to test cases passed
- Example: 8/10 tests pass = 80 points

**Hint Penalty**
- -5 points per hint used
- Maximum penalty: -25 points (5 hints)
- Encourages independent problem-solving

**Final Score**
```
Final Score = max(0, Test Score - Hint Penalty)
```

---

##  Educational Use Cases

- **Coding Interviews**: Test debugging skills with realistic bugs
- **Code Review Training**: Learn to read complex call chains
- **Legacy Code Workshops**: Practice dealing with messy codebases
- **Programming Courses**: Automated debugging assignments
- **Self-Study**: Practice with AI-guided learning

---

##  Security Notes

- Student code executes in the same Python environment
- For production: use Docker containers or sandboxing
- AI helper is constrained but determined students may find workarounds
- All submissions logged locally in challenge_state.json

---

##  Tips for Instructors

1. **Pre-generate challenges** for variety (different repos, different runs)
2. **Review generated challenges** using detailed_explanation.txt
3. **Test challenges yourself** before giving to students
4. **Monitor hint usage** to identify struggling students
5. **Use different nesting levels** for different skill levels:
   - Beginners: Level 1-2
   - Intermediate: Level 3-4
   - Advanced: Level 5-6 with --refactoring
6. **Choose good repositories**:
   - Pure Python (no external dependencies)
   - Utility libraries work best (boltons, more-itertools)
   - Avoid frameworks (Django, Flask) - too many dependencies

---

##  Recommended Repositories

Good targets for challenges:
- [mahmoud/boltons](https://github.com/mahmoud/boltons) - Pure Python utilities
- [more-itertools/more-itertools](https://github.com/more-itertools/more-itertools) - Iterator tools
- [jsonpickle/jsonpickle](https://github.com/jsonpickle/jsonpickle) - JSON serialization
- [python-attrs/attrs](https://github.com/python-attrs/attrs) - Class utilities

Avoid:
- Repos with C extensions
- Web frameworks (too many dependencies)
- Repos with heavy external dependencies

---

##  Credits

Built with:
- **LangChain + LangGraph** - AI workflow orchestration
- **OpenAI GPT-4 / GPT-4o** - Code analysis and bug generation
- **Gradio** - Web interface framework
- **GitPython** - Repository management
- **Python AST** - Code structure analysis

---

##  License

Educational use. Ensure you have rights to repositories used.

---

**Happy Debug ging!   **

#!/usr/bin/env python3
"""
Student Interface for Legacy Code Challenge
Web-based GUI for submitting fixes and getting help via AI chat
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime

import gradio as gr
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# Load environment variables
from dotenv import load_dotenv
load_dotenv()


class ChallengeGrader:
    """Grades student submissions and provides feedback."""
    
    def __init__(self, workspace_path: str):
        self.workspace_path = Path(workspace_path)
        self.target_file = None
        self.original_code = None
        self.challenge_info = self._load_challenge_info()
        
    def _load_challenge_info(self) -> dict:
        """Load challenge information from README."""
        readme_path = self.workspace_path / "STUDENT_README.md"
        if not readme_path.exists():
            return {}
        
        with open(readme_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse the README to extract key info
        info = {
            'function_name': None,
            'target_file': None,
        }
        
        for line in content.split('\n'):
            if 'Affected function' in line and '|' in line:
                parts = line.split('|')
                if len(parts) >= 2:
                    info['function_name'] = parts[1].strip().strip('`')
            elif 'File' in line and '|' in line and '.py' in line:
                parts = line.split('|')
                if len(parts) >= 2:
                    info['target_file'] = parts[1].strip().strip('`')
        
        return info
    
    def run_tests(self) -> tuple[bool, str, int]:
        """
        Run challenge_run.py and analyze results.
        Returns: (all_passed, output, score)
        """
        challenge_run = self.workspace_path / "challenge_run.py"
        if not challenge_run.exists():
            return False, "Error: challenge_run.py not found", 0
        
        try:
            result = subprocess.run(
                [sys.executable, str(challenge_run)],
                cwd=str(self.workspace_path),
                capture_output=True,
                text=True,
                timeout=10
            )
            
            output = result.stdout + result.stderr
            
            # Check if all tests passed
            all_passed = "ALL TESTS PASSED" in output
            
            # Count passed tests
            lines = output.split('\n')
            total_tests = 0
            passed_tests = 0
            
            for line in lines:
                if 'Test ' in line and ('PASS' in line or 'FAIL' in line):
                    total_tests += 1
                    if '✓ PASS' in line:
                        passed_tests += 1
            
            # Calculate score
            if total_tests > 0:
                test_score = (passed_tests / total_tests) * 50  # 50 points for tests
            else:
                test_score = 0
            
            # Check if fix is minimal (1-3 lines changed)
            if all_passed:
                minimal_score = 30  # Assume minimal for now
                explanation_score = 20  # Full points if tests pass
            else:
                minimal_score = 0
                explanation_score = 0
            
            total_score = test_score + minimal_score + explanation_score
            
            return all_passed, output, int(total_score)
            
        except subprocess.TimeoutExpired:
            return False, "Error: Tests timed out (infinite loop?)", 0
        except Exception as e:
            return False, f"Error running tests: {str(e)}", 0
    
    def save_submission(self, modified_code: str, explanation: str) -> str:
        """Save the student's submission."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        submission_dir = self.workspace_path / "submissions"
        submission_dir.mkdir(exist_ok=True)
        
        submission_file = submission_dir / f"submission_{timestamp}.json"
        
        submission_data = {
            "timestamp": timestamp,
            "code": modified_code,
            "explanation": explanation,
            "target_file": self.challenge_info.get('target_file', 'unknown')
        }
        
        with open(submission_file, 'w', encoding='utf-8') as f:
            json.dump(submission_data, f, indent=2)
        
        return str(submission_file)


class AIHelper:
    """AI assistant that helps without giving away the solution."""
    
    def __init__(self, workspace_path: str, challenge_info: dict):
        self.workspace_path = Path(workspace_path)
        self.challenge_info = challenge_info
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
        self.conversation_history = []
        
        # Load the buggy code for context
        target_file = challenge_info.get('target_file')
        if target_file:
            file_path = self.workspace_path / target_file
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.buggy_code = f.read()
            else:
                self.buggy_code = ""
        else:
            self.buggy_code = ""
    
    def get_system_prompt(self) -> str:
        """Get the system prompt that constrains the AI."""
        return f"""You are a helpful coding mentor assisting a student with a legacy code debugging challenge.

CHALLENGE CONTEXT:
- The student needs to fix a bug in the function `{self.challenge_info.get('function_name', 'unknown')}`
- The file is `{self.challenge_info.get('target_file', 'unknown')}`
- The fix should be minimal (1-3 lines)

YOUR ROLE:
✅ You CAN:
- Explain general debugging techniques
- Help the student understand how to read complex code
- Explain what specific lines of code do
- Suggest where to look for bugs (general areas)
- Explain Python concepts, syntax, or language features
- Help them understand error messages
- Ask guiding questions to help them think

❌ You CANNOT:
- Tell them exactly which line has the bug
- Show them the corrected code
- Give them the specific fix
- Compare their code to a "correct" version
- Tell them what values to change

IMPORTANT RULES:
1. Be encouraging and supportive
2. Guide with questions, not answers
3. If they ask directly for the solution, politely refuse and redirect
4. Focus on teaching debugging methodology
5. Keep responses concise (2-3 paragraphs max)

The student is working independently - help them learn, don't solve it for them!"""
    
    def chat(self, user_message: str, history: list) -> tuple[str, list]:
        """Process a chat message and return response."""
        
        # Check if user is asking for the solution directly
        solution_keywords = ['solution', 'answer', 'fix it for me', 'what is the bug', 
                            'tell me the fix', 'show me how to fix', 'what should i change']
        
        if any(keyword in user_message.lower() for keyword in solution_keywords):
            response = ("I understand you're stuck, but I can't give you the direct solution - "
                       "that would defeat the learning purpose! 🎓\n\n"
                       "Instead, let me help you with:\n"
                       "1. **Debugging strategy**: Have you tried tracing the function with the test inputs?\n"
                       "2. **Understanding the code**: Is there a specific part that's confusing?\n"
                       "3. **Testing approach**: Have you narrowed down which condition is failing?\n\n"
                       "What specific aspect would you like help understanding?")
            history.append([user_message, response])
            return response, history
        
        # Build conversation context
        messages = [SystemMessage(content=self.get_system_prompt())]
        
        for human, ai in history:
            messages.append(HumanMessage(content=human))
            messages.append(AIMessage(content=ai))
        
        messages.append(HumanMessage(content=user_message))
        
        try:
            response = self.llm.invoke(messages)
            response_text = response.content
            
            # Add to history
            history.append([user_message, response_text])
            
            return response_text, history
            
        except Exception as e:
            error_msg = f"Error: {str(e)}\nPlease try again."
            history.append([user_message, error_msg])
            return error_msg, history


def create_interface(workspace_path: str):
    """Create the Gradio interface."""
    
    workspace = Path(workspace_path)
    grader = ChallengeGrader(workspace_path)
    ai_helper = AIHelper(workspace_path, grader.challenge_info)
    
    # Find the target file
    target_file_name = grader.challenge_info.get('target_file', 'file1.py')
    target_file_path = workspace / target_file_name
    
    # Load the current code
    if target_file_path.exists():
        with open(target_file_path, 'r', encoding='utf-8') as f:
            initial_code = f.read()
    else:
        initial_code = "# File not found"
    
    # Load README
    readme_path = workspace / "STUDENT_README.md"
    if readme_path.exists():
        with open(readme_path, 'r', encoding='utf-8') as f:
            readme_content = f.read()
    else:
        readme_content = "# Challenge README not found"
    
    def submit_solution(code: str, explanation: str):
        """Handle solution submission."""
        if not code.strip():
            return "❌ Error: Code cannot be empty!", "", 0, gr.update()
        
        if not explanation.strip():
            return "⚠️ Warning: Please provide an explanation of your fix.", "", 0, gr.update()
        
        # Save the code to the target file
        try:
            with open(target_file_path, 'w', encoding='utf-8') as f:
                f.write(code)
        except Exception as e:
            return f"❌ Error saving file: {str(e)}", "", 0, gr.update()
        
        # Run tests
        all_passed, output, score = grader.run_tests()
        
        # Save submission
        submission_path = grader.save_submission(code, explanation)
        
        # Format result
        if all_passed:
            result = f"""🎉 **CONGRATULATIONS!** All tests passed!

**Your Score: {score}/100**

Breakdown:
- ✅ All unit tests passed: 50/50
- ✅ Minimal fix (assumed): 30/30
- ✅ Explanation provided: 20/20

Your submission has been saved to:
`{submission_path}`

Great job debugging this legacy code! 🚀"""
        else:
            result = f"""❌ **Tests Failed**

**Your Score: {score}/100**

Some tests are still failing. Keep debugging!

Test Results:
```
{output}
```

💡 Tip: Use the AI Helper chat below for guidance on debugging strategies."""
        
        return result, output, score, gr.update(value=code)
    
    def reset_code():
        """Reset code to original."""
        return initial_code
    
    # Create the Gradio interface
    with gr.Blocks(title="Legacy Code Challenge") as interface:
        gr.Markdown("""
        # 🔧 Legacy Code Challenge - Student Interface
        
        Fix the bug, submit your solution, and get your score!
        """)
        
        with gr.Tabs():
            # Tab 1: Challenge Instructions
            with gr.Tab("📋 Challenge"):
                gr.Markdown(readme_content)
            
            # Tab 2: Code Editor & Submit
            with gr.Tab("💻 Code Editor"):
                gr.Markdown(f"### Editing: `{target_file_name}`")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        code_editor = gr.Code(
                            value=initial_code,
                            language="python",
                            label="Your Code",
                            lines=30
                        )
                        
                        with gr.Row():
                            reset_btn = gr.Button("🔄 Reset to Original", variant="secondary")
                            submit_btn = gr.Button("✅ Submit Solution", variant="primary", size="lg")
                        
                        explanation_box = gr.Textbox(
                            label="Explanation (Required)",
                            placeholder="Explain in one sentence WHY your fix works and what the bug was...",
                            lines=3
                        )
                    
                    with gr.Column(scale=1):
                        result_box = gr.Markdown("### Results\nSubmit your solution to see results here.")
                        score_display = gr.Number(label="Current Score", value=0, interactive=False)
                        test_output = gr.Textbox(label="Test Output", lines=10, max_lines=15)
                
                # Wire up buttons
                reset_btn.click(fn=reset_code, outputs=code_editor)
                submit_btn.click(
                    fn=submit_solution,
                    inputs=[code_editor, explanation_box],
                    outputs=[result_box, test_output, score_display, code_editor]
                )
            
            # Tab 3: AI Helper Chat
            with gr.Tab("🤖 AI Helper"):
                gr.Markdown("""
                ### Ask the AI Helper for Guidance!
                
                The AI can help you with:
                - Understanding the code
                - Debugging strategies
                - Explaining concepts
                
                **Note:** The AI won't give you the direct solution - you need to figure it out! 💪
                """)
                
                chatbot = gr.Chatbot(height=500, label="AI Helper Chat")
                msg = gr.Textbox(
                    label="Your Question",
                    placeholder="Ask about debugging strategies, code understanding, or concepts...",
                    lines=2
                )
                
                with gr.Row():
                    send_btn = gr.Button("📤 Send", variant="primary")
                    clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary")
                
                gr.Examples(
                    examples=[
                        "How should I approach debugging this code?",
                        "Can you explain what this function is supposed to do?",
                        "What's a good strategy for finding off-by-one errors?",
                        "How do I trace through nested conditionals?",
                    ],
                    inputs=msg
                )
                
                # Wire up chat
                def respond(message, chat_history):
                    if not message.strip():
                        return "", chat_history
                    response, updated_history = ai_helper.chat(message, chat_history)
                    return "", updated_history
                
                msg.submit(respond, [msg, chatbot], [msg, chatbot])
                send_btn.click(respond, [msg, chatbot], [msg, chatbot])
                clear_btn.click(lambda: [], None, chatbot)
        
        gr.Markdown("""
        ---
        💡 **Tips:**
        - Read the challenge instructions carefully
        - Run tests frequently to check your progress
        - Use the AI Helper if you're stuck
        - Remember: the fix should be minimal (1-3 lines)!
        """)
    
    return interface


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Launch the student interface for a challenge")
    parser.add_argument(
        "workspace",
        nargs="?",
        default="./workspaces/Ugly_Legacy_code",
        help="Path to the challenge workspace"
    )
    parser.add_argument("--share", action="store_true", help="Create a public share link")
    parser.add_argument("--port", type=int, default=7860, help="Port to run on")
    
    args = parser.parse_args()
    
    workspace_path = Path(args.workspace)
    if not workspace_path.exists():
        print(f"Error: Workspace not found: {workspace_path}")
        sys.exit(1)
    
    print(f"Loading challenge from: {workspace_path}")
    interface = create_interface(str(workspace_path))
    
    print(f"\n🚀 Starting student interface on http://localhost:{args.port}")
    print("Students can now access the challenge through their web browser!\n")
    
    interface.launch(
        server_name="127.0.0.1",
        server_port=args.port,
        share=False,
        inbrowser=False,
        ssl_verify=False
    )


if __name__ == "__main__":
    main()

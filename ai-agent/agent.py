"""
agent.py - SmartData AI Agent (run this file).

The agent follows a simple loop for every message:

    1. PERCEIVE : read the user's message
    2. THINK    : the brain decides which tool to use
    3. ACT      : run the chosen tool
    4. OBSERVE  : collect the tool's result
    5. RESPOND  : show the answer and save it in memory

Run:   python agent.py            (uses Gemini if GEMINI_API_KEY is set)
       python agent.py --offline  (always uses the rule-based brain)
"""

import sys

from brain import choose_brain
from tools import TOOLS, show_help


class SmartDataAgent:
    def __init__(self, brain=None):
        self.brain = brain or choose_brain()
        self.memory = []  # list of (user message, tool used, answer)

    def handle(self, message):
        # 1. PERCEIVE
        message = message.strip()
        if message.lower() in ["history", "memory"]:
            return self.show_history()

        # 2. THINK
        decision = self.brain.decide(message)
        tool_name = decision.get("tool")

        # 3. ACT + 4. OBSERVE
        if tool_name is None:
            answer = decision.get("reply", "I am not sure how to help with that.")
        else:
            function, _ = TOOLS[tool_name]
            try:
                answer = function(**decision.get("args", {}))
            except Exception as error:
                answer = f"The tool '{tool_name}' failed: {error}"

        # 5. RESPOND (and remember)
        self.memory.append((message, tool_name, answer))
        return answer

    def show_history(self):
        if not self.memory:
            return "We have not discussed anything yet."
        lines = [f"{i}. You: {msg}  ->  tool: {tool or 'none'}" for i, (msg, tool, _) in enumerate(self.memory, 1)]
        return "Conversation so far:\n" + "\n".join(lines)


def main():
    agent = SmartDataAgent(choose_brain(offline="--offline" in sys.argv))
    print("=" * 60)
    print(" SmartData AI Agent  |  brain:", agent.brain.name)
    print("=" * 60)
    print(show_help())
    while True:
        try:
            message = input("\nYou: ")
        except (EOFError, KeyboardInterrupt):
            break
        if message.strip().lower() in ["exit", "quit", "bye"]:
            break
        print("Agent:", agent.handle(message))
    print("Goodbye!")


if __name__ == "__main__":
    main()

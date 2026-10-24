"""
brain.py - The "brain" of the agent: it reads the user's message and DECIDES
which tool to use and with which inputs.

Two brains are available:
  1. RuleBasedBrain - works offline, uses keywords and regular expressions.
  2. GeminiBrain    - uses a Large Language Model (Google Gemini) to decide.
                      Used only if a GEMINI_API_KEY is set; if the LLM fails,
                      the agent falls back to the rule-based brain.

Both brains return the same thing: a decision like
    {"tool": "predict_species", "args": {"sepal_length": 5.1, ...}}
"""

import json
import os
import re

from tools import TOOLS

NUMBER = r"-?\d+(?:\.\d+)?"


class RuleBasedBrain:
    name = "Rule-based brain (offline)"

    def decide(self, message):
        text = message.lower()
        numbers = [float(n) for n in re.findall(NUMBER, text)]

        if any(word in text for word in ["predict", "species", "which flower", "classify"]):
            if len(numbers) >= 4:
                keys = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
                return {"tool": "predict_species", "args": dict(zip(keys, numbers[:4]))}
            return {"tool": None, "reply": "Please give 4 numbers: sepal length, sepal width, petal length, petal width."}

        if any(word in text for word in ["calculate", "compute", "what is"]) or re.fullmatch(r"[\d\s+\-*/().%]+", text):
            expression = re.sub(r"[^0-9+\-*/().%\s]", "", text).strip()
            if expression:
                return {"tool": "calculator", "args": {"expression": expression}}

        if any(word in text for word in ["compare", "accuracy", "train", "best model", "models"]):
            return {"tool": "compare_models", "args": {}}
        if any(word in text for word in ["cluster", "group", "k-means", "kmeans"]):
            k = int(numbers[0]) if numbers else 3
            return {"tool": "find_clusters", "args": {"k": k}}
        if any(word in text for word in ["chart", "plot", "graph", "visual", "dashboard"]):
            return {"tool": "make_chart", "args": {}}
        if any(word in text for word in ["summary", "summarise", "summarize", "describe", "dataset", "data"]):
            return {"tool": "dataset_summary", "args": {}}
        if any(word in text for word in ["help", "what can you do", "hello", "hi"]):
            return {"tool": "show_help", "args": {}}

        return {"tool": None, "reply": "I did not understand that. Type 'help' to see what I can do."}


class GeminiBrain:
    """Asks an LLM to pick the tool. The LLM only chooses; Python code still does the work."""

    name = "Gemini LLM brain"

    def __init__(self, api_key, model="gemini-2.0-flash"):
        from google import genai  # imported here so the package is optional

        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.backup = RuleBasedBrain()

    def decide(self, message):
        tool_list = "\n".join(f"- {name}: {desc}" for name, (_, desc) in TOOLS.items())
        prompt = (
            "You are the decision step of a data-analysis agent. Available tools:\n"
            f"{tool_list}\n"
            "predict_species needs args sepal_length, sepal_width, petal_length, petal_width (numbers).\n"
            "calculator needs arg expression (string). find_clusters may take arg k (integer).\n"
            "Reply ONLY with JSON like {\"tool\": \"name\", \"args\": {}}. "
            "If no tool fits, reply {\"tool\": null, \"reply\": \"short answer\"}.\n"
            f"User message: {message}"
        )
        try:
            response = self.client.models.generate_content(model=self.model, contents=prompt)
            raw = response.text.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
            decision = json.loads(raw)
            if decision.get("tool") not in TOOLS and decision.get("tool") is not None:
                raise ValueError("LLM chose an unknown tool")
            return decision
        except Exception:
            # Guardrail: if the LLM fails or answers badly, use the safe offline brain.
            return self.backup.decide(message)


def choose_brain(offline=False):
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key and not offline:
        try:
            return GeminiBrain(api_key)
        except Exception:
            pass
    return RuleBasedBrain()

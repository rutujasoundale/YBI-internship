# SmartData AI Agent

A basic tool-using AI agent built at the end of my AI & Generative AI internship at YBI Foundation.
You type a request in plain English; the agent decides which tool to use, runs it, and replies.

## How it works

```
You ──► PERCEIVE ──► THINK (brain) ──► ACT (tool) ──► OBSERVE ──► RESPOND ──► memory
```

| File | Role |
|---|---|
| `agent.py` | The agent loop and the command-line chat (run this) |
| `brain.py` | Decides which tool to use: offline rule-based brain, or Gemini LLM brain |
| `tools.py` | The tools: dataset summary, chart, model comparison, prediction, K-Means clustering, calculator |

Dataset: the Iris flower dataset (150 flowers, 4 measurements, 3 species), built into scikit-learn.

## Run it

```bash
pip install -r requirements.txt
python agent.py --offline
```

To let a Large Language Model choose the tools, `pip install google-genai`, set `GEMINI_API_KEY`, and run `python agent.py`.
If the LLM fails, the agent automatically falls back to the offline brain.

## Example

```
You: compare models
Agent: Logistic Regression 94.7%, KNN 97.4%, Decision Tree 89.5%, Random Forest 92.1% -> Best model: KNN
You: predict 6.7 3.0 5.2 2.3
Agent: Using KNN (k=5), the flower ... is most likely 'virginica' (confidence 100%).
```

A full sample run is in `outputs/demo_run.txt`.

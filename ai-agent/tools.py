"""
tools.py - The "hands" of the agent.

Each tool is a plain Python function that does ONE job.
The agent's brain decides WHICH tool to call; the tool does the actual work.

Dataset used: the Iris flower dataset (150 flowers, 4 measurements, 3 species).
It ships with scikit-learn, so no download or internet is needed.
"""

import ast
import operator
import os

import matplotlib
matplotlib.use("Agg")  # save charts to files, no pop-up window needed
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Load the data once (Pandas DataFrame)
# ---------------------------------------------------------------------------
_iris = load_iris(as_frame=True)
DF = _iris.frame.copy()
DF.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "target"]
DF["species"] = DF["target"].map(dict(enumerate(_iris.target_names)))
FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

X_train, X_test, y_train, y_test = train_test_split(
    DF[FEATURES], DF["species"], test_size=0.25, random_state=42, stratify=DF["species"]
)

# Models are trained only when first needed and then reused (simple cache).
_trained = {}
_best_model_name = None


# ---------------------------------------------------------------------------
# Tool 1: summarise the dataset  (Pandas - Week 3 learning)
# ---------------------------------------------------------------------------
def dataset_summary():
    counts = {str(k): int(v) for k, v in DF["species"].value_counts().items()}
    means = DF.groupby("species")[FEATURES].mean().round(2)
    text = [
        f"The dataset has {DF.shape[0]} rows and {len(FEATURES)} input columns: {', '.join(FEATURES)}.",
        f"Flowers per species: {counts}.",
        "Average measurements (cm) per species:",
        means.to_string(),
    ]
    return "\n".join(text)


# ---------------------------------------------------------------------------
# Tool 2: draw a chart  (Matplotlib / data dashboard - Week 3-4 learning)
# ---------------------------------------------------------------------------
def make_chart():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name, group in DF.groupby("species"):
        axes[0].scatter(group["petal_length"], group["petal_width"], label=name, alpha=0.7)
    axes[0].set_xlabel("Petal length (cm)")
    axes[0].set_ylabel("Petal width (cm)")
    axes[0].set_title("Petal size by species")
    axes[0].legend()

    DF["species"].value_counts().plot(kind="bar", ax=axes[1], color=["#4C72B0", "#55A868", "#C44E52"])
    axes[1].set_title("Number of flowers per species")
    axes[1].set_ylabel("Count")
    fig.tight_layout()

    path = os.path.join(OUTPUT_DIR, "iris_dashboard.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return f"Chart saved to {path}"


# ---------------------------------------------------------------------------
# Tool 3: train and compare ML models  (Weeks 5-7 learning)
# ---------------------------------------------------------------------------
def compare_models():
    global _best_model_name
    models = {
        "Logistic Regression": LogisticRegression(max_iter=500),
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    }
    scores = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        scores[name] = accuracy_score(y_test, model.predict(X_test))
        _trained[name] = model

    _best_model_name = max(scores, key=scores.get)
    lines = [f"  {name:<20} accuracy = {acc * 100:.1f}%" for name, acc in scores.items()]
    lines.append(f"Best model: {_best_model_name}")
    return "Model comparison on the test set (38 unseen flowers):\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 4: predict the species of a new flower  (multi-class classification - Week 14)
# ---------------------------------------------------------------------------
def predict_species(sepal_length, sepal_width, petal_length, petal_width):
    if _best_model_name is None:
        compare_models()
    model = _trained[_best_model_name]
    row = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]], columns=FEATURES)
    species = model.predict(row)[0]
    confidence = max(model.predict_proba(row)[0]) * 100
    return (
        f"Using {_best_model_name}, the flower with measurements "
        f"({sepal_length}, {sepal_width}, {petal_length}, {petal_width}) is most likely "
        f"'{species}' (confidence {confidence:.0f}%)."
    )


# ---------------------------------------------------------------------------
# Tool 5: find natural groups without labels  (K-Means clustering - Weeks 7-8)
# ---------------------------------------------------------------------------
def find_clusters(k=3):
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    DF["cluster"] = kmeans.fit_predict(DF[FEATURES])
    table = pd.crosstab(DF["cluster"], DF["species"])
    return f"K-Means found {k} groups. How the groups match the real species:\n{table.to_string()}"


# ---------------------------------------------------------------------------
# Tool 6: safe calculator  (Python operators - Week 2)
# Uses ast instead of eval() so that only arithmetic can run.
# ---------------------------------------------------------------------------
_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.USub: operator.neg,
}


def _evaluate(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_evaluate(node.left), _evaluate(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_evaluate(node.operand))
    raise ValueError("Only numbers and + - * / ** % are allowed.")


def calculator(expression):
    try:
        result = _evaluate(ast.parse(expression, mode="eval").body)
        return f"{expression} = {round(result, 4)}"
    except Exception as error:
        return f"Sorry, I could not calculate '{expression}': {error}"


def show_help():
    return (
        "I can do these things:\n"
        "  1. 'summary'                  - describe the dataset\n"
        "  2. 'chart'                    - draw and save a chart\n"
        "  3. 'compare models'           - train 4 ML models and compare accuracy\n"
        "  4. 'predict 5.1 3.5 1.4 0.2'  - predict the species of a flower\n"
        "  5. 'clusters'                 - group flowers with K-Means\n"
        "  6. 'calculate 12*7+3'         - do arithmetic\n"
        "  7. 'history'                  - show what we discussed\n"
        "Type 'exit' to quit."
    )


# The tool registry: tool name -> (function, short description).
# The brain only needs this list to know what the agent is able to do.
TOOLS = {
    "dataset_summary": (dataset_summary, "Describe the Iris dataset"),
    "make_chart": (make_chart, "Draw and save a chart of the data"),
    "compare_models": (compare_models, "Train and compare 4 ML models"),
    "predict_species": (predict_species, "Predict species from 4 measurements"),
    "find_clusters": (find_clusters, "Group flowers using K-Means"),
    "calculator": (calculator, "Evaluate an arithmetic expression"),
    "show_help": (show_help, "List what the agent can do"),
}

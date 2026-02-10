import matplotlib.pyplot as plt
from pathlib import Path
from model_results import MODELS

plt.figure(figsize=(7, 7))

for m in MODELS:
    plt.scatter(m["recall"], m["precision"], s=80)
    plt.text(
        m["recall"] + 0.01,
        m["precision"],
        m["model"],
        fontsize=9,
        verticalalignment="center"
    )

plt.xlabel("Action Recall")
plt.ylabel("Action Precision")
plt.title("Action Detection Trade-offs Across LLMs (v4 Architecture)")

plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.savefig(Path("plots/model_precision_recall_v4.png"), dpi=200)
plt.close()

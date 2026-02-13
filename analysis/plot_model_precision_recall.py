import matplotlib.pyplot as plt
from pathlib import Path
from model_results import MODELS

plt.figure(figsize=(7, 7))

# for m in MODELS:
#     plt.scatter(m["recall"], m["precision"], s=80)
#     plt.text(
#         m["recall"] + 0.01,
#         m["precision"],
#         m["model"],
#         fontsize=9,
#         verticalalignment="center"
#     )

# plt.xlabel("Action Recall")
# plt.ylabel("Action Precision")
# plt.title("Action Detection Trade-offs Across LLMs (v4 Architecture)")

# plt.xlim(0, 1)
# plt.ylim(0, 1)
# plt.grid(True, linestyle="--", alpha=0.6)

# plt.tight_layout()
# plt.savefig(Path("plots/model_precision_recall_v4.png"), dpi=200)
# plt.close()

# Custom label offsets (dx, dy) in "points"
label_offsets = {
    "qwen2.5:7b": (5, 8),   # up-right
    "gpt-4.1":    (7, -3),  # down-right
}

for m in MODELS:
    x, y = m["recall"], m["precision"]
    name = m["model"]

    plt.scatter(x, y, s=80)

    dx, dy = label_offsets.get(name, (6, 6))  # default for others

    plt.annotate(
        name,
        xy=(x, y),
        textcoords="offset points",
        xytext=(dx, dy),
        ha="left",
        va="center",
        # comment out arrowprops if you don't want the small connector line
        # arrowprops=dict(arrowstyle="-", linewidth=0.8, alpha=0.6),
        fontsize=9,
    )

plt.xlabel("Action Recall")
plt.ylabel("Action Precision")
plt.title("Action Detection Trade-offs Across LLMs (v4 Architecture)")

plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid(True, linestyle="--", alpha=0.6)

out = Path("plots/model_precision_recall_v4.png")
out.parent.mkdir(parents=True, exist_ok=True)
plt.tight_layout()
plt.savefig(out, dpi=200)
plt.close()

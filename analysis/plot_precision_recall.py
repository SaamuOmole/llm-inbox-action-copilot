import matplotlib.pyplot as plt
from results_summary import RESULTS

# Create figure
plt.figure(figsize=(6, 6))

# Plot each version
for r in RESULTS:
    plt.scatter(r["recall"], r["precision"])
    plt.text(
        r["recall"] + 0.01,
        r["precision"],
        r["version"],
        fontsize=10,
        verticalalignment="center"
    )

# Axes labels
plt.xlabel("Action Recall")
plt.ylabel("Action Precision")

# Title
plt.title("Action Detection Trade-offs Across Architectures")

# Bounds for clarity
plt.xlim(0, 1)
plt.ylim(0, 1)

# Grid for readability
plt.grid(True, linestyle="--", alpha=0.6)

# Save figure
plt.tight_layout()
plt.savefig("action_precision_recall.png", dpi=200)
plt.close()

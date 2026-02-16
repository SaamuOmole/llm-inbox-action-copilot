import numpy as np
import matplotlib.pyplot as plt
from confusion_matrix_v4 import CM, INTENTS

# Normalize rows
cm_norm = CM / CM.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(cm_norm, cmap="Blues")

# Axis labels
ax.set_xticks(range(len(INTENTS)))
ax.set_yticks(range(len(INTENTS)))
ax.set_xticklabels(INTENTS, rotation=45, ha="right")
ax.set_yticklabels(INTENTS)

ax.set_xlabel("Predicted intent")
ax.set_ylabel("Gold intent")
ax.set_title("Intent Confusion Matrix (v4 - Normalized)")

# Add values inside cells
for i in range(len(INTENTS)):
    for j in range(len(INTENTS)):
        value = cm_norm[i, j]
        if not np.isnan(value):
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="black" if value < 0.6 else "white",
                fontsize=9
            )

# Colorbar
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Proportion")

plt.tight_layout()
plt.savefig("plots/intent_confusion_matrix_v7.png", dpi=200)
plt.close()

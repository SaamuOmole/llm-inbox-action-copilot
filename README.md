# LLM Inbox Action Copilot (Local-First)

A local-first LLM system for **email triage and action detection**, built to explore how large language models can support everyday inbox workflows without relying on cloud APIs.

The project focuses on **detecting whether an email requires action**, classifying the _type_ of action, and understanding the **limits of prompt-only intent classification** through careful evaluation and iterative system design.

---

## Motivation

Modern inboxes mix:

- newsletters and promotions
- confirmations and notifications
- invoices and payments
- meeting requests
- genuine action requests

Rule-based filters are brittle, while end-to-end LLM solutions often hide failure modes behind impressive demos.

This project was built to answer a more practical question:

> _What can a local LLM reliably do for inbox triage, and where do architectural choices matter more than model size?_

---

## Key Features

- **Local-first**: runs entirely on my machine using Ollama
- **Real email data**: emails exported from Thunderbird as `.eml` files
- **Clean preprocessing**: robust HTML → text extraction with reply stripping
- **Weak → gold labeling workflow** with a custom Streamlit UI
- **Multiple LLM architectures** evaluated (single-stage, two-stage, confidence-gated)
- **Transparent evaluation** using accuracy, precision/recall, and confusion matrices

No emails are sent, modified, or uploaded anywhere.

---

## High-Level Pipeline

1. **Email ingestion**

   - Parse `.eml` files
   - Extract headers, subject, sender
   - Clean HTML and isolate the most recent message body

2. **Weak labeling**

   - Heuristic rules generate approximate labels for:
     - intent
     - action presence
   - Used only for bootstrapping and sampling

3. **Gold labeling**

   - A Streamlit app is used to manually label a representative subset
   - Labels focus on _user workflow_, not just content

4. **LLM inference (local)**

   - Local models served via Ollama
   - Strict JSON outputs
   - Deterministic settings (temperature = 0)

5. **Evaluation & iteration**
   - Compare predictions to gold labels
   - Inspect confusion matrices
   - Refine system design (not just prompts)

---

## Label Definitions

### Action Presence

- **True**: the user must do _something_ (reply, schedule, pay, verify, submit, reset, approve)
- **False**: purely informational or ignorable

### Intent Classes

- `meeting_request` – scheduling or availability required
- `invoice_payment` – billing, invoices, payments
- `action_required` – task beyond replying (reset password, submit form, verify account)
- `needs_reply` – a reply completes the task
- `info_only` – confirmations or notifications with no action
- `newsletter` – bulk promotional or broadcast content

---

## LLM Architectures Explored

### v1 – Single-stage classification

- One prompt predicts intent + action presence
- High recall for actions, weaker intent separation

### v2 – Stricter decision rules

- Improved intent accuracy
- Over-conservative action detection

### v3 – Two-stage pipeline

- Stage 1: action detection (high recall)
- Stage 2: intent classification
- Revealed false-positive routing issues

### v4 – Confidence-gated routing (final)

- Stage 1 outputs action presence **and confidence**
- Low-confidence cases routed to non-action path
- Balanced precision/recall and interpretable behavior

**Key insight:**  
Action detection is easier than action _typing_. Architectural choices matter more than model size once a basic competence threshold is reached.

---

## Results (Gold Set: 200 Emails)

### Quantitative comparison of model architecture

The comparison highlights a consistent trade-off between action recall and precision, as expected, as well as between action recall and the intent accuracy across different architectures. All versions were evaluated on the same gold-labeled dataset of 200 real emails.

**Table 1: Model comparison across four pipeline variants**
| Version | Architecture | Action Precision | Action Recall | Action F1 | Action Accuracy | Intent Accuracy | Key Trade-off |
| ------: | --------------------------- | ---------------- | ------------- | --------- | --------------- | --------------- | ----------------------------------- |
| v1 | Single-stage classification | 0.66 | 0.89 | 0.76 | 0.77 | 0.40 | High recall, weak intent separation |
| v2 | Stricter prompt rules | 0.82 | 0.39 | 0.53 | 0.72 | 0.48 | Conservative action detection |
| v3 | Two-stage pipeline | 0.44 | 0.98 | 0.61 | 0.48 | 0.28 | Over-triggers actions |
| v4 | Confidence-gated routing | 0.65 | 0.61 | 0.63 | 0.71 | 0.37 | Balanced precision and recall |

In the Table, v1 achieved a high recall by aggressively flagging emails as actionable, but suffered from over-triggering and weak intent separation. Stricter prompt rules in v2 improved intent accuracy but significantly reduced recall, missing many genuine actions. The two-stage pipeline in v3 shows the highest recall of 98%, where the model consistently over-triggers emails as actionable but very imprecise in doing so. This version also resulted in the lowest intent accuracy, which can be attributed to the 2-stage workflow, where the model already aggressively flags non-actionable emails as actionable.

The final confidence-gated architecture in v4 balances these trade-offs by routing low-confidence cases away from action handling. While it does not maximize any single metric, it produces more stable and interpretable behavior that better reflects real-world inbox workflows, where both false positives and false negatives are costly. This design exposes trade-offs transparently and supports future extensions such as retrieval-augmented context and agent-based decision making, making it a more robust foundation than a purely recall-optimised classifier.

While intent accuracy is modest, error analysis shows most disagreements arise from **workflow ambiguity**, not language misunderstanding (e.g. receipts vs invoices, meeting notifications vs meeting requests, etc.). This mirrors real-world inbox tools, which rely on rules, confidence thresholds, and user feedback rather than pure classification.

<p align="center">
  <img src="./plots/action_precision_recall.png" alt="Action Detection Trade-offs Across Architectures" width="400">
  <br>
  <em>Plot of precision and recall per architecture design</em>
</p>

Each point represents a different system architecture evaluated on the same gold dataset. Designs that maximise recall tend to over-trigger actions, while conservative designs improve precision at the cost of missed actions. The final confidence-gated architecture (v4) balances these trade-offs, producing behavior closer to real-world inbox tools.

<p align="center">
  <img src="./plots/intent_confusion_matrix_v4.png" alt="Action Detection Trade-offs Across Architectures" width="500">
  <br>
  <em>Confusion matrix of gold and predicted intents (200 email inbox samples)</em>
</p>

Above is the normalized intent confusion matrix for the final confidence-gated system (v4). Most errors occur between semantically adjacent workflow categories (e.g. invoices vs informational receipts), reflecting inherent ambiguity in inbox triage rather than language understanding failures.

---

## Why This Project Is Different

- Uses **real personal email data**, not synthetic benchmarks
- Prioritizes **evaluation and error analysis** over model chasing
- Demonstrates when **prompt tuning stops helping**
- Shows how system design can outperform model upgrades
- Entirely **local and privacy-preserving**

---

## Tech Stack

- Python
- Ollama (local LLM inference)
- Streamlit (labeling UI)
- BeautifulSoup / email parser
- FAISS (planned, for RAG)
- JSONL-based datasets and evaluations

---

## Future Work

Planned extensions (intentionally not rushed):

- **RAG**: retrieve related past emails to ground action summaries
- **Agent workflows**: draft replies, propose calendar events, create task objects (user-approved only)
- **Model comparison**: controlled evaluation across different local and hosted LLMs
- **UI layer**: interactive inbox triage demo

---

## Disclaimer

This project is for research and portfolio purposes only.  
No emails are sent, modified, or synced with live email providers.

---

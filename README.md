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
- **Benchmark against cloud-hosted models**: started with local models but extended to benchmark against cloud-hosted LLMs via OpenAI API
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

4. **LLM inference (local and cloud-hosted)**
   - Local models served via Ollama
   - OpenAI Responses API and Structure Outputs
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

### Quantitative Comparison of Model Architecture

The LLM used here is the llama3.1:8b. The comparison below highlights a consistent trade-off between action recall and precision, as expected, as well as between action recall and the intent accuracy across different architectures. All versions were evaluated on the same gold-labeled dataset of 200 real emails.

**Table 1: Model comparison across four pipeline variants**
| Version | Architecture | Action Precision | Action Recall | Action F1 | Action Accuracy | Intent Accuracy | Key Trade-off |
| ------: | --------------------------- | ---------------- | ------------- | --------- | --------------- | --------------- | ----------------------------------- |
| v1 | Single-stage classification | 0.66 | 0.89 | 0.76 | 0.77 | 0.40 | High recall, weak intent separation |
| v2 | Stricter prompt rules | 0.82 | 0.39 | 0.53 | 0.72 | 0.48 | Conservative action detection |
| v3 | Two-stage pipeline | 0.44 | 0.98 | 0.61 | 0.48 | 0.28 | Over-triggers actions |
| ⭐ v4 | Confidence-gated routing | 0.65 | 0.61 | 0.63 | 0.71 | 0.37 | Balanced precision and recall |

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

### Model Comparison and Analysis (v4 architecture)

To evaluate the impact of model choice independently of system design, the final confidence-gated architecture (v4) was benchmarked across multiple LLMs of varying sizes and families. These include both local open-weight models (via Ollama) and cloud-hosted OpenAI models. All models were evaluated on the same gold-labeled dataset of 200 real emails.

| Model          | Intent Acc | Action Prec | Action Rec | Action F1 | Action Acc |
| -------------- | ---------- | ----------- | ---------- | --------- | ---------- |
| llama3.1:8b ⭐ | 0.37       | 0.649       | 0.61       | 0.629     | 0.705      |
| gemma2:9b      | 0.45       | 0.53        | 0.756      | 0.623     | 0.625      |
| mistral:7b     | 0.427      | 0.577       | 0.556      | 0.566     | 0.653      |
| gpt-4.1        | 0.579      | 0.767       | 0.418      | 0.541     | 0.685      |
| gpt-4o-mini    | 0.435      | 0.627       | 0.451      | 0.525     | 0.665      |
| qwen2.5:7b     | 0.45       | 0.775       | 0.378      | 0.508     | 0.7        |
| qwen2.5:14b    | 0.51       | 0.707       | 0.354      | 0.472     | 0.675      |
| llama3.2:3b    | 0.285      | 0.667       | 0.268      | 0.383     | 0.645      |

Across the six local LLMs, including the initial `llama3.1:8b`, performance differences were driven more by model “risk profiles” than by raw capability. Larger models slightly improved intent accuracy but often became overly conservative in action detection, while some mid-sized models favoured recall at the cost of precision.

<p align="center">
  <img src="./plots/model_precision_recall_v4.png" alt="Action Detection Trade-offs Across Architectures" width="400">
  <br>
  <em>Precision–recall trade-offs for action detection across local LLMs using the same confidence-gated architecture (v4). Differences reflect model-specific risk profiles rather than changes in pipeline design.</em>
</p>

The original `llama3.1:8b` baseline remained the most balanced overall, achieving the highest action F1 score. These results suggest that architectural choices and decision thresholds matter more than model size once basic language competence is reached.

Binary action detection is not the hard part for the LLMs. The main difficulty lies in intent separation, arising from the errors in false positives and false negatives classifications. This shows that this task has an inherent ambiguity ceiling i.e., another person will label intents in the emails differently from me, and so would a language model.

#### Key Observations

Intent accuracy improves modestly with stronger instruction-following models, with `gpt-4.1` achieving the highest overall score and `qwen2.5:14b` achieving the highest score among the local LLMs. This suggests an inherent ceiling driven by workflow ambiguity rather than language understanding. The key observations are summarised below:

- Action detection behaviour varies significantly by model family
- `gpt-4.1` achieved the highest intent classification accuracy (0.579), reflecting strong semantic understanding and instruction-following capabilities.
- However, OpenAI models consistently exhibited low recall for actionable emails, prioritising precision over recall. This resulted in missed actions, particularly for implicit obligations such as invoices, confirmations, and follow-up requests.
- `gemma2:9b` favours recall, aggressively flagging potential actions at the cost of false positives.
- `qwen2.5:14b` is more conservative, achieving high precision but missing many genuine actions.
- `llama3.1:8b` provides the most balanced precision–recall trade-off, with the highest Action F1 score (0.629) and strong recall (0.61), making it better aligned with real-world inbox triage priorities.
- Model size alone does not determine performance. Larger models tend to express higher confidence but do not resolve ambiguous inbox categories such as receipts versus invoices or notifications versus requests.

#### Practical Takeaway

Despite being significantly more capable general-purpose models, GPT-4-class models did not outperform carefully calibrated local models on this task. Once basic language competence is reached, model choice primarily shifts risk tolerance rather than correctness.

Across all models, architectural choices and decision thresholds had a greater impact on system behaviour than model selection. The original `llama3.1:8b` baseline achieved the best overall balance (highest action F1) and remained competitive across all metrics, reinforcing the decision to prioritise system design over model scaling. This demonstrates that system design and decision policy can outweigh raw model capability, especially for recall-sensitive tasks. These results underscore a central insight of this project: **effective LLM systems are defined not only by the size of the underlying model itself but by the system architecture and calibration**.

### Retrieval-Augmented Variants (v5 & v6)

After establishing a strong baseline with the confidence-gated two-stage pipeline (v4), I explored whether Retrieval-Augmented Generation (RAG) could improve performance by providing additional contextual signals from previously processed emails. To aid this, embeddings xxx

The motivation was:

- Some emails are short or ambiguous (“Following up...”, “Invoice attached...”, “Meeting updated...”)
- Thread history may clarify intent
- Similar past emails may provide useful pattern signals
- Label-aware retrieval could act as dynamic few-shot examples

Two RAG variants were tested namely, v5 and v6, while retaining the v4 architecture. The v5 variant is the naïve RAG architecture with semantic context injection only, but no labels. The architecture is as follows:

- Stage 1 (action detection): unchanged from v4
- Stage 2 (intent classification) with these injected to the prompt (but conditional upon the confidence level output in Stage i.e., inject if confidence is not high for action detection):
  - Thread context (emails belonging to the same thread id)
  - Top-k semantically similar past emails
- No label information included in retrieved examples

The v6 RAG architecture is the label-aware instance with retrieved few-shot examples designed to address v5’s shortcomings. The RAG pipeline mainly stayed the same but was redesigned so that the contexts also include similar retrieved emails formatted as labeled examples. Gold labels are preferred, where available, with weak labels as fallback.

#### Architecture Evolution

```mermaid
flowchart TD

subgraph v4["v4 — Confidence-Gated Two-Stage Pipeline (Baseline)"]
A4[Email: subject + body] --> B4[Stage 1: Action Detection]
B4 -->|High/Medium Confidence| C4[Stage 2a: Action Intent]
B4 -->|Low Confidence or No Action| D4[Stage 2b: Non-Action Intent]
C4 --> E4[Final Output]
D4 --> E4
end

subgraph v5["v5 — Naïve RAG"]
A5[Email] --> B5[Stage 1]
B5 --> C5[Stage 2 + Retrieved Thread + Similar Emails]
C5 --> D5[Final Output]
end

subgraph v6["v6 — Label-Aware RAG"]
A6[Email] --> B6[Stage 1]
B6 --> C6[Stage 2 + Retrieved Labeled Examples]
C6 --> D6[Final Output]
end
```

Here are the tabulated results for v4, v5 and v6 architectures:

| Version              | Intent Acc | Action Prec | Action Rec | Action F1 | Action Acc |
| -------------------- | ---------- | ----------- | ---------- | --------- | ---------- |
| v6 (label-aware RAG) | 0.395      | 0.65        | 0.634      | 0.642     | 0.71       |
| v5 (naïve RAG)       | 0.365      | 0.65        | 0.634      | 0.642     | 0.71       |
| v4 (baseline)        | 0.37       | 0.649       | 0.61       | 0.629     | 0.705      |

In v5, intent accuracy marginally reduced, compared to v4, while action recall and precision also slightly increased. In the Naïve semantic retrieval, without providing explicit label guidance, the model shifted toward becoming slightly less conservative in detecting actions. The action metrics in v6 are identical to v5, except the slightly improved intent accuracy, which can be attributed to the injected labels which introduced limited supervisory signals.

In both v5 and v6, retrieval provided limited additional signal, resulting in no substantial gains. This demonstrates that RAG is not beneficial for this classification task.

For structured intent detection with a limited label space:

- Prompt design and architectural decisions had greater impact than retrieval augmentation.
- RAG provided only marginal performance gains -- its effectiveness depends on whether retrieval introduces new, task-relevant signal.

As a result, the confidence-gated two-stage pipeline (v4) remains the preferred baseline architecture.

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
- **UI layer**: interactive inbox triage demo

---

## Disclaimer

This project is for research and portfolio purposes only.  
No emails are sent, modified, or synced with live email providers.

---

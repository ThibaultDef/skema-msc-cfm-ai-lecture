# SKEMA CFM AI Agents

**An educational, inspectable AI-agent workbench for cash-receipts variance investigations.**

This Python project supports the **MSc Corporate Financial Management** teaching case at SKEMA Business School. Students investigate one fictional reporting week using approved financial records and supporting documents, compare three LangGraph architectures, and evaluate the quality and limitations of their conclusions.

> **Teaching prototype only.** All case data are fictional. The application is read-only and must not be connected to live treasury or accounting systems without a separate security, access-control, and governance review.

## The case

For **21–27 September 2026**, the company forecast **EUR 500,000** in customer cash receipts and received **EUR 410,000**. The net variance, calculated as **actual − forecast**, is **EUR −90,000**.

| Customer | Forecast | Actual | Variance |
| --- | ---: | ---: | ---: |
| Acme | EUR 200,000 | EUR 80,000 | EUR −120,000 |
| Beta | EUR 150,000 | EUR 150,000 | EUR 0 |
| Cobalt | EUR 100,000 | EUR 80,000 | EUR −20,000 |
| Delta | EUR 50,000 | EUR 100,000 | EUR +50,000 |
| **Total** | **EUR 500,000** | **EUR 410,000** | **EUR −90,000** |

These figures indicate *where* the variance occurred, not necessarily *why*. Supporting evidence includes a partial Acme payment and unconfirmed balance, a disputed Cobalt amount, and an early Delta advance. Some documents are intentionally incomplete or contradictory.

## What students can do

- Compare **Tools Agent**, **RAG Agent**, and **Hybrid Agent** in one Streamlit interface.
- Choose the OpenRouter model and add bounded student instructions.
- Configure permitted financial tools, approved documents, retrieval depth, and execution budgets.
- Toggle optional **SQLite conversation checkpointing**.
- Run classroom scenarios including missing evidence, conflicting evidence, unmatched transactions, and tool failure.
- Inspect execution traces, retrieved passages, financial tool outputs, and the final investigation note; export session results as JSON.
- Discuss whether the evidence supports a conclusion, what remains uncertain, and when to escalate to a human treasury analyst.

## Agent architectures

Each architecture has an **independent LangGraph graph, state, settings, and nodes**; the hybrid graph composes the first two as specialists.

| Agent | Data available to the model through its agent workflow | Design |
| --- | --- | --- |
| **Tools Agent** | Approved read-only finance tools over CSV records | LLM ↔ deterministic financial tools |
| **RAG Agent** | Approved TXT documents only | LLM-directed, potentially iterative document search and evidence review |
| **Hybrid Agent** | Financial and documentary results delegated to its specialists | Parent supervisor graph → Tools/RAG subgraphs → synthesis |

The finance tool allowlist contains `query_forecast`, `query_bank_receipts`, `lookup_open_invoices`, and `reconcile_variance`. The RAG tool is `search_documents`. **The RAG implementation currently uses a local lexical search index**, not embedding-based semantic retrieval or a vector database.

The hybrid supervisor routes tasks to specialized subgraphs. Its initial design consults each specialist at most once per investigation and can require both perspectives for a full note; it is not a free-form multi-agent discussion.

### Controls and limitations

- Financial arithmetic is delegated to deterministic functions; the LLM should not invent amounts.
- Tool/search permissions and per-question budgets are enforced in Python, not only in prompts. The classroom default ceiling is **six calls per question**; the Hybrid Agent partitions its allowance between specialists.
- Actions that modify invoices, forecasts, ledgers, payments, or customer correspondence are outside scope.
- Retrieved documents include source identifiers for review. Source-reference checks **do not guarantee** that every statement is substantively supported.
- Case-specific contradictory evidence should be disclosed, not silently reconciled.
- Human review remains required for material financial conclusions and any follow-up action.

## Requirements

- **Python 3.11+**
- An **OpenRouter API key** and access to a chat model that supports the required tool-calling behavior
- Network access for model calls (the financial CSVs and documentary search are local)

The project depends on LangGraph, LangChain, `langchain-openrouter`, Streamlit, and `python-dotenv`. Optional conversation persistence requires `langgraph-checkpoint-sqlite`.

## Install and run locally

From the project root:

```bash
python -m venv .venv
```

Activate the environment:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
# .venv\Scripts\Activate.ps1
```

Install the package and SQLite checkpoint support:

```bash
python -m pip install -e ".[test]"
python -m pip install langgraph-checkpoint-sqlite
```

Create `.env` in the project root:

```dotenv
OPENROUTER_API_KEY=replace_with_your_openrouter_key
OPENROUTER_MODEL=openai/gpt-4o-mini
# Optional: override SQLite location
# SKEMA_CFM_CHECKPOINT_DB=.local/cfm_checkpoints.sqlite
```

**Never commit `.env` or API keys to version control.** Model availability, pricing, and tool-calling support depend on the selected OpenRouter provider/model.

Start the modular Streamlit application:

```bash
streamlit run streamlit_app.py
```

Then open the URL shown in your terminal (typically `http://localhost:8501`). The root-level `streamlit_app.py` is intentionally a thin entry point importing `skema_cfm_agents.ui.app.main`.

### Installing from PyPI (once published)

If this project is published as `skema-cfm-agents`, installation would be:

```bash
python -m pip install skema-cfm-agents
python -m pip install langgraph-checkpoint-sqlite
```

A convenient `skema-cfm` console command has been **discussed as a distribution option**, but the launcher must be verified/updated to target the modular Streamlit UI before presenting it as the supported student entry point. Until then, `streamlit run streamlit_app.py` is the documented source-checkout command. A PyPI-ready CLI should locate a bundled Streamlit entry script using package-relative paths and pass it to `python -m streamlit run`.

## Using the interface

1. **Select an architecture:** Tools Agent, RAG Agent, or Hybrid Agent.
2. **Choose a scenario:** standard, missing Acme remittance, conflicting Cobalt evidence, unmatched bank transaction, or tool failure.
3. **Configure the agent:** model, allowed tools or documents, passages per search, call budget, and optional instructions. For Hybrid, allocate a finance-call quota; the rest is allocated to RAG search.
4. **Enable conversation memory** if you want follow-up questions to reuse context; otherwise each question is independent, although the visible chat remains on screen.
5. **Ask an investigation question**, for example:

   > Investigate the weekly cash receipts variance. Identify the main customer-level drivers, cite available evidence, flag unresolved conflicts, and recommend the next checks for a treasury analyst.

6. Open **Inspect this execution** to review prompts/trace, retrieved passages, financial tool results, and stopping conditions. Export the session results as JSON for your test log.

**Important:** viewing case CSVs in the interface is separate from the tools the agent actually invoked. Students should assess the agent using the execution trace, not assume it inspected every displayed record.

## Conversation memory (optional)

With memory disabled, the graph executes each question without previous conversation context. With memory enabled, the UI creates a SQLite-backed LangGraph checkpointer and uses a `thread_id` to distinguish conversations.

- **Tools and RAG Agents** have a `messages` state that can retain conversation turns.
- **Hybrid Agent** currently has `question` and `answer` fields rather than a native `messages` history. The UI uses a **limited follow-up adapter**; this should not be confused with complete conversational memory in the supervisor.
- Changing architecture, scenario, or relevant configuration opens a new thread. **New conversation** starts another thread.
- The runner resets per-question call counters and current-turn evidence before invocation, and shows only the latest turn's trace.
- **SQLite checkpoints store the whole graph state**, potentially including prompts, tool outputs, financial evidence, and traces—not just chat text. Do not use sensitive or real financial information in this classroom prototype.

By default the local SQLite file is written to `.local/cfm_checkpoints.sqlite` relative to the launch directory. Use `SKEMA_CFM_CHECKPOINT_DB` to select a writable location. SQLite persistence here is intended for single-user/local teaching exercises, not a production multi-user service.

## Project layout

```text
.
├── pyproject.toml
├── streamlit_app.py                  # Streamlit bootstrap
├── src/skema_cfm_agents/
│   ├── agents/
│   │   ├── tools_agent/
│   │   │   ├── graph.py
│   │   │   ├── settings.py
│   │   │   ├── state.py
│   │   │   └── nodes/
│   │   ├── rag_agent/
│   │   │   ├── graph.py
│   │   │   ├── settings.py
│   │   │   ├── state.py
│   │   │   └── nodes/
│   │   └── hybrid_agent/
│   │       ├── graph.py
│   │       ├── settings.py
│   │       ├── state.py
│   │       └── nodes/
│   ├── ui/
│   │   ├── app.py                    # Streamlit presentation
│   │   ├── runner.py                 # Graph selection and turn execution
│   │   └── memory.py                 # Thread IDs and SQLite saver lifecycle
│   ├── tools/                         # Reusable finance functions
│   ├── case.py                        # Case/scenario loading
│   ├── retrieval.py                   # Local document retrieval
│   ├── models.py                      # OpenRouter model initialization
│   └── case_data/
│       ├── tables/                   # CSV datasets
│       └── documents/                # TXT evidence
├── tests/
└── docs/                             # Project brief and guidance
```

The three agents retain separate orchestration. Shared components include financial calculation functions, the case loader, model initialization, and retrieval implementation. Streamlit does **not** define agent nodes or finance tool logic.

## Teaching data

The shipped case pack contains:

**Structured records**

- `cash_forecast.csv` — expected amounts and payment dates.
- `bank_transactions.csv` — actual bank receipts and remittance references.
- `ar_open_items.csv` — outstanding receivables and recorded statuses.

**Approved documents**

- `DOC-01`: Weekly cash forecast methodology.
- `DOC-02`: Acme remittance advice.
- `DOC-03`: Acme collections tracker entry.
- `DOC-04`: Cobalt dispute email.
- `DOC-05`: Cobalt receivables note.
- `DOC-06`: Delta remittance advice.
- `DOC-07`: Treasury collection and dispute procedure.
- `DOC-08`: Unverified Cobalt follow-up note, intended for a conflict test.

The documents differ in what they establish. For example, an Acme forecast estimate is **not** a payment commitment, and an unverified Cobalt note must **not** supersede the ledger balance without verification.

## Testing

Install test dependencies if not already installed:

```bash
python -m pip install -e ".[test]"
python -m pytest -q
```

To run the interface adapter tests specifically:

```bash
python -m pytest -q tests/test_ui_offline.py
```

The individual agent modules also have offline/mock tests where integrated. **Passing offline tests is not evidence of a successful end-to-end OpenRouter run.** Before classroom distribution, verify actual model tool calls, SQLite follow-up questions, all five scenarios, evidence citations, budget limits, and error behavior in the installed environment.

A useful student test plan covers:

| Scenario | Expected evaluation focus |
| --- | --- |
| Standard investigation | Reconciliation, major drivers, supported citations |
| Missing evidence | Explicit uncertainty and next verification step |
| Conflicting evidence | Conflicting sources identified, no silent override |
| Unmatched transaction | No unsupported invoice matching |
| Tool failure / out-of-scope | Safe stop, transparent failure reporting, no writes |

Students should retain prompts, settings, retrieved evidence, tool traces, results, failures, and improvements for their project deliverables.

## Troubleshooting

**`ModuleNotFoundError: skema_cfm_agents`** — Install the project in the active environment with `python -m pip install -e .`.

**SQLite checkpoint package missing** — Install `langgraph-checkpoint-sqlite` or disable conversation memory.

**`build_graph() got an unexpected keyword argument 'checkpointer'`** — Ensure all three agent graph builders were updated to accept an optional checkpointer and call `builder.compile(checkpointer=checkpointer)`.

**No response from OpenRouter** — Check `OPENROUTER_API_KEY`, the chosen model identifier, account access, network connectivity, and the model's tool-calling capability.

**Chat seems to forget context** — Check the SQLite memory toggle, whether the UI opened a new conversation after a settings change, and whether the selected agent supports native `messages` history (Hybrid currently uses an adapter).

**SQLite cannot write its database** — Choose a writable path with `SKEMA_CFM_CHECKPOINT_DB`.

## Responsible use and project scope

This repository supports a **bounded, read-only, inspectable classroom exercise**. Its output should be treated as a draft investigation note, not an accounting entry, treasury instruction, payment authorization, or verified statement of fact. The agents may omit evidence, retrieve irrelevant passages, misread documents, or produce unsupported explanations. Students are assessed on **numerical correctness, evidence traceability, uncertainty handling, control compliance, and appropriate escalation**—not fluent prose alone.

See the instructor-provided `docs/project_brief.pdf` for full assignment requirements and assessment criteria. Keep instructor answer keys separate from the student distribution.

---

**Status:** Modular teaching prototype. The three-agent backend and Streamlit UI have been developed in separate iterations; validate the fully assembled package, packaging/CLI, and live OpenRouter integration before releasing it to students or publishing it to PyPI.

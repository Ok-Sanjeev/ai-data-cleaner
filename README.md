# 🤖 AI Data Cleaner

### Privacy-Aware AI Planning · Deterministic Data Cleaning · Validated Execution

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.12-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/Pandas-Data%20Processing-150458?logo=pandas" alt="Pandas">
  <img src="https://img.shields.io/badge/FastAPI-Web%20UI%20%26%20REST%20API-009688?logo=fastapi" alt="FastAPI">
  <img src="https://img.shields.io/badge/LLM-Gemini%20%7C%20OpenAI%20%7C%20Claude%20%7C%20Ollama-purple" alt="LLM Providers">
  <img src="https://img.shields.io/badge/Pydantic%20v2-Schema%20Validation-E92063" alt="Pydantic">
  <img src="https://img.shields.io/badge/Pytest-102%20Passed-0A9EDC?logo=pytest" alt="Pytest">
  <img src="https://img.shields.io/badge/Status-Production%20Ready-success" alt="Status">
</p>

**AI Data Cleaner is a production-grade Python automation system that profiles messy CSV datasets, uses a provider-agnostic LLM planning layer (Google Gemini, OpenAI, Claude, or air-gapped local Ollama/vLLM) to recommend targeted cleaning operations under strict privacy constraints, validates plans with bounded retries, enforces local safety guardrails, and executes approved transformations through trusted, deterministic Python functions with step-by-step undo, complete rollback, benchmarked quality evaluation, and an interactive FastAPI web dashboard.**

Rather than letting an LLM execute arbitrary code or write directly to your database, AI Data Cleaner enforces a strict separation of concerns:

**Profile → Privacy Filter → AI Plan → Schema Validation & Retry → Safety Guardrails → User Approval → Checkpointed Execution & Rollback → Validation & Evaluation → Audit Report**

[View Source Code](https://github.com/Ok-Sanjeev/ai-data-cleaner) · [Report an Issue](https://github.com/Ok-Sanjeev/ai-data-cleaner/issues)

---

## 📌 Overview

Real-world datasets frequently suffer from missing values, duplicate records, mixed casing, unparsed dates, dirty numeric strings (with currency symbols and commas), malformed identifiers, and ambiguous constant columns.

Cleaning these datasets manually is tedious and non-reproducible. Conversely, blindly feeding entire CSVs into an LLM compromises privacy, risks silent hallucinations, and can corrupt domain-specific terms (such as converting `"AWS"`, `"SQL"`, or proper company names into lower/title case).

**AI Data Cleaner solves this with a multi-layered, privacy-preserving architecture:**

1. **Local Profiling**: Sniffs delimiters, detects encoding, identifies column data types, calculates IQR outlier statistics, flags inconsistent casing, whitespace, and semantic types (URLs, emails, phone numbers, identifiers, dates).
2. **Zero Raw-Data Leakage**: Strips all cell values, row samples, and category frequencies before sending metadata to Google Gemini.
3. **Bounded Plan Generation with Feedback**: Gemini generates a structured `CleaningPlan`. If the response violates the Pydantic schema, the engine re-prompts the model with precise validation feedback (up to 2 retries) before falling back to safe action recovery.
4. **Local Safety Guardrails**: Prevents destructive title-casing on acronyms, blocks unparseable date conversions on relative text (e.g., `"3+ weeks ago"`), protects provenance/audit metadata in constant columns, and disallows synthetic imputation of URLs or personal IDs.
5. **Human-in-the-Loop Approval**: Presents the guarded plan with clear explanations and allows users to selectively approve (`A,C`), approve `ALL`, reject `NONE`, or quit `Q`.
6. **Local Category Harmonization**: Clusters near-duplicate category values locally via fuzzy matching without exposing labels externally.
7. **Deterministic Execution**: Applies approved vectorized transformations using hardened Pandas routines.
8. **Step-by-Step Undo & Rollback**: Records isolated DataFrame checkpoints before and after every action, enabling interactive single-step undo (`U`), full rollback (`R`), or programmatic state reversal.
9. **Quality Evaluation against Ground Truth**: Calculates cell-level and row-level Precision, Recall, and F1 score against gold-standard data, penalizing over-cleaning and missed errors.
10. **Audit Trail & Artifacts**: Every run generates an isolated artifact directory with run metadata, cleaned data, validation metrics, session checkpoints, and an unresolved data-quality registry.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🔍 **Automated Deep Profiling** | Detects empty/constant columns, whitespace anomalies, date/numeric hints, IQR outliers, and semantic entity types. |
| 🔐 **Strict Privacy-Safe Planning** | Withholds all raw cell values, row samples, and category dictionaries from the AI prompt. |
| 🔁 **Bounded Retries with Feedback** | Automatically recovers from schema errors by sending actionable validation feedback to Gemini. |
| 🛡️ **Multi-Tier Local Guardrails** | Protects acronyms (`AWS`, `SQL`), verifies date parseability, preserves provenance metadata, and blocks invalid outlier clipping. |
| 🔤 **Offline Category Clustering** | Standardizes misspelled or inconsistent category labels locally without external API leaks. |
| 👤 **Granular Human Approval** | Interactive approval gate supporting individual selection (`A`, `B,D`), `ALL`, `NONE`, or `Q`. |
| ⚙️ **Deterministic Execution** | Only trusted Python routines in `actions.py` execute modifications; no arbitrary AI-generated code is run. |
| 🌐 **Provider-Agnostic LLM Layer** | Seamlessly integrates Google Gemini, OpenAI, Anthropic Claude, and Ollama/vLLM for air-gapped offline environments. |
| 🖥️ **Interactive Web UI & REST API** | Full-featured FastAPI service featuring side-by-side visual diffs, distributions viewer, and one-click approvals. |
| ⏪ **Step-by-Step Undo & Rollback** | Checkpoints DataFrame states per action; interactive session allows granular undo (`U`), complete rollback (`R`), and checkpoint diffing. |
| 🎯 **Cleaning Quality Evaluation** | Scores cleaning precision, recall, and F1 against a known ground-truth benchmark, penalizing over-cleaning and missed errors. |
| 📋 **Accurate Unresolved Issue Tracking** | Distinguishes intentional non-imputations (e.g. missing URLs) from execution failures; records preserved data points. |
| 📊 **Run-Level Audit Artifacts** | Automatically saves `profile.json`, `plan.json`, `approval.json`, `cleaned_data.csv`, `cleaning_report.json`, and session checkpoints. |
| 🧪 **Comprehensive Test Suite** | 102 automated unit, integration, and regression tests covering multi-dataset pipelines, API, and providers. |

---

## 🏗️ Architecture

```text
                           ┌────────────────────────┐
                           │       Input CSV        │
                           └───────────┬────────────┘
                                       │
                                       ▼
                           ┌────────────────────────┐
                           │   Dataset Profiler     │
                           │  (Types, Hints, Stats) │
                           └───────────┬────────────┘
                                       │
                                       ▼
                           ┌────────────────────────┐
                           │   Privacy Filtering    │
                           │ (Strip samples/values) │
                           └───────────┬────────────┘
                                       │
                                       ▼
                           ┌────────────────────────┐
              ┌───────────►│     Gemini Planner     │
              │            │  (Structured Planning) │
              │            └───────────┬────────────┘
              │                        │
  Validation Feedback                  ▼
   (Max 2 retries)         ┌────────────────────────┐
              │            │   Pydantic Validation  │
              └────────────┤ (Schema & Constraints) │
                           └───────────┬────────────┘
                                       │ Valid
                                       ▼
                           ┌────────────────────────┐
                           │    Local Guardrails    │
                           │  - Acronym preservation│
                           │  - Date parseability   │
                           │  - Provenance checks   │
                           │  - Sync summary text   │
                           └───────────┬────────────┘
                                       │
                                       ▼
                           ┌────────────────────────┐
                           │   Local Category Asst  │
                           │ (Fuzzy clustering)     │
                           └───────────┬────────────┘
                                       │
                                       ▼
                           ┌────────────────────────┐
                           │  User Approval Gate    │
                           │ (A, B, C / ALL / NONE) │
                           └───────────┬────────────┘
                                       │ Approved Actions
                                       ▼
                            ┌────────────────────────┐
                            │ Deterministic Actions  │
                            │ (Vectorized Pandas)    │
                            └───────────┬────────────┘
                                        │
                                        ▼
               ┌─────────────────────────────────────────────────┐
               │    Checkpointed Session & Undo/Rollback Engine  │
               │    - Deepcopy DataFrame checkpoint per step     │
               │    - Track row/column counts & values replaced  │
               └───────────┬─────────────────────────▲───────────┘
                           │                         │
                           │ Step-by-step Execution  │ Undo ('U') / Rollback ('R')
                           ▼                         │
               ┌─────────────────────────────────────┴───────────┐
               │         Interactive Post-Execution Gate         │
               │  [ENTER] Accept & Save  │  'U' Undo  │  'R' All │
               └───────────┬─────────────────────────────────────┘
                           │ Accepted Final State
                           ▼
               ┌─────────────────────────────────────────────────┐
               │          Validation & Quality Evaluation        │
               │  - Unresolved issues tracking                   │
               │  - Ground-truth Precision / Recall / F1 Score   │
               └───────────┬─────────────────────────────────────┘
                           │
               ┌───────────┴────────────────────────┐
               ▼                                    ▼
    ┌─────────────────────┐              ┌─────────────────────┐
    │  cleaned_data.csv   │              │cleaning_report.json │
    └─────────────────────┘              └─────────────────────┘

---

## 🧹 Supported Cleaning Operations

The cleaning engine supports **15 validated operations**, each strictly checked against column profiles and parameter schemas:

| Operation | Parameter Schema | Description & Safety Rules |
|---|---|---|
| `impute_missing` | `method: "median" \| "mean" \| "mode" \| "constant"`, optional `value` | Imputes missing values. Guardrails enforce mode for categoricals/booleans and forbid mean/median on IDs and contact fields. |
| `drop_missing_rows` | `threshold: float` (optional) | Drops rows exceeding missing value thresholds. |
| `remove_duplicates` | `{}` | Drops exact duplicate records while maintaining dataset integrity. |
| `remove_column` | `{}` | Removes redundant empty or constant columns. Contextualizes metadata/audit columns (`data_source`, `scraped_at`) and prevents dropping the last remaining column. |
| `rename_column` | `new_name: str` | Renames a specific column safely and updates internal column alias maps. |
| `convert_to_date` | `{}` | Converts string dates to `datetime`. Guardrail blocks relative time strings (`"2 days ago"`) that cannot be parsed without a base timestamp. |
| `convert_to_numeric` | `{}` | Strips currency symbols (`$`, `€`, `₹`), unit prefixes, and commas before converting to float/integer. |
| `convert_to_boolean` | `{}` | Standardizes boolean equivalents (`true/false`, `yes/no`, `y/n`, `1/0`) into native boolean values. |
| `normalize_text` | `case: "lower" \| "upper" \| "title"` | Normalizes casing. Guardrails protect acronyms (`SQL`, `AWS`, `AI`), skills, and domain proper nouns from destructive modification. |
| `normalize_whitespace` | `{}` | Trims leading/trailing whitespace and collapses repeated internal spaces. |
| `normalize_column_names` | `{}` | Formats all headers into consistent, clean `snake_case` names. Runs before any column-specific actions. |
| `replace_empty_strings` | `{}` | Converts empty or whitespace-only strings into standard `NaN` missing values. |
| `remove_invalid_values` | `invalid_values: list` (non-empty) | Replaces specific verified invalid placeholder tokens (e.g. `["N/A", "missing", "-999"]`) with `NaN`. |
| `clean_phone_number` | `{}` | Extracts standard numeric sequences and formats telephone numbers. |
| `standardize_categories` | `mapping: dict[str, str]` | Generated locally via fuzzy matching to unify misspelled category labels without leaking values to the AI. |

> **Note on Outliers (`clip_outliers`)**: Statistical outliers detected via IQR are intentionally flagged in the profile and cleaning report for domain review rather than automatically altered, preventing silent corruption of legitimate business figures.

---

## 👤 Interactive Approval Gate

AI Data Cleaner never modifies data without user consent. Every recommendation is displayed with its action type, target column, parameters, and evidence-based reasoning:

```text
Proposed cleaning plan:
Explanation: The dataset contains textual columns with inconsistent casing that could benefit from normalization. There are constant columns providing no discriminative value across records, which can be removed. Missing values in 'job_url' are present but will not be imputed due to the column's nature as a URL.

[A] normalize_text | column='location' | params={'case': 'title'}
    Reason: The 'location' column has inconsistent text casing, affecting data consistency.

[B] remove_column | column='data_source' | params={}
    Reason: The 'data_source' column is constant with a single value across all records. Although it contains zero variance and does not differentiate records in this dataset slice, it represents data provenance/audit metadata. Removal is optional and recommended only if record origin/lineage is unneeded for downstream analysis.

[C] remove_column | column='salary_negotiable' | params={}
    Reason: The 'salary_negotiable' column has a single uniform value across all records. While it offers no variance for distinguishing records within this dataset slice, it represents a uniform business attribute or flag. Removal is proposed for analysis simplification only if this fixed business rule attribute is not needed downstream.

Select recommendations to execute:
  Enter letters to select individual actions (e.g., ABC or A,C)
  ALL     Approve all recommendations
  NONE    Reject all; leave the original dataset unchanged
  Q       Quit without changes

Your choice:
```

Selections are parsed robustly: case-insensitive, punctuation-tolerant (`A, C`, `AC`, `A C`), and verified against available options.

---

## ⏪ Step-by-Step Undo & Rollback Engine

Real-world data cleaning workflows frequently require stepping back when an automated transformation introduces unintended side effects. AI Data Cleaner features a transactional **`CleaningSession`** engine (`session.py`) with per-action checkpointing:

1. **State Checkpointing**: Before and after every approved action executes, an isolated snapshot of the DataFrame is saved along with full action metadata (step number, action name, target column, parameter payload, execution timestamp, and row/column counts).
2. **Interactive Review Gate**: In interactive mode, once actions complete, the user is presented with the complete step history and an interactive prompt:
   ```text
   Session Checkpoints:
     Step 1: normalize_whitespace on 'first_name'
     Step 2: convert_to_numeric on 'salary'
     Step 3: impute_missing on 'age'

   Review changes: [ENTER] to accept and save, 'U' to undo last action, 'R' to rollback all:
   ```
   - Press **`[ENTER]`**: Commits the cleaned dataset to disk and builds audit artifacts.
   - Press **`U`**: Reverts the most recent action, restores the prior DataFrame checkpoint, and decrements the step counter.
   - Press **`R`**: Rolls back the entire session to Step 0 (original data state) immediately.
3. **Programmatic Session API**: For pipeline automation or UI integrations:
   - `session.undo()`: Reverts to step $n-1$.
   - `session.redo()`: Re-applies step $n+1$ if currently undone.
   - `session.rollback_to(step_k)`: Jumps back to any specific checkpoint.
   - `session.rollback_all()`: Returns directly to step 0.
   - `session.diff(step_a, step_b)`: Computes exact cell-level changes, row count deltas, and column differences between any two checkpoints.

---

## 🎯 Cleaning Quality Evaluation & Benchmark

Pipeline testing proves that transformations run without errors, but does not prove that the data is *correctly* cleaned. AI Data Cleaner includes a dedicated **`evaluator.py`** module and gold-standard ground truth benchmark to objectively score cleaning accuracy.

### Benchmark Dataset (`data/benchmark/`)
- **`dirty_sample.csv`** (11 rows): Contains synthetic dirty customer records featuring exact duplicate records, leading/trailing whitespace, inconsistent text casing (`"NEW YORK"`, `"chicago"`), unparsed dates (`"2023/05/14"`, `"12-01-2022"`), currency symbols and commas (`"$50,000"`, `"65,000"`), boolean strings (`"yes"`, `"false"`, `"Y"`), invalid placeholder tokens (`"N/A"`, `"-999"`), and missing numeric fields.
- **`ground_truth.csv`** (10 rows): The verified, perfectly clean gold standard.

### Scoring Metrics & Formulation
Evaluates cell-by-cell alignment using primary key alignment (`customer_id`):
- **True Positive ($TP$)**: Cell was dirty in the original CSV, was modified during cleaning, and now matches ground truth.
- **False Positive ($FP$)**: Cell was altered during cleaning, but does **not** match ground truth (penalizes *over-cleaning* or incorrect modifications).
- **False Negative ($FN$)**: Cell was dirty in the original CSV, but was left unchanged or remains incorrect (penalizes *missed errors*).
- **True Negative ($TN$)**: Cell was clean in the original CSV and was correctly left untouched.

$$\text{Precision} = \frac{TP}{TP + FP} \qquad \text{Recall} = \frac{TP}{TP + FN} \qquad \text{F1 Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

### Quality Scorecard Output
```text
================================================================================
                    CLEANING QUALITY EVALUATION SCORECARD
================================================================================
Overall Metrics:
  Precision:             1.0000 (100.0%)
  Recall:                1.0000 (100.0%)
  F1 Score:              1.0000 (100.0%)
  Accuracy (Original):   0.6286 (62.9%)
  Accuracy (Cleaned):    1.0000 (100.0%)
  Accuracy Delta:        +0.3714 (+37.1%)

Cell-Level Counts:
  True Positives (Repairs):        26
  False Positives (Over-cleaned):  0
  False Negatives (Missed Errors): 0
  True Negatives (Preserved):      44
  Total Cells Evaluated:           70

Row-Level Alignment:
  Ground Truth Rows:    10
  Cleaned Rows:         10
  Duplicate Rows Purged: 1
  Row Precision:        1.0000
  Row Recall:           1.0000
================================================================================
```

---

## 📊 Run Artifacts & Audit Reporting

Every execution generates an isolated directory in `runs/run_<timestamp>/`:

```text
runs/
└── run_20261025_154449/
    ├── profile.json            # Deep statistical & semantic profile
    ├── plan_raw.txt            # Raw LLM response string
    ├── plan.json               # Guarded, validated CleaningPlan
    ├── approval.json           # User selections, rejections, and timestamp
    ├── cleaned_data.csv        # The resulting cleaned dataset
    └── cleaning_report.json    # Full validation metrics & unresolved issues
```

### Unresolved Data-Quality Registry

The `cleaning_report.json` explicitly documents both resolved transformations and preserved anomalies without inventing fake values:

```json
{
  "run_id": "run_20261025_154449",
  "validation": {
    "rows_valid": true,
    "missing_values_reduced": false,
    "missing_reduction_requested": false,
    "approved_imputations_succeeded": true,
    "unexpected_missing_increase": false,
    "unresolved_missing_values_count": 53,
    "unresolved_missing_columns": { "job_url": 53 },
    "overall": true
  },
  "missing_values": {
    "job_url": {
      "before": 53,
      "after": 53,
      "status": "unresolved"
    }
  },
  "unresolved_issues": [
    {
      "issue_type": "missing_values",
      "column": "job_url",
      "count": 53,
      "status": "unresolved_preserved",
      "details": "53 missing URL values preserved without synthetic imputation to avoid inventing fictitious web addresses."
    },
    {
      "issue_type": "relative_or_unparsed_dates",
      "column": "posted_date_raw",
      "status": "preserved_raw",
      "details": "Raw date/temporal strings in 'posted_date_raw' preserved in original format; relative or unstructured dates cannot be safely parsed without reference context."
    }
  ]
}
```

---

## 🌐 Provider-Agnostic LLM Layer

AI Data Cleaner is completely decoupled from any single LLM vendor. Through the unified **`llm_providers.py`** abstraction, you can choose the planning engine that best fits your security, cost, or compliance constraints:

| Provider | Backend | Best For | Configuration |
|---|---|---|---|
| **Google Gemini** | `google-genai` | Default cloud model (`gemini-2.5-flash`), high speed | `GEMINI_API_KEY` |
| **OpenAI** | `openai` | `gpt-4o-mini`, `gpt-4o` with native JSON mode | `OPENAI_API_KEY` |
| **OpenAI-Compatible** | `openai` | **Groq**, **vLLM**, **Together AI**, **LocalAI**, **DeepSeek** | `--base-url <url>` |
| **Anthropic Claude** | `anthropic` | `claude-3-5-haiku`, `claude-3-7-sonnet` | `ANTHROPIC_API_KEY` |
| **Ollama (Air-Gapped)** | Native HTTP | Enterprise, strictly offline, air-gapped on-premise setups | `OLLAMA_HOST` (e.g. `http://localhost:11434`) |
| **Deterministic Mock** | In-memory | Fast CI/CD pipelines, deterministic offline test harnesses | `--provider mock` |

### Running with Different Providers

```powershell
# Using OpenAI GPT-4o-mini
python .\planner.py .\data\test_customers.csv --provider openai --model gpt-4o-mini

# Using Anthropic Claude 3.5 Haiku
python .\planner.py .\data\test_customers.csv --provider anthropic --model claude-3-5-haiku-20241022

# Using a Local Air-Gapped Ollama Instance (Zero External Network)
python .\planner.py .\data\test_customers.csv --provider ollama --model llama3 --base-url http://localhost:11434

# Using vLLM or Groq OpenAI-Compatible Endpoints
python .\planner.py .\data\test_customers.csv --provider openai --model llama-3.3-70b-versatile --base-url https://api.groq.com/openai/v1
```

---

## 🖥️ Interactive Web UI & FastAPI Service

In addition to the interactive terminal CLI, AI Data Cleaner includes a modern, responsive web application and full REST API powered by **FastAPI** (`server.py`).

### Launching the Dashboard

```powershell
# Directly via server.py
python .\server.py --port 8000

# Or via planner CLI flag
python .\planner.py --ui
```

Open your browser to: **`http://127.0.0.1:8000`**

### Web Dashboard Features

1. **Dataset Explorer & Uploader**: Select from bundled sample CSVs (`indian_tech_jobs_2026.csv`, `Messy_Employee_dataset.csv`, `benchmark/dirty_sample.csv`, etc.) or drag-and-drop custom CSV files.
2. **Column Quality & Distribution Viewer**: Inspect data types, semantic type badges (`email`, `phone_number`, `possible_date`), missing value percentage progress bars, and value distribution breakdowns.
3. **One-Click Approval Gate**: Switch between Gemini, Claude, OpenAI, or Ollama in the UI, view evidence-based reasoning, and toggle recommendations individually or with "Approve All" / "Reject All".
4. **Side-by-Side Visual Diff**: Compares original and cleaned tables with green/amber highlighted cells displaying verified modifications, plus summary deltas for rows and missing values.
5. **Interactive Rollback Bar**: Visual checkpoint timeline with one-click **Undo Step**, **Redo Step**, and **Rollback All** buttons.
6. **Benchmark Scorecard**: One-click quality evaluation displaying Precision, Recall, F1 score, and repair breakdowns ($TP, FP, FN, TN$).
7. **One-Click Downloads**: Download the cleaned CSV dataset and audit report JSON directly.

---

## 📁 Project Structure

```text
ai-data-cleaner/
│
├── actions.py                     # Deterministic execution functions (vectorized)
├── ai_privacy.py                  # Profile sanitization & token mapping
├── approval_selection.py          # Input parser for approval gate
├── approval_utils.py              # CLI label formatting utilities
├── category_standardizer.py       # Offline local fuzzy-matching category clusterer
├── evaluator.py                   # Precision/Recall/F1 cleaning quality evaluator
├── llm_providers.py               # Provider-agnostic LLM layer (Gemini, OpenAI, Claude, Ollama)
├── planner.py                     # Main orchestrator, CLI parser, & safety guardrails
├── profiler.py                    # CSV profiler, type inferencer, & format sniffer
├── recommendation_guardrails.py   # Text-normalization & acronym protectors
├── schemas.py                     # Pydantic models (CleaningAction, CleaningPlan)
├── server.py                      # FastAPI backend service & interactive Web UI
├── session.py                     # Transactional checkpoint, undo, & rollback engine
│
├── test_actions.py                # Unit tests for transformations
├── test_ai_privacy.py             # Privacy leak prevention tests
├── test_all_datasets_pipeline.py  # End-to-end multi-dataset test harness
├── test_approval_selection.py     # Approval input validation tests
├── test_approval_utils.py         # Label generation tests
├── test_category_standardizer.py  # Local category harmonization tests
├── test_evaluation.py             # Precision, recall, F1, & benchmark tests
├── test_llm_providers.py          # Provider abstraction tests (Gemini, Claude, OpenAI, Ollama)
├── test_planner_safety_rules.py   # Safety prompt requirement tests
├── test_profiler.py               # Profiler & CSV structure tests
├── test_recommendation_guardrails.py # Text & date safety guardrail tests
├── test_rollback.py               # Stepwise undo, redo, & rollback tests
├── test_schemas.py                # Pydantic schema validation tests
├── test_semantic_detection.py     # Regex & semantic type detection tests
├── test_server_api.py             # FastAPI REST endpoints & Web UI tests
├── test_validation_retries.py     # Retry loop & unresolved issue tests
│
├── data/                          # Input datasets (original data is NEVER overwritten)
│   ├── benchmark/                 # Evaluation gold standard & dirty pairs
│   │   ├── dirty_sample.csv
│   │   └── ground_truth.csv
│   ├── indian_tech_jobs_2026.csv
│   ├── Messy_Employee_dataset.csv
│   ├── ott_viewer_dropoff_retention_us_v1.0.csv
│   ├── test_customers.csv
│   └── robustness_tests/          # Adversarial tests (empty, malformed, no-data)
│
├── runs/                          # Timestamped run output directories
├── .env                           # Local API keys (never committed)
├── .gitignore
└── README.md
```

---

## ⚙️ Installation & Setup

### Prerequisites

- Python 3.10+ (Python 3.12 recommended)
- A Google Gemini API key ([Google AI Studio](https://aistudio.google.com/))

### 1. Clone the repository

```bash
git clone https://github.com/Ok-Sanjeev/ai-data-cleaner.git
cd ai-data-cleaner
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
```

- **Windows (PowerShell)**:
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
- **macOS / Linux**:
  ```bash
  source venv/bin/activate
  ```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install pandas numpy pydantic python-dotenv google-genai pytest
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_actual_gemini_api_key_here
```

---

## ▶️ Usage

### Interactive Cleaning (Default Dataset)

```powershell
python .\planner.py
```

### Cleaning a Specific CSV Dataset

```powershell
python .\planner.py .\data\indian_tech_jobs_2026.csv
```

### Headless / Automated Execution (`--auto-approve`)

Automatically approve and execute the validated plan (ideal for batch jobs or CI/CD pipelines):

```powershell
python .\planner.py .\data\test_customers.csv --auto-approve
```

### Specifying a Custom Output Destination

```powershell
python .\planner.py .\data\test_customers.csv -o .\output\cleaned_customers.csv --auto-approve
```

### Running with Ground-Truth Quality Evaluation

Automatically score cleaning Precision, Recall, and F1 against a known clean dataset:

```powershell
python .\planner.py .\data\benchmark\dirty_sample.csv -g .\data\benchmark\ground_truth.csv --auto-approve
```

### Standalone Quality Evaluation CLI

Evaluate any cleaned dataset against its dirty original and ground truth directly:

```powershell
python .\evaluator.py -d .\data\benchmark\dirty_sample.csv -c .\runs\run_xxx\cleaned_data.csv -g .\data\benchmark\ground_truth.csv
```

---

## 🧪 Automated Testing

The project contains a comprehensive automated test suite with **102 passing tests** covering every layer of the architecture:

```powershell
python -m pytest -v
```

### Current Test Suite Status

```text
============================= test session starts =============================
collected 102 items

test_actions.py ........                                                 [  7%]
test_ai_privacy.py ......                                                [ 13%]
test_all_datasets_pipeline.py ........                                   [ 21%]
test_approval_selection.py .........                                     [ 30%]
test_approval_utils.py ....                                              [ 34%]
test_category_standardizer.py ..                                         [ 36%]
test_evaluation.py ....                                                  [ 40%]
test_llm_providers.py .......                                            [ 47%]
test_planner_safety_rules.py ..                                          [ 49%]
test_profiler.py .........                                               [ 57%]
test_recommendation_guardrails.py .....                                  [ 62%]
test_rollback.py .......                                                 [ 69%]
test_schemas.py ................                                         [ 85%]
test_semantic_detection.py .....                                         [ 90%]
test_server_api.py ....                                                  [ 94%]
test_validation_retries.py ......                                        [100%]

============================ 102 passed in 39.81s =============================
```

### Multi-Dataset Test Matrix

| Dataset | Type / Scale | Verification Focus | Status |
| :--- | :---: | :--- | :---: |
| `indian_tech_jobs_2026.csv` | 23,201 rows · 32 cols | Unparseable relative dates, metadata columns, missing URLs | **PASS** |
| `Messy_Employee_dataset.csv` | 1,000 rows · 10 cols | Text casing, whitespace, dirty types, invalid emails/phones | **PASS** |
| `ott_viewer_dropoff_retention_us_v1.0.csv` | 33,171 rows · 24 cols | High-cardinality retention data, memory stability | **PASS** |
| `test_customers.csv` | 100 rows · 6 cols | Numerical median imputation, duplicate drop | **PASS** |
| `benchmark/dirty_sample.csv` | 11 rows · 7 cols | Precision/Recall/F1 gold-standard benchmark evaluation | **PASS** |
| `robustness_tests/empty.csv` | 0 bytes | Clean exception handling for empty files | **PASS** |
| `robustness_tests/malformed.csv` | Corrupted row | Catches malformed CSV delimiter mismatches | **PASS** |
| `robustness_tests/no_data.csv` | Header only | Rejects files without data rows | **PASS** |

---

## 🗺️ Roadmap & Future Enhancements

- [x] Multi-dataset pipeline testing across diverse domain CSVs
- [x] Bounded retries with validation error feedback for LLM planning
- [x] Local category standardization without leaking category dictionaries
- [x] Text normalization guardrails protecting acronyms and branding
- [x] Accurate recording of unresolved data issues without inventing fake values
- [x] **Stepwise Rollback & Undo**: Transactional `CleaningSession` checkpointing with single-step undo and full rollback
- [x] **Quality Measurement & Benchmark**: Ground-truth evaluation measuring cell-level precision, recall, and F1
- [x] **Provider-Agnostic LLM Engine**: Multi-provider support for Google Gemini, OpenAI, Claude, and local Ollama/vLLM
- [x] **Interactive Web UI & REST API**: FastAPI dashboard with distributions, side-by-side visual diffs, and one-click approvals
- [ ] **Out-of-Core Processing**: Integrate Polars / DuckDB for multi-gigabyte datasets (>10M rows)

---

## 👨‍💻 Author

**Sanjeev Kumar Rai**  
*BE Computer Science & Engineering · 2026 Graduate*  
Focused on data quality engineering, agentic AI pipelines, and reliable Python automation.

- **GitHub:** [@Ok-Sanjeev](https://github.com/Ok-Sanjeev)
- **Repository:** [ai-data-cleaner](https://github.com/Ok-Sanjeev/ai-data-cleaner)

---

## 📄 License

This project is open-source under the [MIT License](LICENSE).

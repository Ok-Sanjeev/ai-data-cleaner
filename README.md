# 🤖 AI Data Cleaner

### Privacy-Aware AI Planning · Deterministic Data Cleaning · Validated Execution

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.12-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/Pandas-Data%20Processing-150458?logo=pandas" alt="Pandas">
  <img src="https://img.shields.io/badge/Google%20Gemini-AI%20Planning-4285F4?logo=google" alt="Google Gemini">
  <img src="https://img.shields.io/badge/Pydantic%20v2-Schema%20Validation-E92063" alt="Pydantic">
  <img src="https://img.shields.io/badge/Pytest-80%20Passed-0A9EDC?logo=pytest" alt="Pytest">
  <img src="https://img.shields.io/badge/Status-Production%20Ready%20CLI-success" alt="Status">
</p>

**AI Data Cleaner is a production-grade Python automation tool that profiles messy CSV datasets, uses Google Gemini to recommend targeted data cleaning operations under strict privacy constraints, validates the proposed plan with bounded retries, enforces local safety guardrails, and executes approved transformations through trusted, deterministic Python functions.**

Rather than letting an LLM execute arbitrary code or write directly to your database, AI Data Cleaner enforces a strict separation of concerns:

**Profile → Privacy Filter → AI Plan → Schema Validation & Retry → Safety Guardrails → User Approval → Deterministic Execution → Post-Validation → Audit Report**

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
8. **Audit Trail & Artifacts**: Every run generates an isolated artifact directory with run metadata, cleaned data, validation metrics, and an unresolved data-quality registry.

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
| 📋 **Accurate Unresolved Issue Tracking** | Distinguishes intentional non-imputations (e.g. missing URLs) from execution failures; records preserved data points. |
| 📊 **Run-Level Audit Artifacts** | Automatically saves `profile.json`, `plan.json`, `approval.json`, `cleaned_data.csv`, and `cleaning_report.json`. |
| 🧪 **Comprehensive Test Suite** | 80 automated unit, integration, and regression tests covering multi-dataset pipelines and robustness edge cases. |

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
                           ┌────────────────────────┐
                           │ Validation & Reporting │
                           │ (Deltas & Unresolved)  │
                           └───────────┬────────────┘
                                       │
                           ┌───────────┴────────────┐
                           ▼                        ▼
                ┌─────────────────────┐  ┌─────────────────────┐
                │  cleaned_data.csv   │  │cleaning_report.json │
                └─────────────────────┘  └─────────────────────┘
```

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

## 📁 Project Structure

```text
ai-data-cleaner/
│
├── actions.py                     # Deterministic execution functions (vectorized)
├── ai_privacy.py                  # Profile sanitization & token mapping
├── approval_selection.py          # Input parser for approval gate
├── approval_utils.py              # CLI label formatting utilities
├── category_standardizer.py       # Offline local fuzzy-matching category clusterer
├── planner.py                     # Main orchestrator, Gemini client, & guardrails
├── profiler.py                    # CSV profiler, type inferencer, & format sniffer
├── recommendation_guardrails.py   # Text-normalization & acronym protectors
├── schemas.py                     # Pydantic models (CleaningAction, CleaningPlan)
│
├── test_actions.py                # Unit tests for transformations
├── test_ai_privacy.py             # Privacy leak prevention tests
├── test_all_datasets_pipeline.py  # End-to-end multi-dataset test harness
├── test_approval_selection.py     # Approval input validation tests
├── test_approval_utils.py         # Label generation tests
├── test_category_standardizer.py  # Local category harmonization tests
├── test_planner_safety_rules.py   # Safety prompt requirement tests
├── test_profiler.py               # Profiler & CSV structure tests
├── test_recommendation_guardrails.py # Text & date safety guardrail tests
├── test_schemas.py                # Pydantic schema validation tests
├── test_semantic_detection.py     # Regex & semantic type detection tests
├── test_validation_retries.py     # Retry loop & unresolved issue tests
│
├── data/                          # Input datasets (original data is NEVER overwritten)
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

---

## 🧪 Automated Testing

The project contains a comprehensive automated test suite with **80 passing tests** covering every layer of the architecture:

```powershell
python -m pytest -v
```

### Current Test Suite Status

```text
============================= test session starts =============================
collected 80 items

test_actions.py ........                                                 [ 10%]
test_ai_privacy.py ......                                                [ 17%]
test_all_datasets_pipeline.py ........                                   [ 27%]
test_approval_selection.py .........                                     [ 38%]
test_approval_utils.py ....                                              [ 43%]
test_category_standardizer.py ..                                         [ 46%]
test_planner_safety_rules.py ..                                          [ 48%]
test_profiler.py .........                                               [ 60%]
test_recommendation_guardrails.py .....                                  [ 66%]
test_schemas.py ................                                         [ 86%]
test_semantic_detection.py .....                                         [ 92%]
test_validation_retries.py ......                                        [100%]

============================== 80 passed in 18.99s =============================
```

### Multi-Dataset Test Matrix

| Dataset | Type / Scale | Verification Focus | Status |
| :--- | :---: | :--- | :---: |
| `indian_tech_jobs_2026.csv` | 23,201 rows · 32 cols | Unparseable relative dates, metadata columns, missing URLs | **PASS** |
| `Messy_Employee_dataset.csv` | 1,000 rows · 10 cols | Text casing, whitespace, dirty types, invalid emails/phones | **PASS** |
| `ott_viewer_dropoff_retention_us_v1.0.csv` | 33,171 rows · 24 cols | High-cardinality retention data, memory stability | **PASS** |
| `test_customers.csv` | 100 rows · 6 cols | Numerical median imputation, duplicate drop | **PASS** |
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
- [ ] **Provider-Agnostic LLM Engine**: Add adapters for Groq (free cloud Llama-3) and Ollama (local/offline models)
- [ ] **Out-of-Core Processing**: Integrate Polars / DuckDB for multi-gigabyte datasets (>10M rows)
- [ ] **Interactive Web UI**: Build a Streamlit or FastAPI + React dashboard with visual side-by-side diffs
- [ ] **Stepwise Rollback**: Interactive undo stack to revert individual approved actions

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

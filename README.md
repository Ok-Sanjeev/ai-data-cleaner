# 🤖 AI Data Cleaner

### Privacy-Aware AI Planning · Deterministic Data Cleaning · Validated Execution

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.12-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/Pandas-Data%20Processing-150458?logo=pandas" alt="Pandas">
  <img src="https://img.shields.io/badge/Google%20Gemini-AI%20Planning-4285F4?logo=google" alt="Google Gemini">
  <img src="https://img.shields.io/badge/Pydantic-Schema%20Validation-E92063" alt="Pydantic">
  <img src="https://img.shields.io/badge/Pytest-Automated%20Testing-0A9EDC?logo=pytest" alt="Pytest">
</p>

**AI Data Cleaner is a Python-based data quality automation project that profiles messy CSV datasets, uses Google Gemini to recommend appropriate cleaning operations, validates the proposed plan, and executes approved transformations through trusted Python functions.**

Rather than allowing an AI model to modify data directly, the project separates AI-generated recommendations from local execution.

**Profile → Recommend → Validate → Approve → Execute → Verify → Report**

[View Source Code](https://github.com/Ok-Sanjeev/ai-data-cleaner) · [Report an Issue](https://github.com/Ok-Sanjeev/ai-data-cleaner/issues)

---

## 📌 Overview

Real-world datasets frequently contain missing values, duplicate records, inconsistent data types, malformed dates, and other data quality issues.

Cleaning these datasets manually can be repetitive, error-prone, and difficult to reproduce.

AI Data Cleaner explores a safer approach to automating this process.

The application:

- Profiles an input CSV locally.
- Detects data quality issues and gathers relevant statistics.
- Sends a privacy-filtered profile to Google Gemini.
- Generates a structured cleaning plan.
- Validates recommendations against predefined schemas and safety rules.
- Presents recommendations for user approval.
- Executes approved transformations using local Python functions.
- Checks the resulting dataset and produces a JSON cleaning report.

### The Core Design Principle

> **AI recommends the transformation. Validated Python code performs it.**

The AI does not generate executable Python or SQL for modifying the dataset. Its recommendations must pass the application's validation and safety checks before execution.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🔍 Automated profiling | Inspects CSV structure and identifies data quality issues. |
| 🧠 AI-assisted planning | Uses Google Gemini to recommend targeted cleaning operations. |
| 🔐 Privacy-aware prompting | Excludes raw cell values and category labels from the AI prompt. |
| 📋 Structured output | Uses Pydantic to validate the proposed cleaning plan. |
| 🛡️ Safety guardrails | Restricts execution to supported operations and validated parameters. |
| 👤 Human-in-the-loop approval | Lets users select individual recommendations before execution. |
| ⚙️ Deterministic execution | Applies transformations using trusted local Python functions. |
| ✅ Post-cleaning validation | Checks the resulting dataset and records validation outcomes. |
| 📊 Run-level artifacts | Keeps profiles, plans, cleaned data, and reports organized by run. |
| 🧪 Automated tests | Tests cleaning logic, privacy controls, selection handling, and safety rules. |

---

## 🏗️ Architecture

```text
                    ┌──────────────────────┐
                    │     Input CSV        │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Dataset Profiler    │
                    │  Local Data Analysis  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  Privacy Filtering   │
                    │  Profile Preparation │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │    Gemini Planner    │
                    │  Cleaning Proposals  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  Schema Validation   │
                    │      Pydantic        │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  Safety Guardrails   │
                    │  Plan Verification   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  User Approval Gate  │
                    │   Select Operations  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  Trusted Action      │
                    │      Registry        │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Deterministic Local  │
                    │   Data Processing    │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Post-Cleaning Checks │
                    └──────────┬───────────┘
                               │
                    ┌──────────┴───────────┐
                    ▼                      ▼
          ┌──────────────────┐   ┌──────────────────┐
          │   Cleaned CSV    │   │   JSON Report    │
          └──────────────────┘   └──────────────────┘
```

### Why separate planning from execution?

AI-generated output can be incomplete, invalid, or inappropriate for a particular dataset.

Separating planning from execution creates distinct validation boundaries:

1. The profiler gathers evidence.
2. The AI proposes operations.
3. The schema validates the structure and parameters.
4. Safety guardrails check the proposed plan.
5. The user approves the operations.
6. Local functions execute the approved transformations.
7. Validation checks and reporting document the result.

This architecture reduces the risk of arbitrary AI-generated instructions being executed against a dataset.

---

## 🔐 Privacy-Aware AI Planning

Privacy is an important design consideration because datasets may contain sensitive or identifying information.

The application profiles the dataset locally and prepares a restricted representation for the AI planner.

### Information excluded from the Gemini prompt

- Raw cell values.
- Sample data rows.
- Category labels.
- Category-frequency keys.

The application can provide original column names, locally detected semantic types, and relevant aggregate metadata to help Gemini understand the dataset's structure.

For example, a profile may indicate that a column contains missing values and has a numeric type without sending the actual values in that column.

### Privacy pipeline

```text
Local Dataset
     │
     ▼
Local Profiling
     │
     ▼
Privacy Filtering
     │
     ▼
Structured Metadata
     │
     ▼
Google Gemini
```

The application uses privacy filtering to limit the information supplied to the model.

**Important:** This is a privacy-aware design, not a guarantee that every dataset or execution environment is risk-free. Column names and aggregate statistics may themselves reveal information. Review the metadata sent to external services before using the application with confidential data.

---

## 🧹 Supported Cleaning Operations

The current AI planning workflow supports the following operations:

| Operation | Purpose |
|---|---|
| `impute_missing` | Fill missing values using a supported imputation method. |
| `remove_duplicates` | Remove duplicate records. |
| `convert_to_date` | Convert eligible columns to date or datetime values. |
| `standardize_categories` | Standardize categorical values through the supported local cleaning logic. |

Recommendations are subject to schema validation and application safety rules.

The AI planner is intentionally restricted from generating category mappings because category labels are excluded from its input. This prevents the model from proposing mappings based on labels it cannot see.

The local implementation may support capabilities that are not available to the AI planner. The supported AI-generated action set should therefore not be interpreted as a complete inventory of every function in the codebase.

---

## 👤 Human-in-the-Loop Approval

AI Data Cleaner includes an interactive approval step so users can decide which recommended transformations should run.

For example, a plan may contain three recommendations:

```text
A. Convert Join_Date to a date type
B. Impute missing values in Age
C. Impute missing values in Salary
```

The user can select individual recommendations instead of accepting the entire plan.

### Available selection options

| Input | Behavior |
|---|---|
| `A` | Execute recommendation A. |
| `AC` | Execute recommendations A and C. |
| `A,C` | Execute recommendations A and C. |
| `A C` | Execute recommendations A and C. |
| `ALL` | Approve all recommendations. |
| `NONE` | Reject all recommendations and leave the original dataset unchanged. |
| `Q` | Quit without applying changes. |

Selection is validated against the recommendations available for the current run. Invalid selections are rejected.

### Why approval matters

Human approval helps prevent unnecessary transformations and gives the user control over potentially destructive operations.

The application defaults to requiring approval. Automatic execution should only be used when explicitly enabled and appropriate for the workflow.

---

## 🛡️ Safety and Validation

The application uses multiple layers of validation instead of trusting model output directly.

### 1. Structured plan validation

Pydantic schemas validate the structure of the cleaning plan and the permitted action parameters.

### 2. Recommendation guardrails

The application checks proposed operations against its supported action set and safety rules.

### 3. Controlled execution

Approved operations are dispatched to trusted local Python functions.

The application does not execute arbitrary Python code or SQL generated by Gemini.

### 4. Explicit user approval

Users can approve selected actions, reject all recommendations, or quit without applying changes.

### 5. Post-cleaning checks

The pipeline validates the resulting dataset and records relevant outcomes in the run report.

### Execution model

```text
Untrusted AI Output
        │
        ▼
Schema Validation
        │
        ▼
Safety Checks
        │
        ▼
User Approval
        │
        ▼
Trusted Local Functions
        │
        ▼
Validated Output
```

These controls improve predictability and testability, but they do not replace backups, code review, or validation against the requirements of a specific dataset.

---

## 📊 Example Workflow

Consider a dataset containing employee records with missing values and inconsistent date formats.

### Input

```text
employee_id,Age,Salary,Join_Date
101,25,45000,2024-01-15
102,,52000,15/02/2024
103,29,,2024-03-10
104,31,61000,20/04/2024
```

### Step 1 — Profile

The application inspects the CSV and gathers information about its structure and data quality.

Potential findings include:

- Missing values in `Age`.
- Missing values in `Salary`.
- Dates represented using different formats.
- Column-level type inconsistencies.

### Step 2 — Generate recommendations

Gemini receives the permitted dataset metadata and recommends appropriate operations.

### Step 3 — Validate and approve

The application validates the plan and displays the recommendations for selection.

### Step 4 — Execute

Only the approved operations are executed through local Python functions.

### Step 5 — Validate and report

The application checks the resulting dataset and records the run's results.

The example above illustrates the workflow; actual recommendations and results depend on the input dataset, the generated plan, and the actions approved by the user.

---

## 📁 Project Structure

```text
ai-data-cleaner/
│
├── actions.py
├── ai_privacy.py
├── approval_selection.py
├── approval_utils.py
├── planner.py
├── profiler.py
├── recommendation_guardrails.py
├── schemas.py
│
├── test_actions.py
├── test_ai_privacy.py
├── test_approval_selection.py
├── test_approval_utils.py
├── test_planner_safety_rules.py
├── test_profiler.py
├── test_recommendation_guardrails.py
├── test_schemas.py
├── test_semantic_detection.py
│
├── data/                   # Local input datasets
├── runs/                   # Generated run artifacts
│   └── run_<timestamp>/
│       ├── profile.json
│       ├── plan_raw.txt
│       ├── plan.json
│       ├── cleaned_data.csv
│       └── cleaning_report.json
│
├── .env                    # Local API credentials
├── .gitignore
├── README.md
└── requirements.txt        # If maintained in the repository
```

The `data/`, `runs/`, and `.env` paths are local working files and should not be committed when they contain private data, generated artifacts, or credentials.

The exact files generated during a run may vary according to the execution path.

---

## ⚙️ Technology Stack

| Technology | Role |
|---|---|
| Python | Application logic and orchestration |
| Pandas | CSV processing and data transformations |
| NumPy | Numerical operations |
| Google Gemini API | AI-assisted cleaning-plan generation |
| Pydantic | Schema and parameter validation |
| python-dotenv | Local environment-variable configuration |
| Pytest | Automated testing |
| Git and GitHub | Version control and project collaboration |

---

## 🚀 Installation

### Prerequisites

- Python 3.12 recommended.
- Git.
- A Google Gemini API key.
- A CSV dataset to process.

### 1. Clone the repository

```bash
git clone https://github.com/Ok-Sanjeev/ai-data-cleaner.git
cd ai-data-cleaner
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
source venv/bin/activate
```

### 3. Install dependencies

If the repository contains a maintained `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

Otherwise, install the required dependencies:

```bash
python -m pip install pandas numpy pydantic python-dotenv google-genai ydata-profiling pytest
```

### 4. Configure the API key

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key
```

Keep your API key private. Do not commit `.env` or include credentials in logs, screenshots, or issue reports.

---

## ▶️ Usage

Activate your virtual environment and run the application from the project root.

### Run the default workflow

```powershell
python .\planner.py
```

The application uses its configured default input dataset.

### Process a specific CSV

If your current `planner.py` supports a positional CSV argument, run:

```powershell
python .\planner.py .\data\your_dataset.csv
```

Replace the example path with your dataset's actual location.

### Select recommendations

When the approval prompt appears, enter the recommendation letters:

```text
Select recommendations to execute:
  Enter letters to select individual actions (e.g., ABC or A,C)
  ALL     Approve all recommendations
  NONE    Reject all; leave the original dataset unchanged
  Q       Quit without changes
```

Review each recommendation before approving it.

---

## 📦 Run Artifacts

The application organizes execution outputs into run-specific directories.

Example:

```text
runs/
└── run_<timestamp>/
    ├── profile.json
    ├── plan_raw.txt
    ├── plan.json
    ├── cleaned_data.csv
    └── cleaning_report.json
```

### Artifact descriptions

| Artifact | Purpose |
|---|---|
| `profile.json` | Records the dataset profile generated during the run. |
| `plan_raw.txt` | Stores the raw AI planner response when generated and saved by the application. |
| `plan.json` | Stores the structured cleaning plan when produced. |
| `cleaned_data.csv` | Contains the resulting dataset. |
| `cleaning_report.json` | Records cleaning outcomes and validation information. |

These artifacts help make individual runs easier to inspect and debug.

Keep in mind that profiles and reports may contain information about dataset structure. Treat them according to the sensitivity of the input data.

---

## 🧪 Automated Testing

The project includes automated tests for its cleaning logic, validation, and safety mechanisms.

Run the complete test suite:

```powershell
python -m pytest -q
```

### Latest recorded test result

```text
47 passed
```

This result reflects the test run recorded during development. Run the suite locally to verify the current checkout.

### Test coverage areas

- Cleaning action behavior.
- Dataset profiling.
- CSV input validation.
- Privacy filtering.
- Semantic detection.
- Approval selection parsing.
- Approval utilities.
- Recommendation guardrails.
- Cleaning-plan schema validation.
- Planner safety rules.

Automated tests help verify local behavior without requiring every test to call the external AI service.

---

## 🧠 Engineering Principles

### 1. AI for decisions, Python for execution

The model recommends a plan; trusted code performs the transformations.

### 2. Privacy-aware data handling

Raw cell values and category labels are excluded from the Gemini planning prompt.

### 3. Evidence-based recommendations

Cleaning actions should be supported by information gathered during profiling rather than applied indiscriminately.

### 4. Human control

Users can review recommendations and choose which actions to execute.

### 5. Minimal transformation

The goal is to improve data quality while avoiding unnecessary changes to valid records.

### 6. Reproducibility and auditability

Structured plans, deterministic operations, and run-level artifacts make it easier to investigate what happened during a cleaning run.

### 7. Testable architecture

Separating profiling, planning, validation, approval, and execution makes individual components easier to test and maintain.

---

## ⚠️ Current Limitations

AI Data Cleaner is an evolving project, and several considerations remain important:

- AI recommendations depend on the quality and completeness of the generated profile.
- Ambiguous data may require human judgment.
- Schema validation cannot guarantee that every semantically valid operation is appropriate for a particular business context.
- Privacy filtering reduces information exposure but does not guarantee complete anonymity.
- Cleaning results should be checked against domain-specific rules before production use.
- Automatic execution should be enabled only when the workflow and its safeguards have been reviewed.
- Support for additional formats, interfaces, and cleaning operations depends on future implementation.

Always preserve the original dataset and verify important transformations before relying on the output.

---

## 🗺️ Roadmap

Potential future improvements include:

- [ ] Expand testing across diverse real-world datasets.
- [ ] Strengthen semantic validation and execution guardrails.
- [ ] Add configurable data-quality metrics.
- [ ] Improve before-and-after quality comparisons.
- [ ] Expand supported cleaning operations.
- [ ] Add more detailed run summaries.
- [ ] Develop an interactive web interface.
- [ ] Add visualizations for data-quality issues.
- [ ] Explore support for additional input formats.

These are planned enhancements, not claims about currently implemented functionality.

---

## 👨‍💻 Author

**Sanjeev Kumar Rai**

BE Computer Science & Engineering · 2026 Graduate

Interested in data analytics, Python automation, data quality engineering, and practical AI applications.

- **GitHub:** [Ok-Sanjeev](https://github.com/Ok-Sanjeev)
- **Project:** [AI Data Cleaner](https://github.com/Ok-Sanjeev/ai-data-cleaner)

---

## ⭐ Support

If you find this project useful or interesting, consider starring the repository.

Contributions, bug reports, and suggestions for improving the safety and reliability of automated data cleaning are welcome.

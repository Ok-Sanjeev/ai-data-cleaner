# 🤖 AI Data Cleaner

### AI-powered, evidence-based data cleaning for messy CSV datasets

[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)](https://www.python.org/) [![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-150458?logo=pandas)](https://pandas.pydata.org/) [![Gemini](https://img.shields.io/badge/Google%20Gemini-AI%20Planning-4285F4?logo=google)](https://ai.google.dev/) [![Pydantic](https://img.shields.io/badge/Pydantic-Validation-E92063)](https://docs.pydantic.dev/) 

> **Give it a messy CSV. It profiles the data, identifies quality problems, creates a cleaning plan, validates that plan, and safely produces a cleaned dataset.**

---

## ✨ What is this?

Real-world datasets are rarely clean.

They often contain:

- Missing values
- Duplicate records
- Inconsistent categories
- Mixed date formats
- Incorrect data types
- Empty strings
- Whitespace problems
- Invalid values
- Outliers
- Inconsistent column names

**AI Data Cleaner** combines **AI-based planning** with a **deterministic and validated execution layer**.

The AI does not directly modify the dataset or execute arbitrary code. Instead, it analyzes the dataset profile and selects appropriate operations from a trusted cleaning library.

> **AI decides → Pydantic validates → Trusted Python code executes → Results are validated**

---

# 🧠 How It Works

```text
                    ┌─────────────────┐
                    │   CSV Dataset   │
                    └────────┬────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │  Universal Profiler │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │  Data Quality       │
                  │      Profile        │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │   Google Gemini     │
                  │   AI Planner        │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Structured Cleaning │
                  │       Plan         │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Pydantic Validation │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Trusted Action      │
                  │     Registry        │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Deterministic       │
                  │ Python Cleaning     │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Post-Cleaning       │
                  │    Validation       │
                  └──────────┬──────────┘
                             │
                    ┌────────┴────────┐
                    ▼                 ▼
           ┌─────────────────┐ ┌─────────────────┐
           │  Cleaned CSV    │ │  JSON Report    │
           └─────────────────┘ └─────────────────┘
---

# 🚀 Key Features

| Feature                        | Description                                    |
| ------------------------------ | ---------------------------------------------- |
| 🔍 **Automatic Profiling**     | Analyzes unfamiliar CSV datasets               |
| 🧠 **AI Planning**             | Gemini identifies required cleaning operations |
| 🛡️ **Safe Execution**         | AI cannot execute arbitrary Python or SQL      |
| 📋 **Structured Plans**        | Cleaning plans validated using Pydantic        |
| ⚙️ **12 Operations**           | Trusted deterministic cleaning functions       |
| 📊 **Before/After Analysis**   | Tracks changes to the dataset                  |
| ✅ **Post-Cleaning Validation** | Verifies cleaning results                      |
| 📄 **JSON Reports**            | Generates detailed run reports                 |
| 🧪 **Automated Testing**       | 19 tests currently passing                     |

---

# 🧹 Supported Cleaning Operations

The cleaning engine currently supports **12 operations**.

### Missing & Duplicate Data

* `impute_missing`
* `remove_duplicates`

### Data Type Conversion

* `convert_to_date`
* `convert_to_numeric`
* `convert_to_boolean`

### Text & Category Cleaning

* `standardize_categories`
* `normalize_text`
* `normalize_whitespace`

### Schema & Value Cleaning

* `normalize_column_names`
* `replace_empty_strings`
* `remove_invalid_values`

### Statistical Cleaning

* `clip_outliers`

The AI does **not** blindly apply all operations.

It selects operations based on evidence found during profiling.

---

# 🔎 Dataset Profiling

Before asking Gemini for a cleaning plan, the profiler analyzes the dataset.

It can identify:

* Missing values
* Duplicate rows
* Possible dates
* Possible numeric columns
* Possible boolean columns
* Email-like values
* Identifier-like columns
* Empty strings
* Whitespace inconsistencies
* Inconsistent categories
* Text-case inconsistencies
* Invalid negative values
* Column-name issues
* Numeric IQR statistics

The resulting profile becomes the evidence used by the AI planner.

---

# 🤖 AI Cleaning Planner

Gemini receives the structured dataset profile and determines:

1. What problems exist
2. Which cleaning operations are necessary
3. Which columns should be changed
4. What parameters should be used
5. Why each operation is appropriate
6. How confident the model is

Example AI-generated action:

```json
{
  "action": "impute_missing",
  "column": "age_years",
  "params": {
    "method": "median"
  },
  "reasoning": "The column contains missing numerical values.",
  "confidence": 0.95
}
```

The model is explicitly instructed to avoid unnecessary transformations.

---

# 🛡️ Safety Architecture

One of the core design decisions is separating **AI decision-making** from **code execution**.

### ❌ The AI does NOT do this:

```text
Gemini
   ↓
Generate Python
   ↓
Execute Python
```

### ✅ Instead:

```text
Gemini
   ↓
Select supported operation
   ↓
Pydantic validation
   ↓
Trusted Action Registry
   ↓
Deterministic Python function
   ↓
Dataset
```

This prevents the model from executing arbitrary code or SQL against the dataset.

---

# 📊 Example

The included customer dataset intentionally contains multiple data-quality problems.

### Problems

```text
205 rows
7 columns

Missing Age:        31
Missing Income:     20
Duplicate rows:      5
Mixed category case
Dates stored as text
Inconsistent column names
```

### AI-generated plan

Gemini selected only the operations supported by the profile:

```text
1. normalize_column_names
2. convert_to_date
3. standardize_categories
4. impute_missing → age_years
5. impute_missing → income
6. remove_duplicates
```

### Result

```text
                    BEFORE          AFTER

Rows                  205             200
Age missing            31               0
Income missing         20               0
Duplicates              5               0

Registration_Date
str                         →     datetime64

Validation
Overall                     →     TRUE
```

This demonstrates that the AI is selecting **targeted operations** instead of applying every available transformation.

---

# 📁 Project Structure

```text
ai-data-cleaner/
│
├── 📂 data/
│   └── test_customers.csv
│
├── 📂 runs/
│   └── <run_id>/
│       ├── profile.json
│       ├── plan_raw.txt
│       ├── plan.json
│       ├── cleaned_data.csv
│       └── cleaning_report.json
│
├── actions.py
├── profiler.py
├── planner.py
├── schemas.py
├── generate_test_data.py
│
├── test_actions.py
├── test_profiler.py
├── test_schemas.py
│
├── .env
├── .gitignore
└── README.md
```

---

# ⚙️ Tech Stack

| Technology          | Purpose                     |
| ------------------- | --------------------------- |
| 🐍 **Python**       | Core application            |
| 🐼 **Pandas**       | Data processing             |
| ✨ **Google Gemini** | AI cleaning-plan generation |
| 🧩 **Pydantic**     | Structured plan validation  |
| 🧪 **Pytest**       | Automated testing           |
| 🔧 **Git / GitHub** | Version control             |

---

# 🛠️ Installation

## 1. Clone the repository

```bash
git clone https://github.com/Ok-Sanjeev/ai-data-cleaner.git
cd ai-data-cleaner
```

## 2. Create a virtual environment

```bash
python -m venv venv
```

### Windows

```powershell
venv\Scripts\Activate.ps1
```

## 3. Install dependencies

```bash
pip install pandas pydantic google-genai python-dotenv pytest
```

## 4. Configure Gemini

Create a `.env` file in the project root:

```text
GEMINI_API_KEY=your_api_key_here
```

---

# ▶️ Usage

Run the cleaner with a CSV file:

```bash
python planner.py data/test_customers.csv
```

Each run creates its own directory:

```text
runs/
└── run_<timestamp>/
    ├── profile.json
    ├── plan_raw.txt
    ├── plan.json
    ├── cleaned_data.csv
    └── cleaning_report.json
```

### Generated Artifacts

| File                   | Purpose                        |
| ---------------------- | ------------------------------ |
| `profile.json`         | Dataset quality profile        |
| `plan_raw.txt`         | Raw Gemini response            |
| `plan.json`            | Validated cleaning plan        |
| `cleaned_data.csv`     | Final cleaned dataset          |
| `cleaning_report.json` | Cleaning and validation report |

---

# 🧪 Testing

Run the complete test suite:

```bash
python -m pytest -q
```

Current result:

```text
19 passed
```

Tests currently cover:

* Cleaning operations
* Invalid operations
* Parameter validation
* Missing-value handling
* Data-type conversions
* Dataset profiling
* CSV structure validation
* Pydantic cleaning plans

---

# 🏗️ Design Principles

## 1. AI for Decisions, Code for Execution

Gemini determines **what should happen**.

Python determines **how it happens**.

This separation provides a safer and more predictable architecture.

---

## 2. Evidence-Based Cleaning

The system should only recommend transformations supported by evidence in the dataset profile.

A transformation should not be performed simply because it is available.

---

## 3. Minimal Transformation

The goal is to improve data quality while preserving as much original information as possible.

```text
Detect problem
     ↓
Choose smallest appropriate operation
     ↓
Execute deterministically
     ↓
Validate result
```

---

## 4. Reproducible Execution

Although the cleaning plan is AI-generated, the actual transformations are performed by deterministic functions.

This makes the execution layer:

* Testable
* Debuggable
* Reproducible
* Auditable

---

# 📈 Current Status

### Core Pipeline: ✅ Functional

Currently implemented:

* ✅ Universal CSV profiling
* ✅ AI-based cleaning-plan generation
* ✅ 12 cleaning operations
* ✅ Pydantic plan validation
* ✅ Trusted action registry
* ✅ Deterministic execution
* ✅ Post-cleaning validation
* ✅ JSON reporting
* ✅ Run artifacts
* ✅ Automated tests
* ✅ 19/19 tests passing

The next development phase focuses on broader testing against unfamiliar and difficult datasets, stronger safety validation, and portfolio-facing improvements.

---

# 🔮 Future Improvements

* [ ] Test against a wider range of unfamiliar datasets
* [ ] Stronger safety and action validation
* [ ] More advanced data-quality scoring
* [ ] Improved post-cleaning verification
* [ ] Better handling of ambiguous data
* [ ] Support for additional file formats
* [ ] Interactive web interface
* [ ] Cleaning-plan visualization
* [ ] Data-quality dashboard
* [ ] Explainable before/after comparisons

---

# 👨‍💻 Author

## Sanjeev Kumar Rai

**AI / Data / Software Engineering**

GitHub: [https://github.com/Ok-Sanjeev](https://github.com/Ok-Sanjeev)

---

⭐ If you find this project interesting, consider giving it a star!

```
```

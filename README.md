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

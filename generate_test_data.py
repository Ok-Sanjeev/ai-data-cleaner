import random
from datetime import datetime, timedelta

import pandas as pd


random.seed(42)


first_names = [
    "Aarav", "Neha", "Rohan", "Priya", "Kunal",
    "Ananya", "Vivek", "Simran", "Aditya", "Meera",
    "Riya", "Arjun", "Rahul", "Ishita", "Nikhil",
    "Pooja", "Varun", "Sneha", "Kabir", "Aditi"
]

last_names = [
    "Sharma", "Verma", "Singh", "Mehta", "Gupta",
    "Rao", "Kumar", "Kaur", "Jain", "Shah",
    "Kapoor", "Malhotra", "Patel", "Das", "Mishra"
]

cities = [
    "Lucknow",
    "Delhi",
    "Mumbai",
    "Pune",
    "Bengaluru",
    "Jaipur"
]

membership_variants = [
    "Active",
    "active",
    "ACTIVE",
    "Inactive",
    "inactive",
    "INACTIVE"
]


rows = []

start_date = datetime(2022, 1, 1)


# Create 200 customer records
for i in range(1, 201):
    first = random.choice(first_names)
    last = random.choice(last_names)

    age = random.randint(21, 60)
    income = random.randint(30000, 120000)

    registration_date = start_date + timedelta(
        days=random.randint(0, 1400)
    )

    rows.append({
        "Customer_ID": f"C{i:04d}",
        "Customer_Name": f"{first} {last}",
        "Age_Years": age,
        "Membership_Status": random.choice(membership_variants),
        "Registration_Date": registration_date,
        "Income": income,
        "City": random.choice(cities)
    })


df = pd.DataFrame(rows)


# Introduce missing Age values
missing_age = random.sample(
    list(df.index),
    30
)

df.loc[missing_age, "Age_Years"] = None


# Introduce missing Income values
missing_income = random.sample(
    list(df.index),
    20
)

df.loc[missing_income, "Income"] = None


# Introduce multiple date formats
date_formats = [
    "%Y-%m-%d",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%Y/%m/%d"
]

df["Registration_Date"] = [
    date.strftime(random.choice(date_formats))
    for date in df["Registration_Date"]
]


# Add 5 duplicate rows
duplicates = df.sample(
    5,
    random_state=42
)

df = pd.concat(
    [df, duplicates],
    ignore_index=True
)


# Shuffle the final dataset
df = df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# Save the dataset
output_path = "data/test_customers.csv"

df.to_csv(
    output_path,
    index=False
)


# Display a summary
print("Dataset created successfully.")
print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\nMissing values:")
print(df.isna().sum())

print("\nDuplicate rows:", df.duplicated().sum())

print("\nFirst 5 rows:")
print(df.head())

print(f"\nSaved to: {output_path}")
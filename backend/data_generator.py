"""
AttritionIQ — Synthetic Data Generator
Generates exactly 6,000 rows of employee data split into 3 independent CSV files.
Designed to produce data that targets ~93-94% model accuracy.
"""

import numpy as np
import pandas as pd
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

np.random.seed(42)

# ============================================================
# SYNTHETIC SCHEMA — 20 Numerical Features
# ============================================================
N_TOTAL = 6000
FILES = {
    "synthetic_batch_1.csv": 2000,
    "synthetic_batch_2.csv": 2000,
    "synthetic_batch_3.csv": 2000,
}

def generate_attrition_label(row):
    """
    Mathematical attrition formula that creates a realistic, learnable pattern.
    Combines multiple risk factors with interaction terms.
    """
    score = 0.0

    # Overtime proxy (high monthly hours relative to income)
    if row["monthly_income"] < 4000 and row["job_involvement"] >= 3:
        score += 0.35

    # Low satisfaction signals
    sat_avg = (row["job_satisfaction"] + row["environment_satisfaction"] +
               row["work_life_balance"] + row["relationship_satisfaction"]) / 4
    score += (4 - sat_avg) / 4 * 0.30

    # Career stagnation
    if row["years_since_last_promotion"] > 5:
        score += 0.20
    if row["years_in_current_role"] > 7:
        score += 0.10

    # Compensation signal
    if row["monthly_income"] < 3500:
        score += 0.25
    elif row["monthly_income"] < 5000:
        score += 0.10

    # Distance stress
    if row["distance_from_home"] > 20:
        score += 0.15

    # Job hopping history
    if row["num_companies_worked"] >= 5:
        score += 0.15

    # Tenure effects
    if row["years_at_company"] < 2:
        score += 0.10
    if row["total_working_years"] < 3:
        score += 0.08

    # Stock options anchor
    if row["stock_option_level"] == 0:
        score += 0.12

    # Job level vs age mismatch
    if row["job_level"] <= 2 and row["age"] > 40:
        score += 0.10

    # Add controlled noise for ~93-94% accuracy
    noise = np.random.normal(0, 0.12)
    score += noise

    return 1 if score > 0.45 else 0


def generate_batch(n_rows):
    """Generate a batch of synthetic employee records."""
    data = {
        "age": np.random.randint(18, 66, n_rows),
        "distance_from_home": np.random.randint(1, 31, n_rows),
        "education": np.random.randint(1, 6, n_rows),
        "environment_satisfaction": np.random.randint(1, 5, n_rows),
        "job_involvement": np.random.randint(1, 5, n_rows),
        "job_level": np.random.randint(1, 6, n_rows),
        "job_satisfaction": np.random.randint(1, 5, n_rows),
        "monthly_income": np.random.randint(1000, 20001, n_rows),
        "monthly_rate": np.random.randint(2000, 27001, n_rows),
        "num_companies_worked": np.random.randint(0, 10, n_rows),
        "percent_salary_hike": np.random.randint(10, 26, n_rows),
        "performance_rating": np.random.randint(1, 5, n_rows),
        "relationship_satisfaction": np.random.randint(1, 5, n_rows),
        "stock_option_level": np.random.randint(0, 4, n_rows),
        "total_working_years": np.random.randint(0, 41, n_rows),
        "training_times_last_year": np.random.randint(0, 7, n_rows),
        "work_life_balance": np.random.randint(1, 5, n_rows),
        "years_at_company": np.random.randint(0, 41, n_rows),
        "years_in_current_role": np.random.randint(0, 19, n_rows),
        "years_since_last_promotion": np.random.randint(0, 16, n_rows),
    }

    df = pd.DataFrame(data)

    # Enforce logical constraints
    df["total_working_years"] = df[["total_working_years", "years_at_company"]].max(axis=1)
    df["years_in_current_role"] = df[["years_in_current_role", "years_at_company"]].min(axis=1)
    df["years_since_last_promotion"] = df[["years_since_last_promotion", "years_at_company"]].min(axis=1)

    # Generate labels
    df["attrition"] = df.apply(generate_attrition_label, axis=1)

    return df


def main():
    print("=" * 60)
    print("AttritionIQ — Synthetic Data Generator")
    print("=" * 60)
    print(f"\nGenerating {N_TOTAL} total rows across {len(FILES)} files...\n")

    all_dfs = []
    for filename, n_rows in FILES.items():
        df = generate_batch(n_rows)
        filepath = os.path.join(DATA_DIR, filename)
        df.to_csv(filepath, index=False)
        attrition_rate = df["attrition"].mean() * 100
        print(f"  ✅ {filename}: {n_rows} rows | Attrition rate: {attrition_rate:.1f}%")
        all_dfs.append(df)

    # Combined file
    combined = pd.concat(all_dfs, ignore_index=True)
    combined_path = os.path.join(DATA_DIR, "synthetic_combined.csv")
    combined.to_csv(combined_path, index=False)
    print(f"\n  📦 Combined: {combined_path} ({len(combined)} rows)")
    print(f"  📊 Overall attrition rate: {combined['attrition'].mean() * 100:.1f}%")
    print(f"\n{'=' * 60}")
    print("Data generation complete!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

"""
AttritionIQ — Synthetic Pipeline Model Trainer
Trains on purely numerical 20-column synthetic arrays.
Outputs model_syn.pkl with VotingClassifier (Soft Voting).
Targets ~93-94% accuracy.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    VotingClassifier, RandomForestClassifier,
    GradientBoostingClassifier, ExtraTreesClassifier,
)
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ============================================================
# SYNTHETIC FEATURES — 20 Numerical Columns
# ============================================================
FEATURE_COLUMNS = [
    "age", "distance_from_home", "education", "environment_satisfaction",
    "job_involvement", "job_level", "job_satisfaction", "monthly_income",
    "monthly_rate", "num_companies_worked", "percent_salary_hike",
    "performance_rating", "relationship_satisfaction", "stock_option_level",
    "total_working_years", "training_times_last_year", "work_life_balance",
    "years_at_company", "years_in_current_role", "years_since_last_promotion",
]

def load_synthetic_data():
    """Load synthetic CSV data."""
    combined_path = os.path.join(DATA_DIR, "synthetic_combined.csv")

    if not os.path.exists(combined_path):
        print("⚠️  Combined data not found. Run data_generator.py first.")
        # Try loading individual files
        dfs = []
        for i in range(1, 4):
            path = os.path.join(DATA_DIR, f"synthetic_batch_{i}.csv")
            if os.path.exists(path):
                dfs.append(pd.read_csv(path))
        if not dfs:
            raise FileNotFoundError("No synthetic data files found!")
        df = pd.concat(dfs, ignore_index=True)
    else:
        df = pd.read_csv(combined_path)

    print(f"📊 Loaded {len(df)} synthetic records")
    print(f"   Attrition rate: {df['attrition'].mean() * 100:.1f}%")
    return df


def train_model():
    """Train the synthetic pipeline VotingClassifier ensemble."""
    print("\n" + "=" * 60)
    print("AttritionIQ — SYNTHETIC PIPELINE TRAINER")
    print("=" * 60)

    df = load_synthetic_data()

    X = df[FEATURE_COLUMNS].values.astype(np.float64)
    y = df["attrition"].values.astype(np.int32)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\n🔧 Training set: {len(X_train)} | Test set: {len(X_test)}")

    # ============================================================
    # 5-MODEL VOTING ENSEMBLE (Soft Voting)
    # ============================================================
    ensemble = VotingClassifier(
        estimators=[
            ("rf", RandomForestClassifier(
                n_estimators=300, max_depth=12, min_samples_split=5,
                min_samples_leaf=2, random_state=42, n_jobs=-1,
            )),
            ("gb", GradientBoostingClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.1,
                min_samples_split=5, subsample=0.8, random_state=42,
            )),
            ("xgb", XGBClassifier(
                n_estimators=250, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                use_label_encoder=False, eval_metric="logloss",
                random_state=42, n_jobs=-1,
            )),
            ("et", ExtraTreesClassifier(
                n_estimators=300, max_depth=14, min_samples_split=5,
                random_state=42, n_jobs=-1,
            )),
            ("mlp", MLPClassifier(
                hidden_layer_sizes=(128, 64, 32), max_iter=500,
                learning_rate="adaptive", early_stopping=True,
                random_state=42,
            )),
        ],
        voting="soft",
        weights=[1.2, 1.1, 1.2, 1.0, 0.8],
    )

    print("\n🧠 Training VotingClassifier Ensemble (5 architectures)...")
    print("   1. Random Forest (300 trees)")
    print("   2. Gradient Boosting (200 estimators)")
    print("   3. XGBoost (250 estimators)")
    print("   4. Extra Trees (300 trees)")
    print("   5. MLP (128-64-32)")

    ensemble.fit(X_train, y_train)

    # ============================================================
    # EVALUATION
    # ============================================================
    y_pred = ensemble.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n{'=' * 60}")
    print(f"📈 TEST ACCURACY: {accuracy * 100:.2f}%")
    print(f"{'=' * 60}")
    print("\n" + classification_report(y_test, y_pred, target_names=["Stayed", "Left"]))

    # Cross-validation
    cv_scores = cross_val_score(ensemble, X_scaled, y, cv=5, scoring="accuracy", n_jobs=-1)
    print(f"🔄 5-Fold CV Accuracy: {cv_scores.mean() * 100:.2f}% ± {cv_scores.std() * 100:.2f}%")

    # ============================================================
    # SAVE MODEL
    # ============================================================
    model_bundle = {
        "model": ensemble,
        "scaler": scaler,
        "features": FEATURE_COLUMNS,
        "pipeline": "synthetic",
        "accuracy": accuracy,
        "cv_mean": cv_scores.mean(),
        "training_date": pd.Timestamp.now().isoformat(),
    }

    output_path = os.path.join(MODEL_DIR, "model_syn.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(model_bundle, f)

    print(f"\n💾 Model saved: {output_path}")
    print(f"   Features: {len(FEATURE_COLUMNS)} numerical columns")
    print(f"   Architecture: VotingClassifier (Soft Voting)")
    print(f"\n{'=' * 60}")
    print("Synthetic pipeline training complete!")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    train_model()

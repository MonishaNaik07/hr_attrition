"""
AttritionIQ — IBM HR Pipeline Model Trainer
Trains on the genuine IBM HR dataset (1,470 rows).
Uses ColumnTransformer (StandardScaler + OneHotEncoder) for complex string logic.
Outputs model_ibm.pkl with VotingClassifier (Soft Voting).
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
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ============================================================
# IBM HR SCHEMA DEFINITION
# ============================================================
NUMERIC_FEATURES = [
    "Age", "DailyRate", "DistanceFromHome", "Education",
    "EnvironmentSatisfaction", "HourlyRate", "JobInvolvement",
    "JobLevel", "JobSatisfaction", "MonthlyIncome", "MonthlyRate",
    "NumCompaniesWorked", "PercentSalaryHike", "PerformanceRating",
    "RelationshipSatisfaction", "StockOptionLevel", "TotalWorkingYears",
    "TrainingTimesLastYear", "WorkLifeBalance", "YearsAtCompany",
    "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager",
]

CATEGORICAL_FEATURES = [
    "BusinessTravel", "Department", "EducationField", "Gender",
    "JobRole", "MaritalStatus", "OverTime",
]

# Columns to drop (constant/irrelevant)
DROP_COLUMNS = [
    "Attrition", "EmployeeCount", "EmployeeNumber",
    "Over18", "StandardHours",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_ibm_data():
    """Load the genuine IBM HR dataset."""
    ibm_path = os.path.join(DATA_DIR, "WA_Fn-UseC_-HR-Employee-Attrition.csv")

    if not os.path.exists(ibm_path):
        print("⚠️  IBM HR dataset not found at:")
        print(f"   {ibm_path}")
        print("\n📥 Download instructions:")
        print("   1. Visit: https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset")
        print("   2. Download WA_Fn-UseC_-HR-Employee-Attrition.csv")
        print(f"   3. Place it in: {DATA_DIR}/")
        raise FileNotFoundError("IBM HR dataset not found!")

    df = pd.read_csv(ibm_path)
    print(f"📊 Loaded IBM HR dataset: {len(df)} records")
    print(f"   Attrition rate: {(df['Attrition'] == 'Yes').mean() * 100:.1f}%")
    return df


def train_model():
    """Train the IBM pipeline VotingClassifier ensemble."""
    print("\n" + "=" * 60)
    print("AttritionIQ — IBM HR PIPELINE TRAINER")
    print("=" * 60)

    df = load_ibm_data()

    # Encode target
    y = (df["Attrition"] == "Yes").astype(int).values

    # Prepare features
    feature_df = df[ALL_FEATURES].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        feature_df, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\n🔧 Training set: {len(X_train)} | Test set: {len(X_test)}")
    print(f"   Numeric features: {len(NUMERIC_FEATURES)}")
    print(f"   Categorical features: {len(CATEGORICAL_FEATURES)}")

    # ============================================================
    # COLUMN TRANSFORMER — StandardScaler + OneHotEncoder
    # ============================================================
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    n_features_out = X_train_processed.shape[1]
    print(f"   Transformed features: {n_features_out} (after encoding)")

    # ============================================================
    # 5-MODEL VOTING ENSEMBLE (Soft Voting)
    # ============================================================
    ensemble = VotingClassifier(
        estimators=[
            ("rf", RandomForestClassifier(
                n_estimators=500, max_depth=10, min_samples_split=4,
                min_samples_leaf=2, class_weight="balanced",
                random_state=42, n_jobs=-1,
            )),
            ("gb", GradientBoostingClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.08,
                min_samples_split=4, subsample=0.85,
                random_state=42,
            )),
            ("xgb", XGBClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.08,
                subsample=0.85, colsample_bytree=0.8,
                scale_pos_weight=3.0,  # Handle class imbalance
                use_label_encoder=False, eval_metric="logloss",
                random_state=42, n_jobs=-1,
            )),
            ("et", ExtraTreesClassifier(
                n_estimators=500, max_depth=12, min_samples_split=4,
                class_weight="balanced", random_state=42, n_jobs=-1,
            )),
            ("mlp", MLPClassifier(
                hidden_layer_sizes=(128, 64, 32), max_iter=600,
                learning_rate="adaptive", early_stopping=True,
                validation_fraction=0.1, random_state=42,
            )),
        ],
        voting="soft",
        weights=[1.2, 1.2, 1.3, 1.0, 0.8],
    )

    print("\n🧠 Training VotingClassifier Ensemble (5 architectures)...")
    print("   1. Random Forest (500 trees, balanced)")
    print("   2. Gradient Boosting (300 estimators)")
    print("   3. XGBoost (300 estimators, scale_pos_weight=3)")
    print("   4. Extra Trees (500 trees, balanced)")
    print("   5. MLP (128-64-32, early stopping)")

    ensemble.fit(X_train_processed, y_train)

    # ============================================================
    # EVALUATION
    # ============================================================
    y_pred = ensemble.predict(X_test_processed)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n{'=' * 60}")
    print(f"📈 TEST ACCURACY: {accuracy * 100:.2f}%")
    print(f"{'=' * 60}")
    print("\n" + classification_report(y_test, y_pred, target_names=["Stayed", "Left"]))

    # Cross-validation on full dataset
    X_full_processed = preprocessor.transform(feature_df)
    cv_scores = cross_val_score(ensemble, X_full_processed, y, cv=5, scoring="accuracy", n_jobs=-1)
    print(f"🔄 5-Fold CV Accuracy: {cv_scores.mean() * 100:.2f}% ± {cv_scores.std() * 100:.2f}%")

    # ============================================================
    # SAVE MODEL (separate from model_syn.pkl)
    # ============================================================
    model_bundle = {
        "model": ensemble,
        "preprocessor": preprocessor,
        "features": ALL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "pipeline": "ibm",
        "accuracy": accuracy,
        "cv_mean": cv_scores.mean(),
        "training_date": pd.Timestamp.now().isoformat(),
    }

    output_path = os.path.join(MODEL_DIR, "model_ibm.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(model_bundle, f)

    print(f"\n💾 Model saved: {output_path}")
    print(f"   Features: {len(NUMERIC_FEATURES)} numeric + {len(CATEGORICAL_FEATURES)} categorical")
    print(f"   Architecture: VotingClassifier (Soft Voting)")
    print(f"   Preprocessor: ColumnTransformer (StandardScaler + OneHotEncoder)")
    print(f"\n{'=' * 60}")
    print("IBM pipeline training complete!")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    train_model()

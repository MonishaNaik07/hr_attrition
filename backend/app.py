"""
AttritionIQ — Enterprise Employee Attrition AI Platform
Flask Backend: Dual-Brain REST API Server
"""

import os
import pickle
import uuid
import csv
import io
from datetime import datetime
from functools import wraps
from flask import Flask, request, jsonify, session
from flask_cors import CORS
import pandas as pd
import numpy as np
from werkzeug.security import generate_password_hash, check_password_hash
from flask import send_from_directory
import os

app = Flask(__name__, static_folder="static", static_url_path="")
app.secret_key = os.environ.get("SECRET_KEY", "attrition_iq_super_secret_key_2024")
CORS(app, supports_credentials=True)

# ============================================================
# IN-MEMORY DATABASES
# ============================================================
users_db = {}          # { username_lowercase: { password_hash, created_at } }
records_db = []        # Persistent prediction records
batches_db = []        # Batch processing summaries

# ============================================================
# DUAL-BRAIN MODEL LOADER
# ============================================================
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
models = {}

def load_models():
    """Load both AI brains into active runtime dictionary."""
    global models

    ibm_path = os.path.join(MODEL_DIR, "model_ibm.pkl")
    syn_path = os.path.join(MODEL_DIR, "model_syn.pkl")

    if os.path.exists(ibm_path):
        with open(ibm_path, "rb") as f:
            models["ibm"] = pickle.load(f)
        print("✅ IBM brain loaded: model_ibm.pkl")
    else:
        print("⚠️  model_ibm.pkl not found — IBM pipeline unavailable")

    if os.path.exists(syn_path):
        with open(syn_path, "rb") as f:
            models["synthetic"] = pickle.load(f)
        print("✅ Synthetic brain loaded: model_syn.pkl")
    else:
        print("⚠️  model_syn.pkl not found — Synthetic pipeline unavailable")

    print(f"🧠 Dual-Brain Active | Models: {list(models.keys())}")

load_models()

# ============================================================
# FLIGHT RISK RULE ENGINE — EXPLAINABILITY
# ============================================================

def compute_flight_risk(features: dict, pipeline: str):
    """
    Maps prediction to a custom Flight Risk rule-engine that reads
    individual inputs and outputs:
      1. Why they are at risk
      2. Actionable HR retention solutions
    """
    risk_factors = []
    retention_actions = []

    def get(keys, default=0):
        for k in keys:
            if k in features:
                try:
                    return float(features[k])
                except (ValueError, TypeError):
                    return features[k]
        return default

    def get_str(keys, default=""):
        for k in keys:
            if k in features:
                return str(features[k])
        return default

    # --- Overtime ---
    overtime = get_str(["OverTime", "overtime"])
    if overtime == "Yes":
        risk_factors.append("⚡ Employee is working overtime — strong burnout indicator")
        retention_actions.extend([
            "🕐 Implement workload redistribution — reduce overtime immediately",
            "💰 Introduce overtime compensation or time-off-in-lieu policy",
        ])

    # --- Job Satisfaction ---
    js = get(["JobSatisfaction", "job_satisfaction"])
    if js <= 2:
        risk_factors.append("📉 Low job satisfaction score — disengagement risk")
        retention_actions.extend([
            "🎯 Launch targeted job enrichment program",
            "🗣️ Schedule 1-on-1 career development discussion",
        ])

    # --- Environment Satisfaction ---
    es = get(["EnvironmentSatisfaction", "environment_satisfaction"])
    if es <= 2:
        risk_factors.append("🏢 Poor work environment satisfaction — cultural misalignment")
        retention_actions.extend([
            "🏢 Conduct workplace environment assessment",
            "☕ Upgrade amenities and collaboration areas",
        ])

    # --- Work-Life Balance ---
    wlb = get(["WorkLifeBalance", "work_life_balance"])
    if wlb <= 2:
        risk_factors.append("⚖️ Poor work-life balance — high stress indicator")
        retention_actions.extend([
            "🏠 Offer hybrid/remote work arrangements",
            "📅 Implement flexible scheduling",
        ])

    # --- Monthly Income ---
    mi = get(["MonthlyIncome", "monthly_income"])
    if mi < 3500:
        risk_factors.append("💰 Below-market compensation — competitive attrition risk")
        retention_actions.extend([
            "💵 Conduct immediate compensation benchmarking",
            "📊 Propose salary adjustment or bonus structure",
        ])

    # --- Distance From Home ---
    dfh = get(["DistanceFromHome", "distance_from_home"])
    if dfh > 15:
        risk_factors.append("🚗 Long commute distance — commute fatigue detected")
        retention_actions.extend([
            "🏠 Evaluate remote work eligibility",
            "🚌 Offer transportation subsidy or relocation assistance",
        ])

    # --- Years Since Last Promotion ---
    yslp = get(["YearsSinceLastPromotion", "years_since_last_promotion"])
    if yslp > 5:
        risk_factors.append("📈 Stagnant career progression — no recent promotion")
        retention_actions.extend([
            "🚀 Create accelerated career development roadmap",
            "🎓 Fund certification or leadership training",
        ])

    # --- Business Travel ---
    bt = get_str(["BusinessTravel", "business_travel"])
    if bt == "Travel_Frequently":
        risk_factors.append("✈️ Frequent business travel — elevated stress & burnout")
        retention_actions.extend([
            "✈️ Reduce travel frequency — shift to virtual meetings",
            "🏨 Upgrade travel accommodations and perks",
        ])

    # --- Stock Options ---
    so = get(["StockOptionLevel", "stock_option_level"])
    if so == 0:
        risk_factors.append("📊 No stock options — weak financial retention anchor")
        retention_actions.append("📈 Introduce equity/stock option grant")

    # --- Num Companies Worked ---
    ncw = get(["NumCompaniesWorked", "num_companies_worked"])
    if ncw >= 5:
        risk_factors.append("🔄 High job-hopping history — pattern of frequent transitions")

    # --- Job Involvement ---
    ji = get(["JobInvolvement", "job_involvement"])
    if ji <= 2:
        risk_factors.append("🎯 Low job involvement — role disconnection")

    if not risk_factors:
        risk_factors.append("✅ No major risk factors detected — employee appears stable")
        retention_actions.append("🌟 Continue current engagement strategy")

    # Deduplicate actions
    retention_actions = list(dict.fromkeys(retention_actions))[:8]

    if len(retention_actions) == 0:
        retention_actions.append("📋 Schedule retention review meeting within 30 days")

    return risk_factors, retention_actions


def predict_single(features: dict, pipeline: str):
    """Run prediction through the appropriate AI brain."""
    model_bundle = models.get(pipeline)
    if model_bundle is None:
        return {"error": f"Pipeline '{pipeline}' model not loaded."}, 400

    clf = model_bundle["model"]
    feature_names = model_bundle["features"]

    # Build feature vector
    row = []
    for fname in feature_names:
        val = features.get(fname, 0)
        try:
            row.append(float(val))
        except (ValueError, TypeError):
            row.append(0.0)

    X = np.array([row])

    if pipeline == "ibm" and "preprocessor" in model_bundle:
        try:
            X = model_bundle["preprocessor"].transform(
                pd.DataFrame([features], columns=feature_names)
            )
        except Exception:
            pass

    prob = clf.predict_proba(X)[0][1]
    risk_level = (
        "Critical" if prob >= 0.75 else
        "High" if prob >= 0.5 else
        "Medium" if prob >= 0.25 else
        "Low"
    )

    risk_factors, retention_actions = compute_flight_risk(features, pipeline)

    return {
        "attritionProbability": round(prob * 100, 1),
        "riskLevel": risk_level,
        "riskFactors": risk_factors,
        "retentionActions": retention_actions,
        "pipeline": pipeline,
    }, 200


# ============================================================
# AUTHENTICATION ROUTES
# ============================================================

@app.route("/api/register", methods=["POST"])
def register():
    data = request.get_json() or {}
    username = data.get("username", "").strip().lower()
    password = data.get("password", "")

    if not username or not password:
        return jsonify({"error": "Username and password are required."}), 400
    if len(password) < 4:
        return jsonify({"error": "Password must be at least 4 characters."}), 400
    if username in users_db:
        return jsonify({"error": "Username already exists."}), 409

    users_db[username] = {
        "password_hash": generate_password_hash(password),
        "created_at": datetime.utcnow().isoformat(),
    }
    return jsonify({"message": "Registration successful! You can now log in."}), 201


@app.route("/api/login", methods=["POST"])
def login():
    data = request.get_json() or {}
    username = data.get("username", "").strip().lower()
    password = data.get("password", "")

    if not username or not password:
        return jsonify({"error": "Username and password are required."}), 400

    user = users_db.get(username)
    if not user or not check_password_hash(user["password_hash"], password):
        return jsonify({"error": "Invalid username or password."}), 401

    session["user"] = username
    return jsonify({"message": "Login successful!", "username": username}), 200


@app.route("/api/logout", methods=["POST"])
def logout():
    session.pop("user", None)
    return jsonify({"message": "Logged out."}), 200


@app.route("/api/me", methods=["GET"])
def me():
    user = session.get("user")
    if not user:
        return jsonify({"error": "Not authenticated."}), 401
    return jsonify({"username": user}), 200


# ============================================================
# PREDICTION ROUTES
# ============================================================

@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json() or {}
    pipeline = data.get("pipeline", "ibm")
    features = data.get("features", {})

    if pipeline not in ("ibm", "synthetic"):
        return jsonify({"error": "Invalid pipeline. Use 'ibm' or 'synthetic'."}), 400

    result, status = predict_single(features, pipeline)
    if status != 200:
        return jsonify(result), status

    record = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.utcnow().isoformat(),
        "type": "individual",
        "pipeline": pipeline,
        "input": features,
        "result": result,
    }
    records_db.append(record)

    return jsonify(result), 200


# ============================================================
# SMART BATCH ANALYZER
# ============================================================

@app.route("/api/upload_csv", methods=["POST"])
def upload_csv():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    file = request.files["file"]
    if not file.filename.endswith(".csv"):
        return jsonify({"error": "Only CSV files accepted."}), 400

    try:
        df = pd.read_csv(file)
    except Exception as e:
        return jsonify({"error": f"CSV parse error: {str(e)}"}), 400

    columns = list(df.columns)

    # Auto-detect pipeline
    ibm_markers = {"OverTime", "BusinessTravel", "JobRole", "Department", "MaritalStatus"}
    synth_markers = {"age", "distance_from_home", "job_satisfaction", "work_life_balance"}

    ibm_match = len(set(c.lower() for c in columns) & {m.lower() for m in ibm_markers})
    synth_match = len(set(c.lower() for c in columns) & {m.lower() for m in synth_markers})

    pipeline = "ibm" if ibm_match >= synth_match else "synthetic"

    if pipeline not in models:
        return jsonify({"error": f"Detected pipeline '{pipeline}' but model not loaded."}), 400

    batch_id = str(uuid.uuid4())
    results = []
    risk_counts = {"Critical": 0, "High": 0, "Medium": 0, "Low": 0}
    total_prob = 0.0

    for _, row in df.iterrows():
        features = row.to_dict()
        # Convert NaN to 0/empty
        for k, v in features.items():
            if pd.isna(v):
                features[k] = 0

        result, status = predict_single(features, pipeline)
        if status == 200:
            total_prob += result["attritionProbability"]
            risk_counts[result["riskLevel"]] += 1
            results.append(result)

            records_db.append({
                "id": str(uuid.uuid4()),
                "timestamp": datetime.utcnow().isoformat(),
                "type": "batch",
                "pipeline": pipeline,
                "input": {k: str(v) for k, v in features.items()},
                "result": result,
                "batchId": batch_id,
            })

    total = len(results)
    batch_summary = {
        "id": batch_id,
        "timestamp": datetime.utcnow().isoformat(),
        "pipeline": pipeline,
        "totalRecords": total,
        "criticalCount": risk_counts["Critical"],
        "highRiskCount": risk_counts["High"],
        "mediumRiskCount": risk_counts["Medium"],
        "lowRiskCount": risk_counts["Low"],
        "avgAttritionProb": round(total_prob / total, 1) if total else 0,
        "fileName": file.filename,
    }
    batches_db.append(batch_summary)

    return jsonify({
        "batch": batch_summary,
        "results": results[:50],  # Preview first 50
        "message": f"Processed {total} records via {pipeline} pipeline.",
    }), 200


# ============================================================
# HISTORY & RECORDS
# ============================================================

@app.route("/api/history", methods=["GET"])
def get_history():
    limit = request.args.get("limit", 100, type=int)
    return jsonify(records_db[-limit:]), 200


@app.route("/api/batches", methods=["GET"])
def get_batches():
    return jsonify(batches_db), 200


@app.route("/api/stats", methods=["GET"])
def get_stats():
    total = len(records_db)
    high_risk = sum(
        1 for r in records_db
        if r.get("result", {}).get("riskLevel") in ("High", "Critical")
    )
    avg = 0
    if total > 0:
        avg = round(
            sum(r["result"]["attritionProbability"] for r in records_db) / total, 1
        )
    return jsonify({
        "totalPredictions": total,
        "highRisk": high_risk,
        "avgProb": avg,
        "totalBatches": len(batches_db),
        "totalRecords": sum(b["totalRecords"] for b in batches_db),
    }), 200


# ============================================================
# SCHEMA ENDPOINTS — For Dynamic Frontend Forms
# ============================================================

IBM_SCHEMA = [
    {"name": "Age", "label": "Age", "type": "number", "min": 18, "max": 60, "default": 35},
    {"name": "BusinessTravel", "label": "Business Travel", "type": "categorical",
     "options": ["Travel_Rarely", "Travel_Frequently", "Non-Travel"], "default": "Travel_Rarely"},
    {"name": "DailyRate", "label": "Daily Rate", "type": "number", "min": 100, "max": 1500, "default": 800},
    {"name": "Department", "label": "Department", "type": "categorical",
     "options": ["Research & Development", "Sales", "Human Resources"], "default": "Research & Development"},
    {"name": "DistanceFromHome", "label": "Distance From Home", "type": "number", "min": 1, "max": 29, "default": 7},
    {"name": "Education", "label": "Education (1-5)", "type": "number", "min": 1, "max": 5, "default": 3},
    {"name": "EducationField", "label": "Education Field", "type": "categorical",
     "options": ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"],
     "default": "Life Sciences"},
    {"name": "EnvironmentSatisfaction", "label": "Environment Satisfaction (1-4)", "type": "number", "min": 1, "max": 4, "default": 3},
    {"name": "Gender", "label": "Gender", "type": "categorical", "options": ["Male", "Female"], "default": "Male"},
    {"name": "HourlyRate", "label": "Hourly Rate", "type": "number", "min": 30, "max": 100, "default": 65},
    {"name": "JobInvolvement", "label": "Job Involvement (1-4)", "type": "number", "min": 1, "max": 4, "default": 3},
    {"name": "JobLevel", "label": "Job Level (1-5)", "type": "number", "min": 1, "max": 5, "default": 2},
    {"name": "JobRole", "label": "Job Role", "type": "categorical",
     "options": ["Sales Executive", "Research Scientist", "Laboratory Technician",
                  "Manufacturing Director", "Healthcare Representative", "Manager",
                  "Sales Representative", "Research Director", "Human Resources"],
     "default": "Research Scientist"},
    {"name": "JobSatisfaction", "label": "Job Satisfaction (1-4)", "type": "number", "min": 1, "max": 4, "default": 3},
    {"name": "MaritalStatus", "label": "Marital Status", "type": "categorical",
     "options": ["Single", "Married", "Divorced"], "default": "Married"},
    {"name": "MonthlyIncome", "label": "Monthly Income", "type": "number", "min": 1000, "max": 20000, "default": 6500},
    {"name": "MonthlyRate", "label": "Monthly Rate", "type": "number", "min": 2000, "max": 27000, "default": 14000},
    {"name": "NumCompaniesWorked", "label": "Num Companies Worked", "type": "number", "min": 0, "max": 9, "default": 2},
    {"name": "OverTime", "label": "OverTime", "type": "categorical", "options": ["Yes", "No"], "default": "No"},
    {"name": "PercentSalaryHike", "label": "Percent Salary Hike", "type": "number", "min": 11, "max": 25, "default": 15},
    {"name": "PerformanceRating", "label": "Performance Rating (1-4)", "type": "number", "min": 1, "max": 4, "default": 3},
    {"name": "RelationshipSatisfaction", "label": "Relationship Satisfaction (1-4)", "type": "number", "min": 1, "max": 4, "default": 3},
    {"name": "StockOptionLevel", "label": "Stock Option Level (0-3)", "type": "number", "min": 0, "max": 3, "default": 1},
    {"name": "TotalWorkingYears", "label": "Total Working Years", "type": "number", "min": 0, "max": 40, "default": 10},
    {"name": "TrainingTimesLastYear", "label": "Training Times Last Year", "type": "number", "min": 0, "max": 6, "default": 3},
    {"name": "WorkLifeBalance", "label": "Work-Life Balance (1-4)", "type": "number", "min": 1, "max": 4, "default": 3},
    {"name": "YearsAtCompany", "label": "Years At Company", "type": "number", "min": 0, "max": 40, "default": 5},
    {"name": "YearsInCurrentRole", "label": "Years In Current Role", "type": "number", "min": 0, "max": 18, "default": 3},
    {"name": "YearsSinceLastPromotion", "label": "Years Since Last Promotion", "type": "number", "min": 0, "max": 15, "default": 2},
    {"name": "YearsWithCurrManager", "label": "Years With Current Manager", "type": "number", "min": 0, "max": 17, "default": 3},
]

SYNTHETIC_SCHEMA = [
    {"name": "age", "label": "Age", "type": "number", "min": 18, "max": 65, "default": 35},
    {"name": "distance_from_home", "label": "Distance From Home (miles)", "type": "number", "min": 1, "max": 30, "default": 7},
    {"name": "education", "label": "Education Level (1-5)", "type": "number", "min": 1, "max": 5, "default": 3},
    {"name": "environment_satisfaction", "label": "Environment Satisfaction (1-4)", "type": "number", "min": 1, "max": 4, "default": 3},
    {"name": "job_involvement", "label": "Job Involvement (1-4)", "type": "number", "min": 1, "max": 4, "default": 3},
    {"name": "job_level", "label": "Job Level (1-5)", "type": "number", "min": 1, "max": 5, "default": 2},
    {"name": "job_satisfaction", "label": "Job Satisfaction (1-4)", "type": "number", "min": 1, "max": 4, "default": 3},
    {"name": "monthly_income", "label": "Monthly Income ($)", "type": "number", "min": 1000, "max": 20000, "default": 6500},
    {"name": "monthly_rate", "label": "Monthly Rate ($)", "type": "number", "min": 2000, "max": 27000, "default": 14000},
    {"name": "num_companies_worked", "label": "Num Companies Worked", "type": "number", "min": 0, "max": 9, "default": 2},
    {"name": "percent_salary_hike", "label": "Percent Salary Hike (%)", "type": "number", "min": 10, "max": 25, "default": 15},
    {"name": "performance_rating", "label": "Performance Rating (1-4)", "type": "number", "min": 1, "max": 4, "default": 3},
    {"name": "relationship_satisfaction", "label": "Relationship Satisfaction (1-4)", "type": "number", "min": 1, "max": 4, "default": 3},
    {"name": "stock_option_level", "label": "Stock Option Level (0-3)", "type": "number", "min": 0, "max": 3, "default": 1},
    {"name": "total_working_years", "label": "Total Working Years", "type": "number", "min": 0, "max": 40, "default": 10},
    {"name": "training_times_last_year", "label": "Training Times Last Year", "type": "number", "min": 0, "max": 6, "default": 3},
    {"name": "work_life_balance", "label": "Work-Life Balance (1-4)", "type": "number", "min": 1, "max": 4, "default": 3},
    {"name": "years_at_company", "label": "Years At Company", "type": "number", "min": 0, "max": 40, "default": 5},
    {"name": "years_in_current_role", "label": "Years In Current Role", "type": "number", "min": 0, "max": 18, "default": 3},
    {"name": "years_since_last_promotion", "label": "Years Since Last Promotion", "type": "number", "min": 0, "max": 15, "default": 2},
]


@app.route("/api/schema/<pipeline>", methods=["GET"])
def get_schema(pipeline):
    if pipeline == "ibm":
        return jsonify(IBM_SCHEMA), 200
    elif pipeline == "synthetic":
        return jsonify(SYNTHETIC_SCHEMA), 200
    return jsonify({"error": "Invalid pipeline."}), 400


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "models_loaded": list(models.keys()),
        "total_predictions": len(records_db),
        "total_batches": len(batches_db),
    }), 200

@app.route("/")
def serve():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/<path:path>")
def static_proxy(path):
    file_path = os.path.join(app.static_folder, path)

    if os.path.exists(file_path):
        return send_from_directory(app.static_folder, path)

    return send_from_directory(app.static_folder, "index.html")
    
@app.route("/")
def serve_frontend():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/<path:path>")
def serve_static(path):
    full_path = os.path.join(app.static_folder, path)

    if os.path.exists(full_path):
        return send_from_directory(app.static_folder, path)

    return send_from_directory(app.static_folder, "index.html")

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n🚀 AttritionIQ Server starting on port {port}")
    print(f"   Dual-Brain Models: {list(models.keys())}")
    print(f"   Endpoints: /api/login | /api/predict | /api/upload_csv\n")
    app.run(host="0.0.0.0", port=port, debug=True)

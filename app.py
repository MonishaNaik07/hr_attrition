# ─────────────────────────────────────────────
# AttritionIQ — FULLY FIXED app.py
# ─────────────────────────────────────────────

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generate_and_train import HybridEnsemble
import __main__; __main__.HybridEnsemble = HybridEnsemble

from flask import Flask, request, jsonify, render_template, session, redirect, Response
import pandas as pd
import numpy as np
import pickle, io
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__, template_folder='templates')
app.secret_key = os.environ.get("SECRET_KEY", "attriq-dev-secret-change-in-prod")
app.config['SESSION_COOKIE_SAMESITE'] = "Lax"
app.config['SESSION_COOKIE_SECURE'] = False

USERS = {
    "admin":      {"password": "admin123",  "role": "Administrator"},
    "hr_manager": {"password": "hr2024",    "role": "HR Manager"},
    "analyst":    {"password": "data2024",  "role": "Data Analyst"},
}

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "ensemble_model.pkl")
model_data = None

def load_model():
    global model_data
    with open(MODEL_PATH, "rb") as f:
        model_data = pickle.load(f)
    print(f"✅ Model loaded — {len(model_data['feature_cols'])} features")

load_model()

@app.route('/')
def index():
    return render_template('index copy.html')

@app.route('/dashboard')
def dashboard():
    if not session.get('user'):
        return redirect('/')
    return render_template('dashboard copy.html')

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    user = USERS.get(data.get('username'))
    if user and user['password'] == data.get('password'):
        session['user'] = data.get('username')
        session['role'] = user['role']
        return jsonify({"success": True, "role": user['role']})
    return jsonify({"success": False}), 401

@app.route('/api/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({"success": True})

@app.route('/api/model-stats')
def model_stats():
    fi = model_data.get('feature_importance', {})
    top_features = [
        {"name": k.replace("_", " ").title(), "importance": round(v * 100, 1)}
        for k, v in list(fi.items())[:10]
    ]
    return jsonify({
        "accuracy":     round(model_data['accuracy'] * 100, 1),
        "auc":          round(model_data.get('auc', 0.91), 3),
        "top_features": top_features,
    })

@app.route('/api/sample-data')
def sample_data():
    sample_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "ibm_hr_sample.csv")
    df = pd.read_csv(sample_path)
    if 'Attrition' in df.columns:
        df = df.drop(columns=['Attrition'])
    out = io.StringIO()
    df.head(50).to_csv(out, index=False)
    return Response(out.getvalue(), mimetype='text/csv',
                    headers={"Content-Disposition": "attachment; filename=sample_employees.csv"})

# ── Feature pipeline — EXACTLY matches generate_and_train.py ──────────

def encode_columns(df):
    df = df.copy()
    le = LabelEncoder()
    cat_cols = ['BusinessTravel','Department','EducationField','Gender','JobRole','MaritalStatus','Over18']
    for col in cat_cols:
        if col in df.columns:
            df[col+'_enc'] = le.fit_transform(df[col].astype(str))
        else:
            df[col+'_enc'] = 0
    df['OverTime_enc'] = (df['OverTime'].astype(str) == 'Yes').astype(int) if 'OverTime' in df.columns else 0
    return df

def engineer_features(df):
    df = df.copy()
    def col(name, default=0):
        return df[name].astype(float) if name in df.columns else pd.Series([float(default)]*len(df), index=df.index)

    ot  = df['OverTime_enc'].astype(float) if 'OverTime_enc' in df.columns else pd.Series([0.0]*len(df), index=df.index)
    mi  = col('MonthlyIncome', 5000)
    age = col('Age', 35)
    yac = col('YearsAtCompany', 5).replace(0, 0.5)
    ysp = col('YearsSinceLastPromotion', 0).replace(0, 0.5)
    ysm = col('YearsWithCurrManager', 0).replace(0, 0.5)
    ncw = col('NumCompaniesWorked', 0)
    js  = col('JobSatisfaction', 3)
    es  = col('EnvironmentSatisfaction', 3)
    wlb = col('WorkLifeBalance', 3)
    rs  = col('RelationshipSatisfaction', 3)
    so  = col('StockOptionLevel', 0)
    tty = col('TrainingTimesLastYear', 2)
    dfh = col('DistanceFromHome', 5)
    jl  = col('JobLevel', 2)
    ji  = col('JobInvolvement', 3)

    df['IncomePerYear']          = mi / yac
    df['Age_Tenure_ratio']       = age / yac
    df['YearsSincePromo_ratio']  = ysp / yac
    df['YearsWithManager_ratio'] = ysm / yac
    df['NumCompanies_Age']       = ncw / age.replace(0, 1)
    df['OverTime_JobSat']        = ot * js
    df['OverTime_WLB']           = ot * wlb
    df['OverTime_EnvSat']        = ot * es
    df['Low_Income_OverTime']    = (mi < mi.median()).astype(int) * ot
    df['StockOption_Income']     = so * mi / 10000
    df['Satisfaction_Score']     = (js + es + wlb + rs) / 4
    df['TotalSatisfaction']      = js + es + wlb + rs + ji
    df['Income_Age_ratio']       = mi / age.replace(0, 1)
    df['DistanceIncome_ratio']   = dfh / (mi / 1000).replace(0, 0.001)
    df['JobLevel_Income']        = jl * mi
    df['TrainingLastYear_sat']   = tty * js
    df['Age_sq']                 = age ** 2
    df['Income_sq']              = (mi / 1000) ** 2
    return df

def risk_label(p):
    if p >= 0.75: return "Critical"
    if p >= 0.50: return "High"
    if p >= 0.25: return "Medium"
    return "Low"

RECOMMENDATIONS = {
    "Critical": [
        {"priority":"High",   "action":"Immediate Retention Conversation",
         "detail":"Schedule a 1-on-1 with HR leadership within 48 hours. Co-create a personalised retention plan."},
        {"priority":"High",   "action":"Emergency Compensation Review",
         "detail":"Conduct immediate market benchmarking and adjust if below-market. Consider a spot bonus."},
        {"priority":"High",   "action":"Reduce Overtime Immediately",
         "detail":"Redistribute workload to bring overtime below 10%. Burnout is the #1 flight-risk driver."},
        {"priority":"Medium", "action":"Clear Promotion Roadmap",
         "detail":"Provide a written, time-bound promotion plan with measurable milestones within 2 weeks."},
    ],
    "High": [
        {"priority":"High",   "action":"Workload & Role Realignment",
         "detail":"Review role fit and project load. Consider a lateral move to a team that better suits career goals."},
        {"priority":"High",   "action":"Flexible Work Arrangement",
         "detail":"Offer remote/hybrid schedule to ease commute stress and improve work-life balance."},
        {"priority":"Medium", "action":"Manager Relationship Mediation",
         "detail":"Facilitate a structured feedback session between employee and direct manager."},
        {"priority":"Medium", "action":"Stock Option or Retention Bonus",
         "detail":"Award stock options or a 12-month retention bonus to create a financial incentive."},
    ],
    "Medium": [
        {"priority":"Medium", "action":"Engagement Program Enrollment",
         "detail":"Enrol in mentorship or employee resource group to strengthen belonging and career development."},
        {"priority":"Medium", "action":"Skill Development Sponsorship",
         "detail":"Sponsor a certification, course, or conference aligned with career aspirations."},
        {"priority":"Low",    "action":"Monthly Manager Check-ins",
         "detail":"Establish structured monthly 1-on-1s to proactively track satisfaction and growth."},
    ],
    "Low": [
        {"priority":"Low","action":"Recognition & Appreciation",
         "detail":"Acknowledge contributions in team meetings and through formal recognition programs."},
        {"priority":"Low","action":"Annual Career Growth Conversation",
         "detail":"Hold a yearly career development review to maintain long-term engagement."},
    ],
}

def build_explanation(row, prob, label):
    reasons, feature_scores = [], {}
    js  = int(row.get('JobSatisfaction',  3))
    es  = int(row.get('EnvironmentSatisfaction', 3))
    wlb = int(row.get('WorkLifeBalance', 3))
    rs  = int(row.get('RelationshipSatisfaction', 3))
    mi  = float(row.get('MonthlyIncome', 5000))
    yac = int(row.get('YearsAtCompany', 5))
    so  = int(row.get('StockOptionLevel', 1))
    dist= int(row.get('DistanceFromHome', 5))
    ncw = int(row.get('NumCompaniesWorked', 1))
    bt  = str(row.get('BusinessTravel', 'Non-Travel'))
    ms  = str(row.get('MaritalStatus', ''))
    ot  = str(row.get('OverTime', 'No'))

    if ot == 'Yes':
        reasons.append("Working overtime — top predictor of burnout and departure")
        feature_scores['OverTime'] = round(min(prob * 120, 95), 1)
    if js <= 2:
        reasons.append(f"Low job satisfaction ({js}/4)")
        feature_scores['Job Sat.'] = round((3-js)/3*90, 1)
    if es <= 2:
        reasons.append(f"Low environment satisfaction ({es}/4)")
        feature_scores['Env Sat.'] = round((3-es)/3*85, 1)
    if wlb <= 2:
        reasons.append(f"Poor work-life balance ({wlb}/4)")
        feature_scores['WLB'] = round((3-wlb)/3*80, 1)
    if rs <= 2:
        reasons.append(f"Low relationship satisfaction ({rs}/4)")
        feature_scores['Rel Sat.'] = round((3-rs)/3*70, 1)
    if mi < 3000:
        reasons.append(f"Monthly income (${int(mi):,}) is below retention threshold")
        feature_scores['Income'] = round(min((5000-mi)/5000*80, 80), 1)
    if so == 0:
        reasons.append("No stock options — low long-term financial commitment")
        feature_scores['Stock'] = 55.0
    if yac <= 1:
        reasons.append(f"Very short tenure ({yac}y) — highest flight risk window")
        feature_scores['Tenure'] = 70.0
    if dist >= 20:
        reasons.append(f"Long commute ({dist} km) impacts daily satisfaction")
        feature_scores['Commute'] = round(dist/29*65, 1)
    if bt == 'Travel_Frequently':
        reasons.append("Frequent travel creates sustained work-life strain")
        feature_scores['Travel'] = 60.0
    if ncw >= 5:
        reasons.append(f"High job mobility ({ncw} prior companies)")
        feature_scores['Mobility'] = round(min(ncw/9*65, 65), 1)
    if ms == 'Single':
        reasons.append("Single — statistically higher geographic mobility")
        feature_scores['Marital'] = 40.0

    if not reasons:
        reasons.append("No significant risk factors — employee profile is stable")

    feature_scores['Model Score'] = round(prob*100, 1)
    feature_scores = dict(sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)[:7])

    dept = row.get('Department','their department')
    role = row.get('JobRole','their role')
    ot_str = "working overtime" if ot == 'Yes' else "not working overtime"
    sat_avg = round((js+es+wlb+rs)/4, 1)
    if reasons[0].startswith("No significant"):
        explanation = (f"Employee in {dept} ({role}) has a {round(prob*100,1)}% predicted attrition probability "
                       f"({label} risk). Profile shows stable satisfaction ({sat_avg}/4), "
                       f"{yac} year(s) of tenure, and positive retention indicators.")
    else:
        explanation = (f"Employee in {dept} ({role}) has a {round(prob*100,1)}% predicted attrition probability "
                       f"({label} risk). They are {ot_str} with overall satisfaction {sat_avg}/4 "
                       f"and {yac} year(s) at company. Key drivers: {reasons[0].lower()}.")

    return {
        "explanation":     explanation,
        "reasons":         reasons[:5],
        "feature_scores":  feature_scores,
        "recommendations": RECOMMENDATIONS.get(label, RECOMMENDATIONS["Low"]),
    }

@app.route('/api/predict/upload', methods=['POST'])
def predict_upload():
    if not session.get('user'):
        return jsonify({"error": "Unauthorized"}), 401
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    try:
        df = pd.read_csv(request.files['file'])
        print(f"CSV loaded: {df.shape}")
    except Exception as e:
        return jsonify({"error": f"Cannot read CSV: {e}"}), 400

    try:
        df_feat = encode_columns(df)
        df_feat = engineer_features(df_feat)

        feature_cols = model_data['feature_cols']
        for c in feature_cols:
            if c not in df_feat.columns:
                df_feat[c] = 0

        X = df_feat[feature_cols].fillna(0)
        print(f"Prediction matrix: {X.shape}")

        probs  = model_data['ensemble'].predict_proba(X)[:, 1]
        labels = [risk_label(p) for p in probs]
        threshold = model_data.get('threshold', 0.65)
        preds  = ["At Risk" if p >= threshold else "Safe" for p in probs]

        risk_dist = {
            "Critical": int(sum(1 for l in labels if l=="Critical")),
            "High":     int(sum(1 for l in labels if l=="High")),
            "Medium":   int(sum(1 for l in labels if l=="Medium")),
            "Low":      int(sum(1 for l in labels if l=="Low")),
        }

        df_temp = df.copy()
        df_temp['_prob'] = probs
        df_temp['_pred'] = preds

# Age band analysis (replaces department chart)
        df_temp['_age_band'] = pd.cut(
            df['Age'],
            bins=[18, 25, 32, 40, 50, 100],
            labels=['18-25', '26-32', '33-40', '41-50', '51+']
        ).astype(str)
        dept_analysis = []
        for band, grp in df_temp.groupby('_age_band'):
            ar = int((grp['_pred'] == 'At Risk').sum())
            total = len(grp)
            dept_analysis.append({
                "department": band,
                "total": total,
                "at_risk": ar,
                "stay": total - ar,
                "risk_pct": round(ar / total * 100, 1)
            })
        dept_analysis.sort(key=lambda x: x['department'])

        # Job level analysis (replaces role chart)
        role_analysis = []
        for level, grp in df_temp.groupby('JobLevel'):
            ar = int((grp['_pred'] == 'At Risk').sum())
            total = len(grp)
            role_analysis.append({
                "role": f"Level {int(level)}",
                "total": total,
                "at_risk": ar,
                "risk_pct": round(ar / total * 100, 1)
            })
        role_analysis.sort(key=lambda x: x['risk_pct'], reverse=True)

        fi = model_data.get('feature_importance', {})
        top_features = [{"name":k.replace("_"," ").title(),"importance":round(v*100,1)}
                        for k,v in list(fi.items())[:10]]

        predictions = []
        for i in range(len(df)):
            prob = float(probs[i])

            if prob < 0.25:
                continue   # ❌ DO NOT SHOW LOW RISK

            row  = df.iloc[i]
            lbl  = labels[i]
            expl = build_explanation(row, prob, lbl)
            predictions.append({
                "employee_id":         int(row.get('EmployeeNumber', i+1)),
                "name":                f"Employee {int(row.get('EmployeeNumber', i+1))}",
                "department":          str(row.get('Department','—')),
                "job_role":            str(row.get('JobRole','—')),
                "age":                 int(row.get('Age', 0)),
                "monthly_income":      int(row.get('MonthlyIncome', 0)),
                "years_at_company":    int(row.get('YearsAtCompany', 0)),
                "overtime":            str(row.get('OverTime','No')),
                "gender":              str(row.get('Gender','—')),
                "marital_status":      str(row.get('MaritalStatus','—')),
                "education":           int(row.get('Education', 0)),
                "education_field":     str(row.get('EducationField','—')),
                "job_level":           int(row.get('JobLevel', 1)),
                "total_working_years": int(row.get('TotalWorkingYears', 0)),
                "years_since_promo":   int(row.get('YearsSinceLastPromotion', 0)),
                "years_with_manager":  int(row.get('YearsWithCurrManager', 0)),
                "business_travel":     str(row.get('BusinessTravel','—')),
                "stock_option":        int(row.get('StockOptionLevel', 0)),
                "percent_salary_hike": int(row.get('PercentSalaryHike', 0)),
                "performance_rating":  int(row.get('PerformanceRating', 3)),
                "training_times":      int(row.get('TrainingTimesLastYear', 0)),
                "job_satisfaction":    int(row.get('JobSatisfaction', 3)),
                "env_satisfaction":    int(row.get('EnvironmentSatisfaction', 3)),
                "work_life_balance":   int(row.get('WorkLifeBalance', 3)),
                "relationship_satisfaction": int(row.get('RelationshipSatisfaction', 3)),
                "job_involvement":     int(row.get('JobInvolvement', 3)),
                "attrition_probability": round(prob*100, 1),
                "risk_label":          lbl,
                "prediction":          preds[i],
                "explanation":         expl["explanation"],
                "reasons":             expl["reasons"],
                "feature_scores":      expl["feature_scores"],
                "recommendations":     expl["recommendations"],
            })

        visible_probs = [p for p in probs if p >= 0.25]

        at_risk_count = int(sum(1 for p in visible_probs if p >= threshold))
        return jsonify({
            "summary": {
                "total_employees": len(visible_probs),
                "at_risk":         at_risk_count,
                "likely_stay": len(visible_probs) - at_risk_count,
                "attrition_rate":  round(at_risk_count/len(df)*100, 1),
                "avg_risk_score":  round(float(np.mean(probs))*100, 1),
            },
            "predictions":         predictions,
            "risk_distribution":   risk_dist,
            "department_analysis": dept_analysis,
            "role_analysis":       role_analysis,
            "top_features":        top_features,
            "model_accuracy":      round(model_data['accuracy']*100, 1),
        })

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5050)
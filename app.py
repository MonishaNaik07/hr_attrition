"""
HR Attrition Intelligence System - Flask Backend
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# CRITICAL: import HybridEnsemble before any pickle.load so deserialization works
from generate_and_train import HybridEnsemble
import __main__; __main__.HybridEnsemble = HybridEnsemble  # patch __main__ namespace
from flask import Flask, request, jsonify, send_from_directory, render_template
import pandas as pd
import numpy as np
import pickle, os, io, json
from sklearn.preprocessing import LabelEncoder
import warnings; warnings.filterwarnings('ignore')

app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# ─── Load Model ──────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'ensemble_model.pkl')
model_data = None

def load_model():
    global model_data
    try:
        with open(MODEL_PATH, 'rb') as f:
            model_data = pickle.load(f)
        print(f"Model loaded. Accuracy: {model_data['accuracy']*100:.1f}%")
    except Exception as e:
        print(f"Warning: Could not load model: {e}")


# ─── Feature Engineering (must match training) ───────────────────────────────
def engineer_features(df):
    df = df.copy()
    le = LabelEncoder()
    for col in ['BusinessTravel','Department','EducationField','Gender','JobRole','MaritalStatus','Over18']:
        if col in df.columns:
            df[col+'_enc'] = le.fit_transform(df[col].astype(str))
        else:
            df[col+'_enc'] = 0
    ot = (df['OverTime']=='Yes').astype(int) if 'OverTime' in df.columns else pd.Series(0, index=df.index)
    df['OverTime_enc'] = ot
    mi  = df.get('MonthlyIncome', pd.Series(5000, index=df.index))
    age = df.get('Age', pd.Series(35, index=df.index))
    yac = df.get('YearsAtCompany', pd.Series(5, index=df.index)).replace(0,0.5)
    ysp = df.get('YearsSinceLastPromotion', pd.Series(1, index=df.index)).replace(0,0.5)
    ysm = df.get('YearsWithCurrManager', pd.Series(3, index=df.index)).replace(0,0.5)
    ncw = df.get('NumCompaniesWorked', pd.Series(2, index=df.index))
    js  = df.get('JobSatisfaction', pd.Series(3, index=df.index))
    es  = df.get('EnvironmentSatisfaction', pd.Series(3, index=df.index))
    wlb = df.get('WorkLifeBalance', pd.Series(3, index=df.index))
    rs  = df.get('RelationshipSatisfaction', pd.Series(3, index=df.index))
    so  = df.get('StockOptionLevel', pd.Series(0, index=df.index))
    tty = df.get('TrainingTimesLastYear', pd.Series(2, index=df.index))
    dfh = df.get('DistanceFromHome', pd.Series(5, index=df.index))
    jl  = df.get('JobLevel', pd.Series(2, index=df.index))
    ji  = df.get('JobInvolvement', pd.Series(3, index=df.index))
    df['IncomePerYear']          = mi / yac
    df['Age_Tenure_ratio']       = age / yac
    df['YearsSincePromo_ratio']  = ysp / yac
    df['YearsWithManager_ratio'] = ysm / yac
    df['NumCompanies_Age']       = ncw / age
    df['OverTime_JobSat']        = ot * js
    df['OverTime_WLB']           = ot * wlb
    df['OverTime_EnvSat']        = ot * es
    df['Low_Income_OverTime']    = (mi < mi.median()).astype(int) * ot
    df['StockOption_Income']     = so * mi / 10000
    df['Satisfaction_Score']     = (js + es + wlb + rs) / 4
    df['TotalSatisfaction']      = js + es + wlb + rs + ji
    df['Income_Age_ratio']       = mi / age
    df['DistanceIncome_ratio']   = dfh / (mi / 1000)
    df['JobLevel_Income']        = jl * mi
    df['TrainingLastYear_sat']   = tty * js
    df['Age_sq']                 = age ** 2
    df['Income_sq']              = (mi / 1000) ** 2
    return df


def get_feature_columns(df):
    drop = ['Attrition','EmployeeCount','EmployeeNumber','Over18','StandardHours',
            'BusinessTravel','Department','EducationField','Gender','JobRole','MaritalStatus','OverTime']
    if model_data:
        return [c for c in model_data['feature_cols'] if c in df.columns]
    return [c for c in df.columns if c not in drop]


# ─── Explainability ──────────────────────────────────────────────────────────
def generate_explanation(row, risk_prob):
    reasons = []
    recs = []

    ot  = row.get('OverTime','No') == 'Yes'
    js  = int(row.get('JobSatisfaction',3))
    es  = int(row.get('EnvironmentSatisfaction',3))
    wlb = int(row.get('WorkLifeBalance',3))
    rs  = int(row.get('RelationshipSatisfaction',3))
    so  = int(row.get('StockOptionLevel',0))
    ms  = row.get('MaritalStatus','Married')
    yac = float(row.get('YearsAtCompany',5))
    ncw = int(row.get('NumCompaniesWorked',2))
    dfh = int(row.get('DistanceFromHome',5))
    bt  = row.get('BusinessTravel','Travel_Rarely')
    jl  = int(row.get('JobLevel',2))
    mi  = float(row.get('MonthlyIncome',5000))
    ji  = int(row.get('JobInvolvement',3))
    ysp = float(row.get('YearsSinceLastPromotion',1))

    if ot:
        reasons.append("Working overtime – linked to burnout and disengagement")
        recs.append({"priority":"High","action":"Review workload and overtime policies","detail":"Redistribute tasks or hire additional staff to reduce overtime burden."})
    if js <= 2:
        reasons.append(f"Low job satisfaction (score: {js}/4)")
        recs.append({"priority":"High","action":"Conduct 1:1 satisfaction review","detail":"Identify specific pain points through structured feedback sessions."})
    if es <= 2:
        reasons.append(f"Poor workplace environment satisfaction (score: {es}/4)")
        recs.append({"priority":"High","action":"Improve work environment","detail":"Address physical workspace, team dynamics, and management style."})
    if wlb <= 2:
        reasons.append(f"Poor work-life balance (score: {wlb}/4)")
        recs.append({"priority":"High","action":"Implement flexible work arrangements","detail":"Offer remote work options or flexible scheduling."})
    if so == 0:
        reasons.append("No stock option allocation")
        recs.append({"priority":"Medium","action":"Review compensation package","detail":"Consider stock options or equity participation to increase retention."})
    if ms == 'Single':
        reasons.append("Single employees show higher mobility tendency")
        recs.append({"priority":"Low","action":"Strengthen team and community engagement","detail":"Social programs and team-building can improve belonging."})
    if yac <= 1:
        reasons.append(f"Very short tenure ({yac:.0f} year) – early-career flight risk")
        recs.append({"priority":"High","action":"Activate onboarding retention program","detail":"Assign mentor, clear 90-day milestones, and early career pathing."})
    if ncw >= 5:
        reasons.append(f"Worked at {ncw} companies – high job-hopping pattern")
        recs.append({"priority":"Medium","action":"Offer long-term career development plan","detail":"Define a multi-year growth roadmap to reduce flight risk."})
    if dfh >= 20:
        reasons.append(f"Long commute ({dfh} km from home)")
        recs.append({"priority":"Medium","action":"Offer remote/hybrid flexibility","detail":"Partial remote work could significantly reduce commute stress."})
    if bt == 'Travel_Frequently':
        reasons.append("Frequent business travel adds to burnout risk")
        recs.append({"priority":"Medium","action":"Review travel requirements","detail":"Minimize non-essential travel; use video conferencing where possible."})
    if jl == 1:
        reasons.append("Entry-level position – limited advancement visibility")
        recs.append({"priority":"Medium","action":"Create clear promotion pathway","detail":"Set explicit milestones for advancement from entry-level roles."})
    if mi < 3000:
        reasons.append(f"Below-market compensation (${mi:,.0f}/month)")
        recs.append({"priority":"High","action":"Conduct salary benchmarking review","detail":"Compare against market rates and adjust compensation accordingly."})
    if ji <= 2:
        reasons.append(f"Low job involvement (score: {ji}/4)")
        recs.append({"priority":"Medium","action":"Increase role engagement","detail":"Assign stretch projects and increase decision-making autonomy."})
    if ysp >= 4:
        reasons.append(f"No promotion in {ysp:.0f} years")
        recs.append({"priority":"Medium","action":"Review promotion eligibility","detail":"Evaluate employee for promotion or role expansion opportunity."})

    if not reasons:
        reasons.append("Profile shows relatively low attrition risk indicators")
        recs.append({"priority":"Low","action":"Maintain engagement","detail":"Regular check-ins and recognition programs to sustain performance."})

    risk_label = "Critical" if risk_prob >= 0.75 else ("High" if risk_prob >= 0.55 else ("Medium" if risk_prob >= 0.35 else "Low"))
    explanation = f"This employee shows a {risk_label.lower()} attrition risk ({risk_prob*100:.0f}%). "
    if len(reasons) > 1:
        explanation += f"Key factors include: {'; '.join(reasons[:3])}."
    else:
        explanation += reasons[0] + "."

    return {"risk_label": risk_label, "explanation": explanation, "reasons": reasons[:5], "recommendations": recs[:4]}


# ─── Routes ───────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory('templates', 'index.html')


@app.route('/dashboard')
def dashboard():
    return send_from_directory('templates', 'dashboard.html')


@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username','')
    password = data.get('password','')
    # Demo credentials
    users = {'admin':'admin123','hr_manager':'hr2024','analyst':'data2024'}
    if username in users and users[username] == password:
        return jsonify({'success': True, 'token': 'demo_token_2024', 'user': username,
                        'role': 'HR Manager' if username == 'hr_manager' else 'Administrator'})
    return jsonify({'success': False, 'message': 'Invalid credentials'}), 401


@app.route('/api/model-stats', methods=['GET'])
def model_stats():
    if not model_data:
        return jsonify({'error': 'Model not loaded'}), 500
    fi = model_data.get('feature_importance', {})
    top_features = list(fi.items())[:15]
    return jsonify({
        'accuracy': round(model_data['accuracy'] * 100, 1),
        'auc': round(model_data['auc'], 4),
        'attrition_rate': round(model_data['attrition_rate'] * 100, 1),
        'n_records': model_data['n_records'],
        'n_features': len(model_data['feature_cols']),
        'models': ['Random Forest (×3)', 'Extra Trees (×2)', 'Gradient Boost v1 (×3)',
                   'Gradient Boost v2 (×2)', 'Neural Network (×1)'],
        'top_features': [{'name': k, 'importance': round(v*100, 2)} for k,v in top_features]
    })


@app.route('/api/predict/upload', methods=['POST'])
def predict_upload():
    if not model_data:
        return jsonify({'error': 'Model not loaded'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    f = request.files['file']
    if not f.filename.endswith('.csv'):
        return jsonify({'error': 'Please upload a CSV file'}), 400

    try:
        df = pd.read_csv(io.StringIO(f.read().decode('utf-8')))
    except Exception as e:
        return jsonify({'error': f'CSV parse error: {e}'}), 400

    if len(df) == 0:
        return jsonify({'error': 'Empty CSV file'}), 400

    # Keep original for display
    df_orig = df.copy()

    # Engineer features
    df_feat = engineer_features(df)
    feat_cols = get_feature_columns(df_feat)
    X = df_feat[feat_cols].fillna(0)

    ensemble = model_data['ensemble']
    probs = ensemble.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)

    results = []
    dept_stats = {}
    role_stats = {}

    for i, (idx, row) in enumerate(df_orig.iterrows()):
        prob = float(probs[i])
        pred = int(preds[i])
        expl = generate_explanation(row.to_dict(), prob)

        emp_id = str(row.get('EmployeeNumber', i+1))
        name = row.get('Name', f'Employee {emp_id}')
        dept = str(row.get('Department', 'Unknown'))
        role = str(row.get('JobRole', 'Unknown'))
        age  = int(row.get('Age', 0))
        mi   = float(row.get('MonthlyIncome', 0))
        yac  = float(row.get('YearsAtCompany', 0))
        ot   = str(row.get('OverTime','No'))

        result = {
            'employee_id': emp_id,
            'name': name,
            'department': dept,
            'job_role': role,
            'age': age,
            'monthly_income': mi,
            'years_at_company': yac,
            'overtime': ot,
            'attrition_probability': round(prob * 100, 1),
            'prediction': 'At Risk' if pred == 1 else 'Likely to Stay',
            'risk_label': expl['risk_label'],
            'explanation': expl['explanation'],
            'reasons': expl['reasons'],
            'recommendations': expl['recommendations'],
            # Full profile
            'gender': str(row.get('Gender','N/A')),
            'marital_status': str(row.get('MaritalStatus','N/A')),
            'education': int(row.get('Education',0)),
            'education_field': str(row.get('EducationField','N/A')),
            'job_level': int(row.get('JobLevel',0)),
            'job_satisfaction': int(row.get('JobSatisfaction',0)),
            'env_satisfaction': int(row.get('EnvironmentSatisfaction',0)),
            'work_life_balance': int(row.get('WorkLifeBalance',0)),
            'relationship_satisfaction': int(row.get('RelationshipSatisfaction',0)),
            'job_involvement': int(row.get('JobInvolvement',0)),
            'stock_option': int(row.get('StockOptionLevel',0)),
            'num_companies': int(row.get('NumCompaniesWorked',0)),
            'total_working_years': int(row.get('TotalWorkingYears',0)),
            'training_times': int(row.get('TrainingTimesLastYear',0)),
            'years_since_promo': float(row.get('YearsSinceLastPromotion',0)),
            'years_with_manager': float(row.get('YearsWithCurrManager',0)),
            'distance_from_home': int(row.get('DistanceFromHome',0)),
            'business_travel': str(row.get('BusinessTravel','N/A')),
            'percent_salary_hike': int(row.get('PercentSalaryHike',0)),
            'performance_rating': int(row.get('PerformanceRating',0)),
            'feature_scores': {
                'Job Satisfaction': round((5 - int(row.get('JobSatisfaction',3))) / 4 * 100),
                'Env Satisfaction': round((5 - int(row.get('EnvironmentSatisfaction',3))) / 4 * 100),
                'Work-Life Balance': round((5 - int(row.get('WorkLifeBalance',3))) / 4 * 100),
                'Overtime':         100 if row.get('OverTime','No') == 'Yes' else 0,
                'Stock Options':    round((4 - int(row.get('StockOptionLevel',0))) / 4 * 100),
                'Short Tenure':     round(max(0, (3 - float(row.get('YearsAtCompany',5))) / 3 * 100)),
                'Low Salary':       round(max(0, (3000 - float(row.get('MonthlyIncome',5000))) / 3000 * 100)),
                'Commute Burden':   round(min(100, int(row.get('DistanceFromHome',5)) / 29 * 100)),
            },
        }
        results.append(result)

        # Dept stats
        if dept not in dept_stats:
            dept_stats[dept] = {'total': 0, 'at_risk': 0, 'prob_sum': 0}
        dept_stats[dept]['total'] += 1
        dept_stats[dept]['at_risk'] += pred
        dept_stats[dept]['prob_sum'] += prob

        # Role stats
        if role not in role_stats:
            role_stats[role] = {'total': 0, 'at_risk': 0}
        role_stats[role]['total'] += 1
        role_stats[role]['at_risk'] += pred

    total = len(results)
    at_risk_count = sum(1 for r in results if r['prediction'] == 'At Risk')
    avg_risk = np.mean(probs) * 100

    dept_chart = []
    for dept, s in dept_stats.items():
        dept_chart.append({
            'department': dept,
            'total': s['total'],
            'at_risk': s['at_risk'],
            'stay': s['total'] - s['at_risk'],
            'risk_pct': round(s['at_risk']/s['total']*100, 1),
            'avg_prob': round(s['prob_sum']/s['total']*100, 1)
        })

    role_chart = sorted(
        [{'role': r, 'total': s['total'], 'at_risk': s['at_risk'],
          'risk_pct': round(s['at_risk']/s['total']*100, 1)} for r, s in role_stats.items()],
        key=lambda x: x['risk_pct'], reverse=True
    )[:10]

    risk_dist = {
        'Critical': sum(1 for r in results if r['risk_label'] == 'Critical'),
        'High': sum(1 for r in results if r['risk_label'] == 'High'),
        'Medium': sum(1 for r in results if r['risk_label'] == 'Medium'),
        'Low': sum(1 for r in results if r['risk_label'] == 'Low'),
    }

    return jsonify({
        'summary': {
            'total_employees': total,
            'at_risk': at_risk_count,
            'likely_stay': total - at_risk_count,
            'attrition_rate': round(at_risk_count/total*100, 1),
            'avg_risk_score': round(avg_risk, 1),
        },
        'risk_distribution': risk_dist,
        'department_analysis': dept_chart,
        'role_analysis': role_chart,
        'predictions': results,
        'model_accuracy': round(model_data['accuracy']*100, 1),
    })


@app.route('/api/sample-data', methods=['GET'])
def sample_data():
    """Return sample CSV data for demo"""
    csv_path = os.path.join(os.path.dirname(__file__), 'data', 'ibm_hr_sample.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path).head(50)
        return df.to_csv(index=False), 200, {'Content-Type': 'text/csv',
                                              'Content-Disposition': 'attachment; filename=sample_employees.csv'}
    return jsonify({'error': 'Sample data not found'}), 404


if __name__ == '__main__':
    load_model()
    app.run(debug=True, host='0.0.0.0', port=5050)

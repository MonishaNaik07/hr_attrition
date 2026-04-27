from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from auth import register_user, login_user, login_required
import joblib
import pandas as pd
import numpy as np
import traceback
import json
import os
from datetime import datetime

app = Flask(__name__)
app.secret_key = "super_secret_hr_key"  # keep as-is, already fixed string — good for session persistence

models = {
    'ibm': {'model': None, 'scaler': None, 'features': {'numerical': [], 'categorical': [], 'cat_options': {}}},
    'syn': {'model': None, 'scaler': None, 'features': []}
}

try:
    # Load IBM Schema
    models['ibm']['model'] = joblib.load('model_ibm.pkl')
    models['ibm']['scaler'] = joblib.load('scaler_ibm.pkl')
    models['ibm']['features'] = joblib.load('features_ibm.pkl')
    
    # Load Synthetic Schema
    models['syn']['model'] = joblib.load('model_syn.pkl')
    models['syn']['scaler'] = joblib.load('scaler_syn.pkl')
    models['syn']['features'] = joblib.load('features_syn.pkl')
    print("Both models loaded successfully!")
except Exception as e:
    print(f"Error loading models. Run training scripts first. {e}")

# Persistent database for records
RECORDS_FILE = 'records.json'

def load_db():
    if os.path.exists(RECORDS_FILE):
        try:
            with open(RECORDS_FILE, 'r') as f:
                return json.load(f)
        except:
            return []
    return []

def save_db(db):
    try:
        with open(RECORDS_FILE, 'w') as f:
            json.dump(db, f)
    except Exception as e:
        print(f"Error saving DB: {e}")

records_db = load_db()

def safe_float(val, default=0):
    try:
        if pd.isna(val): return default
        return float(str(val).replace(',', '').replace('$', '').strip())
    except:
        return default

def generate_explanation_and_solution(data):
    recommendations = []
    
    # Specific Data-Driven Logic
    inc = safe_float(data.get('MonthlyIncome'), 5000)
    if inc < 3500:
        recommendations.append({
            "reason": f"Current income (${inc}) is significantly below the $4,000 threshold for this role.",
            "solution": f"Immediately review {data.get('employee_name', 'Employee')}'s salary tier for a market-rate adjustment."
        })
        
    ot = data.get('OverTime', 'No')
    if ot == 'Yes':
        recommendations.append({
            "reason": "Consistent overtime is increasing burnout risk.",
            "solution": "Redistribute workload or introduce a compensatory off-day policy."
        })

    jsat = safe_float(data.get('JobSatisfaction'), 3)
    if jsat <= 2:
        recommendations.append({
            "reason": f"Low Job Satisfaction score ({jsat}/4) detected.",
            "solution": f"Conduct a focused role-fit assessment for {data.get('employee_name', 'the employee')}."
        })

    tenure = safe_float(data.get('YearsAtCompany'), 5)
    promot = safe_float(data.get('YearsSinceLastPromotion'), 1)
    if tenure > 3 and promot > 2:
        recommendations.append({
            "reason": f"Has been at company for {tenure} years without a promotion in {promot} years.",
            "solution": "Establish a clear 12-month promotion track or vertical growth plan."
        })

    dist = safe_float(data.get('DistanceFromHome'), 1)
    if dist > 20:
        recommendations.append({
            "reason": f"Long-distance commute ({dist}km) is likely impacting work-life balance.",
            "solution": "Transition to a hybrid model (3 days remote) to reduce travel fatigue."
        })
        
    # Default if no specific triggers
    if not recommendations:
        recommendations.append({
            "reason": "Multiple subtle demographic factors (Tenure, Age, Role complexity).",
            "solution": "Implement a proactive stay-interview to identify hidden engagement gaps."
        })
        
    return recommendations

@app.after_request
def add_header(response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "-1"
    return response

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'username' in session:
        return redirect(url_for('home'))
    error = None
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        ok, msg = login_user(username, password)
        if ok:
            session['username'] = username
            return redirect(url_for('home'))
        error = msg
    return render_template('login.html', error=error)


@app.route('/register', methods=['POST'])
def register():
    username = request.form.get('username', '').strip()
    password = request.form.get('password', '')
    confirm  = request.form.get('confirm_password', '')
    if len(username) < 3:
        return render_template('login.html', reg_error="Username must be at least 3 characters")
    if password != confirm:
        return render_template('login.html', reg_error="Passwords do not match")
    if len(password) < 4:
        return render_template('login.html', reg_error="Password must be at least 4 characters")
    ok, msg = register_user(username, password)
    if ok:
        return render_template('login.html', reg_success="Account created! Please log in.")
    return render_template('login.html', reg_error=msg)


@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))


@app.route('/')
@login_required
def home():
    return render_template('index.html', ibm_features=models['ibm']['features'], syn_features=models['syn']['features'])



@app.route('/predict', methods=['POST'])
@login_required
def predict():
    try:
        data = request.json
        model_type = data.get('model_type', 'ibm') # default ibm
        
        if model_type == 'ibm':
            f_dict = models['ibm']['features']
            input_data = {}
            for f in f_dict['numerical']:
                try:
                    input_data[f] = float(data.get(f, 0))
                except ValueError:
                    input_data[f] = 0.0
            for f in f_dict['categorical']:
                default_val = f_dict['cat_options'][f][0] if f_dict['cat_options'].get(f) else ''
                input_data[f] = data.get(f, default_val)
                
            df_input = pd.DataFrame([input_data])
            X_scaled = models['ibm']['scaler'].transform(df_input)
            prediction = models['ibm']['model'].predict(X_scaled)[0]
            probs = models['ibm']['model'].predict_proba(X_scaled)[0]
            
        else:
            # Synthetic
            f_list = models['syn']['features']
            input_data = []
            for f in f_list:
                val = data.get(f, 0)
                try:
                    input_data.append(float(val))
                except ValueError:
                    input_data.append(0.0)
            X_test = np.array([input_data])
            X_scaled = models['syn']['scaler'].transform(X_test)
            prediction = models['syn']['model'].predict(X_scaled)[0]
            probs = models['syn']['model'].predict_proba(X_scaled)[0]
            
        attrition_prob = probs[1] * 100
        stay_prob = probs[0] * 100
        is_attrited = bool(prediction == 1)
        explanation = generate_explanation_and_solution(data) if is_attrited else None
        
        record = {
            'id': len(records_db) + 1,
            'role': 'Admin',
            'prediction': 'Attrited' if is_attrited else 'Stayed',
            'confidence': round(attrition_prob if is_attrited else stay_prob, 2),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'explanation': explanation or generate_explanation_and_solution(data), # Ensure we always have some explanation
            **data
        }
        records_db.append(record)
        save_db(records_db)

        return jsonify({
            'prediction': 'Attrited' if is_attrited else 'Stayed',
            'attrition_probability': round(attrition_prob, 2),
            'stay_probability': round(stay_prob, 2),
            'raw_prediction': int(prediction),
            'explanation': explanation
        })
        
    except Exception as e:
        print("Error during prediction:", traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/upload_csv', methods=['POST'])
@login_required
def upload_csv():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    try:
        df = pd.read_csv(file)
        df.columns = df.columns.str.strip() # Clean whitespaces from headers
        
        # Determine the best fit model
        ibm_num = models['ibm']['features']['numerical']
        ibm_cat = models['ibm']['features']['categorical']
        ibm_all = ibm_num + ibm_cat
        
        syn_all = models['syn']['features']
        
        # Heuristic: Check which schema has more overlap
        df_cols = set(df.columns)
        ibm_overlap = len(df_cols.intersection(set(ibm_all)))
        syn_overlap = len(df_cols.intersection(set(syn_all)))
        
        if ibm_overlap > syn_overlap and ibm_overlap > 5:
            engine = models['ibm']
            required_cols = ibm_all
            # Fill missing required columns with defaults
            for col in required_cols:
                if col not in df.columns:
                    if col in ibm_num: df[col] = 0
                    else: df[col] = engine['features']['cat_options'].get(col, [''])[0]
            X_test = df[required_cols]
            X_scaled = engine['scaler'].transform(X_test)
        elif syn_overlap > 0:
            engine = models['syn']
            required_cols = syn_all
            for col in required_cols:
                if col not in df.columns: df[col] = 0
            X_test = df[required_cols]
            X_scaled = engine['scaler'].transform(X_test)
        else:
            return jsonify({'error': 'The uploaded CSV does not contain recognized HR features (e.g., Age, MonthlyIncome, JobSatisfaction). Please check your column headers.'}), 400
            
        df = df.fillna(0) # Fill all NaNs to prevent JSON errors
        
        # FINAL FIX (handles nested models inside VotingClassifier)
        try:
            def fix_model(model):
                if hasattr(model, "get_xgb_params"):
                    # FORCE FIX (not just params)
                    try:
                        model.use_label_encoder = False
                    except:
                        pass
                    try:
                        model.gpu_id = -1
                    except:
                        pass

                    try:
                        model.predictor = "cpu_predictor"
                    except:
                        pass
                
                if hasattr(model, "estimators_"):
                    for m in model.estimators_:
                        fix_model(m)

            fix_model(engine['model'])
        except:
            pass
        
        preds = engine['model'].predict(X_scaled)
        probs = engine['model'].predict_proba(X_scaled)
        
        results = []
        for i in range(len(preds)):
            row_dict = df.iloc[i].to_dict()
            # Clean up the dictionary for JSON safety
            for k, v in row_dict.items():
                if pd.isna(v): row_dict[k] = ""
                
            is_attrited = bool(preds[i] == 1)
            confidence = round(probs[i][1] * 100 if is_attrited else probs[i][0] * 100, 2)
            
            # Add to records history
            exp = generate_explanation_and_solution(row_dict)
            record = {
                'id': len(records_db) + 1,
                'role': 'Batch CSV',
                'prediction': 'Attrited' if is_attrited else 'Stayed',
                'confidence': confidence,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'explanation': exp,
                **row_dict
            }
            records_db.append(record)
            
            results.append({
                'row_index': i + 1,
                'prediction': 'Attrited' if is_attrited else 'Stayed',
                'risk': confidence
            })
            
        save_db(records_db)
        return jsonify({'success': True, 'results': results, 'total': len(results)})
    except Exception as e:
        print("CSV Upload Error Detail:", traceback.format_exc())
        return jsonify({'error': f"Failed to process CSV: {str(e)}"}), 500

@app.route('/records', methods=['GET'])
@login_required
def get_records():
    return jsonify(records_db[::-1])

@app.route('/clear_records', methods=['POST'])
@login_required
def clear_records():
    global records_db
    records_db = []
    save_db(records_db)
    return jsonify({'success': True})

if __name__ == '__main__':
    app.run(debug=True, port=8080)

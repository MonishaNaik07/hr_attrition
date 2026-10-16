# AttritionIQ — HR Attrition Intelligence System

**AI-powered HR platform for employee attrition prediction and retention insights.**

🔗 **Live Demo:** https://hr-attrition-520m.onrender.com/

AttritionIQ is an AI-powered HR intelligence platform that predicts employee attrition using a five-model hybrid ensemble. Upload employee CSV data to generate risk scores, natural-language explanations, and actionable HR recommendations through an interactive dashboard.

## 🚀 Live Demo for Recruiters

**Try AttritionIQ:** https://hr-attrition-520m.onrender.com/

Explore the deployed application to experience the employee attrition dashboard, risk analysis, predictions, and HR recommendations.

### Demo Credentials

| Username | Password | Role |
|---|---|---|
| `admin` | `admin123` | Administrator |
| `hr_manager` | `hr2024` | HR Manager |
| `analyst` | `data2024` | Analyst |

*These are the demo credentials specified in the project configuration. Verify that they work on the deployed application before sharing them with recruiters.*

## ✨ Key Features

- **Hybrid ensemble prediction:** Combines five machine-learning models using weighted soft voting.
- **Employee risk scoring:** Categorizes attrition risk as Critical, High, Medium, or Low.
- **AI-generated explanations:** Provides natural-language explanations and key risk factors for individual predictions.
- **HR recommendations:** Suggests prioritized actions to support employee retention.
- **Interactive analytics dashboard:** Visualizes attrition risk, retention metrics, departments, and job roles.
- **CSV upload and export:** Upload employee datasets and export filtered prediction results.
- **Model explainability:** Displays feature importance and the factors associated with predictions.

## 🛠️ Tech Stack

- **Language:** Python
- **Backend:** Flask, REST APIs
- **Machine Learning:** scikit-learn, NumPy, SciPy
- **Data Processing:** Pandas
- **Model Persistence:** Joblib
- **Frontend:** HTML, CSS, JavaScript, Jinja templates

## ⚡ Quick Start

### 1. Install dependencies

```bash
pip install flask pandas numpy scikit-learn scipy joblib
```

### 2. Train the model

Run this step for the first-time setup:

```bash
python generate_and_train.py
```

This generates the IBM HR Analytics dataset containing 1,470 records and trains the five-model ensemble.

Expected performance reported by the project: approximately 84% accuracy and 0.92 ROC-AUC.

### 3. Start the server

```bash
python app.py
```

### 4. Open the application

```text
http://localhost:5050
```

## 📁 Project Structure

```text
hr_attrition/
├── app.py                   Flask REST API
├── generate_and_train.py    Dataset generation and model training
├── requirements.txt         Python dependencies
├── models/
│   └── ensemble_model.pkl   Trained five-model hybrid ensemble
├── data/
│   └── ibm_hr_sample.csv    Generated IBM HR dataset
└── templates/
    ├── index.html           Login page
    └── dashboard.html       HR intelligence dashboard
```

## 🧠 Machine Learning Architecture

AttritionIQ uses a **five-model hybrid ensemble with weighted soft voting**.

| Model | Estimators / Architecture | Weight |
|---|---|---|
| Random Forest | 1,000 estimators | ×3 |
| Extra Trees | 1,000 estimators | ×2 |
| Gradient Boosting v1 | 400 estimators | ×3 |
| Gradient Boosting v2 | 400 estimators | ×2 |
| Neural Network MLP | 256 → 128 → 64 → 32 | ×1 |

### Feature Engineering

The system adds 18 engineered features to the 31 original features, creating 49 features for prediction.

Examples include:
- `IncomePerYear` — Monthly income normalized by tenure
- `Age_Tenure_ratio` — Age relative to years at the company
- `YearsSincePromo_ratio` — Promotion staleness
- `OverTime_JobSat` — Overtime and job satisfaction interaction
- `OverTime_WLB` — Overtime and work-life balance interaction
- `Low_Income_OverTime` — Combined low-income and overtime signal
- `Satisfaction_Score` — Mean of four satisfaction dimensions
- `JobLevel_Income` — Seniority and income interaction
- `Age²` and `Income²` — Nonlinear features

### Reported Model Performance

| Metric | Result |
|---|---:|
| Accuracy | 84.0% |
| ROC-AUC | 0.9194 |
| Total features | 49 |
| Original features | 31 |
| Engineered features | 18 |
| Total dataset records | 1,470 |
| Training records | 1,176 |
| Test split | 20% |

*These are the project's reported results. Actual performance should be validated against the dataset, evaluation procedure, and deployment model.*

## 📊 Dashboard Features

- **Login page:** Animated mesh background and demo credential autofill.
- **KPI cards:** Employee count, at-risk count, retention rate, average risk score, and model accuracy.
- **Department analytics:** Stacked bar chart comparing at-risk and stable employees.
- **Risk distribution:** Donut chart showing Critical, High, Medium, and Low risk categories.
- **Job-role ranking:** Horizontal bar chart of job roles by attrition rate.
- **Feature importance:** Top 10 predictive signals.
- **Predictions table:** Search, sorting, risk filtering, and pagination.
- **Employee detail modal:** Risk meter, explanation, risk factors, and HR recommendations.
- **CSV export:** Download filtered predictions.

## 🔍 Explainability

Each employee prediction includes:

1. **Risk label:** Critical, High, Medium, or Low.
2. **Natural-language explanation:** A summary of the predicted risk and contributing factors.
3. **Top risk factors:** Up to five specific factors associated with the prediction.
4. **HR recommendations:** Priority-ranked suggestions with supporting details.

For example, an explanation may identify overtime, low job satisfaction, or a lack of stock options as factors associated with an elevated predicted risk.

These outputs are model-based indicators, not proof that an employee will leave. HR decisions should involve human review and should not rely on automated predictions alone.

## 🔌 API Endpoints

### `POST /api/login`

Authenticates a user.

Example request:

```json
{
  "username": "admin",
  "password": "admin123"
}
```

### `POST /api/predict/upload`

Accepts a CSV file through a multipart form field named `file`.

Returns prediction results, KPIs, department analysis, and risk distribution.

### `GET /api/model-stats`

Returns model accuracy, AUC, feature importances, and model components.

### `GET /api/sample-data`

Downloads a 50-row sample CSV for testing.

## 📄 CSV Input Format

The upload format follows the IBM HR Analytics schema.

```text
Age, Attrition, BusinessTravel, DailyRate, Department, DistanceFromHome,
Education, EducationField, EnvironmentSatisfaction, Gender, HourlyRate,
JobInvolvement, JobLevel, JobRole, JobSatisfaction, MaritalStatus,
MonthlyIncome, NumCompaniesWorked, OverTime, PercentSalaryHike,
PerformanceRating, RelationshipSatisfaction, StockOptionLevel,
TotalWorkingYears, TrainingTimesLastYear, WorkLifeBalance,
YearsAtCompany, YearsInCurrentRole, YearsSinceLastPromotion,
YearsWithCurrManager
```

Missing columns are filled using median or mode defaults, according to the project implementation.

## 🔄 Retrain the Model

To retrain the ensemble from scratch:

```bash
python generate_and_train.py
```

The trained model is saved to `models/ensemble_model.pkl` and loaded by the Flask application on startup.

## 🔮 Future Improvements

- Validate model performance on independent datasets.
- Improve prediction calibration and risk-threshold selection.
- Add stronger authentication, access controls, and audit logging.
- Monitor model drift and changes in employee data.
- Improve privacy protections and fairness evaluation.
- Add explainability and reporting features for HR teams.

## 👨‍💻 Project Summary

AttritionIQ demonstrates end-to-end machine-learning application development, feature engineering, ensemble modelling, Flask API development, data visualization, model explainability, and HR analytics.

**Live Demo:** https://hr-attrition-520m.onrender.com/

**Project:** AttritionIQ — HR Attrition Intelligence System

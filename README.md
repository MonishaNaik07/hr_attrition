# AttritionIQ — HR Attrition Intelligence System

AI-powered HR platform that predicts employee attrition using a 5-model hybrid ensemble.
Upload employee CSV data and instantly get risk scores, AI explanations, and HR recommendations.

---

## Quick Start

### 1. Install dependencies
```bash
pip install flask pandas numpy scikit-learn scipy joblib
```

### 2. Train the model (first time only)
```bash
python generate_and_train.py
```
This generates the IBM HR Analytics dataset (1,470 records) and trains the ensemble model.
Expected output: ~84% accuracy, ~0.92 ROC-AUC.

### 3. Start the server
```bash
python app.py
```

### 4. Open in browser
```
http://localhost:5050
```

### Demo credentials
| Username     | Password   | Role          |
|--------------|------------|---------------|
| admin        | admin123   | Administrator |
| hr_manager   | hr2024     | HR Manager    |
| analyst      | data2024   | Analyst       |

---

## Project Structure

```
hr_attrition/
├── app.py                   Flask REST API
├── generate_and_train.py    Dataset generation + model training
├── requirements.txt         Python dependencies
├── models/
│   └── ensemble_model.pkl   Trained 5-model hybrid ensemble
├── data/
│   └── ibm_hr_sample.csv    Generated IBM HR dataset (1,470 records)
└── templates/
    ├── index.html           Login page
    └── dashboard.html       Main HR intelligence dashboard
```

---

## Machine Learning Architecture

### Hybrid Ensemble (Weighted Soft Voting)
| Model                  | Estimators | Weight |
|------------------------|-----------|--------|
| Random Forest          | 1,000     | ×3     |
| Extra Trees            | 1,000     | ×2     |
| Gradient Boosting v1   | 400       | ×3     |
| Gradient Boosting v2   | 400       | ×2     |
| Neural Network MLP     | 256→128→64→32 | ×1 |

### Feature Engineering (18 engineered features)
- `IncomePerYear` — Monthly income normalized by tenure
- `Age_Tenure_ratio` — Age relative to years at company
- `YearsSincePromo_ratio` — Promotion staleness
- `YearsWithManager_ratio` — Manager tenure ratio
- `NumCompanies_Age` — Job-hopping index
- `OverTime_JobSat` — Interaction: overtime × job satisfaction
- `OverTime_WLB` — Interaction: overtime × work-life balance
- `OverTime_EnvSat` — Interaction: overtime × environment
- `Low_Income_OverTime` — High-risk compound: low pay + overtime
- `StockOption_Income` — Equity-weighted income signal
- `Satisfaction_Score` — Mean of 4 satisfaction dimensions
- `TotalSatisfaction` — Sum including job involvement
- `Income_Age_ratio` — Compensation relative to age
- `DistanceIncome_ratio` — Commute burden relative to pay
- `JobLevel_Income` — Seniority × income signal
- `TrainingLastYear_sat` — Training × satisfaction interaction
- `Age²`, `Income²` — Non-linear terms

### Performance
- **Accuracy**: 84.0% on 20% held-out test set
- **ROC-AUC**: 0.9194
- **Total features**: 49 (31 original + 18 engineered)
- **Training records**: 1,176 (80% of 1,470)

---

## API Endpoints

### POST /api/login
```json
{ "username": "admin", "password": "admin123" }
```

### POST /api/predict/upload
Multipart form upload with `file` field (CSV).
Returns predictions, KPIs, department analysis, risk distribution.

### GET /api/model-stats
Returns accuracy, AUC, feature importances, model components.

### GET /api/sample-data
Downloads a 50-row sample CSV for testing.

---

## CSV Format

The upload CSV should follow the IBM HR Analytics schema:

```
Age, Attrition, BusinessTravel, DailyRate, Department, DistanceFromHome,
Education, EducationField, EnvironmentSatisfaction, Gender, HourlyRate,
JobInvolvement, JobLevel, JobRole, JobSatisfaction, MaritalStatus,
MonthlyIncome, NumCompaniesWorked, OverTime, PercentSalaryHike,
PerformanceRating, RelationshipSatisfaction, StockOptionLevel,
TotalWorkingYears, TrainingTimesLastYear, WorkLifeBalance,
YearsAtCompany, YearsInCurrentRole, YearsSinceLastPromotion,
YearsWithCurrManager
```

Missing columns are filled with median/mode defaults automatically.

---

## Dashboard Features

- **Login page** — Animated mesh background, demo credential autofill
- **KPI cards** — Total employees, at-risk count, retention rate, avg risk score, model accuracy
- **Department chart** — Stacked bar: at-risk vs. stable by department
- **Risk donut** — Critical / High / Medium / Low distribution
- **Role ranking** — Top job roles by attrition rate (horizontal bar)
- **Feature importance** — Top 10 predictive signals with relative bars
- **Predictions table** — Sortable, searchable, filterable by risk level, paginated
- **Employee modal** — Risk meter, AI explanation, reason list, priority HR recommendations
- **Export** — Download filtered predictions as CSV

---

## Explainability

Each prediction includes:
1. **Risk label** — Critical / High / Medium / Low
2. **Natural language explanation** — "This employee shows a high attrition risk (78%). Key factors include: Working overtime; Low job satisfaction (score: 1/4); No stock option allocation."
3. **Top reasons list** — Up to 5 specific risk factors
4. **HR recommendations** — Priority-ranked actions (High/Medium/Low) with detail text

---

## Retrain the Model

To retrain from scratch (e.g. with a new random seed or larger dataset):
```bash
python generate_and_train.py
```
The model is saved to `models/ensemble_model.pkl` and automatically picked up by the Flask app on next startup.

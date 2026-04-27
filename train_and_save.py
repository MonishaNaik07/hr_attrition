import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from attrition_model import build_ensemble_model

def train_and_save():
    print("Loading dataset for production model...")
    # For production prediction, we use dataset 1 
    df = pd.read_csv('dataset_1_tech_dept.csv')
    
    X = df.drop(columns=['Attrition'])
    y = df['Attrition']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("Training production ensemble model...")
    ensemble = build_ensemble_model()
    ensemble.fit(X_scaled, y)
    
    # Save the model and scaler
    joblib.dump(ensemble, 'model_syn.pkl')
    joblib.dump(scaler, 'scaler_syn.pkl')
    
    # Save the feature columns so the web app knows the required order
    joblib.dump(list(X.columns), 'features_syn.pkl')
    
    print("Model, scaler, and features saved successfully.")

if __name__ == '__main__':
    train_and_save()

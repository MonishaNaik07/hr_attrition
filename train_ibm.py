import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
from attrition_model import build_ensemble_model

def train_ibm_model():
    print("Loading GENUINE IBM Dataset (1470 rows)...")
    df = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')
    
    # Target column processing
    y = df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)
    X = df.drop(columns=['Attrition'])
    
    # Identify categorical and numerical columns
    # We will exclude meaningless columns to increase accuracy and performance
    exclude_cols = ['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours']
    X = X.drop(columns=[c for c in exclude_cols if c in X.columns])
    
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(exclude=['object']).columns.tolist()
    
    print(f"Detected {len(categorical_cols)} categorical & {len(numerical_cols)} numerical features.")
    
    # Create preprocessing pipelines
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )
    
    # Preprocess now before saving so we can dump the transformer
    X_processed = preprocessor.fit_transform(X)
    
    print("Training Ensemble Model on Real Data...")
    ensemble = build_ensemble_model()
    ensemble.fit(X_processed, y)
    
    # Save the pipeline objects
    joblib.dump(ensemble, 'model_ibm.pkl')
    joblib.dump(preprocessor, 'scaler_ibm.pkl') # We reuse the scaler file name to mean "preprocessor"
    
    # Save feature structure dictionary so the frontend knows what to render
    joblib.dump({
        'numerical': numerical_cols,
        'categorical': categorical_cols,
        'cat_options': {col: df[col].unique().tolist() for col in categorical_cols}
    }, 'features_ibm.pkl')
    
    print("Successfully compiled and saved Enterprise IBM Model!")

if __name__ == '__main__':
    train_ibm_model()

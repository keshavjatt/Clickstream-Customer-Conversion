# train_model.py (CORRECTED VERSION)
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')

def train_model():
    # DATASET LOAD KARNA
    print("📂 Loading dataset...")
    df = pd.read_csv("data/e-shop clothing 2008.csv", sep=";")
    
    # CREATE NEW TARGET: Will user go to NEXT page (page+1)?
    print("🎯 Creating new target variable...")
    
    # Sort by session ID and order to track user journey
    df = df.sort_values(['session ID', 'order'])
    
    # Create next page for each click in same session
    df['next_page'] = df.groupby('session ID')['page'].shift(-1)
    
    # Remove last click of each session (no next page)
    df = df.dropna(subset=['next_page'])
    
    # Create target: Will next page be 4 or 5?
    df["next_is_conversion"] = df['next_page'].apply(lambda x: 1 if x >= 4 else 0)
    
    # Filter: Only consider when current page is 1, 2, or 3
    df_filtered = df[df['page'] < 4].copy()
    
    print(f"📊 Total samples after filtering: {len(df_filtered)}")
    print(f"📈 Conversion rate (next page >= 4): {df_filtered['next_is_conversion'].mean():.4f}")
    
    # FEATURE SELECTION
    print("🛠️ Feature engineering...")
    
    features = [
        'page',           # Current page number
        'order',          # Sequence in session
        'price',          # Product price
        'price 2',        # Price above average
        'month',          # Month of year
        'day',           # Day of month
        'country',       # Country code
        'page 1 (main category)',  # Main category
        'colour',        # Color of product
        'location',      # Photo location on page
        'model photography'  # Photo type
    ]
    
    # Prepare data
    X = df_filtered[features].copy()
    y = df_filtered["next_is_conversion"]
    
    # Handle categorical features
    categorical_cols = ['country', 'page 1 (main category)', 'colour', 'location', 'model photography']
    
    # Label encoding for categorical variables
    label_encoders = {}
    for col in categorical_cols:
        if col in X.columns:
            le = LabelEncoder()
            X.loc[:, col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
    
    # DATA KO TRAIN AUR TEST MEIN BATANA
    print("🔀 Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Check class distribution
    print(f"\n📊 Class distribution in training:")
    print(f"   No Conversion (0): {(y_train == 0).sum()} samples")
    print(f"   Conversion (1): {(y_train == 1).sum()} samples")
    
    # MODEL TRAIN KARNA
    print("🤖 Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Model performance
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    
    print(f"📊 Training Accuracy: {train_accuracy:.4f}")
    print(f"📊 Test Accuracy: {test_accuracy:.4f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n🎯 Feature Importance:")
    print(feature_importance.to_string())
    
    # More detailed evaluation
    print("\n📊 Detailed Evaluation on Test Set:")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=['Next Page < 4', 'Next Page >= 4']))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("📋 Confusion Matrix:")
    print(f"True Negatives:  {cm[0,0]:6d} | False Positives: {cm[0,1]:6d}")
    print(f"False Negatives: {cm[1,0]:6d} | True Positives:  {cm[1,1]:6d}")
    
    # "model" folder agar nahi hai to banado
    os.makedirs("model", exist_ok=True)
    
    # Train kiya hua model file mein save karo
    joblib.dump({
        'model': model,
        'features': features,
        'label_encoders': label_encoders,
        'feature_importance': feature_importance,
        'categorical_cols': categorical_cols,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'target_description': 'Predicts if NEXT page will be 4 or 5 (conversion)'
    }, "model/model.pkl")
    
    print("\n✅ Model trained and saved at model/model.pkl")
    print("ℹ️ Model predicts: Will NEXT page be page 4 or 5?")
    
    return model

if __name__ == "__main__":
    train_model()
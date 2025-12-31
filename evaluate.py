import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score
import joblib
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

def evaluate_model():
    # DATASET LOAD KARNA
    print("📂 Loading dataset...")
    df = pd.read_csv("data/e-shop clothing 2008.csv", sep=";")
    
    # CONVERSION COLUMN BANANA
    df["conversion"] = df["page"].apply(lambda x: 1 if x >= 4 else 0)
    
    # MODEL LOAD KARNA
    print("🤖 Loading trained model...")
    saved_data = joblib.load("model/model.pkl")
    model = saved_data['model']
    features = saved_data['features']
    label_encoders = saved_data.get('label_encoders', {})
    
    # Prepare test data - COPY bana lo
    X = df[features].copy()
    y = df["conversion"]
    
    # Apply same preprocessing as training
    for col, le in label_encoders.items():
        if col in X.columns:
            # Handle unseen labels using .loc
            def process_value(x):
                str_val = str(x)
                return str_val if str_val in le.classes_ else 'unknown'
            
            X.loc[:, col] = X[col].apply(process_value)
            
            # Add 'unknown' to classes if not present
            if 'unknown' not in le.classes_:
                le.classes_ = np.append(le.classes_, 'unknown')
            
            X.loc[:, col] = le.transform(X[col].astype(str))
    
    # PREDICTION KARNA
    print("🔮 Making predictions...")
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]  # Probability for class 1
    
    # EVALUATION METRICS
    print("\n📊" + "="*40 + " MODEL EVALUATION " + "="*40)
    print(f"✅ Accuracy: {accuracy_score(y, y_pred):.4f}")
    print(f"🎯 Precision: {precision_score(y, y_pred):.4f}")
    print(f"📈 Recall: {recall_score(y, y_pred):.4f}")
    print(f"⚖️ F1-Score: {f1_score(y, y_pred):.4f}")
    
    # ROC-AUC Score
    try:
        roc_auc = roc_auc_score(y, y_pred_proba)
        print(f"📊 ROC-AUC Score: {roc_auc:.4f}")
    except:
        print("📊 ROC-AUC Score: Could not calculate")
    
    # Confusion Matrix
    print("\n📋 Confusion Matrix:")
    cm = confusion_matrix(y, y_pred)
    print(f"True Negatives:  {cm[0,0]:6d} | False Positives: {cm[0,1]:6d}")
    print(f"False Negatives: {cm[1,0]:6d} | True Positives:  {cm[1,1]:6d}")
    
    # Calculate rates
    total = cm.sum()
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\n📈 Rates:")
    print(f"False Positive Rate: {fp/(fp+tn):.4f}")
    print(f"False Negative Rate: {fn/(fn+tp):.4f}")
    print(f"True Positive Rate (Recall): {tp/(tp+fn):.4f}")
    print(f"True Negative Rate: {tn/(tn+fp):.4f}")
    
    # Classification Report
    print("\n📄 Detailed Classification Report:")
    print(classification_report(y, y_pred, target_names=['No Conversion', 'Conversion']))
    
    # Feature Importance
    if 'feature_importance' in saved_data:
        print("\n🎯 Top 10 Important Features:")
        print(saved_data['feature_importance'].head(10).to_string())
    
    # Actual vs Predicted comparison
    results_df = pd.DataFrame({
        'Actual': y,
        'Predicted': y_pred,
        'Probability_Conversion': y_pred_proba
    })
    
    # Sample predictions
    print("\n🔍 Sample Predictions (First 10 rows):")
    sample_df = results_df.head(10).copy()
    sample_df['Prediction_Status'] = sample_df.apply(
        lambda row: '✅ Correct' if row['Actual'] == row['Predicted'] else '❌ Wrong', 
        axis=1
    )
    print(sample_df[['Actual', 'Predicted', 'Probability_Conversion', 'Prediction_Status']])
    
    # Performance by actual class
    print("\n📊 Performance by Actual Class:")
    correct_predictions = results_df[results_df['Actual'] == results_df['Predicted']]
    conversion_correct = len(correct_predictions[correct_predictions['Actual'] == 1])
    no_conversion_correct = len(correct_predictions[correct_predictions['Actual'] == 0])
    
    total_conversion = len(results_df[results_df['Actual'] == 1])
    total_no_conversion = len(results_df[results_df['Actual'] == 0])
    
    print(f"Conversion Accuracy: {conversion_correct}/{total_conversion} = {conversion_correct/total_conversion:.4f}")
    print(f"No Conversion Accuracy: {no_conversion_correct}/{total_no_conversion} = {no_conversion_correct/total_no_conversion:.4f}")
    
    # Calculate error rate
    error_rate = 1 - accuracy_score(y, y_pred)
    print(f"\n⚠️ Overall Error Rate: {error_rate:.4f}")
    
    # Save predictions to CSV for further analysis
    results_df.to_csv("model/predictions_evaluation.csv", index=False)
    print("\n💾 Predictions saved to: model/predictions_evaluation.csv")
    
    return results_df, cm

if __name__ == "__main__":
    evaluate_model()
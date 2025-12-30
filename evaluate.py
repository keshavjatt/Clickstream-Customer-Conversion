import pandas as pd
from sklearn.metrics import accuracy_score
import joblib

# DATASET LOAD KARNA
# CSV file ko read karo (semicolon separated hai)
df = pd.read_csv("data/e-shop clothing 2008.csv", sep=";")

# CONVERSION COLUMN BANANA
# Rule: Agar page 4 ya 5 tak gaya to conversion = 1, nahi to 0
df["conversion"] = df["page"].apply(lambda x: 1 if x >= 4 else 0)


# Model ke liye input features
X = df[["price", "price 2", "page"]]
# Model ka predict karna target
y = df["conversion"]

# TRAINED MODEL LOAD KARNA
model = joblib.load("model/model.pkl")

# PREDICTION KARNA
# Pure dataset par model chalakar predictions lena
y_pred = model.predict(X)

# ACCURACY CALCULATE KARNA
print("✅ Accuracy:", accuracy_score(y, y_pred))
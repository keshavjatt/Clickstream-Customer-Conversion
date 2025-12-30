import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# DATASET LOAD KARNA
df = pd.read_csv("data/e-shop clothing 2008.csv", sep=";")


# User agar checkout/purchase page tak pahucha
df["conversion"] = df["page"].apply(lambda x: 1 if x >= 4 else 0)

# 🎪 FEATURES AUR TARGET SETUP KARNA
X = df[["price", "price 2", "page"]]  # Input features
y = df["conversion"]                  # Output target (0 ya 1)

# DATA KO TRAIN AUR TEST MEIN BATANA
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42  # 42 = same random results har baar
)

# MODEL TRAIN KARNA
model = RandomForestClassifier(random_state=42)  # Random Forest algorithm
model.fit(X_train, y_train)  # Model ko data sikhana

# "model" folder agar nahi hai to banado
os.makedirs("model", exist_ok=True)

# Train kiya hua model file mein save karo
joblib.dump(model, "model/model.pkl")

print("✅ Model trained and saved at model/model.pkl")
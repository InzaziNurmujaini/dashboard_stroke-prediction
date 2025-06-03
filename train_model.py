import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import LabelEncoder

# Load data
df = pd.read_csv("healthcare-dataset-stroke-data.csv")
df.dropna(inplace=True)
df.drop(columns=["id"], inplace=True)

# Encode kolom kategorikal
categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Split fitur dan target
X = df.drop(columns=["stroke"])
y = df["stroke"]

# Seleksi fitur dengan Chi-Square
selector = SelectKBest(score_func=chi2, k=8)
X_selected = selector.fit_transform(X, y)

# Buat dua model
rf_model = RandomForestClassifier(random_state=42)
logreg_model = LogisticRegression(class_weight="balanced", max_iter=1000)

# Fit kedua model
rf_model.fit(X_selected, y)
logreg_model.fit(X_selected, y)

# Simpan ke folder models/
os.makedirs("models", exist_ok=True)

# Simpan Random Forest
with open("models/rf_model.pkl", "wb") as f:
    pickle.dump(rf_model, f)

# Simpan Logistic Regression
with open("models/logreg_model.pkl", "wb") as f:
    pickle.dump(logreg_model, f)

# Simpan selektor chi-square dan encoders
with open("models/chi_selector.pkl", "wb") as f:
    pickle.dump(selector, f)

with open("models/label_encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)

print("Model RF & LogReg + preprocessing berhasil disimpan.")

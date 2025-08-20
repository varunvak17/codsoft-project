# movie_rating_prediction.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle

# ================================
# 1. Load dataset
# ================================
df = pd.read_csv("C:/Users/nanth/Downloads/archive (1)/IMDb Movies India.csv", encoding="latin1")

# ================================
# 2. Data preprocessing
# ================================

# Handle missing values
df = df.dropna(subset=['Rating'])

# Convert Duration to numeric (remove 'min')
df['Duration'] = df['Duration'].astype(str).str.extract(r'(\d+)').astype(float)

# Clean Votes (remove commas, ignore values with $ or M if present)
df['Votes'] = df['Votes'].astype(str).str.replace(",", "")
df['Votes'] = pd.to_numeric(df['Votes'], errors="coerce")

# Fill NaNs with median
df['Duration'] = df['Duration'].fillna(df['Duration'].median())
df['Votes'] = df['Votes'].fillna(df['Votes'].median())

# Encode categorical features
label_encoders = {}
for col in ['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']:
    le = LabelEncoder()
    df[col] = df[col].astype(str)  # Ensure string type
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# ================================
# 3. Train-test split
# ================================
X = df[['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3', 'Duration', 'Votes']]
y = df['Rating']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ================================
# 4. Train Model
# ================================
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ================================
# 5. Predictions & Evaluation
# ================================
y_pred = model.predict(X_test)

print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Root Mean Squared Error:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R² Score:", r2_score(y_test, y_pred))

# ================================
# 6. Visualization
# ================================
plt.figure(figsize=(6,6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
plt.xlabel("True Ratings")
plt.ylabel("Predicted Ratings")
plt.title("True vs Predicted Movie Ratings")
plt.show()

# ================================
# 7. Save model for reuse
# ================================
with open("movie_rating_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved as movie_rating_model.pkl")

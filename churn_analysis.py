# ==============================
# 📌 IMPORT LIBRARIES
# ==============================
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    accuracy_score, f1_score, recall_score,
    classification_report, roc_auc_score,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay
)

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

import xgboost as xgb
import shap

# ==============================
#  LOAD DATA
# ==============================
df = pd.read_excel("data/Telco_customer_churn1.xlsx")

# ==============================
#  DATA CLEANING
# ==============================
df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")

drop_cols = [
    "CustomerID","Count","Country","State","City","Zip Code",
    "Lat Long","Latitude","Longitude","Churn Reason",
    "Churn Score","Churn Label"
]
df = df.drop(columns=drop_cols, errors="ignore")

# ==============================
#  ENCODING
# ==============================
binary_map = {"Yes": 1, "No": 0}

for col in ["Gender", "Partner", "Dependents", "Phone Service", "Paperless Billing"]:
    if col in df.columns:
        df[col] = df[col].map(binary_map)

if "Senior Citizen" in df.columns:
    df["Senior Citizen"] = df["Senior Citizen"].map({"Yes":1, "No":0})

df = pd.get_dummies(df, drop_first=True)

# ==============================
#  FEATURES & TARGET
# ==============================
X = df.drop("Churn Value", axis=1)
y = df["Churn Value"]

# ==============================
#  TRAIN TEST SPLIT
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==============================
#  PREPROCESSING
# ==============================
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numeric_features)
    ],
    remainder="passthrough"
)

# ==============================
#  MODEL PIPELINE
# ==============================
xgb_model = xgb.XGBClassifier(
    random_state=42,
    scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]),
    use_label_encoder=False,
    eval_metric='logloss'
)

pipeline = ImbPipeline([
    ("preprocessor", preprocessor),
    ("smote", SMOTE(random_state=42)),
    ("classifier", xgb_model)
])

# ==============================
#  HYPERPARAMETER TUNING
# ==============================
param_grid = {
    "classifier__n_estimators": [100, 300],
    "classifier__max_depth": [4, 6],
    "classifier__learning_rate": [0.05, 0.1]
}

grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=3,
    scoring="f1",
    n_jobs=-1
)

grid.fit(X_train, y_train)

best_model = grid.best_estimator_

print("Best Parameters:", grid.best_params_)

# ==============================
#  CROSS VALIDATION
# ==============================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for train_idx, val_idx in cv.split(X, y):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    best_model.fit(X_tr, y_tr)
    preds = best_model.predict(X_val)
    cv_scores.append(f1_score(y_val, preds))

print("CV F1 Score:", np.mean(cv_scores))

# ==============================
#  EVALUATION
# ==============================
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

print("\n Model Performance")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))

print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ==============================
#  GRAPH 1: CONFUSION MATRIX
# ==============================
ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test)
plt.title("Confusion Matrix")
plt.show()

# ==============================
#  GRAPH 2: ROC CURVE
# ==============================
RocCurveDisplay.from_estimator(best_model, X_test, y_test)
plt.title("ROC Curve")
plt.show()

# ==============================
#  GRAPH 3: PRECISION-RECALL
# ==============================
PrecisionRecallDisplay.from_estimator(best_model, X_test, y_test)
plt.title("Precision-Recall Curve")
plt.show()

# ==============================
#  GRAPH 4: FEATURE IMPORTANCE
# ==============================
feature_names = best_model.named_steps["preprocessor"].get_feature_names_out()

importance = best_model.named_steps["classifier"].feature_importances_

feat_imp = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
}).sort_values(by="Importance", ascending=False)

feat_imp.head(10).plot(kind="barh", x="Feature", y="Importance")
plt.title("Top 10 Feature Importance")
plt.gca().invert_yaxis()
plt.show()

# ==============================
#  GRAPH 5: SHAP
# ==============================
X_test_transformed = best_model.named_steps["preprocessor"].transform(X_test)

explainer = shap.TreeExplainer(best_model.named_steps["classifier"])
shap_values = explainer.shap_values(X_test_transformed)

shap.summary_plot(
    shap_values,
    X_test_transformed,
    feature_names=feature_names,
    max_display=10
)

# ==============================
#  SAVE MODEL
# ==============================
joblib.dump(best_model, "churn_model.pkl")

print("\n Model saved successfully!")




print(X.columns)



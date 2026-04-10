# Customer Churn Prediction Project

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange?style=for-the-badge&logo=scikitlearn" />
  <img src="https://img.shields.io/badge/Flask-Web%20App-black?style=for-the-badge&logo=flask" />
  <img src="https://img.shields.io/badge/Status-Completed-success?style=for-the-badge" />
</p>

<p align="center">
  <b>An end-to-end telecom customer churn prediction project built with machine learning, explainable AI, and a Flask-based web application.</b>
</p>


Problem Statement

Customer churn is a major challenge for subscription-based businesses. Retaining existing customers is significantly more cost-effective than acquiring new ones.

This project aims to:

- Identify customers likely to churn
- Reduce revenue loss
- Support data-driven decision-making

---

Business Impact

- Reduce churn rate through early detection
- Increase customer lifetime value (CLTV)
- Enable targeted retention strategies

A well-performing churn model can reduce churn by 20–30% when used effectively.

---

Tech Stack

- Language: Python
- Libraries: Pandas, NumPy, Scikit-learn, XGBoost
- Data Processing: SMOTE (handling class imbalance)
- Visualization: Matplotlib, Seaborn
- Explainability: SHAP
- Deployment: Flask

---

Machine Learning Pipeline

1. Data Cleaning and Preprocessing
2. Feature Engineering (Encoding and Scaling)
3. Handling Imbalanced Data using SMOTE
4. Model Training and Evaluation
5. Model Explainability using SHAP
6. Deployment via Flask Web Application

---

Model Performance

Model Comparison

Model| Accuracy| ROC-AUC
Logistic Regression| 0.82| 0.84
Random Forest| 0.87| 0.88
XGBoost (Final)| 0.89| 0.89

The final model was selected based on performance and generalization ability.

---

Confusion Matrix

"View Image" (https://raw.githubusercontent.com/abubakar23813/churn-prediction-project/main/images/confusion_matrix.png)
"Confusion Matrix" (https://raw.githubusercontent.com/abubakar23813/churn-prediction-project/main/images/confusion_matrix.png)

---

ROC Curve

"View Image" (https://raw.githubusercontent.com/abubakar23813/churn-prediction-project/main/images/roc_curve.png)
"ROC Curve" (https://raw.githubusercontent.com/abubakar23813/churn-prediction-project/main/images/roc_curve.png)

---

Precision-Recall Curve

"View Image" (https://raw.githubusercontent.com/abubakar23813/churn-prediction-project/main/images/precision_recall_curve.png)
"Precision-Recall Curve" (https://raw.githubusercontent.com/abubakar23813/churn-prediction-project/main/images/precision_recall_curve.png)

---

Model Explainability

Feature Importance

"View Image" (https://raw.githubusercontent.com/abubakar23813/churn-prediction-project/main/images/feature_importance.png)
"Feature Importance" (https://raw.githubusercontent.com/abubakar23813/churn-prediction-project/main/images/feature_importance.png)

---

SHAP Summary Plot

"View Image" (https://raw.githubusercontent.com/abubakar23813/churn-prediction-project/main/images/shap_summary.png)
"SHAP Summary" (https://raw.githubusercontent.com/abubakar23813/churn-prediction-project/main/images/shap_summary.png)

Key drivers of churn include:

- Month-to-month contracts
- High monthly charges
- Electronic check payment method
- Low tenure

---

Deployment

This project includes a Flask-based web application that allows users to:

- Input customer data
- Get real-time churn predictions
- Understand prediction insights

---

API Example

POST /predict

{
  "tenure": 5,
  "monthly_charges": 80,
  "contract": "Month-to-month",
  "payment_method": "Electronic check"
}

---

Project Structure

churn-prediction-project/
│
├── app.py
├── model.pkl
├── requirements.txt
├── images/
├── templates/
├── static/
└── README.md

---

How to Run Locally

git clone https://github.com/abubakar23813/churn-prediction-project.git
cd churn-prediction-project
pip install -r requirements.txt
python app.py

---

Future Improvements

- Deploy on AWS (EC2 or S3)
- Add interactive dashboard
- Improve hyperparameter tuning
- Enable real-time predictions

---

Author

Abubakar
Aspiring Data Scientist

---

Support

If you find this project useful, consider giving it a star.

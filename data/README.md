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

---

## Overview

Customer churn is one of the most important business challenges in the telecom industry. When customers leave a service, it directly impacts recurring revenue, growth, and long-term customer value. This project focuses on predicting whether a customer is likely to churn based on customer demographics, account information, service subscriptions, billing behavior, and usage-related features.

The project follows a complete machine learning workflow, including data preprocessing, feature engineering, model training, evaluation, explainability, and deployment readiness. It is built as both a practical business solution and a strong portfolio project.

---

## Business Objective

The core objective of this project is to identify customers who are at high risk of churn so that businesses can take proactive retention actions before the customer leaves.

Using machine learning, this system helps transform raw customer data into predictive insights that can support business decisions such as retention campaigns, service improvement, pricing optimization, and targeted customer engagement.

---

## Project Highlights

- Built a machine learning model for telecom customer churn prediction
- Performed preprocessing, encoding, and feature transformation
- Evaluated the model using ROC Curve, Precision-Recall Curve, and Confusion Matrix
- Applied feature importance analysis to identify key churn drivers
- Used SHAP values to improve model interpretability
- Integrated the prediction pipeline into a Flask web application
- Structured the project for deployment and GitHub presentation

---

## Tech Stack

- **Programming Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, SHAP
- **Framework:** Flask
- **Model Type:** Classification
- **Deployment Style:** Web-based prediction interface
- **Version Control:** Git & GitHub

---

## Workflow

1. Data collection and understanding  
2. Data cleaning and preprocessing  
3. Exploratory data analysis  
4. Feature engineering and encoding  
5. Model training and evaluation  
6. Feature importance analysis  
7. SHAP-based model explainability  
8. Flask web app integration  

---

## Dataset Features

The model uses customer-related features such as:

- Gender
- Senior Citizen
- Partner
- Dependents
- Tenure Months
- Phone Service
- Multiple Lines
- Internet Service
- Online Security
- Online Backup
- Device Protection
- Tech Support
- Streaming TV
- Streaming Movies
- Contract
- Paperless Billing
- Payment Method
- Monthly Charges
- Total Charges
- CLTV

These variables help the model capture behavioral, financial, and service-related patterns associated with customer churn.

---

## Model Performance

The model shows strong predictive performance for churn classification. Based on the uploaded evaluation visuals, the ROC-AUC score is **0.89**, while the Average Precision score is **0.72**, indicating that the model performs well in separating churning and non-churning customers while maintaining useful precision-recall balance [file:37][file:36].

### ROC Curve
"ROC Curve" (https://github.com/abubakar23813/churn-prediction-project/blob/main/images/ROC%20Curve.png)
The ROC Curve measures the model’s ability to distinguish between churn and non-churn classes across different classification thresholds. The uploaded graph reports an AUC of **0.89**, which reflects strong overall discrimination performance [file:37].

### Precision-Recall Curve
"PR Curve" (https://github.com/abubakar23813/churn-prediction-project/blob/main/images/Precision_Recall%20Curve.png)
The Precision-Recall Curve is especially useful for churn prediction because such datasets are commonly imbalanced. The uploaded graph reports an Average Precision of **0.72**, which shows that the model maintains good precision as recall increases [file:36].


### Confusion Matrix
"Confusion Matrix" (https://raw.githubusercontent.com/abubakar23813/churn-prediction-project/main/images/confusion_matrix.png)
The confusion matrix gives a direct overview of prediction outcomes. According to the uploaded matrix, the model produced **744 true negatives**, **291 false positives**, **36 false negatives**, and **338 true positives**, which shows that the classifier is effective at detecting churn while still making some trade-offs between sensitivity and false alarms [file:35].

---

## Feature Importance
"Feature Importance" (https://github.com/abubakar23813/churn-prediction-project/blob/main/images/Top%2010%20Feature%20Importance.png)
Feature importance analysis reveals which variables contribute most to the model’s predictions. In the uploaded chart, **Contract_Two year** is the most influential feature, followed by **Contract_One year**, **Payment Method_Electronic check**, and multiple service-related variables [file:33].

## Model Explainability with SHAP
"SHAP" (https://github.com/abubakar23813/churn-prediction-project/blob/main/images/Shap%20Value.png)
To make the model more transparent, SHAP analysis was used to understand how features influence predictions. The uploaded SHAP summary plot shows both feature importance and the direction of impact, making it easier to explain why the model predicts churn for specific customer profiles [file:34].


## Key Business Insights

The feature importance and SHAP analysis suggest that contract-related variables are among the strongest churn drivers. Long-term contracts such as one-year and two-year plans appear highly influential in reducing churn, while payment method and service behavior also play a major role in prediction outcomes [file:33][file:34].

Other strong signals include tenure, dependents, internet service type, streaming-related features, paperless billing, and online security. These insights indicate that effective churn reduction strategies should focus on contract retention, payment trust, bundled services, and customer support quality [file:33][file:34].

---

## Web Application

This project also includes a Flask-based web application where users can input customer details and generate churn predictions. The current app structure can be extended further by adding probability outputs, business recommendations, and customer-level explainability features.

---

## Project Structure

```bash
CHURN-PREDICTION-PROJECT/
├── data/
│   ├── Telco_customer_churn1.xlsx   # Raw dataset
│   ├── README.md                    # Data documentation
│   └── requirments.txt              # Project dependencies
├── images/                          # Visual Evaluation Metrics
│   ├── confusion_matrix.png
│   ├── Precision_Recall Curve.png
│   ├── ROC Curve.png
│   ├── Shap Value.png
│   └── Top 10 Feature Importance.png
├── templates/
│   └── index.html                   # Flask Web Frontend
├── app.py                           # Application Server
├── churn_analysis.py                # Analysis & Training Pipeline
├── churn_model.pkl                  # Serialized Model
└── columns.pkl                      # Feature Metadata

Installation & Usage
1.Clone the repo:  git clone https://github.com/yourusername/CHURN-PREDICTION-PROJECT.git
cd CHURN-PREDICTION-PROJECT
2.Install requirements:  pip install -r data/requirments.txt
3.Launch the Application: python app.py

Access the app at http://127.0.0.1:5000

Tech Stack
Language: Python 3.9+
Libraries: Pandas, Scikit-Learn, SHAP, Matplotlib, Seaborn
Deployment: Flask, Jinja2 (HTML Templates)
Storage: Pickle (Model Serialization)
✍️ Author
[ARHAM QURAISHI]
Data Science Enthusiast | B.Com Student
Note: This project is part of my Data Science portfolio focusing on business analytics and predictive modeling.

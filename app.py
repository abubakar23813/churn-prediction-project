from flask import Flask, render_template, request
import pickle
import pandas as pd
import joblib

app = Flask(__name__)

# Load model
model = joblib.load(open("churn_model.pkl", 'rb'))

# Load trained columns
model_columns = pickle.load(open('columns.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = request.form.to_dict()

        # Convert to DataFrame
        df = pd.DataFrame([data])

        #  Convert numeric
        df['Tenure'] = df['Tenure'].astype(float)
        df['MonthlyCharges'] = df['MonthlyCharges'].astype(float)
        df['TotalCharges'] = df['TotalCharges'].astype(float)
        df['CLTV'] = df['CLTV'].astype(float)

        #  Convert binary
        binary_cols = ['Gender','SeniorCitizen','Partner','Dependents','PhoneService','PaperlessBilling']
        for col in binary_cols:
            df[col] = df[col].astype(int)

        #  One-hot encoding
        df = pd.get_dummies(df)

        #  Add missing columns
        for col in model_columns:
            if col not in df.columns:
                df[col] = 0

        #  Ensure same order
        df = df[model_columns]
        print(df.shape)
        print(type(model))

        # Fill any NA
        df = df.fillna(0)

        # Prediction
        prediction = model.predict(df)[0]

        if prediction == 1:
            result = " High Chance of Churn"
        else:
            result = " Customer Safe"

        return render_template('index.html', prediction_text=result)

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")


if __name__ == "__main__":
    app.run(debug=True)
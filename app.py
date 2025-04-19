from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load("xgb_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input
        jun_sep = float(request.form['jun_sep'])
        curve_number = float(request.form['curve_number'])
        retention = float(request.form['retention'])
        runoff = float(request.form['runoff'])

        # Create input DataFrame
        input_df = pd.DataFrame([[jun_sep, curve_number, retention, runoff]],
                                columns=['Jun-Sep', 'Curve Number', 'Potential Maximum Retention', 'Surface Runoff'])

        # Scale input
        input_scaled = scaler.transform(input_df)

        # Make prediction
        prob = model.predict_proba(input_scaled)[:, 1][0]
        is_flood = prob > 0.5
        prediction = "🚨 Flood Likely" if is_flood else "✅ No Flood Expected"

        return render_template('result.html',
                               prediction=prediction,
                               probability=f"{prob:.2f}",
                               values=input_df.to_dict(orient='records')[0],
                               flood=is_flood)
    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    app.run(debug=True)
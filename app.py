from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model properly
with open('house_price_prediction.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Corrected to request.form, removed MEDV from input
    features = [
        float(request.form['CRIM']),
        float(request.form['ZN']),
        float(request.form['INDUS']),
        float(request.form['CHAS']),
        float(request.form['NOX']),
        float(request.form['RM']),
        float(request.form['AGE']),
        float(request.form['DIS']),
        float(request.form['RAD']),
        float(request.form['TAX']),
        float(request.form['PTRATION']),
        float(request.form['BLACK']),
        float(request.form['LSTAT']),
        float(request.form['MEDV'])
    ]

    features_array = np.array([features])

    prediction = model.predict(features_array)
    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text=f'Predicted price: ${output}k')

if __name__ == "__main__":
    app.run(debug=True)

from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)
model = pickle.load(open('airline_pipeline.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get all values from the form
        f = request.form
        
        # Manually constructing the list to ensure the order is EXACT
        # This order must match X_train.columns from your Colab
        data = [
            float(f['Age']),
            float(f['Distance']),
            float(f['wifi']),
            float(f['time']),
            float(f['booking']),
            float(f['gate']),
            float(f['food']),
            float(f['boarding']),
            float(f['seat']),
            float(f['ent']),
            float(f['onboard']),
            float(f['legroom']),
            float(f['baggage']),
            float(f['checkin']),
            float(f['service']),
            float(f['clean']),
            float(f['dep_delay']),
            float(f['arr_delay']),
            # Binary/Encoded columns
            1.0 if f['Gender'] == 'Female' else 0.0,
            1.0 if f['Gender'] == 'Male' else 0.0,
            1.0 if f['CustType'] == 'Loyal' else 0.0,
            1.0 if f['CustType'] == 'disloyal' else 0.0,
            1.0 if f['TravelType'] == 'Business' else 0.0,
            1.0 if f['TravelType'] == 'Personal' else 0.0,
            1.0 if f['Class'] == 'Business' else 0.0  # Example 25th column
        ]

        # Convert to DataFrame because Pipeline expects column names
        # Replace these strings with your actual X_train.columns names
        cols = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else None
        input_df = pd.DataFrame([data], columns=cols)
        
        prediction = model.predict(input_df)
        result = "Satisfied" if prediction[0] == 1 else "Neutral or Dissatisfied"
        
        return render_template('index.html', prediction_text=f'Result: {result}')
    
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
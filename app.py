from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import os
import pickle

app = Flask(__name__)

# Load or train the KMeans model
model_file = 'model.pkl'

if os.path.exists(model_file):
    with open(model_file, 'rb') as f:
        kmeans = pickle.load(f)
else:
    np.random.seed(42)
    data = {
        'AnnualIncome': np.random.randint(0, 100000, 100),
        'SpendingScore': np.random.randint(0, 100, 100)
    }
    df = pd.DataFrame(data)
    X = df.values
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X)
    
    # Save the trained model
    with open(model_file, 'wb') as f:
        pickle.dump(kmeans, f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        print("Form submitted")
        # Get user input from the form
        age = request.form['age']
        income = request.form['income']
        spending_score = request.form['spending_score']
        
        # Validate input (ensure numeric values)
        if not (age.isdigit() and income.isdigit() and spending_score.isdigit()):
            return "Error: Please enter numeric values for age, income, and spending score."
        
        # Convert input to float
        age = float(age)
        income = float(income)
        spending_score = float(spending_score)
        
        # Make prediction using the pre-fitted machine learning model
        user_cluster = kmeans.predict([[income, spending_score]])  # Including both income and spending score
        
        # Store the response in an Excel sheet
        response_df = pd.DataFrame({'Age': [age], 'AnnualIncome': [income], 'SpendingScore': [spending_score], 'Cluster': user_cluster})
        
        # Save the response to Excel
        try:
            response_df.to_excel('user_response.xlsx', index=False, header=not os.path.exists('user_response.xlsx'))
        except Exception as e:
            print(f"Error saving response to Excel: {e}")
        
        return render_template('result.html', cluster=user_cluster[0])


if __name__ == "__main__":
    app.run(debug=True)

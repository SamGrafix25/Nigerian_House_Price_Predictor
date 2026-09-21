from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the saved model and column structure
with open('house_price_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('model_columns.pkl', 'rb') as f:
    model_columns = pickle.load(f)


states = [col.replace('state_', '') for col in model_columns if col.startswith('state_')]
titles = [col.replace('title_', '') for col in model_columns if col.startswith('title_')]
towns = [col.replace('town_grouped_', '') for col in model_columns if col.startswith('town_grouped_')]
towns = sorted([t for t in towns if t != 'Other'])
towns.append('Other')  

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    form_data = {}

    if request.method == 'POST':
        form_data = request.form
        # Get form inputs
        bedrooms = float(request.form['bedrooms'])
        bathrooms = float(request.form['bathrooms'])
        toilets = float(request.form['toilets'])
        parking_space = float(request.form['parking_space'])
        title = request.form['title']
        state = request.form['state']
        town = request.form['town']
        
        # Build a single-row dataframe matching training data structure
        input_data = pd.DataFrame([[bedrooms, bathrooms, toilets, parking_space]], 
                                    columns=['bedrooms', 'bathrooms', 'toilets', 'parking_space'])
        
        # Add one-hot encoded columns, all starting at 0
        for col in model_columns:
            if col not in input_data.columns:
                input_data[col] = 0
        
        # Set the correct dummy column to 1 based on user's title/state choice
        title_col = f'title_{title}'
        state_col = f'state_{state}'
        town_col = f'town_grouped_{town}'
        if title_col in input_data.columns:
            input_data[title_col] = 1
        if state_col in input_data.columns:
            input_data[state_col] = 1
        if town_col in input_data.columns:
            input_data[town_col] = 1
        
        # Reorder columns to match training exactly
        input_data = input_data[model_columns]
        
        # Predict (remember: model was trained on log(price), so convert back)
        log_pred = model.predict(input_data)[0]
        prediction = round(np.exp(log_pred))
    
    return render_template('index.html', prediction=prediction, states=states, titles=titles, towns=towns, form_data=form_data)

if __name__ == '__main__':
    app.run(debug=True)
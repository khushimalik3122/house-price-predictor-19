from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    location = request.form['location']
    sqft = float(request.form['sqft'])
    bath = int(request.form['bath'])
    bhk = int(request.form['bhk'])

    # Dummy example - replace with your actual model input logic
    input_features = np.array([[sqft, bath, bhk]])  
    prediction = model.predict(input_features)[0]

    return render_template('index.html', prediction_text=f'Predicted Price: ₹ {round(prediction, 2)} Lakhs')

if __name__ == "__main__":
    app.run(debug=True)




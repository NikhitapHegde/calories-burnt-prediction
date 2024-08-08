from flask import Flask, request, jsonify
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor  # Assuming RandomForest is chosen

# Create a trained RandomForestRegressor object (replace with your training logic)
model = RandomForestRegressor()

app = Flask(__name__)

scaler = StandardScaler()  # Create a global scaler object

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from the form data
    age = float(request.form.get('age'))
    gender = request.form.get('gender')  # Assuming a dropdown for gender
    height = float(request.form.get('height'))
    heartrate = int(request.form.get('heartrate'))
    bodytemp = float(request.form.get('bodytemp'))

    # Preprocess user input (replace with your actual logic)
    user_data = scaler.transform([[age, height, heartrate, bodytemp]])[0]

    # Make prediction using the chosen model (RandomForestRegressor)
    predicted_calories = model.predict(user_data.reshape(1, -1))[0]

    # Return the predicted calorie expenditure in JSON format
    return jsonify({'predicted_calories': predicted_calories})

if __name__ == '__main__':
    app.run(debug=True)

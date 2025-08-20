from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load model
model = joblib.load("model/california_housing_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Extract features from form
        features = [float(x) for x in request.form.values()]
        features_array = np.array(features).reshape(1, -1)

        # Prediction
        prediction = model.predict(features_array)[0]

        return render_template("index.html", prediction_text=f"Predicted House Price: ${prediction:,.2f}")
    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {e}")

if __name__ == "__main__":
    app.run(debug=True)

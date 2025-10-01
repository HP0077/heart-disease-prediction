from flask import Flask, render_template, request, jsonify, url_for
import pickle
import numpy as np
import os

app = Flask(__name__)

# Loading the trained model 
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
try:
    model = pickle.load(open(MODEL_PATH, "rb"))
except Exception as e:
    print("Error loading model:", e)
    model = None  # Prevent crash if model is missing

@app.route("/")
def home():
    """Render the landing page(Front page ke liye)"""
    return render_template("index.html")  

@app.route("/predict-page")
def predict_page():
    """Render the prediction page(Main page ke liye)"""
    return render_template("home.html")  

@app.route("/predict", methods=["POST"])
def predict():
    """Process user input and return prediction result"""
    try:
        data = request.get_json() if request.is_json else request.form.to_dict()

        if not data:
            return jsonify({"error": "No data received"}), 400

        # Here we will convert the input data to numpy array
        input_features = np.array([[float(data.get(k, 0)) for k in [
            "male", "age", "education", "currentSmoker", "cigsPerDay",
            "BPMeds", "prevalentStroke", "prevalentHyp", "diabetes",
            "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"
        ]]])

        if model is None:
            return jsonify({"error": "Model not loaded. Check server logs."}), 500

        prediction = model.predict(input_features)[0]
        result = "High risk" if prediction == 1 else "Low risk"

        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)





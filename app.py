import warnings
import os
import joblib
from flask import Flask, request, render_template

# Ignore sklearn version warnings
warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)

# --- FILE PATH CONFIGURATION ---
# This ensures the app finds your models regardless of where you run it
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "marks_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

# --- LOAD AI MODELS ---
try:
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print("✅ Umang's AI Models Loaded Successfully")
    else:
        print("❌ ERROR: Model or Scaler file not found in the directory!")
except Exception as e:
    print(f"❌ ERROR LOADING MODELS: {e}")


# --- ROUTES ---

@app.route("/", methods=["GET"])
def home():
    """Renders the main input dashboard."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Processes input data and returns a personalized report."""
    try:
        # 1. Retrieve data from the form
        name = request.form.get("name", "Student")
        attendence = float(request.form.get("attendence", 0))
        hours = float(request.form.get("hours", 0))
        marks = float(request.form.get("marks", 0))
        sleep = float(request.form.get("sleep", 0))

        # 2. Prepare data for the model 
        # (Note: keeping your attendence/2 logic as requested)
        processed_attendance = attendence / 2
        input_data = [[processed_attendance, hours, marks, sleep]]
        
        # 3. Scale and Predict
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        
        # 4. Final Formatting (Clamping between 0-100)
        final_score = round(max(0, min(prediction, 100)), 2)

        # 5. Render the Results Page with Umang's Suggestion Logic
        return render_template(
            "predict.html",
            name=name,
            prediction_text=f"{final_score}%",
            attendence=attendence,
            hours=hours,
            marks=marks,
            sleep=sleep
        )

    except Exception as e:
        # Error handling to prevent the app from stopping
        print(f"Prediction Error: {e}")
        return f"<h3>Oops! Something went wrong.</h3><p>{str(e)}</p><a href='/'>Go Back</a>"


if __name__ == "__main__":
    # debug=True allows for live updates and shows errors in the browser
    app.run(debug=True)
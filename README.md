# UMANG_GPT-Grade-Prediction-Tool-
Hey, I'm Umang Singh, I built Umang GPT to bridge the gap between lifestyle habits and academic results, As a developer, I focus on creating clean, efficient, and data-driven web applications.
# 🚀 Umang GPT: Grade Prediction Tool

**Umang GPT** is a professional-grade Machine Learning web application designed to help students forecast their academic performance. By analyzing lifestyle and academic metrics, the tool provides a predicted final score and actionable insights to improve performance.



---

## ✨ Features
* **AI-Powered Predictions:** Uses a trained Machine Learning model to calculate potential marks.
* **Smart Suggestions:** Compares user input against "Ideal Benchmarks" (8h sleep, 86% attendance, 12h study) and provides real-time advice.
* **Professional Dashboard:** Clean, responsive UI with a split-screen design.
* **Report Export:** Built-in functionality to download or print the analysis as a PDF report.
* **Developer Profile:** Integrated "About Me" section with social connectivity.

## 🛠️ Tech Stack
* **Frontend:** HTML5, CSS3 (Custom Gradients), Bootstrap 5.
* **Backend:** Python, Flask.
* **Machine Learning:** Scikit-Learn, Joblib (Random Forest/Linear Regression model).
* **Typography:** Google Fonts (Poppins).

## 📂 Project Structure
```text
/
├── app.py              # Main Flask application logic
├── marks_model.pkl      # Trained Machine Learning model
├── scaler.pkl           # Data scaling file for ML consistency
├── templates/
│   ├── index.html       # Home dashboard & Developer profile
│   └── predict.html     # Results & Suggestion engine
└── README.md            # Project documentation

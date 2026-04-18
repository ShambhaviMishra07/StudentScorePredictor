# ================================================
# PHASE 7 - Flask Web Application
# StudentPerformanceProject/app/app.py
# ================================================

from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# ----------------------------------------
# Load saved model and scaler
# ----------------------------------------
with open('../models/random_forest.pkl', 'rb') as f:
    model = pickle.load(f)

with open('../models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# ----------------------------------------
# Grade and remark logic
# ----------------------------------------
def get_grade(score):
    if score >= 85:   return 'A+'
    elif score >= 75: return 'A'
    elif score >= 63: return 'B'
    elif score >= 50: return 'C'
    elif score >= 40: return 'D'
    else:             return 'F'

def get_remark(score):
    if score >= 85:   return 'Outstanding! 🌟'
    elif score >= 75: return 'Excellent! 👏'
    elif score >= 63: return 'Good Performance 👍'
    elif score >= 50: return 'Average - Can Improve 📚'
    elif score >= 40: return 'Passed but Needs Effort ⚠️'
    else:             return 'Failed - Needs Serious Attention ❌'

def get_status(score):
    return 'PASS' if score >= 40 else 'FAIL'

# ----------------------------------------
# Routes
# ----------------------------------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get values from the form
        attendance     = float(request.form['attendance'])
        study_hours    = float(request.form['study_hours'])
        sleep_hours    = float(request.form['sleep_hours'])
        internal_marks = float(request.form['internal_marks'])
        assignments    = float(request.form['assignments'])

        # Prepare and scale input
        input_data   = np.array([[attendance, study_hours,
                                   sleep_hours, internal_marks,
                                   assignments]])
        input_scaled = scaler.transform(input_data)

        # Predict
        prediction = model.predict(input_scaled)[0]
        prediction = round(float(np.clip(prediction, 0, 100)), 2)

        grade  = get_grade(prediction)
        status = get_status(prediction)
        remark = get_remark(prediction)

        return render_template('index.html',
            prediction    = prediction,
            grade         = grade,
            status        = status,
            remark        = remark,
            attendance    = attendance,
            study_hours   = study_hours,
            sleep_hours   = sleep_hours,
            internal_marks= internal_marks,
            assignments   = assignments,
            show_result   = True
        )

    except Exception as e:
        return render_template('index.html',
            error = f"Something went wrong: {str(e)}",
            show_result = False
        )

if __name__ == '__main__':
    app.run(debug=True)
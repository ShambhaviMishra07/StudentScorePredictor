# ================================================
# PHASE 6 - Prediction Script
# StudentPerformanceProject/src/predict.py
# ================================================

import pickle
import numpy as np

print("=" * 50)
print("   🎓 STUDENT PERFORMANCE PREDICTION SYSTEM")
print("=" * 50)

# ----------------------------------------
# Load saved model and scaler
# ----------------------------------------
try:
    with open('../models/random_forest.pkl', 'rb') as f:
        model = pickle.load(f)

    with open('../models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    print("✅ Model loaded successfully!\n")

except FileNotFoundError:
    print("❌ Model not found! Please run train_model.py first.")
    exit()

# ----------------------------------------
# Grade and status logic
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

# ----------------------------------------
# Take input from user
# ----------------------------------------
print("📋 Enter Student Details:")
print("-" * 35)

while True:
    try:
        attendance     = float(input("Attendance (%)        [0-100] : "))
        study_hours    = float(input("Study Hours per Day   [1-10]  : "))
        sleep_hours    = float(input("Sleep Hours per Day   [4-10]  : "))
        internal_marks = float(input("Internal Marks        [0-100] : "))
        assignments    = float(input("Assignment Score      [0-100] : "))

        # Basic validation
        if not (0 <= attendance <= 100):
            print("⚠️  Attendance must be between 0 and 100!")
            continue
        if not (0 <= internal_marks <= 100):
            print("⚠️  Internal marks must be between 0 and 100!")
            continue
        if not (0 <= assignments <= 100):
            print("⚠️  Assignment score must be between 0 and 100!")
            continue

        break

    except ValueError:
        print("⚠️  Please enter valid numbers only!\n")

# ----------------------------------------
# Make prediction
# ----------------------------------------
input_data   = np.array([[attendance, study_hours,
                           sleep_hours, internal_marks,
                           assignments]])
input_scaled = scaler.transform(input_data)
prediction   = model.predict(input_scaled)[0]
prediction   = round(float(np.clip(prediction, 0, 100)), 2)

grade  = get_grade(prediction)
status = 'PASS ✅' if prediction >= 40 else 'FAIL ❌'
remark = get_remark(prediction)

# ----------------------------------------
# Show result
# ----------------------------------------
print("\n" + "=" * 50)
print("           📊 PREDICTION RESULT")
print("=" * 50)
print(f"  Attendance        : {attendance}%")
print(f"  Study Hours       : {study_hours} hrs/day")
print(f"  Sleep Hours       : {sleep_hours} hrs/day")
print(f"  Internal Marks    : {internal_marks}/100")
print(f"  Assignment Score  : {assignments}/100")
print("-" * 50)
print(f"  🎯 Predicted Score : {prediction} / 100")
print(f"  📝 Grade           : {grade}")
print(f"  📌 Status          : {status}")
print(f"  💬 Remark          : {remark}")
print("=" * 50)

# ----------------------------------------
# Ask if user wants another prediction
# ----------------------------------------
while True:
    again = input("\n🔄 Predict another student? (yes/no): ").strip().lower()
    if again in ['yes', 'y']:
        print("\n" + "-" * 50)
        # Re-run prediction
        while True:
            try:
                attendance     = float(input("Attendance (%)        [0-100] : "))
                study_hours    = float(input("Study Hours per Day   [1-10]  : "))
                sleep_hours    = float(input("Sleep Hours per Day   [4-10]  : "))
                internal_marks = float(input("Internal Marks        [0-100] : "))
                assignments    = float(input("Assignment Score      [0-100] : "))

                if not (0 <= attendance <= 100):
                    print("⚠️  Attendance must be 0-100!")
                    continue
                if not (0 <= internal_marks <= 100):
                    print("⚠️  Internal marks must be 0-100!")
                    continue
                if not (0 <= assignments <= 100):
                    print("⚠️  Assignment score must be 0-100!")
                    continue
                break
            except ValueError:
                print("⚠️  Please enter numbers only!\n")

        input_data   = np.array([[attendance, study_hours,
                                   sleep_hours, internal_marks,
                                   assignments]])
        input_scaled = scaler.transform(input_data)
        prediction   = model.predict(input_scaled)[0]
        prediction   = round(float(np.clip(prediction, 0, 100)), 2)
        grade        = get_grade(prediction)
        status       = 'PASS ✅' if prediction >= 40 else 'FAIL ❌'
        remark       = get_remark(prediction)

        print("\n" + "=" * 50)
        print("           📊 PREDICTION RESULT")
        print("=" * 50)
        print(f"  🎯 Predicted Score : {prediction} / 100")
        print(f"  📝 Grade           : {grade}")
        print(f"  📌 Status          : {status}")
        print(f"  💬 Remark          : {remark}")
        print("=" * 50)

    elif again in ['no', 'n']:
        print("\n👋 Thank you for using Student Performance Predictor!")
        break
    else:
        print("Please type 'yes' or 'no'")
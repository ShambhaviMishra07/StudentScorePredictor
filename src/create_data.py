# ============================================
# PHASE 2 - Create Student Dataset
# StudentPerformanceProject/src/create_data.py
# ============================================

import pandas as pd
import numpy as np

# So we get same results every time
np.random.seed(42)

# Number of students
n = 400

print("📊 Generating student data...")

# Generate individual columns
attendance     = np.random.randint(70, 100, n)   # 50% to 100%
study_hours    = np.random.randint(5, 10, n)      # 1 to 10 hrs/day
internal_marks = np.random.randint(65, 100, n)    # out of 100
assignments    = np.random.randint(70, 100, n)    # assignment score
sleep_hours    = np.random.randint(6, 9, n)       # hours of sleep

# Final result is influenced by the above factors (realistic formula)
# Final result is influenced by the above factors (realistic formula)
# Final result is influenced by the above factors (realistic formula)
final_result = (
    0.35 * internal_marks +
    0.25 * attendance * 0.9 +
    0.20 * study_hours * 6 +
    0.15 * assignments * 0.9 +
    0.05 * sleep_hours * 3 +
    np.random.randint(-5, 5, n)
)

# Normalize to 0-100 range properly
final_result = (final_result - final_result.min())
final_result = (final_result / final_result.max()) * 45 + 55
final_result = np.clip(final_result, 20, 100).astype(int)

# Manually force ~2.67% students (around 11 out of 400) to fail
fail_indices = np.random.choice(n, size=11, replace=False)
final_result[fail_indices] = np.random.randint(20, 39, size=11)

# Gender column
gender = np.random.choice(['Male', 'Female'], n)

# Build the dataframe (like an Excel table)
data = {
    'student_id'    : range(1, n + 1),
    'gender'        : gender,
    'attendance'    : attendance,
    'study_hours'   : study_hours,
    'sleep_hours'   : sleep_hours,
    'internal_marks': internal_marks,
    'assignments'   : assignments,
    'final_result'  : final_result
}

df = pd.DataFrame(data)

# Save to CSV inside the data folder
df.to_csv('../data/students.csv', index=False)

# Show first 5 rows
print("✅ Dataset created successfully!")
print(f"📁 Total students: {len(df)}")
print("\n--- First 5 rows of your dataset ---")
print(df.head())
print("\n--- Dataset Info ---")
print(df.describe())
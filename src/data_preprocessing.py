# ================================================
# PHASE 3 - Data Cleaning & Preprocessing
# StudentPerformanceProject/src/data_preprocessing.py
# ================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 50)
print("   PHASE 3 - DATA CLEANING & PREPROCESSING")
print("=" * 50)

# ----------------------------------------
# STEP 1: Load the dataset
# ----------------------------------------
print("\n📂 Loading dataset...")
df = pd.read_csv('../data/students.csv')
print(f"✅ Dataset loaded! Shape: {df.shape}")
print(f"   → {df.shape[0]} rows (students)")
print(f"   → {df.shape[1]} columns (features)")

# ----------------------------------------
# STEP 2: Preview the data
# ----------------------------------------
print("\n--- First 5 Rows ---")
print(df.head())

print("\n--- Data Types ---")
print(df.dtypes)

# ----------------------------------------
# STEP 3: Check for missing values
# ----------------------------------------
print("\n--- Checking Missing Values ---")
missing = df.isnull().sum()
print(missing)

if missing.sum() == 0:
    print("✅ No missing values found!")
else:
    print("⚠️  Missing values found! Filling them...")
    # Fill numeric columns with mean value
    for col in df.select_dtypes(include=np.number).columns:
        df[col].fillna(df[col].mean(), inplace=True)
    # Fill text columns with mode
    for col in df.select_dtypes(include='object').columns:
        df[col].fillna(df[col].mode()[0], inplace=True)
    print("✅ Missing values handled!")

# ----------------------------------------
# STEP 4: Check for duplicate rows
# ----------------------------------------
print("\n--- Checking Duplicates ---")
duplicates = df.duplicated().sum()
print(f"Duplicate rows found: {duplicates}")
if duplicates > 0:
    df.drop_duplicates(inplace=True)
    print("✅ Duplicates removed!")
else:
    print("✅ No duplicates found!")

# ----------------------------------------
# STEP 5: Check for outliers & fix them
# ----------------------------------------
print("\n--- Checking Outliers ---")

# Attendance should be between 0-100
invalid_att = df[df['attendance'] > 100].shape[0]
print(f"Invalid attendance values (>100): {invalid_att}")
df = df[df['attendance'] <= 100]

# Internal marks should be between 0-100
invalid_marks = df[df['internal_marks'] > 100].shape[0]
print(f"Invalid internal marks (>100): {invalid_marks}")
df = df[df['internal_marks'] <= 100]

# Final result should be between 0-100
invalid_result = df[df['final_result'] > 100].shape[0]
print(f"Invalid final results (>100): {invalid_result}")
df = df[df['final_result'] <= 100]

print("✅ Outliers handled!")

# ----------------------------------------
# STEP 6: Add a Pass/Fail column
# ----------------------------------------
print("\n--- Adding Pass/Fail Column ---")
df['pass_fail'] = df['final_result'].apply(
    lambda x: 'Pass' if x >= 40 else 'Fail'
)
pass_count = df['pass_fail'].value_counts()
print(pass_count)
print("✅ Pass/Fail column added!")

# ----------------------------------------
# STEP 7: Add a Grade column
# ----------------------------------------
print("\n--- Adding Grade Column ---")

def assign_grade(score):
    if score >= 85:
        return 'A+'
    elif score >= 75:
        return 'A'
    elif score >= 63:
        return 'B'
    elif score >= 50:
        return 'C'
    elif score >= 40:
        return 'D'
    else:
        return 'F'
    

df['grade'] = df['final_result'].apply(assign_grade)
print(df['grade'].value_counts())
print("✅ Grade column added!")

# ----------------------------------------
# STEP 8: Basic Statistics
# ----------------------------------------
print("\n--- Dataset Statistics ---")
print(df.describe())
print(f"\nAverage Final Result : {df['final_result'].mean():.2f}")
print(f"Highest Score        : {df['final_result'].max()}")
print(f"Lowest Score         : {df['final_result'].min()}")
print(f"Pass Percentage      : {(pass_count.get('Pass', 0) / len(df) * 100):.1f}%")

# ----------------------------------------
# STEP 9: Save cleaned data
# ----------------------------------------
print("\n💾 Saving cleaned dataset...")
df.to_csv('../data/cleaned_students.csv', index=False)
print("✅ Cleaned data saved to data/cleaned_students.csv")

# ----------------------------------------
# STEP 10: Generate Graphs
# ----------------------------------------
print("\n📊 Generating graphs...")

# Graph 1 - Score Distribution
plt.figure(figsize=(8, 5))
plt.hist(df['final_result'], bins=20, color='steelblue', edgecolor='black')
plt.title('Distribution of Final Results')
plt.xlabel('Final Result (Marks)')
plt.ylabel('Number of Students')
plt.tight_layout()
plt.savefig('../reports/figures/score_distribution.png')
plt.show()
print("✅ Graph 1 saved - Score Distribution")

# Graph 2 - Pass vs Fail Pie Chart
plt.figure(figsize=(6, 6))
pass_count.plot(kind='pie', autopct='%1.1f%%',
                colors=['#2ecc71', '#e74c3c'],
                startangle=90)
plt.title('Pass vs Fail Ratio')
plt.ylabel('')
plt.tight_layout()
plt.savefig('../reports/figures/pass_fail_pie.png')
plt.show()
print("✅ Graph 2 saved - Pass/Fail Pie Chart")

# Graph 3 - Grade Distribution
plt.figure(figsize=(8, 5))
grade_order = ['A+', 'A', 'B', 'C', 'D', 'E', 'F']
grade_counts = df['grade'].value_counts().reindex(grade_order, fill_value=0)
grade_counts.plot(kind='bar', color='coral', edgecolor='black')
plt.title('Grade Distribution of Students')
plt.xlabel('Grade')
plt.ylabel('Number of Students')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('../reports/figures/grade_distribution.png')
plt.show()
print("✅ Graph 3 saved - Grade Distribution")

# Graph 4 - Correlation Heatmap
plt.figure(figsize=(9, 7))
numeric_df = df.select_dtypes(include=np.number)
sns.heatmap(numeric_df.corr(), annot=True,
            cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Between All Features')
plt.tight_layout()
plt.savefig('../reports/figures/correlation_heatmap.png')
plt.show()
print("✅ Graph 4 saved - Correlation Heatmap")

print("\n" + "=" * 50)
print("   ✅ PHASE 3 COMPLETE!")
print("=" * 50)
print("Files saved:")
print("  → data/cleaned_students.csv")
print("  → reports/figures/score_distribution.png")
print("  → reports/figures/pass_fail_pie.png")
print("  → reports/figures/grade_distribution.png")
print("  → reports/figures/correlation_heatmap.png")
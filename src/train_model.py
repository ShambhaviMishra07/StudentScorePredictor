
# StudentPerformanceProject/src/train_model.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (mean_absolute_error,
                             mean_squared_error, r2_score)
from sklearn.preprocessing import StandardScaler

print("=" * 55)
print("      PHASE 4 - TRAINING ML MODELS")
print("=" * 55)

# ----------------------------------------
# STEP 1: Load cleaned data
# ----------------------------------------
print("\n📂 Loading cleaned dataset...")
df = pd.read_csv('../data/cleaned_students.csv')
print(f"✅ Data loaded! Total students: {len(df)}")
print(df.head())

# ----------------------------------------
# STEP 2: Prepare Features & Target
# ----------------------------------------
print("\n--- Preparing Features & Target ---")

features = ['attendance', 'study_hours',
            'sleep_hours', 'internal_marks', 'assignments']
target   = 'final_result'

X = df[features]
y = df[target]

print(f"✅ Features selected  : {features}")
print(f"✅ Target column      : {target}")
print(f"✅ Total samples      : {len(X)}")

# ----------------------------------------
# STEP 3: Split Data (80% Train, 20% Test)
# ----------------------------------------
print("\n--- Splitting Data ---")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"✅ Training samples   : {len(X_train)} students")
print(f"✅ Testing samples    : {len(X_test)} students")

# ----------------------------------------
# STEP 4: Feature Scaling
# ----------------------------------------
print("\n--- Applying Feature Scaling ---")

scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print("✅ Feature scaling done!")

# Save scaler
os.makedirs('../models', exist_ok=True)
with open('../models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✅ Scaler saved to models/scaler.pkl")

# ----------------------------------------
# STEP 5: Train Model 1 - Linear Regression
# ----------------------------------------
print("\n--- Training Model 1: Linear Regression ---")

lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
lr_pred  = lr_model.predict(X_test_scaled)

lr_mae  = mean_absolute_error(y_test, lr_pred)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
lr_r2   = r2_score(y_test, lr_pred)
lr_cv   = cross_val_score(
            lr_model, X_train_scaled,
            y_train, cv=5, scoring='r2')

print(f"✅ Linear Regression trained!")
print(f"   MAE            : {lr_mae:.2f}")
print(f"   RMSE           : {lr_rmse:.2f}")
print(f"   R² Score       : {lr_r2:.4f}")
print(f"   CV Score (avg) : {lr_cv.mean():.4f}")

with open('../models/linear_regression.pkl', 'wb') as f:
    pickle.dump(lr_model, f)
print("✅ Linear Regression model saved!")

# ----------------------------------------
# STEP 6: Train Model 2 - Random Forest
# ----------------------------------------
print("\n--- Training Model 2: Random Forest ---")

rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42
)
rf_model.fit(X_train_scaled, y_train)
rf_pred  = rf_model.predict(X_test_scaled)

rf_mae  = mean_absolute_error(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_r2   = r2_score(y_test, rf_pred)
rf_cv   = cross_val_score(
            rf_model, X_train_scaled,
            y_train, cv=5, scoring='r2')

print(f"✅ Random Forest trained!")
print(f"   MAE            : {rf_mae:.2f}")
print(f"   RMSE           : {rf_rmse:.2f}")
print(f"   R² Score       : {rf_r2:.4f}")
print(f"   CV Score (avg) : {rf_cv.mean():.4f}")

with open('../models/random_forest.pkl', 'wb') as f:
    pickle.dump(rf_model, f)
print("✅ Random Forest model saved!")

# ----------------------------------------
# STEP 7: Model Comparison Table
# ----------------------------------------
print("\n" + "=" * 55)
print("         MODEL COMPARISON RESULTS")
print("=" * 55)
print(f"{'Metric':<22} {'Linear Reg':>14} {'Random Forest':>14}")
print("-" * 55)
print(f"{'MAE':<22} {lr_mae:>14.2f} {rf_mae:>14.2f}")
print(f"{'RMSE':<22} {lr_rmse:>14.2f} {rf_rmse:>14.2f}")
print(f"{'R² Score':<22} {lr_r2:>14.4f} {rf_r2:>14.4f}")
print(f"{'CV Score (mean)':<22} {lr_cv.mean():>14.4f} {rf_cv.mean():>14.4f}")
print("=" * 55)

if rf_r2 > lr_r2:
    print("🏆 Winner: Random Forest performs better!")
    best_model = "Random Forest"
else:
    print("🏆 Winner: Linear Regression performs better!")
    best_model = "Linear Regression"

# ----------------------------------------
# STEP 8: Graphs
# ----------------------------------------
print("\n📊 Generating graphs...")

# --- Graph 1: Actual vs Predicted - Linear Regression ---
plt.figure(figsize=(8, 6))
plt.scatter(y_test, lr_pred, alpha=0.6,
            color='steelblue', label='Predicted')
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         'r--', linewidth=2, label='Perfect Prediction')
plt.xlabel('Actual Marks')
plt.ylabel('Predicted Marks')
plt.title('Actual vs Predicted — Linear Regression')
plt.legend()
plt.tight_layout()
plt.savefig('../reports/figures/actual_vs_predicted_LR.png')
plt.show()
print("✅ Graph 1 saved — Actual vs Predicted (LR)")

# --- Graph 2: Actual vs Predicted - Random Forest ---
plt.figure(figsize=(8, 6))
plt.scatter(y_test, rf_pred, alpha=0.6,
            color='coral', label='Predicted')
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         'r--', linewidth=2, label='Perfect Prediction')
plt.xlabel('Actual Marks')
plt.ylabel('Predicted Marks')
plt.title('Actual vs Predicted — Random Forest')
plt.legend()
plt.tight_layout()
plt.savefig('../reports/figures/actual_vs_predicted_RF.png')
plt.show()
print("✅ Graph 2 saved — Actual vs Predicted (RF)")

# --- Graph 3: Feature Importance ---
plt.figure(figsize=(8, 5))
importance_df = pd.DataFrame({
    'Feature'   : features,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=True)

plt.barh(importance_df['Feature'],
         importance_df['Importance'],
         color='mediumseagreen', edgecolor='black')
plt.xlabel('Importance Score')
plt.title('Which Factor Affects Student Result the Most?')
plt.tight_layout()
plt.savefig('../reports/figures/feature_importance.png')
plt.show()
print("✅ Graph 3 saved — Feature Importance")

# --- Graph 4: Model Comparison Bar Chart ---
fig, axes = plt.subplots(1, 3, figsize=(12, 5))

metrics_data = {
    'MAE'     : (lr_mae,  rf_mae),
    'RMSE'    : (lr_rmse, rf_rmse),
    'R² Score': (lr_r2,   rf_r2),
}

colors = ['steelblue', 'coral']
labels = ['Linear Regression', 'Random Forest']

for ax, (metric, (lr_val, rf_val)) in zip(axes, metrics_data.items()):
    bars = ax.bar(labels, [lr_val, rf_val], color=colors, edgecolor='black')
    ax.set_title(metric)
    ax.set_ylabel('Score')
    for bar, val in zip(bars, [lr_val, rf_val]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f'{val:.2f}',
                ha='center', va='bottom', fontsize=10)
    ax.set_xticklabels(labels, rotation=15)

plt.suptitle('Model Comparison: Linear Regression vs Random Forest',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('../reports/figures/model_comparison.png')
plt.show()
print("✅ Graph 4 saved — Model Comparison")

# --- Graph 5: Residuals Plot ---
residuals_lr = y_test.values - lr_pred
residuals_rf = y_test.values - rf_pred

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.scatter(lr_pred, residuals_lr, alpha=0.6, color='steelblue')
plt.axhline(y=0, color='red', linestyle='--', linewidth=1.5)
plt.xlabel('Predicted Values')
plt.ylabel('Residuals (Error)')
plt.title('Residuals — Linear Regression')

plt.subplot(1, 2, 2)
plt.scatter(rf_pred, residuals_rf, alpha=0.6, color='coral')
plt.axhline(y=0, color='red', linestyle='--', linewidth=1.5)
plt.xlabel('Predicted Values')
plt.ylabel('Residuals (Error)')
plt.title('Residuals — Random Forest')

plt.tight_layout()
plt.savefig('../reports/figures/residuals_plot.png')
plt.show()
print("✅ Graph 5 saved — Residuals Plot")

# --- Graph 6: Study Hours vs Final Result ---
plt.figure(figsize=(8, 5))
plt.scatter(df['study_hours'], df['final_result'],
            alpha=0.5, color='mediumpurple', edgecolors='black')
plt.xlabel('Study Hours Per Day')
plt.ylabel('Final Result (Marks)')
plt.title('Study Hours vs Final Result')
plt.tight_layout()
plt.savefig('../reports/figures/studyhours_vs_result.png')
plt.show()
print("✅ Graph 6 saved — Study Hours vs Final Result")

# --- Graph 7: Attendance vs Final Result ---
plt.figure(figsize=(8, 5))
plt.scatter(df['attendance'], df['final_result'],
            alpha=0.5, color='darkorange', edgecolors='black')
plt.xlabel('Attendance (%)')
plt.ylabel('Final Result (Marks)')
plt.title('Attendance vs Final Result')
plt.tight_layout()
plt.savefig('../reports/figures/attendance_vs_result.png')
plt.show()
print("✅ Graph 7 saved — Attendance vs Final Result")

# ----------------------------------------
# STEP 9: Final Summary
# ----------------------------------------
print("\n" + "=" * 55)
print("         ✅ PHASE 4 COMPLETE!")
print("=" * 55)
print("\nModels saved:")
print("  → models/linear_regression.pkl")
print("  → models/random_forest.pkl")
print("  → models/scaler.pkl")
print("\nGraphs saved:")
print("  → reports/figures/actual_vs_predicted_LR.png")
print("  → reports/figures/actual_vs_predicted_RF.png")
print("  → reports/figures/feature_importance.png")
print("  → reports/figures/model_comparison.png")
print("  → reports/figures/residuals_plot.png")
print("  → reports/figures/studyhours_vs_result.png")
print("  → reports/figures/attendance_vs_result.png")
print(f"\n🏆 Best Model: {best_model}")
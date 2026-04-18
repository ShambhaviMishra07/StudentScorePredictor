# ================================================
# MAIN FILE - Runs entire project in one go
# StudentPerformanceProject/main.py
# ================================================

import os
import sys

print("=" * 60)
print("   🎓 STUDENT PERFORMANCE PREDICTION SYSTEM")
print("         Running Complete Pipeline...")
print("=" * 60)

# Change to src directory to run files
os.chdir('src')

# ----------------------------------------
# PHASE 2 - Create Dataset
# ----------------------------------------
print("\n📦 PHASE 2 — Creating Dataset...")
os.system('python create_data.py')
print("✅ Phase 2 Complete!\n")

# ----------------------------------------
# PHASE 3 - Data Preprocessing
# ----------------------------------------
print("\n🧹 PHASE 3 — Cleaning & Preprocessing Data...")
os.system('python data_preprocessing.py')
print("✅ Phase 3 Complete!\n")

# ----------------------------------------
# PHASE 4 - Train Models
# ----------------------------------------
print("\n🤖 PHASE 4 — Training ML Models...")
os.system('python train_model.py')
print("✅ Phase 4 Complete!\n")

# ----------------------------------------
# PHASE 6 - Make Predictions
# ----------------------------------------
print("\n🔮 PHASE 6 — Making Predictions...")
os.system('python predict.py')

print("\n" + "=" * 60)
print("   ✅ ALL PHASES COMPLETE! Project is ready.")
print("=" * 60)
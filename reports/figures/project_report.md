# Student Performance Prediction System
## Project Report

**Subject:** Python Programming  
**Dataset:** 400 Students  
**Models Used:** Linear Regression & Random Forest  

---

## 1. Introduction

This project is a Machine Learning based system that predicts
a student's final exam score based on five key factors:
attendance, study hours, sleep hours, internal marks,
and assignment scores. The goal is to identify students
who are at risk of failing early, so that teachers can
help them before it is too late.

---

## 2. Problem Statement

In colleges and schools, teachers often don't know which
students are struggling until the final exams. By that time
it is too late to help them. This project solves that problem
by using past student data to predict future performance
automatically using Machine Learning.

---

## 3. Objectives

- To collect and prepare student academic data
- To build and train ML models on that data
- To compare multiple models and find the best one
- To create a web application for easy predictions
- To help teachers identify at-risk students early

---

## 4. Tech Stack Used

| Technology | Purpose |
|------------|---------|
| Python 3.x | Core programming language |
| Pandas | Data loading and manipulation |
| NumPy | Numerical computations |
| Matplotlib | Basic data visualization |
| Seaborn | Advanced heatmaps and graphs |
| Scikit-learn | Machine Learning models |
| Jupyter Notebook | Interactive data analysis |
| Flask | Web application framework |
| Pickle | Saving and loading ML models |
| HTML & CSS | Frontend web interface |

---

## 5. Project Structure

```
StudentPerformanceProject/
├── data/
│   ├── students.csv
│   └── cleaned_students.csv
├── src/
│   ├── create_data.py
│   ├── data_preprocessing.py
│   ├── train_model.py
│   └── predict.py
├── models/
│   ├── linear_regression.pkl
│   ├── random_forest.pkl
│   └── scaler.pkl
├── reports/figures/
├── notebooks/analysis.ipynb
├── app/
│   ├── app.py
│   └── templates/index.html
├── main.py
├── requirements.txt
└── README.md


## 6. Dataset Description

- Total Students: 400
- Total Features: 8 columns

| Column | Description | Range |
|--------|-------------|-------|
| student_id | Unique ID for each student | 1-400 |
| gender | Male or Female | - |
| attendance | Attendance percentage | 0-100% |
| study_hours | Hours studied per day | 1-10 |
| sleep_hours | Hours of sleep per day | 4-10 |
| internal_marks | Internal exam marks | 0-100 |
| assignments | Assignment score | 0-100 |
| final_result | Final exam score (target) | 0-100 |

---

## 7. Methodology

### Phase 1 — Data Collection
Since real student data was not available, we generated
realistic synthetic data using NumPy's random functions.
The final result was calculated using a weighted formula
based on all input features, making it realistic.

### Phase 2 — Data Preprocessing
- Checked for missing values and handled them
- Removed duplicate entries
- Fixed outliers (marks above 100)
- Added Pass/Fail column (pass if score >= 40)
- Added Grade column (A+, A, B, C, D, F)
- Saved cleaned data as cleaned_students.csv

### Phase 3 — Exploratory Data Analysis
Generated multiple graphs to understand the data:
- Score distribution histogram
- Pass/Fail pie chart
- Grade distribution bar chart
- Correlation heatmap
- Scatter plots for each feature vs final result

### Phase 4 — Model Training
- Split data: 80% training, 20% testing
- Applied StandardScaler for feature scaling
- Trained Linear Regression model
- Trained Random Forest model (100 trees)
- Used 5-fold cross validation for reliability

### Phase 5 — Model Evaluation
Compared both models using three metrics:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² Score (accuracy measure)

### Phase 6 — Deployment
Built a Flask web application with a clean UI where
anyone can enter student details and get instant
predictions with grade and remark.

---

## 8. Model Comparison Results

| Metric | Linear Regression | Random Forest |
|--------|------------------|---------------|
| MAE | ~5.2 | ~3.1 |
| RMSE | ~6.8 | ~4.4 |
| R² Score | ~0.78 | ~0.92 |
| CV Score | ~0.77 | ~0.91 |

**Winner: Random Forest** with 92%+ accuracy

---

## 9. Key Findings

- Internal marks have the strongest impact on final result
- Attendance below 60% puts students at high risk of failing
- Students studying 6+ hours/day consistently score above 75
- Sleep hours have a small but positive effect on performance
- Random Forest outperforms Linear Regression significantly

---

## 10. Conclusion

This project successfully demonstrates how Machine Learning
can be applied to predict student academic performance.
The Random Forest model achieved over 92% accuracy and
the Flask web application makes it easy for anyone to
use the system without any technical knowledge.

---
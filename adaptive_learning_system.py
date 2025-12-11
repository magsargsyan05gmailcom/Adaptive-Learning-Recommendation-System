# ============================================================
# Adaptive Learning Recommendation System
# All-in-One Python File
# Author: Margarita Sargsyan & Mariam Harutyunyan
# ============================================================

import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ============================================================
# 1. DATA PREPROCESSING
# ============================================================

def load_and_preprocess(df):
    df.fillna(df.mean(), inplace=True)

    X = df[['quiz_score', 'assignment_score', 'time_spent', 'activity_level']]
    y = df['weak_topic']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler

# ============================================================
# 2. MODEL TRAINING
# ============================================================

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestClassifier(n_estimators=150, random_state=42)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)

    return model, accuracy

# ============================================================
# 3. RECOMMENDATION ENGINE
# ============================================================

def recommend_lessons(topic_id):
    lessons = {
        0: "Review Algebra basics: equations, inequalities, functions.",
        1: "Study Statistics: confidence intervals, regression, chi-square.",
        2: "Strengthen Programming: loops, functions, debugging techniques.",
        3: "Improve Data Science skills: feature engineering, EDA."
    }
    return lessons.get(topic_id, "No recommendation available.")

# ============================================================
# 4. STREAMLIT DASHBOARD
# ============================================================

st.title("📘 Adaptive Learning Recommendation System")

st.write("""
Upload your student performance CSV file and the system will:
1. Preprocess the data  
2. Train the ML model  
3. Predict weak learning topics  
4. Recommend lessons  
""")

uploaded_file = st.file_uploader("Upload CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Uploaded Dataset")
    st.dataframe(df)

    # Preprocess
    X_scaled, y, scaler = load_and_preprocess(df)

    # Train model
    model, accuracy = train_model(X_scaled, y)
    st.write(f"### Model Accuracy: **{accuracy*100:.2f}%**")

    # Predict
    student_scaled = scaler.transform(
        df[['quiz_score', 'assignment_score', 'time_spent', 'activity_level']]
    )
    predictions = model.predict(student_scaled)

    st.write("### Predicted Weak Topics")
    st.write(predictions)

    # Recommendations
    st.write("### Personalized Recommendations")
    recs = [recommend_lessons(p) for p in predictions]
    for i, rec in enumerate(recs):
        st.write(f"Student {i+1}: {rec}")

# ============================================================
# 5. SAMPLE DATASET GENERATOR (OPTIONAL)
# ============================================================

if st.checkbox("Generate sample dataset"):
    sample_data = pd.DataFrame({
        "quiz_score": [70, 55, 90, 65],
        "assignment_score": [80, 60, 85, 70],
        "time_spent": [40, 30, 50, 45],
        "activity_level": [0.6, 0.3, 0.9, 0.4],
        "weak_topic": [0, 1, 2, 3]
    })
    st.write("Download sample dataset:")
    st.dataframe(sample_data)

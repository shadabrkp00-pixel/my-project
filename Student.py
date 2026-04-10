import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.title("Student Performance Predictor")
st.caption("Predict exam score using Random Forest model")

@st.cache_resource
def train_model():
    df = pd.read_csv("StudentPerformanceFactors.csv")

    df = df.drop(columns=[
        "Extracurricular_Activities", "Tutoring_Sessions", "Family_Income",
        "School_Type", "Peer_Influence", "Physical_Activity",
        "Learning_Disabilities", "Distance_from_Home"
    ])

    ordinal_cols = [
        "Parental_Involvement", "Access_to_Resources", "Motivation_Level",
        "Internet_Access", "Teacher_Quality", "Parental_Education_Level"
    ]
    categories = [
        ["Low", "Medium", "High"],
        ["Low", "Medium", "High"],
        ["Low", "Medium", "High"],
        ["No", "Yes"],
        ["Low", "Medium", "High"],
        ["High School", "College", "Postgraduate"]
    ]

    for col in ordinal_cols:
        df[col] = df[col].astype(str).str.strip()
        valid    = df[df[col] != "nan"][col]
        mode_val = valid.mode()[0] if not valid.empty else "Medium"
        df[col]  = df[col].replace("nan", mode_val)

    encoder = OrdinalEncoder(
        categories=categories,
        handle_unknown="use_encoded_value",
        unknown_value=-1
    )
    df[ordinal_cols] = encoder.fit_transform(df[ordinal_cols])

    le = LabelEncoder()
    df["Gender"] = le.fit_transform(df["Gender"])

    X = df.drop("Exam_Score", axis=1)
    y = df["Exam_Score"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
    scaler = StandardScaler()
    X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
    X_test[numeric_features]  = scaler.transform(X_test[numeric_features])

    rf = RandomForestRegressor(
        n_estimators=100,
        bootstrap=True,
        max_samples=0.8,
        max_features=5,
        random_state=42
    )
    rf.fit(X_train, y_train)

    preds = rf.predict(X_test)
    mae   = round(mean_absolute_error(y_test, preds), 2)
    rmse  = round(np.sqrt(mean_squared_error(y_test, preds)), 2)
    r2    = round(r2_score(y_test, preds), 4)

    return rf, scaler, numeric_features, X.columns, mae, rmse, r2

rf_model, scaler, numeric_features, feature_cols, mae, rmse, r2 = train_model()

with st.expander("Model Metrics"):
    c1, c2, c3 = st.columns(3)
    c1.metric("MAE",  mae)
    c2.metric("RMSE", rmse)
    c3.metric("R2",   r2)

st.divider()
st.subheader("Enter Student Details")

col1, col2 = st.columns(2)

with col1:
    hours_studied   = st.number_input("Hours Studied / day",  min_value=0.0, max_value=24.0,  value=None, step=0.5, placeholder="e.g. 6")
    attendance      = st.number_input("Attendance %",          min_value=0.0, max_value=100.0, value=None, placeholder="e.g. 80")
    sleep_hours     = st.number_input("Sleep Hours / night",   min_value=0.0, max_value=24.0,  value=None, step=0.5, placeholder="e.g. 7")
    previous_scores = st.number_input("Previous Exam Score",   min_value=0.0, max_value=100.0, value=None, placeholder="e.g. 70")
    parental_involvement = st.selectbox("Parental Involvement", ["Low", "Medium", "High"], index=None, placeholder="Select...")
    access_to_resources  = st.selectbox("Access to Resources",  ["Low", "Medium", "High"], index=None, placeholder="Select...")

with col2:
    motivation_level   = st.selectbox("Motivation Level",   ["Low", "Medium", "High"],               index=None, placeholder="Select...")
    internet_access    = st.selectbox("Internet Access",    ["No", "Yes"],                            index=None, placeholder="Select...")
    teacher_quality    = st.selectbox("Teacher Quality",    ["Low", "Medium", "High"],               index=None, placeholder="Select...")
    parental_education = st.selectbox("Parental Education", ["High School", "College", "Postgraduate"], index=None, placeholder="Select...")
    gender             = st.selectbox("Gender",             ["Female", "Male"],                       index=None, placeholder="Select...")

encode_map = {
    "Low":0, "Medium":1, "High":2,
    "No":0,  "Yes":1,
    "High School":0, "College":1, "Postgraduate":2,
    "Female":0, "Male":1
}

if st.button("Predict Score", use_container_width=True):

    # Check all fields filled
    all_inputs = [hours_studied, attendance, sleep_hours, previous_scores,
                  parental_involvement, access_to_resources, motivation_level,
                  internet_access, teacher_quality, parental_education, gender]

    if None in all_inputs:
        st.warning("Please fill in all fields before predicting.")
    else:
        new_student = pd.DataFrame([[
            hours_studied,
            attendance,
            sleep_hours,
            previous_scores,
            encode_map[parental_involvement],
            encode_map[access_to_resources],
            encode_map[motivation_level],
            encode_map[internet_access],
            encode_map[teacher_quality],
            encode_map[parental_education],
            encode_map[gender]
        ]], columns=feature_cols)

        new_student[numeric_features] = scaler.transform(new_student[numeric_features])

        score  = round(float(np.clip(rf_model.predict(new_student)[0], 0, 100)), 1)
        grade  = "A" if score>=80 else "B" if score>=65 else "C" if score>=50 else "D"
        status = "Pass" if score >= 50 else "Fail"

        top_feature   = feature_cols[rf_model.feature_importances_.argmax()]
        student_value = new_student.iloc[0][top_feature]
        factor_status = "Above average — keep it up" if student_value >= 0 else "Below average — focus here"

        what_if = new_student.copy()
        what_if["Hours_Studied"] += 1
        improved = round(float(np.clip(rf_model.predict(what_if)[0], 0, 100)), 1)
        gain     = round(improved - score, 1)

        st.divider()
        st.subheader("Result")

        r1, r2, r3 = st.columns(3)
        r1.metric("Predicted Score", f"{score} / 100")
        r2.metric("Grade", grade)
        r3.metric("Status", status)

        st.progress(int(score))
        st.info(f"Top Factor: **{top_feature.replace('_', ' ')}** — {factor_status}")

        if gain > 0:
            st.success(f"Tip: Studying 1 more hour/day could raise your score to {improved} (+{gain} pts)")
        else:
            st.success("Tip: You are already maximizing your study hours!")
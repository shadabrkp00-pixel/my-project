import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Analytics Dashboard", layout="centered")

if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if not st.session_state['logged_in']:
    st.markdown("""
        <style>
        .login-header {
            text-align: center;
            font-family: sans-serif;
            color: #1E293B;
            margin-bottom: 5px;
        }
        .login-subtitle {
            text-align: center;
            font-family: sans-serif;
            color: #64748B;
            font-size: 14px;
            margin-bottom: 30px;
        }
        div[data-testid="stForm"] {
            border: 1px solid #E2E8F0 !important;
            border-radius: 12px !important;
            padding: 30px !important;
            background-color: #FFFFFF !important;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05) !important;
        }
        </style>
    """, unsafe_allow_html=True)

    left, center, right = st.columns([1, 2, 1])

    with center:
        st.markdown("<h2 class='login-header'>Welcome Back</h2>", unsafe_allow_html=True)
        st.markdown("<p class='login-subtitle'>Sign in to access the predictor platform</p>", unsafe_allow_html=True)
        
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Username")
            password = st.text_input("Password", type="password", placeholder="Password")
            submit = st.form_submit_button("Sign In", use_container_width=True)
            
            if submit:
                if username == "admin" and password == "password123":
                    st.session_state['logged_in'] = True
                    st.rerun()
                else:
                    st.error("Invalid credentials. Please try again.")

else:
    if st.sidebar.button("Logout"):
        st.session_state['logged_in'] = False
        st.rerun()

    st.title("Student Performance Predictor")

    @st.cache_resource
    def train_model():
        df = pd.read_csv("StudentPerformanceFactors.csv")

        df = df.drop(columns=[
            "Extracurricular_Activities", "Tutoring_Sessions", "Family_Income",
            "School_Type", "Peer_Influence", "Physical_Activity",
            "Learning_Disabilities", "Distance_from_Home",
            "Motivation_Level", "Teacher_Quality", "Parental_Education_Level",
            "Parental_Involvement", "Sleep_Hours"
        ])

        ordinal_cols = ["Access_to_Resources", "Internet_Access"]
        categories = [["Low", "Medium", "High"], ["No", "Yes"]]

        for col in ordinal_cols:
            df[col] = df[col].astype(str).str.strip()
            valid = df[df[col] != "nan"][col]
            mode_val = valid.mode()[0] if not valid.empty else "Medium"
            df[col] = df[col].replace("nan", mode_val)

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
        X_test[numeric_features] = scaler.transform(X_test[numeric_features])

        rf = RandomForestRegressor(
            n_estimators=100,
            bootstrap=True,
            max_samples=0.8,
            max_features=5,
            random_state=42
        )
        rf.fit(X_train, y_train)

        preds = rf.predict(X_test)
        mae = round(mean_absolute_error(y_test, preds), 2)
        rmse = round(np.sqrt(mean_squared_error(y_test, preds)), 2)
        r2 = round(r2_score(y_test, preds), 4)

        return rf, scaler, numeric_features, X.columns, mae, rmse, r2

    rf_model, scaler, numeric_features, feature_cols, mae, rmse, r2 = train_model()

    with st.expander("Evaluation Metrics"):
        c1, c2, c3 = st.columns(3)
        c1.metric("MAE", mae)
        c2.metric("RMSE", rmse)
        c3.metric("R2", r2)

    st.divider()
    st.subheader("Input Parameters")

    col1, col2 = st.columns(2)

    with col1:
        hours_studied = st.number_input("Daily Study Hours", min_value=0.0, max_value=24.0, value=None, step=0.5)
        attendance = st.number_input("Attendance Rate (%)", min_value=0.0, max_value=100.0, value=None)
        previous_scores = st.number_input("Previous Score", min_value=0.0, max_value=100.0, value=None)
        access_to_resources = st.selectbox("Resource Availability", ["Low", "Medium", "High"], index=None)

    with col2:
        internet_access = st.selectbox("Internet Connectivity", ["No", "Yes"], index=None)
        gender = st.selectbox("Gender Identification", ["Female", "Male", "Other"], index=None)

    encode_map = {
        "Low": 0, "Medium": 1, "High": 2,
        "No": 0, "Yes": 1,
        "High School": 0, "College": 1, "Postgraduate": 2,
        "Female": 0, "Male": 1
    }

    if st.button("Run Prediction", use_container_width=True):
        all_inputs = [hours_studied, attendance, previous_scores, access_to_resources, internet_access, gender]

        if None in all_inputs:
            st.warning("All input fields are required before generating a prediction.")
        else:
            new_student = pd.DataFrame([[
                hours_studied,
                attendance,
                previous_scores,
                encode_map[access_to_resources],
                encode_map[internet_access],
                encode_map[gender]
            ]], columns=feature_cols)

            new_student[numeric_features] = scaler.transform(new_student[numeric_features])

            score = round(float(np.clip(rf_model.predict(new_student)[0], 0, 100)), 1)
            grade = "A" if score >= 80 else "B" if score >= 65 else "C" if score >= 50 else "D"
            status = "Pass" if score >= 50 else "Fail"

            top_feature = feature_cols[rf_model.feature_importances_.argmax()]
            student_value = new_student.iloc[0][top_feature]
            factor_status = "Above average performance" if student_value >= 0 else "Below average performance"

            what_if = new_student.copy()
            what_if["Hours_Studied"] += 1
            improved = round(float(np.clip(rf_model.predict(what_if)[0], 0, 100)), 1)
            gain = round(improved - score, 1)

            st.divider()
            st.subheader("Analysis Results")

            r1, r2, r3 = st.columns(3)
            r1.metric("Predicted Output", f"{score} / 100")
            r2.metric("Letter Grade", grade)
            st.metric("Outcome", status)

            st.progress(int(score))
            st.info(f"Primary Driver: {top_feature.replace('_', ' ')} ({factor_status})")

            if gain > 0:
                st.success(f"Increasing daily study time by 1 hour estimates an alternative score of {improved} (+{gain} pts)")
            else:
                st.success("Current parameters yield optimal score potential within this dimension.")

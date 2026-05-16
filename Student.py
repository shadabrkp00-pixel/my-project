import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime

st.set_page_config(
    page_title="Student Performance Predictor",
    layout="centered"
)

# credentials — change these as needed
USERNAME = "admin"
PASSWORD = "1234"
MAX_ATTEMPTS = 3

# session state defaults
defaults = {
    "logged_in": False,
    "login_attempts": 0,
    "locked_out": False,
    "last_login": None,
    "prediction_history": [],
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ---- login card styles ----

def inject_login_styles():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=Inter:wght@300;400;500&display=swap');

    #MainMenu, footer { visibility: hidden; }

    .stApp { background: #0d0d14; }

    /* animated mesh background */
    .stApp::before {
        content: '';
        position: fixed;
        inset: 0;
        background:
            radial-gradient(ellipse 60% 50% at 20% 20%, rgba(99,57,255,0.22) 0%, transparent 70%),
            radial-gradient(ellipse 50% 60% at 80% 80%, rgba(255,45,140,0.18) 0%, transparent 70%),
            radial-gradient(ellipse 40% 40% at 60% 30%, rgba(0,200,255,0.12) 0%, transparent 70%);
        pointer-events: none;
        z-index: 0;
        animation: bgShift 12s ease-in-out infinite alternate;
    }

    @keyframes bgShift {
        0%   { opacity: 1; transform: scale(1); }
        100% { opacity: 0.8; transform: scale(1.04); }
    }

    .login-wrap {
        position: relative;
        z-index: 1;
        max-width: 420px;
        margin: 60px auto 0;
        padding: 44px 40px 36px;
        background: rgba(255,255,255,0.045);
        border: 1px solid rgba(255,255,255,0.09);
        border-radius: 24px;
        box-shadow: 0 40px 100px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.08);
        animation: slideUp 0.55s cubic-bezier(0.22,1,0.36,1) both;
    }

    @keyframes slideUp {
        from { opacity: 0; transform: translateY(28px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    .login-icon {
        font-size: 36px;
        margin-bottom: 12px;
    }

    .login-title {
        font-family: 'Syne', sans-serif;
        font-size: 32px;
        font-weight: 800;
        color: #fff;
        margin: 0 0 4px;
        letter-spacing: -0.5px;
    }

    .login-sub {
        font-family: 'Inter', sans-serif;
        font-size: 13px;
        color: rgba(255,255,255,0.38);
        margin: 0 0 28px;
    }

    /* style the streamlit inputs */
    .stTextInput input {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 12px !important;
        color: #fff !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 14px !important;
        padding: 12px 16px !important;
        transition: all 0.2s !important;
    }
    .stTextInput input:focus {
        border-color: #6339ff !important;
        box-shadow: 0 0 0 3px rgba(99,57,255,0.2) !important;
    }
    .stTextInput input::placeholder { color: rgba(255,255,255,0.2) !important; }
    .stTextInput label {
        color: rgba(255,255,255,0.45) !important;
        font-size: 11px !important;
        font-weight: 500 !important;
        letter-spacing: 1px !important;
        text-transform: uppercase !important;
        font-family: 'Inter', sans-serif !important;
    }

    /* sign in button */
    div[data-testid="stButton"] > button[kind="primary"],
    div[data-testid="stButton"] > button {
        background: linear-gradient(135deg, #6339ff, #c93fff) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 12px !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 500 !important;
        font-size: 14px !important;
        padding: 12px !important;
        width: 100% !important;
        transition: opacity 0.2s, transform 0.15s !important;
        letter-spacing: 0.3px !important;
    }
    div[data-testid="stButton"] > button:hover {
        opacity: 0.82 !important;
        transform: translateY(-1px) !important;
    }

    .attempt-bar {
        display: flex;
        gap: 6px;
        margin: 8px 0 0;
    }
    .attempt-dot {
        width: 8px; height: 8px;
        border-radius: 50%;
        background: rgba(255,255,255,0.12);
    }
    .attempt-dot.used { background: #ff4d6d; }

    .lockout-box {
        background: rgba(255,30,60,0.1);
        border: 1px solid rgba(255,30,60,0.3);
        border-radius: 10px;
        padding: 12px 14px;
        font-family: 'Inter', sans-serif;
        font-size: 13px;
        color: #ff6b81;
        text-align: center;
        margin-top: 10px;
    }

    .divider-line {
        border: none;
        border-top: 1px solid rgba(255,255,255,0.07);
        margin: 20px 0 14px;
    }

    .login-footer {
        font-family: 'Inter', sans-serif;
        font-size: 11px;
        color: rgba(255,255,255,0.2);
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)


def login_card():
    inject_login_styles()

    st.markdown("""
    <div class="login-wrap">
       
        <div class="login-title">Student Portal</div>
        <div class="login-sub">Sign in to access the performance predictor</div>
    </div>
    """, unsafe_allow_html=True)

    # inputs sit below the decorative header card
    with st.container():
        if st.session_state.locked_out:
            st.markdown('<div class="lockout-box"> Too many failed attempts. Refresh the page to try again.</div>', unsafe_allow_html=True)
            return

        username = st.text_input("Username", placeholder="admin", key="login_user")
        password = st.text_input("Password", placeholder="••••••••", type="password", key="login_pass")

        # show remaining attempts as dots
        if st.session_state.login_attempts > 0:
            dots = ""
            for i in range(MAX_ATTEMPTS):
                cls = "attempt-dot used" if i < st.session_state.login_attempts else "attempt-dot"
                dots += f'<div class="{cls}"></div>'
            remaining = MAX_ATTEMPTS - st.session_state.login_attempts
            st.markdown(
                f'<div style="font-family:Inter,sans-serif;font-size:11px;color:rgba(255,100,100,0.8);margin-bottom:4px;">'
                f'Wrong password — {remaining} attempt{"s" if remaining != 1 else ""} left</div>'
                f'<div class="attempt-bar">{dots}</div>',
                unsafe_allow_html=True
            )

        if st.button("Sign In →", use_container_width=True):
            if username == USERNAME and password == PASSWORD:
                st.session_state.logged_in = True
                st.session_state.login_attempts = 0
                st.session_state.last_login = datetime.now().strftime("%d %b %Y, %I:%M %p")
                st.rerun()
            else:
                st.session_state.login_attempts += 1
                if st.session_state.login_attempts >= MAX_ATTEMPTS:
                    st.session_state.locked_out = True
                st.rerun()

        st.markdown('<hr class="divider-line">', unsafe_allow_html=True)
        st.markdown('<div class="login-footer">Default credentials: admin / 1234 &nbsp;·&nbsp; © 2025 SPP</div>', unsafe_allow_html=True)


# ---- model training (cached so it only runs once) ----

@st.cache_resource
def train_model():
    df = pd.read_csv("StudentPerformanceFactors.csv")

    df = df.drop(columns=[
        "Extracurricular_Activities", "Tutoring_Sessions", "Family_Income",
        "School_Type", "Peer_Influence", "Physical_Activity",
        "Learning_Disabilities", "Distance_from_Home",
        "Parental_Involvement", "Teacher_Quality", "Parental_Education_Level",
        "Sleep_Hours"
    ])

    ordinal_cols = ["Access_to_Resources", "Motivation_Level", "Internet_Access"]
    categories = [
        ["Low", "Medium", "High"],
        ["Low", "Medium", "High"],
        ["No", "Yes"]
    ]

    for col in ordinal_cols:
        df[col] = df[col].astype(str).str.strip()
        valid = df[df[col] != "nan"][col]
        mode_val = valid.mode()[0] if not valid.empty else "Medium"
        df[col] = df[col].replace("nan", mode_val)

    encoder = OrdinalEncoder(categories=categories, handle_unknown="use_encoded_value", unknown_value=-1)
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

    rf = RandomForestRegressor(n_estimators=100, bootstrap=True, max_samples=0.8, max_features=5, random_state=42)
    rf.fit(X_train, y_train)

    preds = rf.predict(X_test)
    mae  = round(mean_absolute_error(y_test, preds), 2)
    rmse = round(np.sqrt(mean_squared_error(y_test, preds)), 2)
    r2   = round(r2_score(y_test, preds), 4)

    return rf, scaler, numeric_features, X.columns, mae, rmse, r2


# ---- score gauge (SVG donut) ----

def score_gauge(score):
    pct = score / 100
    color = "#22c55e" if score >= 80 else "#f59e0b" if score >= 50 else "#ef4444"
    circumference = 2 * 3.14159 * 54
    dash = pct * circumference

    st.markdown(f"""
    <div style="display:flex;justify-content:center;margin:16px 0 8px;">
        <svg width="140" height="140" viewBox="0 0 120 120">
            <circle cx="60" cy="60" r="54" fill="none" stroke="rgba(255,255,255,0.07)" stroke-width="10"/>
            <circle cx="60" cy="60" r="54" fill="none" stroke="{color}" stroke-width="10"
                stroke-dasharray="{dash:.1f} {circumference:.1f}"
                stroke-dashoffset="{circumference/4:.1f}"
                stroke-linecap="round"
                style="transition: stroke-dasharray 1s ease;">
            </circle>
            <text x="60" y="56" text-anchor="middle" fill="#fff"
                font-family="Syne,sans-serif" font-size="22" font-weight="800">{score}</text>
            <text x="60" y="72" text-anchor="middle" fill="rgba(255,255,255,0.4)"
                font-family="Inter,sans-serif" font-size="9">out of 100</text>
        </svg>
    </div>
    """, unsafe_allow_html=True)


# ---- grade badge ----

def grade_badge(grade, status):
    colors = {"A": "#22c55e", "B": "#3b82f6", "C": "#f59e0b", "D": "#ef4444"}
    color = colors.get(grade, "#888")
    status_color = "#22c55e" if status == "Pass" else "#ef4444"

    st.markdown(f"""
    <div style="display:flex;gap:10px;justify-content:center;margin:8px 0 16px;">
        <span style="background:{color}22;border:1px solid {color}55;color:{color};
            padding:5px 18px;border-radius:99px;font-family:Inter,sans-serif;
            font-weight:600;font-size:14px;letter-spacing:0.5px;">
            Grade {grade}
        </span>
        <span style="background:{status_color}22;border:1px solid {status_color}55;color:{status_color};
            padding:5px 18px;border-radius:99px;font-family:Inter,sans-serif;
            font-weight:600;font-size:14px;letter-spacing:0.5px;">
            {status}
        </span>
    </div>
    """, unsafe_allow_html=True)


# ---- prediction history table ----

def show_history():
    if not st.session_state.prediction_history:
        return
    with st.expander(f" Prediction History ({len(st.session_state.prediction_history)} runs)"):
        hist_df = pd.DataFrame(st.session_state.prediction_history)
        st.dataframe(hist_df, use_container_width=True, hide_index=True)
        if st.button("Clear History"):
            st.session_state.prediction_history = []
            st.rerun()


# ---- main app ----

def main_app():
    rf_model, scaler, numeric_features, feature_cols, mae, rmse, r2 = train_model()

    # sidebar
    with st.sidebar:
        st.markdown("###  Admin")
        if st.session_state.last_login:
            st.caption(f"Last login: {st.session_state.last_login}")
        st.divider()

        # model metrics in sidebar instead of taking up main space
        st.markdown("**Model Performance**")
        st.metric("MAE", mae,  help="Mean Absolute Error — lower is better")
        st.metric("RMSE", rmse, help="Root Mean Squared Error")
        st.metric("R²",  r2,   help="Explained variance — closer to 1 is better")
        st.divider()

        if st.button(" Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.rerun()

    st.title(" Student Performance Predictor")
    st.caption("Random Forest model · fill in the form below and hit Predict")

    st.subheader("Student Details")
    col1, col2 = st.columns(2)

    with col1:
        hours_studied   = st.number_input("Hours Studied / day",  min_value=0.0, max_value=24.0,  value=None, step=0.5, placeholder="e.g. 6")
        attendance      = st.number_input("Attendance %",          min_value=0.0, max_value=100.0, value=None, placeholder="e.g. 80")
        previous_scores = st.number_input("Previous Exam Score",   min_value=0.0, max_value=100.0, value=None, placeholder="e.g. 70")

    with col2:
        access_to_resources = st.selectbox("Access to Resources", ["Low", "Medium", "High"], index=None, placeholder="Select...")
        motivation_level    = st.selectbox("Motivation Level",    ["Low", "Medium", "High"], index=None, placeholder="Select...")
        internet_access     = st.selectbox("Internet Access",     ["No", "Yes"],             index=None, placeholder="Select...")
        gender              = st.selectbox("Gender",              ["Female", "Male", "Other"], index=None, placeholder="Select...")

    encode_map = {
        "Low": 0, "Medium": 1, "High": 2,
        "No": 0, "Yes": 1,
        "Female": 0, "Male": 1
    }

    if st.button("Predict Score", use_container_width=True, type="primary"):
        all_inputs = [hours_studied, attendance, previous_scores,
                      access_to_resources, motivation_level, internet_access, gender]

        if None in all_inputs:
            st.warning("Please fill in all fields before predicting.")
        else:
            new_student = pd.DataFrame([[
                hours_studied,
                attendance,
                previous_scores,
                encode_map[access_to_resources],
                encode_map[motivation_level],
                encode_map[internet_access],
                encode_map[gender]
            ]], columns=feature_cols)

            new_student[numeric_features] = scaler.transform(new_student[numeric_features])

            score  = round(float(np.clip(rf_model.predict(new_student)[0], 0, 100)), 1)
            grade  = "A" if score >= 80 else "B" if score >= 65 else "C" if score >= 50 else "D"
            status = "Pass" if score >= 50 else "Fail"

            # what-if: +1 study hour
            what_if = new_student.copy()
            what_if["Hours_Studied"] += 1
            improved = round(float(np.clip(rf_model.predict(what_if)[0], 0, 100)), 1)
            gain = round(improved - score, 1)

            # top contributing factor
            top_feature = feature_cols[rf_model.feature_importances_.argmax()]
            student_val = new_student.iloc[0][top_feature]
            factor_msg  = "above average — keep it up " if student_val >= 0 else "below average — focus here "

            # save to history
            st.session_state.prediction_history.append({
                "Time": datetime.now().strftime("%H:%M:%S"),
                "Score": score,
                "Grade": grade,
                "Status": status,
                "Hours Studied": hours_studied,
                "Attendance %": attendance,
                "Prev Score": previous_scores,
            })

            st.divider()
            st.subheader("Result")

            # visual gauge + grade badge
            score_gauge(score)
            grade_badge(grade, status)

            # what-if tip
            if gain > 0:
                st.success(f" Studying **1 more hour/day** could push your score to **{improved}** (+{gain} pts)")
            else:
                st.success(" You're already maximising study hours — great work!")

            # top factor callout
            st.info(f" Top influencing factor: **{top_feature.replace('_', ' ')}** — {factor_msg}")

    st.divider()
    show_history()


# ---- router ----

if st.session_state.logged_in:
    main_app()
else:
    login_card()

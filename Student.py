import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import hashlib
import json
import os

# ─────────────────────────────────────────────
# USER DATABASE  (stored in users.json)
# ─────────────────────────────────────────────
USERS_FILE = "users.json"

def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    # Default admin account
    return {
        "admin": {
            "password": hash_password("admin123"),
            "role": "admin",
            "name": "Administrator"
        }
    }

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def authenticate(username, password):
    users = load_users()
    if username in users:
        if users[username]["password"] == hash_password(password):
            return users[username]
    return None

def register_user(username, password, name):
    users = load_users()
    if username in users:
        return False, "Username already exists."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    if not username.strip():
        return False, "Username cannot be empty."
    users[username] = {
        "password": hash_password(password),
        "role": "user",
        "name": name
    }
    save_users(users)
    return True, "Account created successfully!"

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="",
    layout="centered"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
}

/* Card container */
.auth-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 20px;
    padding: 2.5rem;
    margin: 2rem auto;
    max-width: 480px;
    box-shadow: 0 20px 60px rgba(0,0,0,0.4), 0 0 0 1px rgba(255,255,255,0.05);
}

.auth-title {
    text-align: center;
    font-size: 2rem;
    font-weight: 700;
    color: #e2e8f0;
    margin-bottom: 0.3rem;
}

.auth-subtitle {
    text-align: center;
    font-size: 0.95rem;
    color: #94a3b8;
    margin-bottom: 2rem;
}

/* Style Streamlit inputs */
div[data-testid="stTextInput"] input {
    border-radius: 10px !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    background: rgba(255,255,255,0.05) !important;
    color: #e2e8f0 !important;
    padding: 0.6rem 1rem !important;
}

div[data-testid="stTextInput"] input:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 2px rgba(99,102,241,0.3) !important;
}

/* Primary button */
div[data-testid="stButton"] > button[kind="primary"],
div[data-testid="stButton"] > button {
    width: 100%;
    border-radius: 10px !important;
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 0.65rem 1rem !important;
    transition: transform 0.15s, box-shadow 0.15s !important;
}

div[data-testid="stButton"] > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(99,102,241,0.5) !important;
}

.divider-text {
    text-align: center;
    color: #64748b;
    font-size: 0.85rem;
    margin: 1rem 0;
    position: relative;
}

.divider-text::before,
.divider-text::after {
    content: "";
    position: absolute;
    top: 50%;
    width: 40%;
    height: 1px;
    background: rgba(255,255,255,0.1);
}

.divider-text::before { left: 0; }
.divider-text::after  { right: 0; }

/* Logout btn in sidebar */
section[data-testid="stSidebar"] div[data-testid="stButton"] > button {
    background: rgba(239,68,68,0.15) !important;
    color: #f87171 !important;
    border: 1px solid rgba(239,68,68,0.3) !important;
    width: auto !important;
}

section[data-testid="stSidebar"] div[data-testid="stButton"] > button:hover {
    background: rgba(239,68,68,0.25) !important;
    transform: none !important;
    box-shadow: none !important;
}

.user-badge {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white;
    border-radius: 50%;
    width: 42px;
    height: 42px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 1.1rem;
    margin-right: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username  = ""
    st.session_state.user_info = {}
    st.session_state.auth_mode = "login"   # "login" | "register"

# ─────────────────────────────────────────────
# AUTH PAGES
# ─────────────────────────────────────────────
def show_login():
    st.markdown('<div class="auth-title">🎓 Welcome Back</div>', unsafe_allow_html=True)
    st.markdown('<div class="auth-subtitle">Sign in to your account</div>', unsafe_allow_html=True)

    username = st.text_input("Username", placeholder="Enter your username", key="login_user")
    password = st.text_input("Password", placeholder="Enter your password", type="password", key="login_pass")

    st.write("")
    if st.button("Sign In", use_container_width=True):
        if not username or not password:
            st.error("Please enter both username and password.")
        else:
            user = authenticate(username, password)
            if user:
                st.session_state.logged_in = True
                st.session_state.username  = username
                st.session_state.user_info = user
                st.success(f"Welcome back, {user['name']}!")
                st.rerun()
            else:
                st.error("Invalid username or password.")

    st.markdown('<div class="divider-text">or</div>', unsafe_allow_html=True)

    if st.button("Create a new account", use_container_width=True):
        st.session_state.auth_mode = "register"
        st.rerun()

    st.markdown("---")
    st.caption("**Default admin credentials:** username: `admin` / password: `admin123`")


def show_register():
    st.markdown('<div class="auth-title">🎓 Create Account</div>', unsafe_allow_html=True)
    st.markdown('<div class="auth-subtitle">Join to start predicting scores</div>', unsafe_allow_html=True)

    name     = st.text_input("Full Name",        placeholder="e.g. Rahul Sharma",  key="reg_name")
    username = st.text_input("Username",         placeholder="Choose a username",   key="reg_user")
    password = st.text_input("Password",         placeholder="Min. 6 characters",  type="password", key="reg_pass")
    confirm  = st.text_input("Confirm Password", placeholder="Repeat password",     type="password", key="reg_confirm")

    st.write("")
    if st.button("Register", use_container_width=True):
        if not all([name, username, password, confirm]):
            st.error("Please fill in all fields.")
        elif password != confirm:
            st.error("Passwords do not match.")
        else:
            ok, msg = register_user(username, password, name)
            if ok:
                st.success(msg + " Please sign in.")
                st.session_state.auth_mode = "login"
                st.rerun()
            else:
                st.error(msg)

    st.markdown('<div class="divider-text">or</div>', unsafe_allow_html=True)

    if st.button("Back to Sign In", use_container_width=True):
        st.session_state.auth_mode = "login"
        st.rerun()


def show_auth_page():
    # Centered layout
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        if st.session_state.auth_mode == "login":
            show_login()
        else:
            show_register()


# ─────────────────────────────────────────────
# MODEL TRAINING
# ─────────────────────────────────────────────
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


# ─────────────────────────────────────────────
# MAIN APP  (shown after login)
# ─────────────────────────────────────────────
def show_main_app():
    # ── Sidebar ──────────────────────────────
    with st.sidebar:
        user = st.session_state.user_info
        initial = user["name"][0].upper()
        st.markdown(
            f'<div style="display:flex;align-items:center;margin-bottom:0.5rem;">'
            f'<span class="user-badge">{initial}</span>'
            f'<div><div style="font-weight:600;font-size:1rem;">{user["name"]}</div>'
            f'<div style="font-size:0.8rem;color:#94a3b8;">@{st.session_state.username}</div></div>'
            f'</div>',
            unsafe_allow_html=True
        )
        st.markdown(f"`Role: {user['role']}`")
        st.divider()
        if st.button(" Sign Out"):
            st.session_state.logged_in = False
            st.session_state.username  = ""
            st.session_state.user_info = {}
            st.rerun()

    # ── Page ─────────────────────────────────
    st.title(" Student Performance Predictor")
    st.caption("Predict exam score using a Random Forest model")

    rf_model, scaler, numeric_features, feature_cols, mae, rmse, r2 = train_model()

    with st.expander(" Model Metrics"):
        c1, c2, c3 = st.columns(3)
        c1.metric("MAE",  mae)
        c2.metric("RMSE", rmse)
        c3.metric("R²",   r2)

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

    if st.button(" Predict Score", use_container_width=True):
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
            status = "Pass " if score >= 50 else "Fail "

            top_feature   = feature_cols[rf_model.feature_importances_.argmax()]
            student_value = new_student.iloc[0][top_feature]
            factor_status = "Above average — keep it up " if student_value >= 0 else "Below average — focus here "

            what_if = new_student.copy()
            what_if["Hours_Studied"] += 1
            improved = round(float(np.clip(rf_model.predict(what_if)[0], 0, 100)), 1)
            gain     = round(improved - score, 1)

            st.divider()
            st.subheader(" Result")

            r1, r2_col, r3 = st.columns(3)
            r1.metric("Predicted Score", f"{score} / 100")
            r2_col.metric("Grade", grade)
            r3.metric("Status", status)

            st.progress(int(score))
            st.info(f"Top Factor: **{top_feature.replace('_', ' ')}** — {factor_status}")

            if gain > 0:
                st.success(f" Tip: Studying 1 more hour/day could raise your score to {improved} (+{gain} pts)")
            else:
                st.success(" Tip: You are already maximizing your study hours!")


# ─────────────────────────────────────────────
# ROUTER
# ─────────────────────────────────────────────
if st.session_state.logged_in:
    show_main_app()
else:
    show_auth_page()

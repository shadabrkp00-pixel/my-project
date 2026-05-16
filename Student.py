import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Student Performance Predictor", layout="centered")

# ── Credentials (change as needed) ───────────────────────────────────────────
VALID_USERNAME = "admin"
VALID_PASSWORD = "1234"

# ── Session state init ────────────────────────────────────────────────────────
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "login_error" not in st.session_state:
    st.session_state.login_error = False


# ══════════════════════════════════════════════════════════════════════════════
#  LOGIN PAGE
# ══════════════════════════════════════════════════════════════════════════════
def show_login():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500&display=swap');

    /* Hide Streamlit chrome */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding: 0 !important; max-width: 100% !important; }

    /* Full-screen background */
    .stApp {
        background: #0a0a0f;
        min-height: 100vh;
    }

    /* Animated gradient orbs */
    .bg-orb {
        position: fixed;
        border-radius: 50%;
        filter: blur(80px);
        opacity: 0.35;
        animation: drift 8s ease-in-out infinite alternate;
        pointer-events: none;
        z-index: 0;
    }
    .orb1 { width:500px; height:500px; background:#6c3fff; top:-120px; left:-100px; animation-delay:0s; }
    .orb2 { width:400px; height:400px; background:#ff3cac; bottom:-100px; right:-80px; animation-delay:2s; }
    .orb3 { width:300px; height:300px; background:#00d4ff; top:40%; left:40%; animation-delay:4s; }

    @keyframes drift {
        0%   { transform: translate(0,0) scale(1); }
        100% { transform: translate(30px,20px) scale(1.08); }
    }

    /* Card wrapper */
    .login-card {
        position: relative;
        z-index: 10;
        background: rgba(255,255,255,0.04);
        backdrop-filter: blur(24px);
        -webkit-backdrop-filter: blur(24px);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 28px;
        padding: 52px 48px 12px;
        max-width: 440px;
        margin: 72px auto 0;
        box-shadow: 0 32px 80px rgba(0,0,0,0.55), 0 0 0 1px rgba(255,255,255,0.05);
        animation: fadeUp 0.7s cubic-bezier(.22,1,.36,1) both;
    }

    @keyframes fadeUp {
        from { opacity:0; transform:translateY(32px); }
        to   { opacity:1; transform:translateY(0); }
    }

    /* Badge */
    .badge {
        display: inline-block;
        background: linear-gradient(135deg,#6c3fff,#ff3cac);
        color: #fff;
        font-family: 'DM Sans', sans-serif;
        font-size: 11px;
        font-weight: 500;
        letter-spacing: 2px;
        text-transform: uppercase;
        padding: 5px 14px;
        border-radius: 99px;
        margin-bottom: 20px;
    }

    .login-title {
        font-family: 'Playfair Display', serif;
        font-size: 40px;
        font-weight: 900;
        color: #ffffff;
        line-height: 1.1;
        margin: 0 0 6px;
    }
    .login-sub {
        font-family: 'DM Sans', sans-serif;
        font-size: 14px;
        color: rgba(255,255,255,0.40);
        margin: 0 0 32px;
        font-weight: 300;
    }

    /* Streamlit input overrides */
    .stTextInput > div > div > input {
        background: rgba(255,255,255,0.06) !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        border-radius: 14px !important;
        color: #ffffff !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 15px !important;
        padding: 14px 18px !important;
        transition: border-color 0.2s, box-shadow 0.2s !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #6c3fff !important;
        box-shadow: 0 0 0 3px rgba(108,63,255,0.25) !important;
        outline: none !important;
    }
    .stTextInput > div > div > input::placeholder {
        color: rgba(255,255,255,0.22) !important;
    }
    .stTextInput label {
        font-family: 'DM Sans', sans-serif !important;
        font-size: 11px !important;
        font-weight: 500 !important;
        letter-spacing: 1.4px !important;
        text-transform: uppercase !important;
        color: rgba(255,255,255,0.45) !important;
    }

    /* Streamlit button overrides */
    div[data-testid="stButton"] > button {
        width: 100% !important;
        background: linear-gradient(135deg,#6c3fff 0%,#ff3cac 100%) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 14px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 15px !important;
        font-weight: 500 !important;
        padding: 14px 0 !important;
        letter-spacing: 0.5px !important;
        cursor: pointer !important;
        transition: opacity 0.2s, transform 0.15s !important;
        margin-top: 6px !important;
    }
    div[data-testid="stButton"] > button:hover {
        opacity: 0.85 !important;
        transform: translateY(-1px) !important;
    }

    /* Error box */
    .err-box {
        background: rgba(255,60,100,0.12);
        border: 1px solid rgba(255,60,100,0.35);
        border-radius: 12px;
        padding: 12px 16px;
        font-family: 'DM Sans', sans-serif;
        font-size: 13px;
        color: #ff7096;
        margin-top: 10px;
        text-align: center;
    }

    /* Hint + footer */
    .hint-row {
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 22px 0 0;
    }
    .hint-line { flex:1; height:1px; background:rgba(255,255,255,0.08); }
    .hint-text {
        font-family: 'DM Sans', sans-serif;
        font-size: 11px;
        color: rgba(255,255,255,0.25);
        white-space: nowrap;
    }
    .login-footer {
        font-family: 'DM Sans', sans-serif;
        font-size: 11px;
        color: rgba(255,255,255,0.18);
        text-align: center;
        margin-top: 18px;
        padding-bottom: 8px;
    }
    </style>

    <!-- background orbs -->
    <div class="bg-orb orb1"></div>
    <div class="bg-orb orb2"></div>
    <div class="bg-orb orb3"></div>

    <div class="login-card">
        <div class="badge">✦ Student Portal</div>
        <div class="login-title">Welcome<br>Back.</div>
        <div class="login-sub">Sign in to access the predictor</div>
    </div>
    """, unsafe_allow_html=True)

    # Inputs rendered inside the visual card area
    with st.container():
        # Nudge inputs to overlap with the card visually
        st.markdown("""
        <style>
        section[data-testid="stMain"] > div > div > div > div:nth-child(2) {
            max-width: 440px;
            margin: 0 auto;
            padding: 0 48px 32px;
            background: rgba(255,255,255,0.04);
            backdrop-filter: blur(24px);
            border: 1px solid rgba(255,255,255,0.10);
            border-top: none;
            border-radius: 0 0 28px 28px;
            margin-top: -4px;
        }
        </style>
        """, unsafe_allow_html=True)

        username = st.text_input("Username", placeholder="e.g.  admin", key="usr")
        password = st.text_input("Password", placeholder="••••••••", type="password", key="pwd")

        if st.button("Sign In  →", use_container_width=True):
            if username == VALID_USERNAME and password == VALID_PASSWORD:
                st.session_state.logged_in = True
                st.session_state.login_error = False
                st.rerun()
            else:
                st.session_state.login_error = True

        if st.session_state.login_error:
            st.markdown('<div class="err-box">⚠ Incorrect username or password. Try again.</div>',
                        unsafe_allow_html=True)

        st.markdown("""
        <div class="hint-row">
            <div class="hint-line"></div>
            <div class="hint-text">default credentials: admin / 1234</div>
            <div class="hint-line"></div>
        </div>
        <div class="login-footer">© 2025 Student Performance Predictor</div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN APP
# ══════════════════════════════════════════════════════════════════════════════
def show_app():
    with st.sidebar:
        st.markdown("### 👤 Logged in as `admin`")
        if st.button("🚪 Logout"):
            st.session_state.logged_in = False
            st.rerun()

    st.title("Student Performance Predictor")
    st.caption("Predict exam score using Random Forest model")

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

        ordinal_cols = [
            "Access_to_Resources", "Motivation_Level",
            "Internet_Access"
        ]
        categories = [
            ["Low", "Medium", "High"],
            ["Low", "Medium", "High"],
            ["No", "Yes"]
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
        previous_scores = st.number_input("Previous Exam Score",   min_value=0.0, max_value=100.0, value=None, placeholder="e.g. 70")

    with col2:
        access_to_resources = st.selectbox("Access to Resources", ["Low", "Medium", "High"], index=None, placeholder="Select...")
        motivation_level    = st.selectbox("Motivation Level",    ["Low", "Medium", "High"], index=None, placeholder="Select...")
        internet_access     = st.selectbox("Internet Access",     ["No", "Yes"],             index=None, placeholder="Select...")
        gender              = st.selectbox("Gender",              ["Female", "Male", "Other"], index=None, placeholder="Select...")

    encode_map = {
        "Low":0, "Medium":1, "High":2,
        "No":0,  "Yes":1,
        "Female":0, "Male":1
    }

    if st.button("Predict Score", use_container_width=True):
        all_inputs = [hours_studied, attendance, previous_scores,
                      access_to_resources, motivation_level,
                      internet_access, gender]

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

            r1, r2_col, r3 = st.columns(3)
            r1.metric("Predicted Score", f"{score} / 100")
            r2_col.metric("Grade", grade)
            r3.metric("Status", status)

            st.progress(int(score))
            st.info(f"Top Factor: **{top_feature.replace('_', ' ')}** — {factor_status}")

            if gain > 0:
                st.success(f"Tip: Studying 1 more hour/day could raise your score to {improved} (+{gain} pts)")
            else:
                st.success("Tip: You are already maximizing your study hours!")


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTER
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.logged_in:
    show_app()
else:
    show_login()

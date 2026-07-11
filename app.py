import streamlit as st
import pandas as pd
import joblib

# ─────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Health Insurance Claim Predictor",
    page_icon="🏥",
    layout="wide",
)

# ─────────────────────────────────────────────
# Custom CSS Styling
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); min-height: 100vh; }

    .stApp { background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%); }

    .hero-box {
        background: linear-gradient(135deg, rgba(99,102,241,0.25), rgba(168,85,247,0.15));
        border: 1px solid rgba(99,102,241,0.4);
        border-radius: 20px;
        padding: 2rem 2.5rem;
        margin-bottom: 2rem;
        backdrop-filter: blur(10px);
    }
    .hero-box h1 { color: #e0d7ff; font-size: 2.2rem; margin: 0; font-weight: 700; }
    .hero-box p  { color: #a5b4fc; margin: 0.4rem 0 0; font-size: 1rem; }

    .section-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 1.5rem 2rem;
        margin-bottom: 1.5rem;
        backdrop-filter: blur(8px);
    }
    .section-title {
        color: #a5b4fc;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid rgba(99,102,241,0.3);
    }

    /* Result boxes */
    .result-approved {
        background: linear-gradient(135deg, rgba(16,185,129,0.2), rgba(5,150,105,0.1));
        border: 1px solid rgba(16,185,129,0.5);
        border-radius: 16px;
        padding: 1.5rem 2rem;
        margin-top: 1.5rem;
        text-align: center;
    }
    .result-denied {
        background: linear-gradient(135deg, rgba(239,68,68,0.2), rgba(185,28,28,0.1));
        border: 1px solid rgba(239,68,68,0.5);
        border-radius: 16px;
        padding: 1.5rem 2rem;
        margin-top: 1.5rem;
        text-align: center;
    }
    .result-title { font-size: 1.6rem; font-weight: 700; margin: 0; }
    .result-sub   { font-size: 0.95rem; margin: 0.4rem 0 0; opacity: 0.8; }
    .claim-amount { font-size: 3rem; font-weight: 700; color: #34d399; margin: 1rem 0 0; }
    .confidence-badge {
        display: inline-block;
        background: rgba(99,102,241,0.3);
        border: 1px solid rgba(99,102,241,0.5);
        border-radius: 20px;
        padding: 0.3rem 1rem;
        font-size: 0.85rem;
        color: #c4b5fd;
        margin-top: 0.8rem;
    }

    div[data-testid="stNumberInput"] label,
    div[data-testid="stSelectbox"] label,
    
    /* Radio labels and options */
div[data-testid="stRadio"] label,
div[data-testid="stRadio"] span,
div[data-testid="stRadio"] p,
div[role="radiogroup"] label,
div[role="radiogroup"] span,
div[role="radiogroup"] p {
    color: white !important; font-size: 0.9rem ;
}

    div.stButton > button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 3rem;
        font-size: 1.05rem;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        width: 100%;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 20px rgba(99,102,241,0.4);
    }
    div.stButton > button:hover {
        background: linear-gradient(135deg, #4f46e5, #7c3aed);
        box-shadow: 0 6px 30px rgba(99,102,241,0.6);
        transform: translateY(-1px);
    }
    div[data-testid="stProgress"] > div { background-color: rgba(99,102,241,0.2) !important; }
    div[data-testid="stProgress"] > div > div { background: linear-gradient(90deg, #6366f1, #8b5cf6) !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Load Models & Feature Lists
# ─────────────────────────────────────────────
@st.cache_resource
def load_models(cache_buster=3): # Incremented to force cache clear
    try:
        classifier              = joblib.load("xgb_classifier.pkl")
        regressor               = joblib.load("xgb_regressor.pkl")
        classification_features = joblib.load("classification_features.pkl")
        regression_features     = joblib.load("regression_features.pkl")
        return classifier, regressor, classification_features, regression_features
    except FileNotFoundError as e:
        st.error(f"❌ Model file not found: {e}. Make sure all .pkl files are in the same directory as app.py.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Failed to load models: {e}")
        st.stop()

classifier, regressor, classification_features, regression_features = load_models(cache_buster=2)

# ─────────────────────────────────────────────
# Complete Dropdown Lists (from training data)
# ─────────────────────────────────────────────
CITIES = [
    'Atlanta', 'AtlanticCity', 'Bakersfield', 'Baltimore', 'Bloomington',
    'Boston', 'Brimingham', 'Brookings', 'Buffalo', 'Cambridge', 'Canton',
    'Carlsbad', 'Charleston', 'Charlotte', 'Chicago', 'Cincinnati',
    'Cleveland', 'Columbia', 'Columbus', 'Denver', 'Escabana', 'Eureka',
    'FallsCity', 'Fargo', 'Florence', 'Fresno', 'Georgia', 'GrandForks',
    'Harrisburg', 'Hartford', 'Houston', 'Huntsville', 'Indianapolis',
    'IowaCity', 'JeffersonCity', 'KanasCity', 'Kingman', 'Kingsport',
    'Knoxville', 'LasVegas', 'Lincoln', 'LosAngeles', 'Louisville',
    'Lovelock', 'Macon', 'Mandan', 'Marshall', 'Memphis', 'Mexicali',
    'Miami', 'Minneapolis', 'Minot', 'Montrose', 'Nashville', 'NewOrleans',
    'NewYork', 'Newport', 'Oceanside', 'Oklahoma', 'Orlando', 'Oxnard',
    'PanamaCity', 'Pheonix', 'Phildelphia', 'Pittsburg', 'Portland',
    'Prescott', 'Providence', 'Raleigh', 'Reno', 'Rochester', 'Salina',
    'SanDeigo', 'SanFrancisco', 'SanJose', 'SanLuis', 'SantaFe',
    'SantaRosa', 'SilverCity', 'Springfield', 'Stamford', 'Syracuse',
    'Tampa', 'Trenton', 'Tucson', 'Warwick', 'WashingtonDC', 'Waterloo',
    'Worcester', 'York', 'Youngstown'
]

JOB_TITLES = [
    'Academician', 'Accountant', 'Actor', 'Analyst', 'Architect',
    'Beautician', 'Blogger', 'Buisnessman', 'CA', 'CEO', 'Chef', 'Clerks',
    'Dancer', 'DataScientist', 'DefencePersonnels', 'Doctor', 'Engineer',
    'Farmer', 'FashionDesigner', 'FilmDirector', 'FilmMaker', 'GovEmployee',
    'HomeMakers', 'HouseKeeper', 'ITProfessional', 'Journalist', 'Labourer',
    'Lawyer', 'Manager', 'Photographer', 'Police', 'Politician', 'Singer',
    'Student', 'Technician'
]

DISEASES = [
    'NoDisease', 'Alzheimer', 'Arthritis', 'Cancer', 'Diabetes',
    'Epilepsy', 'EyeDisease', 'HeartDisease', 'High BP', 'Obesity'
]

# ─────────────────────────────────────────────
# Hero Header
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero-box">
    <h1>🏥 Health Insurance Claim Predictor</h1>
    <p>Fill in the patient details below. Our AI will first determine claim eligibility, then estimate the claim amount if approved.</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Input Form — Two Columns Layout
# ─────────────────────────────────────────────
col_left, col_right = st.columns(2, gap="large")

with col_left:
    # ── Personal Details ──────────────────────
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">👤 Personal Details</div>', unsafe_allow_html=True)

    age = st.number_input(
        "Age", min_value=18, max_value=64, value=35, step=1,
        help="Patient age (18–64)"
    )
    sex = st.selectbox("Gender", ["male", "female"])
    weight = st.number_input(
        "Weight (kg)", min_value=34, max_value=95, value=65, step=1,
        help="Body weight in kilograms (34–95 kg)"
    )
    bmi = st.number_input(
        "BMI (Body Mass Index)", min_value=10.0, max_value=60.0, value=22.5, step=0.1,
        help="BMI = weight(kg) / height(m)²"
    )
    no_of_dependents = st.number_input(
        "Number of Dependents", min_value=0, max_value=10, value=1, step=1,
        help="Family members covered under this policy"
    )
    city = st.selectbox("City", CITIES, index=CITIES.index("NewYork"))
    job_title = st.selectbox("Occupation", JOB_TITLES, index=JOB_TITLES.index("Engineer"))

    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    # ── Health Details ────────────────────────
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🩺 Health & Lifestyle Details</div>', unsafe_allow_html=True)

    bloodpressure = st.number_input(
        "Blood Pressure (mmHg)", min_value=60, max_value=200, value=80, step=1,
        help="Diastolic blood pressure in mmHg"
    )
    hereditary_diseases = st.selectbox(
        "Hereditary Disease", DISEASES, index=0,
        help="Select 'NoDisease' if no hereditary conditions"
    )
    smoker = st.radio(
        "Smoker?", options=[0, 1], format_func=lambda x: "Yes 🚬" if x == 1 else "No ✅",
        horizontal=True, index=0
    )
    diabetes = st.radio(
        "Diabetes?", options=[0, 1], format_func=lambda x: "Yes ⚠️" if x == 1 else "No ✅",
        horizontal=True, index=0
    )
    regular_ex = st.radio(
        "Regular Exercise?", options=[0, 1], format_func=lambda x: "Yes 💪" if x == 1 else "No 🛋️",
        horizontal=True, index=0
    )

    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Predict Button
# ─────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    predict_clicked = st.button("🔍 Predict Claim", use_container_width=True)

# ─────────────────────────────────────────────
# Prediction Logic
# ─────────────────────────────────────────────
if predict_clicked:
    with st.spinner("Analyzing your information..."):

        # Build raw input DataFrame
        raw_data = pd.DataFrame({
            "age":                  [float(age)],
            "sex":                  [1 if sex == "male" else 0],
            "weight":               [weight],
            "bmi":                  [bmi],
            "no_of_dependents":     [no_of_dependents],
            "smoker":               [smoker],
            "bloodpressure":        [bloodpressure],
            "diabetes":             [diabetes],
            "regular_ex":           [regular_ex],
            "city":                 [city],
            "hereditary_diseases":  [hereditary_diseases],
            "job_title":            [job_title],
        })

        # ── One-hot encode categorical columns ────
        data_encoded = pd.get_dummies(raw_data)

        # ── Classification Prediction ─────────────
        # Add all missing training columns at once (avoids DataFrame fragmentation)
        missing_clf_cols = {col: [0] for col in classification_features if col not in data_encoded.columns}
        clf_input = pd.concat([data_encoded, pd.DataFrame(missing_clf_cols)], axis=1)
        clf_input = clf_input[classification_features]

        approval       = classifier.predict(clf_input)[0]
        approval_proba = classifier.predict_proba(clf_input)[0]  # [prob_denied, prob_approved]
        confidence     = float(approval_proba[int(approval)]) * 100  # cast to Python float

        print("=== DEBUG INFERENCE ===")
        print("Raw Data:\n", raw_data)
        print("Prediction:", approval, "Confidence:", confidence)
        print("=======================")

    # ── Show Results ──────────────────────────────
    if approval == 1:
        # --- Claim Approved ---
        st.markdown(f"""
        <div class="result-approved">
            <div class="result-title" style="color:#34d399;">✅ Claim Approved</div>
            <div class="result-sub" style="color:#6ee7b7;">
                The model predicts this claim will be approved.
            </div>
            <div class="confidence-badge">🎯 Confidence: {confidence:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

        # ── Regression Prediction ──────────────────
        missing_reg_cols = {col: [0] for col in regression_features if col not in data_encoded.columns}
        reg_input = pd.concat([data_encoded, pd.DataFrame(missing_reg_cols)], axis=1)
        reg_input = reg_input[regression_features]

        amount = float(regressor.predict(reg_input)[0])  # cast to Python float

        st.markdown(f"""
        <div class="section-card" style="margin-top:1.5rem; text-align:center;">
            <div class="section-title">💰 Estimated Claim Amount</div>
            <div class="claim-amount">₹ {amount:,.2f}</div>
            <div style="color:#a5b4fc; font-size:0.9rem; margin-top:0.5rem;">
                Predicted insurance payout based on patient profile
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Progress bar (normalized to max Rs.63,770) — must be Python float for Streamlit
        normalized = float(min(amount / 63770.0, 1.0))
        st.markdown('<div style="color:#a5b4fc; font-size:0.85rem; margin-top:1rem;">Claim amount relative to maximum (&#8377;63,770)</div>', unsafe_allow_html=True)
        st.progress(normalized)

    else:
        # --- Claim Denied ---
        st.markdown(f"""
        <div class="result-denied">
            <div class="result-title" style="color:#f87171;">❌ Claim Not Approved</div>
            <div class="result-sub" style="color:#fca5a5;">
                Based on the provided information, this claim does not meet approval criteria.
            </div>
            <div class="confidence-badge" style="background:rgba(239,68,68,0.2); border-color:rgba(239,68,68,0.4); color:#fca5a5;">
                🎯 Confidence: {confidence:.1f}%
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Input Summary Table ────────────────────
    with st.expander("📋 View Input Summary"):
        summary = {
            "Field": [
                "Age", "Gender", "Weight (kg)", "BMI", "No. of Dependents",
                "City", "Occupation", "Blood Pressure", "Hereditary Disease",
                "Smoker", "Diabetes", "Regular Exercise"
            ],
            "Value": [
                age, sex.capitalize(), weight, bmi, no_of_dependents,
                city, job_title, bloodpressure, hereditary_diseases,
                "Yes" if smoker else "No",
                "Yes" if diabetes else "No",
                "Yes" if regular_ex else "No"
            ]
        }
        st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; color:rgba(165,180,252,0.5); font-size:0.8rem; margin-top:3rem; padding-top:1rem; border-top:1px solid rgba(255,255,255,0.07);">
    Powered by XGBoost · Built with Streamlit · For informational purposes only
</div>
""", unsafe_allow_html=True)
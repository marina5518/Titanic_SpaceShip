import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Spaceship Titanic – Predictor",
    page_icon="🚀",
    layout="wide",
)

# ── Load Artifacts ────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model         = joblib.load("best_model.pkl")
    scaler        = joblib.load("scaler.pkl")
    feature_names = joblib.load("feature_names.pkl")
    return model, scaler, feature_names

try:
    model, scaler, feature_names = load_artifacts()
    artifacts_loaded = True
except FileNotFoundError:
    artifacts_loaded = False

# ── Helpers ───────────────────────────────────────────────────────────────────
SPENDING_COLS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]

DECK_OPTIONS  = ["A", "B", "C", "D", "E", "F", "G", "T"]
SIDE_OPTIONS  = ["P", "S"]
PLANET_OPTIONS = ["Earth", "Europa", "Mars"]
DEST_OPTIONS   = ["TRAPPIST-1e", "55 Cancri e", "PSO J318.5-22"]


def build_feature_row(inputs: dict) -> pd.DataFrame:
    """
    Replicate the exact feature-engineering + OHE pipeline from the notebook,
    then return a single-row DataFrame aligned to feature_names.
    """
    d = inputs.copy()

    # Derived features
    spending = [d[c] for c in SPENDING_COLS]
    d["TotalSpending"] = sum(spending)
    d["ServicesUsed"]  = sum(1 for v in spending if v > 0)

    # CryoSleep: if spending > 0 force False
    if d["TotalSpending"] > 0:
        d["CryoSleep"] = 0
    d["CryoSleep"] = int(d["CryoSleep"])
    d["VIP"]       = int(d["VIP"])

    row = pd.DataFrame([d])

    # One-Hot Encode categoricals (drop_first=True mirrors notebook)
    cat_cols = ["HomePlanet", "Destination", "Deck", "Side"]
    row = pd.get_dummies(row, columns=cat_cols, drop_first=True)

    # Align to training feature columns – fill missing dummies with 0
    for col in feature_names:
        if col not in row.columns:
            row[col] = 0
    row = row[feature_names]

    # Cast to int (notebook does this)
    row = row.astype(int)
    return row


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/9/96/Exoplanet_icon.png/240px-Exoplanet_icon.png", width=80)
    st.title("🚀 Spaceship Titanic")
    st.markdown("Predict whether a passenger was **transported** to an alternate dimension.")
    st.divider()
    st.markdown(
        "**Model:** Tuned Random Forest / SVM  \n"
        "**Dataset:** [Kaggle Competition](https://www.kaggle.com/c/spaceship-titanic)"
    )

# ── Main Content ──────────────────────────────────────────────────────────────
st.title("🛸 Spaceship Titanic — Passenger Transport Predictor")
st.markdown("Fill in the passenger details below and click **Predict** to see the result.")

if not artifacts_loaded:
    st.error(
        "⚠️ Model files not found (`best_model.pkl`, `scaler.pkl`, `feature_names.pkl`).  \n"
        "Run **Cell 21** in the notebook first to export these files, then place them "
        "in the same folder as `app.py`."
    )
    st.stop()

# ── Input Form ────────────────────────────────────────────────────────────────
with st.form("prediction_form"):
    st.subheader("👤 Passenger Information")

    c1, c2, c3 = st.columns(3)
    with c1:
        home_planet  = st.selectbox("Home Planet",  PLANET_OPTIONS)
        destination  = st.selectbox("Destination",  DEST_OPTIONS)
        deck         = st.selectbox("Cabin Deck",   DECK_OPTIONS)

    with c2:
        side        = st.selectbox("Cabin Side",   SIDE_OPTIONS)
        age         = st.number_input("Age",  min_value=0, max_value=120, value=28)
        cryo_sleep  = st.checkbox("CryoSleep", value=False)

    with c3:
        vip = st.checkbox("VIP", value=False)
        st.markdown("")   # spacer

    st.divider()
    st.subheader("💳 Onboard Spending (credits)")

    s1, s2, s3, s4, s5 = st.columns(5)
    room_service  = s1.number_input("Room Service",   min_value=0.0, value=0.0, step=10.0)
    food_court    = s2.number_input("Food Court",     min_value=0.0, value=0.0, step=10.0)
    shopping_mall = s3.number_input("Shopping Mall",  min_value=0.0, value=0.0, step=10.0)
    spa           = s4.number_input("Spa",            min_value=0.0, value=0.0, step=10.0)
    vr_deck       = s5.number_input("VR Deck",        min_value=0.0, value=0.0, step=10.0)

    submitted = st.form_submit_button("🔮 Predict", use_container_width=True, type="primary")

# ── Prediction ────────────────────────────────────────────────────────────────
if submitted:
    inputs = {
        "HomePlanet":   home_planet,
        "CryoSleep":    cryo_sleep,
        "Destination":  destination,
        "Age":          age,
        "VIP":          vip,
        "RoomService":  room_service,
        "FoodCourt":    food_court,
        "ShoppingMall": shopping_mall,
        "Spa":          spa,
        "VRDeck":       vr_deck,
        "Deck":         deck,
        "Side":         side,
    }

    row = build_feature_row(inputs)

    # Detect model type – SVM / LogReg need scaling
    model_name = type(model).__name__
    needs_scaling = model_name in ("SVC", "LogisticRegression")
    X_input = scaler.transform(row) if needs_scaling else row.values

    prediction  = model.predict(X_input)[0]
    probability = model.predict_proba(X_input)[0]

    st.divider()
    st.subheader("🎯 Prediction Result")

    res_col, prob_col = st.columns([1, 2])

    with res_col:
        if prediction == 1:
            st.success("### ✅ TRANSPORTED\nThis passenger was transported to the alternate dimension.")
        else:
            st.error("### ❌ NOT TRANSPORTED\nThis passenger was NOT transported.")

    with prob_col:
        not_transported_pct = probability[0] * 100
        transported_pct     = probability[1] * 100

        st.markdown("**Prediction Confidence**")
        st.progress(int(transported_pct),
                    text=f"Transported: {transported_pct:.1f}%")
        st.progress(int(not_transported_pct),
                    text=f"Not Transported: {not_transported_pct:.1f}%")

    with st.expander("📊 Passenger Summary"):
        summary_df = pd.DataFrame([{
            "Home Planet":    home_planet,
            "Destination":    destination,
            "Age":            age,
            "CryoSleep":      cryo_sleep,
            "VIP":            vip,
            "Deck / Side":    f"{deck} / {side}",
            "Total Spending": f"{room_service+food_court+shopping_mall+spa+vr_deck:,.0f} credits",
        }]).T.rename(columns={0: "Value"})
        st.dataframe(summary_df, use_container_width=True)

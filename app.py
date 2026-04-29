
# Step 3: Streamlit UI for the Technical Paper Presentation Predictor
# Run this with: streamlit run app.py

import streamlit as st
import pickle
import numpy as np

# ---- Page config ----
st.set_page_config(page_title="Paper Presentation Predictor", page_icon="📄")

# ---- Load the trained model ----
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Model file not found! Please run train_model.py first.")
    st.stop()

# ---- Title and description ----
st.title("📄 Technical Paper Presentation Predictor")
st.write("Rate the student's presentation on 8 parameters (scale: 1 to 10).")
st.write("---")

# ---- Input sliders ----
st.subheader("Enter Scores (1 = Poor, 10 = Excellent)")

col1, col2 = st.columns(2)

with col1:
    paper_quality       = st.slider("Paper Quality",       min_value=1, max_value=10, value=5)
    ppt_design          = st.slider("PPT Design",          min_value=1, max_value=10, value=5)
    technical_depth     = st.slider("Technical Depth",     min_value=1, max_value=10, value=5)
    time_management     = st.slider("Time Management",     min_value=1, max_value=10, value=5)

with col2:
    presentation_skills = st.slider("Presentation Skills", min_value=1, max_value=10, value=5)
    content_clarity     = st.slider("Content Clarity",     min_value=1, max_value=10, value=5)
    qa_handling         = st.slider("Q&A Handling",        min_value=1, max_value=10, value=5)
    confidence_level    = st.slider("Confidence Level",    min_value=1, max_value=10, value=5)

st.write("---")

# ---- Predict button ----
if st.button("🔍 Predict Performance Category"):

    # Prepare input as a numpy array (same order as training data)
    user_input = np.array([[
        paper_quality,
        presentation_skills,
        ppt_design,
        content_clarity,
        technical_depth,
        qa_handling,
        time_management,
        confidence_level
    ]])

    # Make prediction
    result = model.predict(user_input)[0]

    # Show result with color
    if result == 'High':
        st.success(f"✅ Predicted Category: **{result}**  — Excellent performance!")
    elif result == 'Medium':
        st.warning(f"⚠️ Predicted Category: **{result}**  — Average performance, needs improvement.")
    else:
        st.error(f"❌ Predicted Category: **{result}**  — Poor performance, significant improvement needed.")



# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import joblib
from PIL import Image
import streamlit.components.v1 as components

# Load model
model = joblib.load("enhanced_physical_health_model.pkl")

# Page config
st.set_page_config(
    page_title="DualCare: AI Health Assistant",
    layout="centered",
    page_icon="💡"
)

# Sidebar with logo
with st.sidebar:
    logo = Image.open("dualcare_logo.jpeg")
    st.image(logo, width=160)
    st.markdown("""
        ### Welcome to DualCare
        Your all-in-one assistant for:
        - Mental Well-being
        - Physical Health
    """)

# Page title
st.markdown("<h1 style='text-align:center;'>DualCare: AI Health Assistant</h1>", unsafe_allow_html=True)

# --- Layout: Two cards side by side ---
col1, col2 = st.columns(2)

# --- Mental Health Chatbot ---
with col1:
    with st.container():
        st.markdown("### Mental Health Chatbot")
        st.markdown("Chat with the bot for emotional support based on your mood:")
        components.html("""
        <script src="https://cdn.botpress.cloud/webchat/v3.2/inject.js" defer></script>
        <script src="https://files.bpcontent.cloud/2025/07/17/16/20250717162635-FQK11ES7.js" defer></script>
        """, height=480)

# --- Physical Health Checker ---
with col2:
    st.markdown("### Symptom Checker")
    st.markdown("Select your symptoms and we’ll predict your condition:")

    symptoms = {
        "Fever": "High body temperature or chills.",
        "Cough": "Persistent dry or wet cough.",
        "Fatigue": "Tiredness or lack of energy.",
        "Headache": "Pain in head or neck.",
        "Nausea": "Sensation of vomiting.",
        "Body Pain": "Muscle/joint aches.",
        "Chills": "Cold feeling with shivering.",
        "Loss of Smell": "Anosmia or reduced smell.",
        "Diarrhea": "Loose bowel movements.",
        "Vomiting": "Expelling stomach contents."
    }

user_inputs = []
with st.form("symptom_form"):
    for label, help_text in symptoms.items():
        user_input = st.selectbox(f"{label}:", ["Select", "Yes", "No"], help=help_text)
        user_inputs.append(1 if user_input == "Yes" else 0)
    submitted = st.form_submit_button("🔍 Predict")


# --- Original Full Definitions of Tips ---
tips = {
    'Flu': (
        "Rest: Get plenty of sleep.\n"
        "Hydration: Drink lots of fluids (water, soup, herbal tea).\n"
        "Medicine: Use over-the-counter meds for fever and body aches.\n"
        "Avoid: Cold exposure and crowded areas if contagious."
    ),
    'Cold': (
        "Stay Warm: Use a scarf or warm clothing.\n"
        "Steam: Inhale steam to clear nasal blockage.\n"
        "Rest: Let your body recover naturally.\n"
        "Natural Remedies: Honey, ginger, and tulsi can help."
    ),
    'Migraine': (
        "Avoid Bright Light: Rest in a dark, quiet room.\n"
        "Pain Relievers: Use prescribed medications.\n"
        "Relaxation: Try deep breathing or meditation.\n"
        "Track Triggers: Keep a diary of foods/situations causing migraines."
    ),
    'Food Poisoning': (
        "Hydrate: Drink ORS (oral rehydration salts) or coconut water.\n"
        "Eat Light: Begin with bananas, toast, or rice.\n"
        "Hygiene: Wash hands and clean utensils properly.\n"
        "Consult Doctor: If vomiting/diarrhea persists for 2+ days."
    ),
    'Typhoid': (
        "Antibiotics: Take the full prescribed course.\n"
        "Soft Diet: Eat porridge, khichdi, or soup.\n"
        "Boil Water: Only drink purified or boiled water.\n"
        "Avoid Raw Food: Especially street food or cut fruits."
    ),
    'Malaria': (
        "Rest & Medication: Start anti-malarial drugs early.\n"
        "Mosquito Protection: Use repellents and nets.\n"
        "Monitor Fever: Keep track of spikes.\n"
        "Hydration: Keep fluids up to avoid weakness."
    ),
    'COVID-19': (
        "Isolate: Avoid contact until negative.\n"
        "Sanitize: Wash hands and disinfect surfaces.\n"
        "Monitor Symptoms: Especially breathing and O2 levels.\n"
        "Seek Help: For persistent fever or chest tightness."
    ),
    'Dengue': (
        "Rest: Avoid exertion.\n"
        "Fluids: Coconut water, ORS, and papaya leaf juice (optional).\n"
        "Monitor Platelets: Get blood tests regularly.\n"
        "No Aspirin: Avoid blood-thinning meds."
    )
}

# --- Show prediction ---
if submitted:
    if sum(user_inputs) == 0:
        st.warning("⚠️ You have selected 'No' for all symptoms. It's unlikely that you are ill.")
        st.info("Please select at least one symptom to get a meaningful prediction.")
    else:
        x_input = np.array([user_inputs])
        prediction = model.predict(x_input)[0]
        probs = model.predict_proba(x_input)[0]
        classes = model.classes_

        st.markdown("---")
        st.subheader(f"Prediction: {prediction}")
        st.markdown("#### Prevention Tips:")
        st.text(tips.get(prediction, "Please consult a doctor for accurate diagnosis."))

        # Probability chart
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(classes, probs, color="#70d6ff")
        ax.set_title("Prediction Probabilities", fontsize=12)
        ax.set_ylabel("Confidence Level")
        for i, p in enumerate(probs):
            ax.text(i, p + 0.01, f"{p:.2f}", ha='center')
        ax.set_ylim(0, 1.1)
        st.pyplot(fig)

# --- Clean Footer ---
st.markdown("---")
st.markdown("<p style='text-align:center;'>DualCare © 2025</p>", unsafe_allow_html=True)

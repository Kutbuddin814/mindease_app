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

# --- Full Original Definitions of Tips (Plain, No Contact, No Icons) ---
tips = {
    'Flu': (
        "Rest well to let your body fight off the virus.\n"
        "Drink plenty of fluids to stay hydrated.\n"
        "Take medications like paracetamol for fever and body aches.\n"
        "Avoid exposure to cold and crowded places."
    ),
    'Cold': (
        "Keep yourself warm with appropriate clothing.\n"
        "Inhale steam to relieve nasal congestion.\n"
        "Allow your body time to rest and recover.\n"
        "Use natural remedies such as honey and ginger."
    ),
    'Migraine': (
        "Stay in a dark, quiet room to ease the pain.\n"
        "Take doctor-prescribed pain relief medication.\n"
        "Practice deep breathing and relaxation techniques.\n"
        "Note down possible triggers like stress or certain foods."
    ),
    'Food Poisoning': (
        "Rehydrate with ORS or clean water frequently.\n"
        "Eat bland foods like bananas, toast, or rice.\n"
        "Maintain proper hygiene while preparing or eating food.\n"
        "See a doctor if symptoms continue for more than two days."
    ),
    'Typhoid': (
        "Take all antibiotic doses as prescribed by your doctor.\n"
        "Stick to soft foods like porridge and soup.\n"
        "Boil or purify drinking water before consumption.\n"
        "Avoid eating raw fruits and street food."
    ),
    'Malaria': (
        "Begin anti-malarial medication early.\n"
        "Use mosquito repellents and nets for protection.\n"
        "Track and record any fever spikes.\n"
        "Stay hydrated to support recovery."
    ),
    'COVID-19': (
        "Self-isolate until you test negative.\n"
        "Regularly wash hands and clean surfaces.\n"
        "Monitor symptoms like breathlessness or low oxygen.\n"
        "Contact medical services if symptoms worsen."
    ),
    'Dengue': (
        "Get adequate rest and sleep.\n"
        "Drink fluids like water, ORS, or papaya leaf juice.\n"
        "Regularly monitor platelet levels through blood tests.\n"
        "Avoid painkillers that thin the blood like aspirin."
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

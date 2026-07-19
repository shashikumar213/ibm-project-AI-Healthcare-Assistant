
import streamlit as st
import pickle
import pandas as pd
from pathlib import Path
import spacy
from spacy.cli import download

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")
import re
from google import genai
from dotenv import load_dotenv
import os

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)
# -----------------------------
# Load Model & Feature Files
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent

# Load SpaCy NLP Model
nlp = spacy.load("en_core_web_sm")

# Session State
if "detected_symptoms" not in st.session_state:
    st.session_state.detected_symptoms = []

model_path = BASE_DIR / "healthcare_model.pkl"
features_path = BASE_DIR / "symptom_features.pkl"

with open(model_path, "rb") as file:
    model = pickle.load(file)

with open(features_path, "rb") as file:
    symptom_features = pickle.load(file)

# -----------------------------
# Streamlit Page
# -----------------------------
st.set_page_config(
    page_title="AI Healthcare Assistant",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 AI Healthcare Assistant")

st.write(
    "Select your symptoms to analyze possible health condition patterns."
)

st.markdown("---")

# -----------------------------
# NLP Section
# -----------------------------
st.subheader("🤖 NLP Symptom Analyzer")

user_text = st.text_area(
    "Describe your symptoms in English",
    placeholder="Example: I have fever, cough and chest pain for 3 days."
)

# Use Session State
detected_symptoms = st.session_state.detected_symptoms

# -----------------------------
# Symptom Synonyms
# -----------------------------
symptom_synonyms = {
    "high temperature": "fever",
    "temperature": "fever",
    "feverish": "fever",

    "breathing problem": "shortness of breath",
    "difficulty breathing": "shortness of breath",
    "breathlessness": "shortness of breath",

    "head ache": "headache",

    "stomach ache": "abdominal pain",
    "stomach pain": "abdominal pain",

    "throwing up": "vomiting",
    "feeling sick": "nausea",

    "blocked nose": "nasal congestion"
}
# -----------------------------
# AI Follow-up Question Bank for later advancement
# -----------------------------
QUESTION_BANK = {

    "Chest Pain": [
        "How old are you?",
        "Is the pain on the left side?",
        "Does the pain spread to your left arm?",
        "Are you having difficulty breathing?",
        "How long have you had the pain?"
    ],

    "Fever": [
        "What is your body temperature?",
        "Since when do you have fever?",
        "Do you have chills?",
        "Do you also have cough?",
        "Do you have body pain?"
    ],

    "Cough": [
        "Is your cough dry or with mucus?",
        "How many days have you had the cough?",
        "Do you have fever?",
        "Do you have chest pain?",
        "Do you have difficulty breathing?"
    ],

    "Headache": [
        "Where is the headache located?",
        "Is the pain severe?",
        "Do you have nausea?",
        "Do you have blurred vision?",
        "How long have you had the headache?"
    ],

    "Shortness Of Breath": [
        "Does it happen while walking or at rest?",
        "Do you have chest pain?",
        "Do you have cough?",
        "Do you have asthma?",
        "How long have you had breathing difficulty?"
    ]
}
symptom_names = {
    feature: feature.replace("symptom_", "")
                    .replace("_", " ")
                    .title()
    for feature in symptom_features
}

if st.button("Process Text"):

    if user_text.strip() == "":
        st.warning("Please enter your symptoms.")

    else:

        text = user_text.lower()

        # Replace synonym words
        for key, value in symptom_synonyms.items():
            text = text.replace(key, value)

        # NLP Processing
        doc = nlp(text)

        tokens = [token.text for token in doc]

        lemmas = [
            token.lemma_
            for token in doc
            if not token.is_stop 
            and not token.is_punct
        ]

        st.subheader("Word Tokenization")
        st.info(" | ".join(tokens))

        st.subheader("After Lemmatization")
        st.info(" | ".join(lemmas))

        # -------------------------
        # Age Extraction
        # -------------------------
        age_match = re.search(
            r"\b(\d{1,3})\s*(?:years?|yrs?)?\s*(?:old)?\b",
            text
        )

        age = None

        if age_match:
            age = int(age_match.group(1))

        st.subheader("Extracted Age")

        if age:
            st.success(f"{age} Years")
        else:
            st.info("Age not found")

        # -------------------------
        # Duration Extraction
        # -------------------------
        duration_match = re.search(
            r"(\d+)\s*(day|days|week|weeks|month|months)",
            text
        )

        st.subheader("Duration")

        if duration_match:
            st.success(
                f"{duration_match.group(1)} {duration_match.group(2)}"
            )
        else:
            st.info("Duration not found")

        # -------------------------
        # Symptom Detection
        # -------------------------
        filtered_text = " ".join(lemmas)

        detected_symptoms = []

        for feature in symptom_features:

            symptom = (
                feature.replace("symptom_", "")
                .replace("_", " ")
                .lower()
            )

            if symptom in filtered_text:
                detected_symptoms.append(symptom.title())

        # Save symptoms in Session State
        st.session_state.detected_symptoms = detected_symptoms

        st.subheader("Detected Symptoms")

        if detected_symptoms:
            detected_symptoms = sorted(set(detected_symptoms))
            st.success(", ".join(detected_symptoms))
        else:
            st.error("No symptom detected.")

st.markdown("---")

st.subheader("OR Manual Symptom Selection")

selected_names = st.multiselect(
    "Select Symptoms",
    list(symptom_names.values())
)

if st.button("Analyze Symptoms"):

    detected_symptoms = st.session_state.detected_symptoms

    if len(selected_names) == 0 and len(detected_symptoms) == 0:
        st.warning("Please enter text or select at least one symptom.")

    else:

        # Create blank input
        user_data = pd.DataFrame(
            [[0] * len(symptom_features)],
            columns=symptom_features
        )

        # Manual + NLP symptoms
        for feature, clean_name in symptom_names.items():

            if clean_name in selected_names:
                user_data.loc[0, feature] = 1

            if clean_name.lower() in [s.lower() for s in detected_symptoms]:
                user_data.loc[0, feature] = 1

        # Prediction
        probabilities = model.predict_proba(user_data)[0]

        results = pd.DataFrame({
            "Condition": model.classes_,
            "Confidence": probabilities * 100
        })

        results = results.sort_values(
            by="Confidence",
            ascending=False
        ).head(3)

        # Results
        st.subheader("🩺 AI Analysis Result")

        for _, row in results.iterrows():

            st.write(
                f"### {row['Condition'].replace('_', ' ')}"
            )

            st.progress(float(row["Confidence"] / 100))

            if row["Confidence"] >= 70:
                st.success(f"Confidence: {row['Confidence']:.2f}%")

            elif row["Confidence"] >= 40:
                st.warning(f"Confidence: {row['Confidence']:.2f}%")

            else:
                st.error(f"Confidence: {row['Confidence']:.2f}%")

        # =====================================================
        # 🤖 Gemini AI Explanation
        # =====================================================

        top_prediction = results.iloc[0]["Condition"].replace("_", " ")
        confidence = results.iloc[0]["Confidence"]

        prompt = f"""
You are an AI Healthcare Assistant.

Patient Symptoms:
{", ".join(detected_symptoms)}

Top Predicted Disease:
{top_prediction}

Confidence:
{confidence:.2f}%

Explain in simple English.

Your response should include:

1. What this disease is.
2. Why these symptoms may indicate this disease.
3. General precautions.
4. When to consult a doctor.

Do NOT say the disease is confirmed.
Clearly mention this is only an educational AI prediction.
"""

        try:

           response = client.models.generate_content(
              model="gemini-3.1-flash-lite",
               contents=prompt
            )

           st.markdown("---")
           st.subheader("🤖 AI Explanation")

           st.write(response.text)


        except Exception as e:

            st.error("Gemini AI Error")
            st.exception(e)

        st.markdown("---")

        st.warning(
            "⚠️ This AI system is an educational decision-support tool "
            "and does not provide a confirmed medical diagnosis."
        )

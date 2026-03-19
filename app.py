import streamlit as st
import os
import time
import pickle
import pandas as pd
import speech_recognition as sr
from deep_translator import GoogleTranslator
import plotly.graph_objects as go
from audio_recorder_streamlit import audio_recorder
import importlib.util

# Try to import our training module
try:
    import train_model
    TRAIN_MODULE_AVAILABLE = True
except ImportError:
    TRAIN_MODULE_AVAILABLE = False

# --- Page Configuration ---
st.set_page_config(
    page_title="Multilingual Speech Sentiment Analytics UI",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS for Premium Design ---
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
        font-family: 'Inter', sans-serif;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #E2E8F0;
        font-weight: 600;
        letter-spacing: -0.5px;
    }
    
    /* Metrics Container */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #1A202C 0%, #2D3748 100%);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #4A5568;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    div[data-testid="metric-container"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.4);
    }
    
    /* Primary Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #4F46E5 0%, #7C3AED 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(124, 58, 237, 0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(124, 58, 237, 0.5);
    }
    
    /* Info/Warning Boxes */
    .stAlert {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# --- State Initialization ---
if 'model' not in st.session_state:
    st.session_state.model = None
if 'vectorizer' not in st.session_state:
    st.session_state.vectorizer = None

# --- Helper Functions ---
@st.cache_resource
def load_models():
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "model.pkl")
        vec_path = os.path.join(base_dir, "vectorizer.pkl")
        
        if os.path.exists(model_path) and os.path.exists(vec_path):
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            with open(vec_path, "rb") as f:
                vectorizer = pickle.load(f)
            return model, vectorizer
        return None, None
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

model, vectorizer = load_models()
if model is not None and vectorizer is not None:
    st.session_state.model = model
    st.session_state.vectorizer = vectorizer

# --- Application Layout ---
st.title("🎙️ Multilingual Speech Sentiment Analytics")
st.markdown("Speak in your preferred language or type text. The system will transcribe, translate to English, and predict the sentiment.")

if st.session_state.model is None:
    st.warning("⚠️ The Machine Learning model is not loaded! Please ensure model.pkl and vectorizer.pkl are present.")

LANGUAGES = {
    "English": {"code": "en-US", "trans": "en"},
    "French": {"code": "fr-FR", "trans": "fr"},
    "Hindi": {"code": "hi-IN", "trans": "hi"},
    "Bengali": {"code": "bn-IN", "trans": "bn"},
    "Kannada": {"code": "kn-IN", "trans": "kn"},
    "Telugu": {"code": "te-IN", "trans": "te"},
    "Punjabi": {"code": "pa-IN", "trans": "pa"}
}

selected_lang = st.selectbox("Select your language", list(LANGUAGES.keys()))

st.markdown("---")
st.subheader("🎤 Option 1: Record Audio")
st.markdown("Click the microphone below to **start/stop** recording directly from your browser!")
col_btn, _ = st.columns([1, 4])
with col_btn:
    audio_bytes = audio_recorder(text="", recording_color="#E21818", neutral_color="#4F46E5", icon_name="microphone", icon_size="2x")

st.markdown("---")
st.subheader("⌨️ Option 2: Type Text (Fallback)")
manual_text = st.text_input("Type your sentence here if recording fails:")
analyze_clicked = st.button("Analyze Typed Text")

def process_text(text, is_english):
    st.markdown("### 📝 Input Text")
    st.info(text)
    
    # Translation
    translated_text = text
    if not is_english:
        with st.spinner("Translating to English for sentiment prediction..."):
            translated_text = GoogleTranslator(source='auto', target='en').translate(text)
            
        st.markdown("### 🌐 English Translation")
        st.success(translated_text)
        
    # Prediction
    with st.spinner("Analyzing Sentiment..."):
        vectorized_text = st.session_state.vectorizer.transform([translated_text])
        prediction = st.session_state.model.predict(vectorized_text)[0]
        probabilities = st.session_state.model.predict_proba(vectorized_text)[0]
        
    st.markdown("### 🧠 AI Analysis")
    if prediction == 1:
        st.success(f"**Sentiment**: POSITIVE 😊 (Confidence: {probabilities[1]:.2%})")
    else:
        st.error(f"**Sentiment**: NEGATIVE 😞 (Confidence: {probabilities[0]:.2%})")

if audio_bytes:
    if st.session_state.model is None:
        st.error("Cannot predict sentiment: Model is missing.")
    else:
        recognizer = sr.Recognizer()
        
        with st.spinner("Processing audio..."):
            try:
                with open("temp_audio.wav", "wb") as f:
                    f.write(audio_bytes)
                
                with sr.AudioFile("temp_audio.wav") as source:
                    audio = recognizer.record(source)
                    
                st.success("✅ Audio captured successfully!")
                
                with st.spinner("Transcribing audio to text..."):
                    lang_code = LANGUAGES[selected_lang]["code"]
                    transcribed_text = recognizer.recognize_google(audio, language=lang_code)
                
                process_text(transcribed_text, selected_lang == "English")
                    
            except sr.UnknownValueError:
                st.warning("🤷 Could not understand audio. The speech might have been too quiet or garbled.")
            except sr.RequestError as e:
                st.error(f"🔌 Speech service error (Check your internet connection): {e}")
            except Exception as e:
                st.error(f"❌ An unexpected error occurred: {e}")

elif analyze_clicked and manual_text:
    if st.session_state.model is None:
        st.error("Cannot predict sentiment: Model is missing.")
    else:
        process_text(manual_text, selected_lang == "English")

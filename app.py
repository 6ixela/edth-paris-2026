import streamlit as st
import whisper
import tempfile
import json
from jiwer import wer
from src.preprocess import ptt_preprocess

st.set_page_config(page_title="Radio STT", page_icon="📡", layout="wide")
st.title("📡 Radio Speech-to-Text — Baseline vs Amélioré")

@st.cache_resource
def load_model():
    return whisper.load_model("small")

model = load_model()

uploaded = st.file_uploader("Upload un fichier audio radio", type=["wav", "mp3"])

if uploaded:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    st.audio(uploaded)
    prompt = "Radio communication, callsigns, Alpha Bravo Charlie Delta Echo Foxtrot."

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### ❌ Whisper Vanilla")
        with st.spinner("Transcription..."):
            result_baseline = model.transcribe(tmp_path, language="en")
            st.write(result_baseline["text"])

    with col2:
        st.markdown("### ✅ Notre Pipeline")
        with st.spinner("Pre-processing + transcription..."):
            processed_path = tmp_path.replace(".wav", "_processed.wav")
            ptt_preprocess(tmp_path, processed_path)
            result_improved = model.transcribe(processed_path, language="en", initial_prompt=prompt)
            st.write(result_improved["text"])

    st.markdown("---")
    reference = st.text_input("Transcription de référence (pour calculer WER):")
    if reference:
        wer_b = wer(reference.lower(), result_baseline["text"].lower())
        wer_i = wer(reference.lower(), result_improved["text"].lower())
        m1, m2, m3 = st.columns(3)
        m1.metric("WER Baseline", f"{wer_b:.1%}")
        m2.metric("WER Amélioré", f"{wer_i:.1%}")
        m3.metric("Gain", f"{wer_b - wer_i:.1%}")

st.markdown("---")
st.markdown("### 📊 Résultats globaux")
try:
    with open("./data/results.json") as f:
        results = json.load(f)
    col1, col2, col3 = st.columns(3)
    col1.metric("WER Baseline", f"{results['wer_baseline']:.1%}")
    col2.metric("WER Amélioré", f"{results['wer_improved']:.1%}")
    col3.metric("Amélioration", f"{results['gain']:.1%}")
except:
    st.info("Lance d'abord `python src/evaluate.py` pour générer les résultats")

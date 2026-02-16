import streamlit as st
import numpy as np
import time
import torch
from langdetect import detect, DetectorFactory  # type: ignore
from typing import Tuple, Optional
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Language dictionary mapping (NLLB uses 3-letter codes)
NLLB_LANGUAGE_DICT = {
    'af': 'afr_Latn', 'ar': 'ary_Arab', 'bg': 'bul_Cyrl', 'bn': 'ben_Beng', 'ca': 'cat_Latn',
    'cs': 'ces_Latn', 'da': 'dan_Latn', 'de': 'deu_Latn', 'el': 'ell_Grek', 'en': 'eng_Latn',
    'es': 'spa_Latn', 'et': 'est_Latn', 'fa': 'pes_Arab', 'fi': 'fin_Latn', 'fr': 'fra_Latn',
    'gu': 'guj_Gujr', 'he': 'heb_Hebr', 'hi': 'hin_Deva', 'hr': 'hrv_Latn', 'hu': 'hun_Latn',
    'id': 'ind_Latn', 'it': 'ita_Latn', 'ja': 'jpn_Jpan', 'kn': 'kan_Knda', 'ko': 'kor_Hang',
    'lt': 'lit_Latn', 'lv': 'lav_Latn', 'mk': 'mkd_Cyrl', 'ml': 'mal_Mlym', 'mr': 'mar_Deva',
    'ne': 'npi_Deva', 'nl': 'nld_Latn', 'no': 'nob_Latn', 'pa': 'pan_Guru', 'pl': 'pol_Latn',
    'pt': 'por_Latn', 'ro': 'ron_Latn', 'ru': 'rus_Cyrl', 'sk': 'slk_Latn', 'sl': 'slv_Latn',
    'so': 'som_Latn', 'sq': 'als_Latn', 'sv': 'swe_Latn', 'sw': 'swh_Latn', 'ta': 'tam_Taml',
    'te': 'tel_Telu', 'th': 'tha_Thai', 'tr': 'tur_Latn', 'uk': 'ukr_Cyrl', 'ur': 'urd_Arab',
    'vi': 'vie_Latn', 'zh-cn': 'zho_Hans', 'zh-tw': 'zho_Hant'
}

# Display names map
LANGUAGE_NAMES = {
    'af': 'Afrikaans', 'ar': 'Arabic', 'bg': 'Bulgarian', 'bn': 'Bengali', 'ca': 'Catalan',
    'cs': 'Czech', 'cy': 'Welsh', 'da': 'Danish', 'de': 'German', 'el': 'Greek',
    'en': 'English', 'es': 'Spanish', 'et': 'Estonian', 'fa': 'Persian', 'fi': 'Finnish',
    'fr': 'French', 'gu': 'Gujarati', 'he': 'Hebrew', 'hi': 'Hindi', 'hr': 'Croatian',
    'hu': 'Hungarian', 'id': 'Indonesian', 'it': 'Italian', 'ja': 'Japanese', 'kn': 'Kannada',
    'ko': 'Korean', 'lt': 'Lithuanian', 'lv': 'Latvian', 'mk': 'Macedonian', 'ml': 'Malayalam',
    'mr': 'Marathi', 'ne': 'Nepali', 'nl': 'Dutch', 'no': 'Norwegian', 'pa': 'Punjabi',
    'pl': 'Polish', 'pt': 'Portuguese', 'ro': 'Romanian', 'ru': 'Russian', 'sk': 'Slovak',
    'sl': 'Slovenian', 'so': 'Somali', 'sq': 'Albanian', 'sv': 'Swedish', 'sw': 'Swahili',
    'ta': 'Tamil', 'te': 'Telugu', 'th': 'Thai', 'tr': 'Turkish', 'uk': 'Ukrainian', 
    'ur': 'Urdu', 'vi': 'Vietnamese', 'zh-cn': 'Chinese (Simplified)', 'zh-tw': 'Chinese (Traditional)'
}

st.set_page_config(page_title="Multilingual Translator", page_icon="🌐", layout="wide")

# This model is small (~300MB) and very reliable on free tiers
model_name = "Helsinki-NLP/opus-mt-mul-en"

@st.cache_resource
def load_model():
    try:
        # Check for both 'token' and 'TOKEN'
        hf_token = st.secrets.get("TOKEN") or st.secrets.get("token")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        # Loading in standard FP32 for better CPU compatibility on Streamlit
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, 
            token=hf_token
        )
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

model, tokenizer = load_model()

if model is None or tokenizer is None:
    st.stop()

def translate(text, src_lang_code):
    try:
        inputs = tokenizer(text, return_tensors="pt", padding=True)
        # Standard generation params for Opus-MT
        generated_tokens = model.generate(**inputs, max_length=512)
        return tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    except Exception as e:
        st.error(f"Translation error: {str(e)}")
        return ""

def detect_language(text: str) -> Optional[str]:
    try:
        return detect(text)
    except:
        return None

st.success("Welcome to the Multilingual Translation App")
st.title("🌍 Multilingual to English Translator 🌎")

input_text = st.text_area("Enter text to translate:", height=150)

st.markdown("""
<style>
.big-font { font-size:20px !important; }
.green-border { border: 2px solid #28a745; padding: 10px; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

language_options = ["Auto Detect"] + list(LANGUAGE_NAMES.values())
selected_language_name = st.selectbox("Select Source Language", language_options)

if st.button("Translate"):
    if input_text:
        if selected_language_name == "Auto Detect":
            detected_lang = detect_language(input_text)
            if detected_lang == 'en':
                st.info("The text is already in English.")
                translated_text = input_text
            else:
                display_name = LANGUAGE_NAMES.get(detected_lang, detected_lang)
                st.info(f"Detected language: {display_name}")
                with st.spinner("Translating..."):
                    translated_text = translate(input_text, detected_lang)
        else:
            src_lang_code = [k for k, v in LANGUAGE_NAMES.items() if v == selected_language_name][0]
            st.info(f"Selected language: {selected_language_name}")
            with st.spinner("Translating..."):
                translated_text = translate(input_text, src_lang_code)

        if translated_text:
            st.markdown(f'<div class="green-border"><p class="big-font">{translated_text}</p></div>', unsafe_allow_html=True)
    else:
        st.warning("Please enter some text.")

st.markdown("---")
st.markdown("## 🚀 Features")
st.markdown("- Powered by Helsinki-NLP Opus-MT")
st.markdown("- Fast translation (optimized for web)")
st.markdown("- Supports multiple languages to English translation")

st.markdown("## 📊 Translation Statistics")
col3, col4, col5 = st.columns(3)
with col3:
    st.metric(label="Status", value="Ready")
with col4:
    st.metric(label="Supported Languages", value="100+")
with col5:
    st.metric(label="Model Size", value="~300MB")

st.markdown("---")
st.markdown("Built with ❤️ using Streamlit and Hugging Face Transformers")

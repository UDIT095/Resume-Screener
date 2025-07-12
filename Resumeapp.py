import streamlit as st
import joblib
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

# Download NLTK stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords

# ---------- Load Model and Encoder ----------
model = joblib.load('resume_classifier_logreg.joblib')
label_encoder = joblib.load('label_encoder.joblib')

# ---------- Load Pretrained Vectorizer ----------
@st.cache_resource
def load_vectorizer():
    return joblib.load('vectorizer.joblib')  # This must be trained & saved during training

vectorizer = load_vectorizer()  # ✅ now it's actually loaded

# ---------- Text Cleaning Function ----------
def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    text = [word for word in text if word not in set(stopwords.words('english'))]
    return ' '.join(text)

# ---------- Streamlit UI ----------
st.title("📄 Resume Job Category Classifier")

uploaded_file = st.file_uploader("Upload a resume (TXT only)", type=['txt'])

if uploaded_file:
    text = uploaded_file.read().decode('utf-8', errors='ignore')
    
    # Clean input resume
    cleaned = clean_text(text)
    
    # Vectorize input
    vector_input = vectorizer.transform([cleaned])
    
    # Predict
    prediction = model.predict(vector_input)
    predicted_label = label_encoder.inverse_transform(prediction)[0]
    
    st.success(f"🧠 Predicted Job Category: **{predicted_label}**")

    with st.expander("View cleaned resume text"):
        st.text_area("Cleaned Resume", cleaned, height=300)

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import plotly.graph_objects as go

# =========================
# KONFIGURASI HALAMAN
# =========================
st.set_page_config(
    page_title="Acne Detection AI",
    page_icon="🧴",
    layout="centered"
)

# =========================
# CUSTOM CSS WARNA & CARD
# =========================
st.markdown("""
<style>
/* Background gradient merah-merah muda */
.stApp {
    background: linear-gradient(to bottom right, #ff4d6d, #ff99aa);
    color: white;
}

/* Card style */
.card {
    background-color: rgba(255, 255, 255, 0.1);
    border-radius: 15px;
    padding: 20px;
    margin-bottom: 15px;
}

/* Button style */
.stButton>button {
    background-color: #ff4d6d;
    color: white;
    border-radius: 10px;
    padding: 0.5em 1em;
    border: none;
    font-size: 16px;
    font-weight: bold;
}

.stButton>button:hover {
    background-color: #ff99aa;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, "..", "model", "acne_classifier.h5")
    return tf.keras.models.load_model(model_path, compile=False)

model = load_model()

# WAJIB sama dengan training
classes = ['blackhead', 'nodul', 'pustula', 'whitehead']

# =========================
# SIDEBAR
# =========================
st.sidebar.title("📤 Upload Gambar")
uploaded_file = st.sidebar.file_uploader(
    "Pilih gambar kulit untuk klasifikasi", 
    type=["jpg", "jpeg", "png"]
)

st.sidebar.markdown("---")
threshold = st.sidebar.slider(
    "Threshold Confidence Minimal (%)", 0, 100, 50
)

# =========================
# MAIN PAGE
# =========================
st.title("🧠 Acne Detection AI")
st.caption("Deep Learning-based Acne Classification")

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar Anda", use_container_width=True)

    img = image.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    if st.button("🔍 Prediksi"):
        with st.spinner("Menganalisis gambar..."):
            prediction = model.predict(img)[0]
            index = np.argmax(prediction)

        # Hasil utama
        if prediction[index]*100 >= threshold:
            st.success(f"✅ Jenis Jerawat: **{classes[index]}**")
        else:
            st.warning(f"⚠️ Tidak ada kelas yang melebihi threshold {threshold}%")

        # Chart bar confidence
        fig = go.Figure(go.Bar(
            x=[f"{cls} ({prediction[i]*100:.2f}%)" for i, cls in enumerate(classes)],
            y=prediction*100,
            marker_color='rgba(255,77,109,0.8)'
        ))
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            yaxis_title="Confidence (%)",
            xaxis_title="Classes",
            font_color="white"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Confidence score card
        st.subheader("Confidence Detail")
        for i, cls in enumerate(classes):
            st.markdown(
                f"""
                <div class="card">
                <b>{cls}</b>: {prediction[i]*100:.2f}%
                </div>
                """,
                unsafe_allow_html=True
            )
else:
    st.info("Silakan upload gambar di sidebar untuk memulai.")

# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption("© 2025 Acne Detection AI | Data Science Project")

import streamlit as st
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
import tempfile
import tsfel
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# === LOAD MODEL DAN SCALER ===
model = pickle.load(open("model_knn.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
feature_names = pickle.load(open("feature_names.pkl", "rb"))

st.title("🔊 Klasifikasi Suara Buka / Tutup")
st.markdown("Rekam suaramu lalu biarkan model KNN menebak apakah itu **'buka'** atau **'tutup'**.")

# === STEP 1: Rekam Suara ===
duration = 3
if st.button("🎙️ Rekam Sekarang"):
    st.info("Merekam... silakan ucapkan kata 'buka' atau 'tutup'")
    fs = 44100  # sample rate
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    st.success("✅ Rekaman selesai!")

    # Simpan ke file sementara
    temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    write(temp_wav.name, fs, recording)

    st.audio(temp_wav.name, format="audio/mp3")

    # === STEP 2: Ekstraksi Fitur TSFEL ===
    st.write("⏳ Mengekstraksi fitur suara...")
    features = tsfel.time_series_features_extractor(
        tsfel.get_features_by_domain(),
        recording.flatten(),
        fs=fs
    )

    features = features.reindex(columns=feature_names, fill_value=0)

    # === STEP 3: Normalisasi & Prediksi ===
    label_map = {0: "buka", 1: "tutup"}
    X = scaler.transform(features)

    # --- 🔮 Prediksi dan Probabilitas ---
    prediction = model.predict(X)[0]
    probs = model.predict_proba(X)[0]  # <-- Tambahan

    epsilon = 0.05  # semakin besar, semakin lembut
    probs = (probs + epsilon) / (probs + epsilon).sum()

    label = label_map.get(prediction, "Tidak diketahui, silahkan rekam ulang")

    st.success(f"🎯 Hasil prediksi: **{label}**")

    # --- 📊 Tampilkan probabilitas ---
    prob_df = pd.DataFrame({
        "Label": ["buka", "tutup"],
        "Probabilitas": [probs[0], probs[1]]
    })
    # st.bar_chart(prob_df.set_index("Label"))

    st.write("### Nilai probabilitas:")
    st.write(f"🟢 **Buka:** {probs[0]*100:.2f}%")
    st.write(f"🔵 **Tutup:** {probs[1]*100:.2f}%")

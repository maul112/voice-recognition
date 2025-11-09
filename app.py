import streamlit as st
from audio_recorder_streamlit import audio_recorder
import numpy as np
import pandas as pd
import soundfile as sf
from pydub import AudioSegment
import joblib
import librosa
import os
import io
import tsfel
import tempfile
import hashlib
import time
import matplotlib.pyplot as plt

st.title("🎙️ Voice Classifier: Buka / Tutup")
st.markdown("""
Rekam suaramu lalu biarkan model KNN menebak apakah itu **'buka'** atau **'tutup'**.  
Empat label yang dikenali:
- 🟢 **Buka Dio**
- 🟢 **Buka Maulana**
- 🔵 **Tutup Dio**
- 🔵 **Tutup Maulana**
""")

# === Load Model ===
try:
    base_dir = os.path.dirname(__file__)
    model_action = joblib.load(os.path.join(base_dir, "./rf/rf_model_action.pkl"))
    model_speaker = joblib.load(os.path.join(base_dir, "./rf/rf_model_speaker.pkl"))
    action_scaler = joblib.load(os.path.join(base_dir, "./rf/action_scaler.pkl"))
    speaker_scaler = joblib.load(os.path.join(base_dir, "./rf/speaker_scaler.pkl"))
    action_feature_names = joblib.load(os.path.join(base_dir, "./rf/action_feature_names.pkl"))
    speaker_feature_names = joblib.load(os.path.join(base_dir, "./rf/speaker_feature_names.pkl"))
    action_encoder = joblib.load(os.path.join(base_dir, "./rf/action_encoder.pkl"))
    speaker_encoder = joblib.load(os.path.join(base_dir, "./rf/speaker_encoder.pkl"))

except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

if "last_hash" not in st.session_state:
    st.session_state.last_hash = None


def analyze_audio(audio_bytes):
    """Fungsi utama analisis audio"""
    try:
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes), format="wav")
        samples = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)

        # Normalisasi sesuai bit depth
        if audio_segment.sample_width == 2:
            samples = samples / 32768.0
        elif audio_segment.sample_width == 4:
            samples = samples / 2147483648.0

        # Stereo ke mono
        if audio_segment.channels == 2:
            samples = samples.reshape((-1, 2)).mean(axis=1)

        sr = audio_segment.frame_rate

        # Resample ke 16 kHz
        if sr != 16000:
            samples = librosa.resample(samples, orig_sr=sr, target_sr=16000)
            sr = 16000

        samples = samples / (np.max(np.abs(samples)) + 1e-8)

        # Simpan sementara untuk diputar
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            sf.write(tmp.name, samples, sr)
            st.audio(tmp.name, format="audio/mp3")
        os.remove(tmp.name)

        # === Ekstraksi fitur ===
        st.write("⏳ Mengekstraksi fitur statistik...")
        cfg_statistikal = tsfel.get_features_by_domain("statistical")
        cfg_temporal = tsfel.get_features_by_domain("temporal")
        cfg_spectral = tsfel.get_features_by_domain("spectral")
        features_statistikal = tsfel.time_series_features_extractor(cfg_statistikal, samples, fs=sr, verbose=0)
        features_temporal = tsfel.time_series_features_extractor(cfg_temporal, samples, fs=sr, verbose=0)
        features_spectral = tsfel.time_series_features_extractor(cfg_spectral, samples, fs=sr, verbose=0)
        features = pd.concat([features_statistikal, features_temporal, features_spectral], axis=1)
        features1 = features.reindex(columns=action_feature_names, fill_value=0)
        features2 = features.reindex(columns=speaker_feature_names, fill_value=0)

        # === Prediksi ===
        X1 = pd.DataFrame(action_scaler.transform(features1), columns=action_feature_names)
        X2 = pd.DataFrame(speaker_scaler.transform(features2), columns=speaker_feature_names)
        prediction1 = model_action.predict(X1)[0]
        prediction2 = model_speaker.predict(X2)[0]
        # probs1 = model_action.predict_proba(X1)[0]
        # probs2 = model_speaker.predict_proba(X2)[0]

        # Smooth probabilitas
        label1 = action_encoder.inverse_transform([prediction1])[0]
        label2 = speaker_encoder.inverse_transform([prediction2])[0]
        st.success(f"🎯 Hasil prediksi: **{label1} {label2}**")
        
        # === Probabilitas Gabungan ===
        # epsilon = 0.05

        # # Smooth probabilitas (menghindari 0)
        # probs1 = (probs1 + epsilon) / (probs1 + epsilon).sum()
        # probs2 = (probs2 + epsilon) / (probs2 + epsilon).sum()

        # # Hitung kombinasi semua label (4 total)
        # combined_labels = []
        # combined_probs = []

        # for i, action_label in enumerate(action_encoder.classes_):
        #     for j, speaker_label in enumerate(speaker_encoder.classes_):
        #         combined_labels.append(f"{action_label} {speaker_label}")
        #         combined_probs.append(probs1[i] * probs2[j])  # Probabilitas gabungan

        # # Normalisasi agar total = 1
        # combined_probs = np.array(combined_probs)
        # combined_probs = combined_probs / combined_probs.sum()

        # Buat dataframe hasil
        # prob_df = pd.DataFrame({
        #     "Label": combined_labels,
        #     "Probabilitas (%)": combined_probs * 100
        # }).sort_values(by="Probabilitas (%)", ascending=False)

        # Tampilkan dataframe
        # st.write("📊 **Probabilitas Gabungan (Smooth):**")
        # st.dataframe(prob_df, hide_index=True, use_container_width=True)

        # Visualisasi bar chart
        # fig, ax = plt.subplots(figsize=(6, 3))
        # ax.bar(prob_df["Label"], prob_df["Probabilitas (%)"], color=["#2ecc71", "#27ae60", "#3498db", "#2980b9"])
        # ax.set_ylabel("Persentase (%)")
        # ax.set_ylim(0, 100)
        # plt.xticks(rotation=20, ha="right")
        # st.pyplot(fig)

    except Exception as e:
        st.error(f"Error saat analisis audio: {e}")


# === Rekaman Langsung ===
st.info("🎙️ Tekan tombol mikrofon di bawah, ucapkan **'BUKA'** atau **'TUTUP'**, lalu lepas tombol.")

audio_bytes = audio_recorder(
    text="Klik untuk merekam",
    recording_color="#e74c3c",
    neutral_color="#3498db",
    icon_name="microphone",
    icon_size="2x",
    pause_threshold=2.0,
    sample_rate=16000
)

if audio_bytes:
    # ✅ FIX untuk rekaman ulang
    # Buat hash unik berdasarkan isi audio + waktu (agar setiap rekaman baru dianggap berbeda)
    current_hash = hashlib.md5(audio_bytes + str(time.time()).encode()).hexdigest()

    if current_hash != st.session_state.last_hash:
        st.session_state.last_hash = current_hash
        st.success("✅ Audio baru diterima! Menganalisis...")
        with st.spinner("🔄 Memproses audio..."):
            analyze_audio(audio_bytes)
    else:
        st.info("ℹ️ Audio ini sudah dianalisis. Rekam ulang untuk hasil baru.")
else:
    st.info("👆 Klik tombol mikrofon di atas untuk mulai merekam.")

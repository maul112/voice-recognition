import streamlit as st
from audio_recorder_streamlit import audio_recorder
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import soundfile as sf
from pydub import AudioSegment
import joblib
import librosa
import os
import io
import tsfel

st.title("🎙️ Voice Classifier: Buka / Tutup")
st.markdown("Rekam suaramu lalu biarkan model KNN menebak apakah itu **'buka'** atau **'tutup'**.")

model_path = os.path.join(os.path.dirname(__file__), "model_knn.pkl")
scaler_path = os.path.join(os.path.dirname(__file__), "scaler.pkl")
feature_names_path = os.path.join(os.path.dirname(__file__), "feature_names.pkl")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
feature_names = joblib.load(feature_names_path)

le = LabelEncoder()
y_encoded = le.fit_transform(["buka", "tutup"])

if "last_audio" not in st.session_state:
    st.session_state.last_audio = None

mode = "🎙️ Rekam langsung"

def analyze_audio(audio_bytes, source="rekaman"):
    try:
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes), format="wav")
        samples = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)

        if audio_segment.sample_width == 2:
            samples = samples / 32768.0
        elif audio_segment.sample_width == 4:
            samples = samples / 2147483648.0

        if audio_segment.channels == 2:
            samples = samples.reshape((-1, 2)).mean(axis=1)

        sr = audio_segment.frame_rate

        # Resample ke 16kHz jika perlu
        if sr != 16000:
            samples = librosa.resample(samples, orig_sr=sr, target_sr=16000)
            sr = 16000
        
        # Normalisasi
        samples = samples / (np.max(np.abs(samples)) + 1e-8)

        # Simpan untuk preview
        sf.write("temp_audio.wav", samples, sr)
        st.audio("temp_audio.wav", format="audio/wav")

        st.write("⏳ Mengekstraksi fitur...")
        cfg = tsfel.get_features_by_domain("statistical")
        features = tsfel.time_series_features_extractor(cfg, samples, fs=sr, verbose=0)

        # Pastikan kolom sama dengan feature_names
        features = features.reindex(columns=feature_names, fill_value=0)

        # === Prediksi ===
        label_map = {0: 'buka_dio', 1: 'buka_maulana', 2: 'tutup_dio', 3: 'tutup_maulana'}
        X = scaler.transform(features)
        prediction = model.predict(X)[0]
        probs = model.predict_proba(X)[0]  # <-- Tambahan

        epsilon = 0.05  # semakin besar, semakin lembut
        probs = (probs + epsilon) / (probs + epsilon).sum()
        label = label_map.get(prediction, "Tidak diketahui, silahkan rekam ulang")

        st.success(f"🎯 Hasil prediksi: **{label.replace("_", " ")}**")

        # --- 📊 Tampilkan probabilitas ---
        prob_df = pd.DataFrame({
            "Label": label_map.values(),
            "Probabilitas": [probs[0], probs[1], probs[2], probs[3]]
        })
        # st.bar_chart(prob_df.set_index("Label"))

        st.write("Nilai probabilitas:")
        st.write(f"🟢 **Buka Dio:** {probs[0]*100:.2f}%")
        st.write(f"🟢 **Buka Maulana:** {probs[1]*100:.2f}%")
        st.write(f"🔵 **Tutup Dio:** {probs[2]*100:.2f}%")
        st.write(f"🔵 **Tutup Maulana:** {probs[3]*100:.2f}%")

    except Exception as e:
        st.error(f"❌ Error saat analisis: {e}")

if mode == "🎙️ Rekam langsung":
    st.info("🎙️ Tekan tombol mikrofon di bawah, ucapkan **'BUKA'** atau **'TUTUP'** dengan jelas, lalu tekan stop.")
    
    # Audio recorder widget
    audio_bytes = audio_recorder(
        text="Klik untuk merekam",
        recording_color="#e74c3c",
        neutral_color="#3498db",
        icon_name="microphone",
        icon_size="2x",
        pause_threshold=2.0,
        sample_rate=16000
    )
    
    # Jika ada audio baru yang direkam
    if audio_bytes:
        # Cek apakah ini audio baru (berbeda dari sebelumnya)
        if audio_bytes != st.session_state.last_audio:
            st.session_state.last_audio = audio_bytes
            
            st.success("✅ Audio berhasil direkam! Menganalisis...")
            
            # Analisis otomatis
            with st.spinner("🔄 Memproses audio..."):
                analyze_audio(audio_bytes, source="rekaman")
        else:
            # Audio sama dengan sebelumnya, tampilkan tombol analisis ulang
            st.info("ℹ️ Audio sudah dianalisis. Rekam ulang untuk prediksi baru.")
            
            if st.button("🔄 Analisis Ulang", type="secondary"):
                with st.spinner("🔄 Memproses audio..."):
                    analyze_audio(audio_bytes, source="rekaman")
    else:
        st.info("👆 Klik tombol mikrofon di atas untuk mulai merekam")

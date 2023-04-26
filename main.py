import streamlit as st
import librosa
import io
import matplotlib.pyplot as plt
import librosa.display
import pickle
import numpy as np


def feature_extractor(audio_data):
    mfccs_features = librosa.feature.mfcc(y=audio_data, sr=22050, n_mfcc=64)

    return np.mean(mfccs_features.T, axis=0)


def get_features(audio, sample_rate):
    if audio.shape[0] % sample_rate != 0:
        audio = np.pad(audio, (0, sample_rate - audio.shape[0] % sample_rate))
    audio_split = np.array(np.split(audio, audio.shape[0] / sample_rate, axis=0))
    X = []
    for array in audio_split:
        X.append(feature_extractor(array))
    return X


uploaded_file = st.file_uploader(label='Upload audio clip', type='wav')

if uploaded_file is not None:
    st.audio(uploaded_file.read(), format='audio/wav')
    audio, sample_rate = librosa.load(io.BytesIO(uploaded_file.getvalue()))
    X = get_features(audio, sample_rate)
    with open('./models/rfm.pkl', 'rb') as f:
        random_forest = pickle.load(f)
    predictions = np.array([1 if proba > 0.4 else 0 for _, proba in random_forest.predict_proba(X)])
    if predictions.mean() == 0:
        st.write('No snoring detected!')
    else:
        st.write(f'{predictions.sum()} seconds of snoring was detected')
    fig, ax = plt.subplots(figsize=(16, 9), dpi=100)
    ax = librosa.display.waveshow(audio, sr=sample_rate)
    st.pyplot(fig)

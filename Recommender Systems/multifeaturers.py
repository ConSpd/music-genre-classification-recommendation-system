import librosa, librosa.display, librosa.feature
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
import math

CSV_PATH = "/Users/conspd/School/Thesis/Data/features_30_sec.csv"
sr = 22050
hop_length = 512
n_fft = 2048

def get_song_list():
    with open(CSV_PATH) as f:
        df = pd.read_csv(CSV_PATH)
        df.drop(['length','label','harmony_mean','harmony_var','perceptr_mean','perceptr_var'],axis=1,inplace=True)
        data = np.array(df.values.tolist())
    return data

def extract_features(y, start, finish):
    # Chroma
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_stft_mean = np.mean(chroma_stft)
    chroma_stft_var = np.var(chroma_stft)

    # RMS
    rms = librosa.feature.rms(y=y)
    rms_mean = np.mean(rms)
    rms_var = np.var(rms)

    # Spectral Centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_centroid_mean = np.mean(spectral_centroid)
    spectral_centroid_var = np.var(spectral_centroid)

    # Spectral Bandwidth
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spectral_bandwidth_mean = np.mean(spectral_bandwidth)
    spectral_bandwidth_var = np.var(spectral_bandwidth)

    # Rolloff
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    rolloff_mean = np.mean(rolloff)
    rolloff_var = np.var(rolloff)

    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = np.mean(zcr)
    zcr_var = np.var(zcr)

    # Tempo
    tempo = librosa.feature.tempo(y=y, sr=sr)[0]

    # MFCC
    mfcc = librosa.feature.mfcc(y=y[start:finish], sr=sr)
    mfcc_mean = [np.mean(m) for m in mfcc]
    mfcc_var = [np.var(m) for m in mfcc]

    features = [chroma_stft_mean, chroma_stft_var,
                rms_mean, rms_var,
                spectral_centroid_mean, spectral_centroid_var,
                spectral_bandwidth_mean, spectral_bandwidth_var,
                rolloff_mean, rolloff_var,
                zcr_mean, zcr_var,
                tempo]
    for m, v in zip(mfcc_mean, mfcc_var):
        features.append(m)
        features.append(v)

    return features

def load_song(song_path):
    signal, sr = librosa.load(song_path, sr=22050)
    if len(signal) > 1984512:
        features = extract_features(signal, 882000, 1543504)
    else:
        features = extract_features(signal, 0, 661504)
    return features


def find_most_similar_song(song, song_list):
    cosine_similarity_score = [1 - cosine(song, np.array(s[1:]).astype(float)) for s in song_list]
    top_indices = sorted(enumerate(cosine_similarity_score), key=lambda x: x[1], reverse=True)[:5]
    top_indices_only = [index for index, _ in top_indices]

    # With Euclidean
    EMD_similarity_score = [math.dist(song, np.array(s[1:]).astype(float)) for s in song_list]
    EMD_top_indices_only = sorted(range(len(EMD_similarity_score)), key=lambda sub: EMD_similarity_score[sub])[:5]

    return top_indices_only, EMD_top_indices_only

def print_results(cosine_top, emd_top, song_list):
    for k, i in enumerate(cosine_top):
        print(f'Top {k + 1} = {song_list[i][0]}')
    print("Results with Euclidean Distance")
    for k, i in enumerate(emd_top):
        print(f'Top EMD{k + 1} = {song_list[i][0]}')

def run_app():
    song_list = get_song_list()
    while True:
        song_path = input("Input Path:\n")
        song = load_song(song_path)
        cosine_top, emd_top = find_most_similar_song(song, song_list)
        print_results(cosine_top, emd_top, song_list)

if __name__ == "__main__":
    run_app()
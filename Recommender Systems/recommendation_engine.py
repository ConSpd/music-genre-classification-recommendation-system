import json
import numpy as np
import librosa, librosa.feature
import math
from scipy.spatial.distance import cosine

DATASET_PATH = "recommendation_data.json"
SAMPLES_PER_TRACK = 22050 * 30
num_samples_per_segment = int(SAMPLES_PER_TRACK / 10)


def load_data(dataset_path):
    # convert lists into numpy arrays
    inputs = np.array(data["mfcc"])
    return inputs


def load_song(song_path):
    signal, sr = librosa.load(song_path, sr=22050)
    if len(signal) > 1984512:
        mfcc = librosa.feature.mfcc(y=signal[882000:1543504], sr=sr, n_mfcc=40, n_fft=2048, hop_length=512).T
    else:
        mfcc = librosa.feature.mfcc(y=signal[:661504], sr=sr, n_mfcc=40, n_fft=2048, hop_length=512).T
    return mfcc


def find_most_similar_song(song, list_of_songs):
    cosine_similarity_score = [1 - cosine(song.flatten(), s.flatten()) for s in list_of_songs]
    top_indices = sorted(enumerate(cosine_similarity_score), key=lambda x: x[1], reverse=True)[:5]
    top_indices_only = [index for index, _ in top_indices]

    # With Euclidean
    EMD_similarity_score = [math.dist(song.flatten(), s.flatten()) for s in list_of_songs]
    EMD_top_indices_only = sorted(range(len(EMD_similarity_score)), key=lambda sub: EMD_similarity_score[sub])[:5]

    return top_indices_only, EMD_top_indices_only


if __name__=="__main__":
    with open(DATASET_PATH,"r") as fp:
        data = json.load(fp)

    list_of_songs = np.array(data["mfcc"])
    song_metadata = {"name": np.array(data["name"]), "path": np.array(data["path"])}

    input_song = input("Input path:\n")
    while input_song != "":
        # Load Song
        song = load_song(input_song)
        index, EMD_index = find_most_similar_song(song, list_of_songs)
        print("Results with Cosine Similarity")
        for k, i in enumerate(index):
            print(f'Top {k+1} = {song_metadata["name"][i]}, path {song_metadata["path"][i]}')
        print("Results with Euclidean Distance")
        for k, i in enumerate(EMD_index):
            print(f'Top with EMD{k+1} = {song_metadata["name"][i]}, path {song_metadata["path"][i]}')
        input_song = input("Input path:\n")


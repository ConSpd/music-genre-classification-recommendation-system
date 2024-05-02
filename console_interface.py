import json
import numpy as np
import keras
import librosa, librosa.feature
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine

DATASET_PATH = "MGC/data.json"
DATASET_PATH_RS = "Recommender Systems/recommendation_data.json"
SAMPLES_PER_TRACK = 22050 * 30
num_samples_per_segment = int(SAMPLES_PER_TRACK / 10)


def load_data(dataset_path):
    with open(dataset_path,"r") as fp:
        data = json.load(fp)
        inputs = np.array(data["mfcc"])
        targets = np.array(data["labels"])
        return inputs, targets


def plot_results(prediction):
    x_axis = ["pop", "metal", "disco", "blues", "reggae", "classical", "rock", "hiphop", "country", "jazz"]
    y_axis = []
    for p in prediction:
        y_axis.append(p)
    plt.figure(figsize=(10, 6))
    plt.bar(x_axis, y_axis)
    plt.xlabel('Genres')
    plt.ylabel('Score')
    plt.show()


def predict(model, song, mfcc_rs):
    sum_preds = np.zeros(10)
    for s in song:
        s = s[np.newaxis, ...]
        prediction = model.predict(s)
        preds = np.asarray(prediction[0])
        sum_preds = np.add(sum_preds, preds)
    plot_results(sum_preds)
    genres = ["pop", "metal", "disco", "blues", "reggae", "classical", "rock", "hiphop", "country", "jazz"]
    predicted_genre = np.argmax(sum_preds)  # Genre of Music that was mostly predicted
    index = find_most_similar_song(mfcc_rs, list_of_songs, genres[predicted_genre])
    for k, i in enumerate(index):
        print(f'Top {k + 1} = {song_metadata["name"][i]}, path {song_metadata["path"][i]}')

def find_most_similar_song(song, list_of_songs, predicted_genre):
    cosine_similarity_score = []
    i = 0
    for s in list_of_songs:
        if predicted_genre in song_metadata["name"][i]:
            cosine_similarity_score.append(1 - cosine(song.flatten(), s.flatten()))
        else:
            cosine_similarity_score.append(-1)
        i = i + 1
    top_indices = sorted(enumerate(cosine_similarity_score), key=lambda x: x[1], reverse=True)[:5]
    top_indices_only = [index for index, _ in top_indices]
    return top_indices_only

def load_song(song_path):
    signal, sr = librosa.load(song_path, sr=22050)
    mfcc_table = []
    for s in range(10):
        start_sample = num_samples_per_segment * s
        finish_sample = start_sample + num_samples_per_segment
        mfcc = librosa.feature.mfcc(y=signal[start_sample:finish_sample], n_mfcc=13, n_fft=2048, hop_length=512).T
        mfcc = mfcc[..., np.newaxis]
        mfcc_table.append(mfcc)

    # For RS
    if len(signal) > 1984512:
        mfcc_rs = librosa.feature.mfcc(y=signal[882000:1543504], sr=sr, n_mfcc=40, n_fft=2048, hop_length=512).T
    else:
        mfcc_rs = librosa.feature.mfcc(y=signal[:661504], sr=sr, n_mfcc=40, n_fft=2048, hop_length=512).T

    return mfcc_table, mfcc_rs



if __name__=="__main__":
    # Load the Model
    model = keras.models.load_model('MGC/mymodel.keras')
    input_song = input("Give path of song:\n")

    with open(DATASET_PATH_RS,"r") as fp:
        data = json.load(fp)
    list_of_songs = np.array(data["mfcc"])
    song_metadata = {"name": np.array(data["name"]), "path": np.array(data["path"])}

    while input_song != "":
        # Load song
        song, mfcc_rs = load_song(input_song)
        predict(model, song, mfcc_rs)
        input_song = input("Give path of song:\n")

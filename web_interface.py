from flask import Flask, render_template, request
import os
import numpy as np
import keras
import librosa, librosa.feature
from werkzeug.utils import secure_filename
import json
from scipy.spatial.distance import cosine

DATASET_PATH = "MGC/data.json"
DATASET_PATH_RS = "Recommender Systems/recommendation_data.json"
SAMPLES_PER_TRACK = 22050 * 30
num_samples_per_segment = int(SAMPLES_PER_TRACK / 10)
UPLOAD_FOLDER = "/Users/conspd/Desktop/Python/uploads"
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/songupload', methods=['POST'])
def song_upload():
    file = request.files['audio']
    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    song, mfcc_rs = load_song(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    return predict(model, song, mfcc_rs)

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

    # For MGC
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


def predict(model, song, mfcc_rs):
    sum_preds = np.zeros(10)
    for s in song:
        s = s[np.newaxis, ...]
        prediction = model.predict(s)
        preds = np.asarray(prediction[0])
        sum_preds = np.add(sum_preds, preds)
    genres = ["pop", "metal", "disco", "blues", "reggae", "classical", "rock", "hiphop", "country", "jazz"]
    predicted_genre = np.argmax(sum_preds) # Genre of Music that was mostly predicted
    index = find_most_similar_song(mfcc_rs, list_of_songs, genres[predicted_genre])
    list_of_paths = []
    list_of_names = []
    for k, i in enumerate(index):
        # list_of_paths.append(song_metadata["path"][i]+'/'+song_metadata["name"][i])
        list_of_paths.append(song_metadata["path"][i] + '/' + song_metadata["name"][i])
        list_of_names.append(song_metadata["name"][i])

    return render_template('showresults.html', sum_preds=sum_preds, genres=genres, paths=zip(list_of_paths,list_of_names))

if __name__=="__main__":
    # Load the Model
    model = keras.models.load_model('MGC/mymodel.keras')

    # Load data for RS
    with open(DATASET_PATH_RS,"r") as fp:
        data = json.load(fp)
    list_of_songs = np.array(data["mfcc"])
    song_metadata = {"name": np.array(data["name"]), "path": np.array(data["path"])}

    app.run(debug=True)

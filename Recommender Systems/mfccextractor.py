import os
import librosa
import librosa.feature
import math
import json


DATASET_PATH = "/Users/conspd/School/Thesis/Data/genres_original"
JSON_PATH = "recommendation_data.json"


def save_mfcc(dataset_path, json_path, n_mfcc=40, n_fft=2048, hop_length=512):
    data = {
        "name": [],
        "mfcc": [],
        "path": [],
    }

    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        if dirpath is not dataset_path:
            dirpath_components = dirpath.split("/")
            semantic_label = dirpath_components[-1]

            print(f"\nProcessing {semantic_label}")

            for f in filenames:
                file_path = os.path.join(dirpath, f)
                signal, sr = librosa.load(file_path, sr=22050)

                # process segments extracting mfcc and storing data
                mfcc = librosa.feature.mfcc(y=signal[:661504],
                                            sr=22050,
                                            n_mfcc=n_mfcc,
                                            n_fft=n_fft,
                                            hop_length=hop_length)
                mfcc = mfcc.T
                if len(mfcc) == 1293:
                    data["mfcc"].append(mfcc.tolist())
                    data["name"].append(f)
                    data["path"].append(dirpath)

    with open(JSON_PATH, "w") as fp:
        json.dump(data, fp, indent=4)


if __name__=="__main__":
    save_mfcc(DATASET_PATH, JSON_PATH)
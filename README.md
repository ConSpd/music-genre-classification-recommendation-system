# music-genre-classification-recommendation-system
A Deep Learning approach to music genre classification that recognises the genre of an input song and recommends 5 similar songs.

## Data Set and Feature Extraction
The Dataset used for the classification and recommendation system is the GTZAN Dataset that consists of 10 genres where each one contains 100 songs of 30 second duration. The extracted features are the MFCC both for the Music Genre Classification and for the Recommendation System.

## Music Genre Classification
The MGC category contains The MFCC extractor `preprocess.py` and 3 Deep Neural Architectures that tackle the MGC part.
1. Multilayer Perceptron is `mlpclassifier.py`. The `overfitsolve.py` solves the problem of overfitting but still an MLP model.
2. Convolutional Neural Network is `cnnclassifier.py`, the CNN is the model with the highest accuracy and also the one that is used in the final app.
3. Long-Short Term Memory Network is `lstm.py`.

## Music Recommendation System
In MRS we use 2 approaches, one using MFCC only comparisson and the other using multi-feature csv dataset that is provided by the GTZAN dataset.
1. MFCC apprach is contained in `recommendation_engine.py`.
2. Multi-feature approach is contained in `multifeaturers.py`.

## Final Approach
Using CNN Network and MFCC based Recommendation Engine we built a Web App using the Flask Framework and the source code is located at `web_interface.py` file.

## Running of Application
**Main Page**<br>
<img src=https://github.com/ConSpd/music-genre-classification-recommendation-system/assets/74179715/aab2f7ff-52e6-45ea-8b61-667cb40f0ab6 width=500><br><br>

**Results of Rock song**<br>
<img src=https://github.com/ConSpd/music-genre-classification-recommendation-system/assets/74179715/ac64a7c4-2fb9-4e29-9bf6-0a684e9ee22f width=500><br><br>

**Results of Metal song**<br>
<img src=https://github.com/ConSpd/music-genre-classification-recommendation-system/assets/74179715/fa9d24ee-abab-422c-94e9-fe371b404544 width=500> 

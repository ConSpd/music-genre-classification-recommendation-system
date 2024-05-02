import json
import numpy as np
import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

DATASET_PATH = "data.json"

def load_data(dataset_path):
    with open(dataset_path,"r") as fp:
        data = json.load(fp)
        # convert lists into numpy arrays
        inputs = np.array(data["mfcc"])
        targets = np.array(data["labels"])

        return inputs, targets

def plot_history(history):
    fig, axs = plt.subplots(2)

    # accuracy subplot
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy evaluation")

    # error subplot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="lower right")
    axs[1].set_title("Error val")

    plt.show()

def print_metrics(inputs_test):
    predictions = model.predict(inputs_test)
    predicted_labels = np.argmax(predictions, axis=1)

    # calculate metrics
    accuracy = accuracy_score(targets_test, predicted_labels)
    precision = precision_score(targets_test, predicted_labels, average="weighted")
    recall = recall_score(targets_test, predicted_labels, average="weighted")
    f1 = f1_score(targets_test, predicted_labels, average="weighted")

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

def prep_data(inputs, targets):
    inputs_train, inputs_test, targets_train, targets_test \
        = train_test_split(inputs, targets, test_size=0.45)
    inputs_test, inputs_val, targets_test, targets_val \
        = train_test_split(inputs_test, targets_test, test_size=0.45)
    inputs_train = inputs_train[..., np.newaxis]
    inputs_val = inputs_val[..., np.newaxis]
    inputs_test = inputs_test[..., np.newaxis]

    return inputs_train, inputs_val, inputs_test,\
           targets_train, targets_val, targets_test

def predict(model, input, expected_output):

    input = input[np.newaxis, ...] # array shape (1, 130, 13, 1)

    # perform prediction
    prediction = model.predict(input)

    # get index with max value
    predicted_index = np.argmax(prediction, axis=1)

    print("Target: {}, Predicted label: {}".format(expected_output, predicted_index))

def build_model(input_shape):
    # create model
    model = keras.Sequential()

    # first conv layer
    model.add(keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPool2D((3,3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # second conv layer
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # third conv layer
    model.add(keras.layers.Conv2D(32, (2, 2), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # flatten output of conv layer and feed it into dense layer
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.3))

    # output layer that uses softmax
    model.add(keras.layers.Dense(10, activation='softmax'))

    return model

if __name__=="__main__":
    inputs, targets = load_data(DATASET_PATH)
    inputs_train, inputs_val, inputs_test, targets_train, targets_val, targets_test = prep_data(inputs, targets)

    # Build CNN
    input_shape = (inputs_train.shape[1], inputs_train.shape[2], inputs_train.shape[3])
    model = build_model(input_shape)

    # Compile the CNN
    optimizer = keras.optimizers.legacy.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
                  loss="sparse_categorical_crossentropy",
                  metrics=['accuracy'])

    # Train Model
    history = model.fit(inputs_train, targets_train,
              validation_data=(inputs_val, targets_val),
              batch_size=32, epochs=50)
    model.summary()
    error, accuracy = model.evaluate(inputs_test, targets_test, verbose=1)
    print(f"Accuracy on test = {accuracy}")
    plot_history(history)
    print_metrics(inputs_test)

    # # pick a sample to predict from the test set
    # X_to_predict = inputs_test[100]
    # y_to_predict = targets_test[100]
    #
    # # predict sample
    # predict(model, X_to_predict, y_to_predict)

    # model.save('mymodel.keras')

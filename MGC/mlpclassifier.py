import json
import numpy as np
import keras
from sklearn.model_selection import train_test_split
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

if __name__=="__main__":
    # load data
    inputs, targets = load_data(DATASET_PATH)

    # split the data into train and test sets
    inputs_train, inputs_test, targets_train, targets_test = train_test_split(inputs, targets, test_size=0.3)

    # build the network achitecture
    model = keras.Sequential([
        # input layer
        keras.layers.Flatten(input_shape=(inputs.shape[1], inputs.shape[2])),

        # 1st hidden layer
        keras.layers.Dense(512, activation="relu"),

        # 2nd hidden layer
        keras.layers.Dense(256, activation="relu"),

        # 3rd hidden layer
        keras.layers.Dense(64, activation="relu"),

        # output layer
        keras.layers.Dense(10, activation="softmax")
    ])

    # compile network
    optimizer = keras.optimizers.legacy.Adam(learning_rate=0.0001)
    model.compile(optimizer,
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    model.summary()

    # train the network
    history = model.fit(inputs_train,
              targets_train,
              validation_data=(inputs_test, targets_test),
              epochs=50,
              batch_size=32)

    plot_history(history)

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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

from os.path import exists
from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection as sk_model
from PIL import Image


def clean_ans_data(answers_raw_data):
    """
    Milestone 2
    Remove the two entries that don’t have images. Should now have a length of 2 less
    :param answers_raw_data: pandas data frame with answer data
    :return: answers data without corresponding images
    """
    file_list = answers_raw_data.iterrows()
    missing_files = []
    for _, x in file_list:
        fileName = x["BallotID"]
        filePath = "writein_crops/" + fileName + ".jpg"
        file_exists = exists(filePath)
        if not file_exists:
            missing_files += [fileName]
    return answers_raw_data[~answers_raw_data["BallotID"].isin(missing_files)]


def img_to_arr(answers_cleaned):
    """
    Milestone 3
    Read all images into a single, three-dimensional numpy array
    :param answers_cleaned:
    :return: img_array
    """
    img_array = np.empty((len(answers_cleaned), 53, 358))
    file_list = answers_cleaned.iterrows()
    counter = 0
    for _, x in file_list:
        fileName = x["BallotID"]
        img = np.array(Image.open("writein_crops/" + fileName + ".jpg"))
        img_array[counter, :, :] = img[:53, :358] / 255
        counter += 1
    return img_array


def single_output_neuron(train_set_x, train_set_y, validation_set_x, validation_set_y):
    """
    Milestone 6 & 7
    Simple single output neuron with sigmoid activation.
    Compiles model and fits to data
    Accuracy usually around 67%
    :param train_set_x: Ballot ID
    :param train_set_y: 1 or 0
    :param validation_set_x: Ballot ID
    :param validation_set_y: 1 or 0
    :return: trained model
    """
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[53, 358]),
        keras.layers.Dense(64, activation='sigmoid'),
    ])
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics='accuracy'
    )
    model.fit(
        x=train_set_x,
        y=train_set_y,
        epochs=10,
        validation_data=(validation_set_x, validation_set_y),
        callbacks=keras.callbacks.ReduceLROnPlateau(monitor='accuracy', factor=0.2, patience=1)
    )
    return model


def convolutional_neural_network(train_set_x, train_set_y, validation_set_x, validation_set_y):
    """
    Milestone 8 & 9
    More complicated convolutional neural network
    Compiles model and fits data
    Plots learning curve history
    Highest accuracy 97% plateaus after running too much data
        or running code several times in a row
    :param train_set_x: Ballot ID
    :param train_set_y: 1 or 0
    :param validation_set_x: Ballot ID
    :param validation_set_y: 1 or 0
    :return: trained model and model history
    """
    model = keras.models.Sequential([
        keras.layers.Conv2D(32, 10, activation='relu', input_shape=(53, 358, 1)),
        keras.layers.Conv2D(32, 5, activation='relu', input_shape=(53, 358, 1)),
        keras.layers.Conv2D(32, 1, activation='relu', input_shape=(53, 358, 1)),
        keras.layers.MaxPooling2D(1),
        keras.layers.Flatten(input_shape=[53, 358]),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(2, activation='softmax')
    ])
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics='accuracy'
    )
    history = model.fit(
        x=train_set_x,
        y=train_set_y,
        epochs=10,
        validation_data=(validation_set_x, validation_set_y),
        callbacks=keras.callbacks.ReduceLROnPlateau(monitor='accuracy', factor=0.7, patience=1)
    )
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()
    return model


def main():
    """
    Runs all functions
    Milestones 1, 4, 5 out of functions
    1: read answers into pandas dataframe
    4: set aside 20% of data for testing
    5: set aside 20% of remaining data for validation
    """
    answers_raw_data = pd.read_csv('answers.csv', nrows=10000)
    # answers_raw_data = pd.read_csv('answers.csv')
    answers_cleaned = clean_ans_data(answers_raw_data)
    img_array = img_to_arr(answers_cleaned)
    train_set_x, test_set_x, train_set_y, test_set_y = sk_model.train_test_split(img_array, answers_cleaned["raiford"],
                                                                                 test_size=0.2, train_size=0.8)
    test_set_x, validation_set_x, test_set_y, validation_set_y = sk_model.train_test_split(train_set_x, train_set_y,
                                                                                           test_size=0.2,
                                                                                           train_size=0.8)
    # model = single_output_neuron(train_set_x, train_set_y, validation_set_x, validation_set_y)
    model = convolutional_neural_network(train_set_x, train_set_y, validation_set_x, validation_set_y)
    results = model.evaluate(test_set_x, test_set_y)
    print("test loss, test acc: ", results)


main()

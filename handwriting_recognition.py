from os.path import exists
import keras as keras
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


def single_output_neuron(trainSetX, trainSetY, validationSetX, validationSetY):
    """
    Milestone 6 & 7
    Simple single output neuron with sigmoid activation.
    Compiles model and fits to data
    Accuracy usually around 67%
    :param trainSetX: Ballot ID
    :param trainSetY: 1 or 0
    :param validationSetX: Ballot ID
    :param validationSetY: 1 or 0
    :return: trained model
    """
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[53, 358]),
        keras.Input(shape=(53, 358)),
        keras.layers.Dense(64, activation='sigmoid'),
    ])
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics='accuracy'
    )
    model.fit(
        x=trainSetX,
        y=trainSetY,
        epochs=10,
        validation_data=(validationSetX, validationSetY),
        callbacks=keras.callbacks.ReduceLROnPlateau(monitor='accuracy', factor=0.2, patience=1)
    )
    return model


def convolutional_neural_network(trainSetX, trainSetY, validationSetX, validationSetY):
    """
    Milestone 8 & 9
    More complicated convolutional neural network
    Compiles model and fits data
    Plots learning curve history
    Highest accuracy 97% plateaus after running too much data
        or running code several times in a row
    :param trainSetX: Ballot ID
    :param trainSetY: 1 or 0
    :param validationSetX: Ballot ID
    :param validationSetY: 1 or 0
    :return: trained model and model history
    """
    model = keras.models.Sequential([
        keras.layers.Conv2D(32, 12, activation='relu', input_shape=(53, 358, 1)),
        keras.layers.MaxPooling2D(1),
        keras.layers.Flatten(input_shape=[53, 358]),
        keras.Input(shape=(53, 358)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(2, activation='softmax')
    ])
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics='accuracy'
    )
    history = model.fit(
        x=trainSetX,
        y=trainSetY,
        epochs=10,
        validation_data=(validationSetX, validationSetY),
        callbacks=keras.callbacks.ReduceLROnPlateau(monitor='accuracy', factor=0.2, patience=1)
    )
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()
    return model


def main():
    # Milestone 1:
    # Read answers.csv into a pandas dataframe. It should have a length of 47703.
    answers_raw_data = pd.read_csv('answers.csv', nrows=1000)
    answers_cleaned = clean_ans_data(answers_raw_data)
    img_array = img_to_arr(answers_cleaned)
    # Milestone 4 & 5
    trainSetX, testSetX, trainSetY, testSetY = sk_model.train_test_split(img_array, answers_cleaned["raiford"],
                                                                         test_size=0.2, train_size=0.8)
    trainSetX, validationSetX, trainSetY, validationSetY = sk_model.train_test_split(trainSetX, trainSetY,
                                                                                     test_size=0.2, train_size=0.8)
    # single_output_neuron(trainSetX, trainSetY, validationSetX, validationSetY)
    model = convolutional_neural_network(trainSetX, trainSetY, validationSetX, validationSetY)
    results = model.evaluate(testSetX, testSetY)
    print("test loss, test acc: ", results)


main()
from os.path import exists
import keras as keras
import pandas as pd
import numpy as np
import sklearn as sk
import sklearn.model_selection as sk_model
from PIL import Image
import tensorflow as tf

# Milestone 1:
# Read answers.csv into a pandas dataframe. It should have a length of 47703.
# answers_raw_data = pd.read_csv('answers.csv')
answers_raw_data = pd.read_csv('answers.csv', nrows=1000)

# Milestone 2:
# Remove the two entries that don’t have images. Should now have a length of 2 less
file_list = answers_raw_data.iterrows()
missing_files = []
for _, x in file_list:
    fileName = x["BallotID"]
    filePath = "writein_crops/" + fileName + ".jpg"
    file_exists = exists(filePath)
    if not file_exists:
        missing_files += [fileName]
answers_cleaned = answers_raw_data[~answers_raw_data["BallotID"].isin(missing_files)]

# Milestone 3:
# Read all the images into a single, three-dimensional numpy array (img_array).
img_array = np.empty((len(answers_cleaned), 53, 358))
file_list = answers_cleaned.iterrows()
counter = 0
for _, x in file_list:
    fileName = x["BallotID"]
    img = np.array(Image.open("writein_crops/" + fileName + ".jpg"))
    img_array[counter, :, :] = img[:53, :358] / 255
    counter += 1

# Milestone 4 & 5:
# Set aside 20% of the data for testing.
# Set aside 20% of the remaining data for validation.
# trainSetX & testSetX & validationSetX = array of images
# trainSetY & testSetY & validationSetY = 0 or 1 in Raiford column of answers.csv
trainSetX, testSetX, trainSetY, testSetY = sk_model.train_test_split(img_array, answers_cleaned["raiford"],
                                                                     test_size=0.2, train_size=0.8)
trainSetX, validationSetX, trainSetY, validationSetY = sk_model.train_test_split(trainSetX, trainSetY,
                                                                                 test_size=0.2, train_size=0.8)

# Milestone 6 & 7
model = keras.models.Sequential([
    keras.layers.Conv2D(32, 11, activation='sigmoid', input_shape=(53, 358, 1)),
    keras.layers.MaxPooling2D(1),
    keras.layers.Conv2D(32, 11, activation='sigmoid', input_shape=(53, 358, 1)),
    keras.layers.MaxPooling2D(1),
    keras.layers.Flatten(input_shape=[53, 358]),
    keras.Input(shape=(53, 358)),
    keras.layers.Dense(32, activation='sigmoid'),
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
    validation_data=(validationSetX, validationSetY)
)
model.summary()







# history = model.fit(x=trainSetX, y=trainSetY, epochs=10, validation_data=(validationSetX, validationSetY))

# print(model.output_shape)
# print(model.compute_output_signature)



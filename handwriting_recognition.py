#Imports
from os.path import exists
import keras as keras
import pandas as pd
import numpy as np
import sklearn as sk
import sklearn.model_selection as skmodel
from sklearn.model_selection import train_test_split
from PIL import Image

# Milestone 1:
# Read answers.csv into a pandas dataframe. It should have a length of 47703.
answers_raw_data = pd.read_csv('answers.csv')
print(len(answers_raw_data))

# Milestone 2:
# Remove the two entries that don’t have images. Should now have a length of 2 less
file_list = answers_raw_data.iterrows()
missing_files = []
for _, x in file_list:
    fileName = x["BallotID"]
    filePath = "writein_crops/" + fileName + ".jpg"
    file_exists = exists(filePath)
    if file_exists == False:
        missing_files+=[fileName]
answers_cleaned = answers_raw_data[~answers_raw_data["BallotID"].isin(missing_files)]
print(len(answers_cleaned))

# Milestone 3:
# Read all of the images into a single, three-dimensional numpy array (img_array).
img_array = np.empty(len(answers_cleaned),53,358)
file_list = answers_cleaned.iterrows()
counter = 0
for _, x in file_list:
    fileName = x["BallotID"]
    img = np.array(Image.open("writein_crops/" + fileName + ".jpg"))
    img_array[counter,:,:] = img[:53,:358] / 255
    counter+=1

# Milestone 4 & 5:
# Set aside 20% of the data for testing.
# Set aside 20% of the remaining data for validation.
#trainSetX & testSetX & validationSetX = array of images
#trainSetY & testSetY & validationSetY = 0 or 1 in Raiford column of answers.csv
trainSetX, testSetX, trainSetY, testSetY = train_test_split(img_array,answers_cleaned["raiford"],test_size=0.2,train_size=0.8)
trainSetX, validationSetX, trainSetY, validationSetY = train_test_split(trainSetX,trainSetY,test_size=0.2,train_size=0.8)

#NOTES for next steps: USE Tensorflow Keras (big machine learning library)
#Likely in textbook
#Start by defining model
model=keras.models.sequential()
#Start with dense layer with sigmoid function
#For compile model.compile
#For fitting model model.fit (needs inputs)

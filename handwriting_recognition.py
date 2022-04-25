import pandas as pd
import numpy as np
import sklearn as sk

answers_raw_data = pd.read_csv('answers.csv')

#Training set = 60%
#Testing Set = 20%
#Validation Set = 20%
train_set, test_set = train_test_split(answers,test_size=0.2,train_size=0.8)
train_set, validation_set = train_test_split(train_set,test_size=0.25,train_size=0.75)

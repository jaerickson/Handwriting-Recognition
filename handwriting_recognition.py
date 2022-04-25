import pandas as pd
import numpy as np

answers_raw_data = pd.read_csv('answers.csv')

#Training set = 60%
#Testing Set = 20%
#Validation Set = 20%
train_set, test_set, validation_set = train_test_split(answers_raw_data,test_size=0.2,train_size=0.6,validation_size=0.2)

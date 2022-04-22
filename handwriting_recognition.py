#Imports
import pandas as pd

#Creates a variable "answers" that holds data from answers.csv
answers = pd.read_csv ('answers.csv')

answers = answers[~answers['BallotID'].isin(nonexistent)]
#Print data as a test
print(answers)

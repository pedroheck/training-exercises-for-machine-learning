import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from numpy import random

data = pd.read_csv('Beginner/Asteroid Classification/data/asteroids.csv')

SEED = 1234 # any number
random.seed(SEED) # setting the seed for reproducibility (everytime we run the code, we get the same results)

x = data.drop(columns=['Name', 'Neo Reference ID', 'Close Approach Date', 'Epoch Date Close Approach', 'Orbit ID', 'Orbit Determination Date', 'Orbiting Body', 'Equinox', 'Hazardous'])
labels = data['Hazardous']

train_x, test_x, train_y, test_y = train_test_split(x, labels, test_size=0.2, stratify=labels)

# Creating a model and fitting it to the training data
model = RandomForestClassifier(n_estimators=100)
model.fit(train_x, train_y)

# Predicting the test data
score = model.score(test_x, test_y)

# Creating a dummy classifier to compare our results


dummy_model = DummyClassifier(strategy='most_frequent')
dummy_model.fit(train_x, train_y)
dummy_score = dummy_model.score(test_x, test_y)

print("The dummy model's score is %.2f%%" % (dummy_score * 100))
print("Our model's score is %.2f%%" % (score * 100))
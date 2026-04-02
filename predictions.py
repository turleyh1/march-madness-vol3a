# import things
import pandas as pd
import numpy as np

from the_model import clean, add_features, train_and_test

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# clean data
# engineer new features
# find and train best model
# use model to predict a single game
# loop it for multiple games (or do it individually for each game)
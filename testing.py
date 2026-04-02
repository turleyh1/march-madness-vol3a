# import things
import pandas as pd
import numpy as np

from the_model import clean, add_features, train_and_test, predict_single_game

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

school_stats, season_results = clean("school_stats.csv", "2026_season_results.csv")
stats_list = ["Rk", "W-L%", "FG%", "TRB", "TOV", "SOS", "SRS"]
comparison_df = add_features(school_stats, season_results, stats_list)
results = train_and_test(comparison_df)
print(results)



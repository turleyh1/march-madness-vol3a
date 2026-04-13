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
basic_stats = ["Rk", "W-L%", "FG%", "TRB", "TOV", "SOS", "SRS"]



# differernt stats to check
defense_stats = ["Rk", "W-L%", "SOS", "SRS", "TRB", "ORB", "BLK", "STL", "TOV"]
""" This list is meant to measure the strength of a teams defense
and see if that is a better indicator of wins then anything else.
Rk, W-L%, SOS, SRS are just defaults that I think basically all lists
should include for basic stats on teams. After that I would want to add
something in the subtract TRB and ORB that way we only get defensive 
rebounds, and then the other stats are other defensive stats to use."""

offense_stats = ["Rk", "W-L%", "SOS", "SRS", "FG%", "3P%", "FT%", "ORB", "AST"]
""" This list is the opposite of the one above. It is testing how well the offensive
power of a team is at predicting wins. Again the basic stats are included, followed,
by purely offensive stats. """

well_rounded_stats = ["Rk", "W-L%", "SOS", "SRS", "FG%", "3P%", "FT%", "TRB", "AST", "BLK", "STL", "TOV"]
""" This list combines the defensive and offensive lists into one. I choose to only include TRB because for
now I don't care about whether it was offensive or defensive since we are combining them, but maybe later I 
will want to seperate them."""

# kyles thoughts
# ORB for winning, TOV for losing (compare aginst each other)
# avg possesion time (so do schools that use the whole shot clock do better?)
# second chance points (% conversion on second chance points)
# look at per game vs per minute stats 

# could also look at home vs away stats 


comparison_df = add_features(school_stats, season_results, well_rounded_stats)
results = train_and_test(comparison_df)
# print(results)
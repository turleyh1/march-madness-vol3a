# import things
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


############## clean data function #################
def clean(school_stats_file, season_results_file):
    # read in data
    school_stats = pd.read_csv(school_stats_file)
    season_results = pd.read_csv(season_results_file)

    # remove the redundent column headers
    column_list = season_results.columns.tolist()
    # print(column_list)
    season_results = season_results[~season_results.isin(column_list).any(axis=1)]

    # rename the columns to make more sense
    season_results = season_results.rename(columns={'Visitor/Neutral': "Visitor", 
                                                    'PTS': 'PTS_v', 
                                                    'Home/Neutral': 'Home', ''
                                                    'PTS.1': 'PTS_h'})
    
    # print(season_results.columns)
    
    # remove rows that are missing values in the points columns or missing teams
    season_results = season_results.dropna(subset=['Home', 'Visitor', 'PTS_v', 'PTS_h'])
    season_results = season_results.reset_index(drop=True)

    # check what type of data are in the data sets
    #print(season26_results.dtypes)
    #print(school_stats.dtypes)

    # change data types to what they should be for season results (school stats is fine)
    season_results = season_results.astype({
        "Date": str,
        "Visitor": str,
        "PTS_v": float,
        "Home": str,
        "PTS_h": float,
        "OT": str,
        "Notes": str,
    })


    # I need to add in data cleaning for school stats, if the school name is missing remove column
    # make sure all data type is float besides school name

    return school_stats, season_results


#################### feature engineering function ####################
def add_features(school_stats_file, season_results_file, stats_list):
    # # defensive rebound stuff
    # school_stats_file['DRB'] = school_stats_file['TRB'] - school_stats_file['ORB']

    # stats_list.remove("TRB")
    # stats_list.remove("ORB")
    # stats_list.append("DRB")


    # merge school stats and season results
    # first merge Visitor stats
    merged_df = pd.merge(
        season_results_file,
        school_stats_file,
        left_on='Visitor',
        right_on='School'
    )

    # then again but this time on home stats
    final_df = pd.merge(
        merged_df,
        school_stats_file,
        left_on='Home',
        right_on='School',
        suffixes=('_v', '_h')
    )

    # drop duplicates from merge
    final_df = final_df.drop(['School_v', 'School_h'], axis=1)

    # print(final_df.head()
    # print(final_df.columns)

    # calculate if Home team won (1) or if Visitors won (0)
    final_df["Win"] = (final_df['PTS_h'] > final_df["PTS_v"]).astype(int)


    # remove unnecessary columns
    final_df = remove_cols(stats_list, final_df)
    
    # subtract the home team and visitor teams stats
    for stat in stats_list:
        final_df[f'Diff_{stat}'] = final_df[f'{stat}_h'] - final_df[f'{stat}_v']


    return final_df


#################### Train and Test ML ####################
def train_and_test(comparison_df):
    # only get diff stats to put into training model
    features = [col for col in comparison_df.columns if col.startswith('Diff_')]
    X = comparison_df[features]
    y = comparison_df["Win"]

    # split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2)

    # dictionary of models to try
    models = {
        "Random Forest": RandomForestClassifier(random_state=402),
        "K-Neighbors": make_pipeline(StandardScaler(), KNeighborsClassifier()),
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "XGBoost": XGBClassifier(eval_metric='logloss')
    }

    results = {}
    best_score = 0
    best_name = None

    # loop through different models to see which is the best
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        # print(f"\nClassification report for {name}:")
        # print(classification_report(y_test, predictions))

        # calculate score
        score = accuracy_score(y_test, predictions)

        results[name] = model

        # find best model
        if score > best_score:
            best_score = best_score
            best_name = name

    return results[best_name]




################### Predict Whole Tournament #############################
def simulate_tournament(teams_list, school_stats, stats_list, model):
    current_round_teams = teams_list
    round_number = 1
    
    # go until there is only one team left
    while len(current_round_teams) > 1:
        print(f"\n--- Round {round_number} ---")
        winners = []
        
        # loop through teams 2 at a time (Team 0 vs Team 1, Team 2 vs Team 3, etc.)
        for i in range(0, len(current_round_teams), 2):
            team1 = current_round_teams[i]
            team2 = current_round_teams[i+1]
            
            winner = predict_single_game(team1, team2, school_stats, stats_list, model)
            print(f"{team1} vs {team2} -> Winner: {winner}")
            winners.append(winner)
        
        # The winners of this round become the players for the next round
        current_round_teams = winners
        round_number += 1
        
    print(f"\n🏆 TOURNAMENT CHAMPION: {current_round_teams[0]} 🏆")
    return current_round_teams[0]





##################### Auxillary functions ###############################
def remove_cols(stats_list, data_set):
        """arguments:
                list: a list of stat column names that you want to keep
            returns the data set with only those columns remaining """
        # change stats list to inlclude _h and _v
        new_stats_list = []
        for stat in stats_list:
            new_stat_h = stat + '_h'
            new_stat_v = stat + '_v'

            new_stats_list.append(new_stat_v)
            new_stats_list.append(new_stat_h)


        # includes these ones are absolutely necessary but aren't stats
        required_cols = ["Visitor", "Home", "Win"]
        for col in required_cols:
            if col not in new_stats_list:
                new_stats_list.insert(0, col)

        new_data_set = data_set[new_stats_list]

        return new_data_set

def predict_single_game(Team_H, Team_V, school_stats_file, stats_list, model):
    # get home and visitor stats from lists
    home_stats = school_stats_file[school_stats_file['School'] == Team_H]
    visitor_stats = school_stats_file[school_stats_file['School'] == Team_V]
    
    # don't do anything if lists are empty for whatever reason
    if home_stats.empty or visitor_stats.empty:
        print(f"Error: Could not find {Team_H if home_stats.empty else Team_V} in stats file.")
        return

    # create difference stats for game data
    game_data = {}
    for stat in stats_list:
        game_data[f'Diff_{stat}'] = home_stats[stat].values[0] - visitor_stats[stat].values[0]

    single_game_df = pd.DataFrame([game_data])
    

    # keep order the same
    feature_order = [f'Diff_{s}' for s in stats_list]
    single_game_df = single_game_df[feature_order]

    prediction = model.predict(single_game_df)
    
    # get the probability that the home team wins or looses
    prob = model.predict_proba(single_game_df)[0][1] 

    winner = Team_H if prediction[0] == 1 else Team_V
    conf = prob if prediction[0] == 1 else (1 - prob)
    
    # print(f"Prediction: {Team_V} @ {Team_H} -> Winner: {winner} ({conf:.1%} confidence)")
    
    return winner






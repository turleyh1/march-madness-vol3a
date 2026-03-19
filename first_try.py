# import things
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix




def the_model(school_stats_file, season_results_file, stats_list, Team_A, Team_B):
    """This takes a csv file with school stats and a csv file
    with season results, as well as a list of stats you want to include
    in the analysis. It also takes which two teams you want to predict
    the winner of.
    It will return the prediction of who will win between team A and team B """

    # read in data
    school_stats = pd.read_csv("school_stats_file")
    season_results = pd.read_csv("season_results_file")

    ############ clean data ############
    # remove the redundent column headers
    column_list = season_results.columns.tolist()
    # print(column_list)
    season_results = season_results[~season_results.isin(column_list).any(axis=1)]

    # rename the columns to make more sense
    season_results = season_results.rename(columns={'Visitor/Neutral': "Vistor", 
                                                    'PTS': 'PTS_v', 
                                                    'Home/Neutral': 'Home', ''
                                                    'PTS.1': 'PTS_h'})
    
    # remove rows that are missing values in the points columns or missing teams
    season_results = season_results.dropna(subset=['Home', 'Visitor', 'PTS_v', 'PTS_h'])
    season_results = season_results.reset_index(drop=True)

    # check what type of data are in the data sets
    #print(season26_results.dtypes)
    #print(school_stats.dtypes)


    # change data types to what they should be for season results (school stats is fine)
    season_results = season_results.astype({
        "Date": str,
        "Vistor": str,
        "PTS_v": float,
        "Home": str,
        "PTS_h": float,
        "OT": str,
        "Notes": str,
    })

    #################### feature engineering ####################

    # merge school stats and season results
    # first merge vistor stats
    merged_df = pd.merge(
        season_results,
        school_stats_file,
        left_on='Vistor',
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

    # calculate if Home team won (1) or if Vistors won (0)
    final_df["Win"] = (final_df['PTS_h'] > final_df["PTS_v"]).astype(int)


    # remove unnecessary columns
    final_df = remove_cols(stats_list, final_df)
    
    # compares just two schools
    comparison_df = school_comparison(Team_A, Team_B, final_df, stats_list)

    #################### Train and Test ML ####################
    # only get diff stats to put into training model
    features = [col for col in comparison_df.columns if col.startswith('Diff_')]
    X = comparison_df[features]
    y = comparison_df["Win"]

    # split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2)

    # fit to RandomForestClassifier
    rfc = RandomForestClassifier(random_state=43)
    rfc.fit(X_train, y_train)
    rfc_predicted = rfc.predict(X_test)

    # fit to KNeighborsClassifier
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    knn_predicted = knn.predict(X_test)

    # compute the confusion matrixes
    CM_rfc = confusion_matrix(y_test, rfc_predicted)
    CM_knn = confusion_matrix(y_test, knn_predicted)

    # print out reports
    print(f"Classification report for Random Forest")
    print(classification_report(y_test, rfc_predicted))
    print(f"Classification report for K Neighbors")
    print(classification_report(y_test, knn_predicted))


    return rfc_predicted, knn_predicted



stats_list = ["Rk", "W-L%", "FG%", "TRB", "TOV", "SOS", "SRS"]
print(the_model("school_stats.csv", "2026_season_results.csv", stats_list, "Arizona NCAA", "Arizona State"))













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
        required_cols = ["Vistor", "Home", "Win"]
        for col in required_cols:
            if col not in new_stats_list:
                new_stats_list.insert(0, col)

        new_data_set = data_set[new_stats_list]

        return new_data_set

# stats_list = ["Rk", "W-L%"]
#print(remove_cols(stats_list, final_df))



def school_comparison(team_A, team_B, data_set, stats_list):
    for stat in stats_list:
        data_set[f'Diff_{stat}'] = data_set[f'{stat}_h'] - data_set[f'{stat}_v']

    return data_set


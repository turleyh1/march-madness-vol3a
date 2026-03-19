# import things
import pandas as pd
import numpy as np

# import data
school_stats = pd.read_csv("school_stats.csv")
season26_results = pd.read_csv("2026_season_results.csv")

#################### we need to clean the data set ####################

# remove the redundent column headers
column_list = season26_results.columns.tolist()
# print(column_list)
season26_results = season26_results[~season26_results.isin(column_list).any(axis=1)]

# rename the columns to make more sense
season26_results = season26_results.rename(columns={'Visitor/Neutral': "Vistor", 
                                                    'PTS': 'PTS_v', 
                                                    'Home/Neutral': 'Home', ''
                                                    'PTS.1': 'PTS_h'})


# check season results for missing data
#print(season26_results.isna().sum())
# so 4 rows are missing info about points so we need to drop those
season26_results = season26_results.dropna(subset=['PTS_v', 'PTS_h'])
season26_results = season26_results.reset_index(drop=True)

# now check school stats for missing data
#print(school_stats.isna().sum())
# we straight chillin, all the data is there

# check what type of data are in the data sets
#print(season26_results.dtypes)
#print(school_stats.dtypes)


# change data types to what they should be for season results (school stats is fine)
season26_results = season26_results.astype({
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
    season26_results,
    school_stats,
    left_on='Vistor',
    right_on='School'
)

# then again but this time on home stats
final_df = pd.merge(
    merged_df,
    school_stats,
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
# print(remove_cols(stats_list, school_stats))

# we need a function that compares just two schools
def school_comparison(team_A, team_B, data_set, stats_list):
    for stat in stats_list:
        final_df[f'Diff_{stat}'] = final_df[f'{stat}_h'] - final_df[f'{stat}_v']

    return final_df


#################### Train and Test ML ####################
# only get diff stats to put into training model
features = [col for col in final_df.columns if col.startswith('Diff_')]
X = final_df[features]
y = final_df["Win"]


# import things
import pandas as pd
import numpy as np

from the_model import clean, add_features, train_and_test, simulate_tournament, predict_single_game

def tournament(school_stats_file, season_results_file, stats_list, teams_list):
    school_stats, season_results = clean(school_stats_file, season_results_file)
    comparison_df = add_features(school_stats, season_results, stats_list)
    model = train_and_test(comparison_df)  
    simulate_tournament(teams_list, school_stats, stats_list, model)




offense_stats = ["Rk", "W-L%", "SOS", "SRS", "FG%", "3P%", "FT%", "ORB", "AST"]
schedule = [
    "Duke NCAA", "Siena NCAA", "Ohio State NCAA", "Texas Christian NCAA", 
    "St. John's (NY) NCAA", "Northern Iowa NCAA", "Kansas NCAA", "California Baptist NCAA", 
    "Louisville NCAA", "South Florida NCAA", "Michigan State NCAA", "North Dakota State NCAA", 
    "UCLA NCAA", "UCF NCAA", "Connecticut NCAA", "Furman NCAA", "Florida NCAA", 
    "Prairie View A&M NCAA", "Clemson NCAA", "Iowa NCAA", "Vanderbilt NCAA", 
    "McNeese NCAA", "Nebraska NCAA", "Troy NCAA", "North Carolina NCAA", 
    "Virginia Commonwealth NCAA", "Illinois NCAA", "Pennsylvania NCAA", 
    "Saint Mary's NCAA", "Texas A&M NCAA", "Houston NCAA", "Idaho NCAA", 
    "Arizona NCAA", "Long Island University NCAA", "Villanova NCAA", "Utah State NCAA", 
    "Wisconsin NCAA", "High Point NCAA", "Arkansas NCAA", "Hawaii NCAA", "Brigham Young NCAA", 
    "Texas NCAA", "Gonzaga NCAA", "Kennesaw State NCAA", "Miami (FL) NCAA", 
    "Missouri NCAA", "Purdue NCAA", "Queens (NC) NCAA", "Michigan NCAA", "Howard NCAA", 
    "Georgia NCAA", "Saint Louis NCAA", "Texas Tech NCAA", "Akron NCAA", 
    "Alabama NCAA", "Hofstra NCAA", "Tennessee NCAA", "Miami (OH) NCAA", 
    "Virginia NCAA", "Wright State NCAA", "Kentucky NCAA", "Santa Clara NCAA", 
    "Iowa State NCAA", "Tennessee State NCAA"
]

tournament("school_stats.csv", "2026_season_results.csv", offense_stats, schedule)
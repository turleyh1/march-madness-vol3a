# visualitation.py - Fixed for special characters in team names
from the_model2 import clean, add_features, train_and_test, simulate_tournament, predict_single_game
from graphviz import Digraph
import pandas as pd
import math
import html

# File paths
school_stats_file = "school_stats.csv"
season_results_file = "2026_season_results.csv"

# Stats to use
stats_list = ["Rk", "W-L%", "SOS", "SRS", "FG%", "3P%", "FT%", "ORB", "AST"]

# Load and prepare data
school_stats, season_results = clean(school_stats_file, season_results_file)
final_df = add_features(school_stats, season_results, stats_list)
model = train_and_test(final_df)

# Get teams from school_stats
my_teams = schedule = [
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

# get actual tournament results
results = [
    ("Duke NCAA", "Texas Christian NCAA", "St. John's (NY) NCAA", "Kansas NCAA",
    "Louisville NCAA", "Michigan State NCAA", "UCLA NCAA", "Connecticut NCAA", "Florida NCAA", 
    "Iowa NCAA", "Vanderbilt NCAA", "Nebraska NCAA",
    "Virginia Commonwealth NCAA", "Illinois NCAA","Texas A&M NCAA", "Houston NCAA",
    "Arizona NCAA", "Utah State NCAA", "High Point NCAA", "Arkansas NCAA",
    "Texas NCAA", "Gonzaga NCAA", "Miami (FL) NCAA", "Purdue NCAA", "Michigan NCAA", 
    "Saint Louis NCAA", "Texas Tech NCAA","Alabama NCAA", "Tennessee NCAA",
    "Virginia NCAA", "Kentucky NCAA","Iowa State NCAA"), 

    ("Duke NCAA", "St. John's (NY) NCAA",
    "Michigan State NCAA","Connecticut NCAA",
    "Iowa NCAA","Nebraska NCAA",
    "Illinois NCAA","Houston NCAA",
    "Arizona NCAA", "Arkansas NCAA",
    "Texas NCAA", "Purdue NCAA", 
    "Michigan NCAA", "Alabama NCAA", 
    "Tennessee NCAA","Iowa State NCAA"),

    ("Duke NCAA","Connecticut NCAA",
    "Iowa NCAA","Illinois NCAA",
    "Arizona NCAA","Purdue NCAA", 
    "Michigan NCAA","Tennessee NCAA"),

    ("Connecticut NCAA","Illinois NCAA",
    "Arizona NCAA","Michigan NCAA"),

    ("Connecticut NCAA","Michigan NCAA"),

    ("Michigan NCAA",)
]

# Run tournament simulation
champion, history = simulate_tournament(my_teams, school_stats, stats_list, model)

print(f"\n🏆 CHAMPION: {champion} 🏆")

# Extract all teams from history
all_teams = set()
for match in history:
    all_teams.add(match['team1'])
    all_teams.add(match['team2'])
    all_teams.add(match['winner'])


# Calculate tournament structure
max_round = max([match['round'] for match in history])
num_rounds = max_round
num_teams = len(all_teams)

# print(f"Tournament structure: {num_rounds} rounds, {num_teams} teams")

# Group matches by round
matches_by_round = {}
for match in history:
    if match['round'] not in matches_by_round:
        matches_by_round[match['round']] = []
    matches_by_round[match['round']].append(match)

# Function to escape special characters for HTML
def escape_team_name(name):
    """Escape special characters in team names for HTML display"""
    # First escape HTML entities
    escaped = html.escape(name)
    # Replace spaces for better display
    return escaped

# Create visualization

dot = Digraph(format='png', graph_attr={
    'rankdir': 'LR',
    'splines': 'ortho',
    'nodesep': '0.3',
    'ranksep': '0.5',
    'fontname': 'Arial'
})

# Create nodes for each match
# for round_num in range(1, num_rounds + 1):
#     if round_num in matches_by_round:
#         matches = matches_by_round[round_num]
        
#         for idx, match in enumerate(matches):
#             match_id = f"R{round_num}_M{idx}"
            
#             # Escape team names
#             team1_escaped = escape_team_name(match['team1'])
#             team2_escaped = escape_team_name(match['team2'])
#             winner_escaped = escape_team_name(match['winner'])
            
#             # Use simpler label format without HTML table to avoid parsing issues
#             label = f"Round {round_num}\\n{team1_escaped} vs {team2_escaped}\\nWinner: {winner_escaped}"
            
#             # Add color coding
#             dot.node(match_id, label, shape='box', style='filled', 
#                     fillcolor='lightyellow', fontname='Arial', fontsize='10')
            
#             # # Color winner differently
#             # winner_id = f"{match_id}_winner"
#             # dot.node(winner_id, f"🏆 {winner_escaped} 🏆", shape='plaintext', 
#             #         fontcolor='green', fontsize='9', fontname='Arial')

# Assuming 'results' is your list of winner tuples provided above
# Note: Python indices start at 0, so Round 1 is results[0]

for round_num in range(1, num_rounds + 1):
    if round_num in matches_by_round:
        matches = matches_by_round[round_num]
        
        # Get the actual winners for this specific round from your list
        # We use round_num - 1 because your list is 0-indexed
        actual_winners_this_round = results[round_num - 1] if (round_num - 1) < len(results) else []
        
        # To determine if the "Right Teams" played, we check the previous round's actual winners
        # For Round 1, the "Right Teams" are always correct based on your 'my_teams' setup
        actual_winners_prev_round = results[round_num - 2] if round_num > 1 else my_teams

        for idx, match in enumerate(matches):
            match_id = f"R{round_num}_M{idx}"
            t1, t2 = match['team1'], match['team2']
            model_winner = match['winner']
            
            # 1. Check if the correct teams made it to this match
            # (i.e., were both teams actually winners in the previous round?)
            right_teams = (t1 in actual_winners_prev_round) and (t2 in actual_winners_prev_round)
            
            # 2. Check if the model's winner is correct for this round
            right_winner = model_winner in actual_winners_this_round

            # 3. Assign Colors based on your rules
            if right_teams and right_winner:
                bg_color = "green"  # Correct teams, Correct winner
            elif right_teams and not right_winner:
                bg_color = "red"         # Correct teams, Wrong winner
            elif not right_teams and right_winner:
                bg_color = "yellow"      # Wrong teams, Correct winner
            else:
                bg_color = "orange"      # Wrong teams, Wrong winner

            # Create Label
            label = f"Round {round_num}\\n{t1} vs {t2}\\nWinner: {model_winner}"
            
            dot.node(match_id, label, shape='box', style='filled', 
                     fillcolor=bg_color, fontname='Arial', fontsize='10')

# Add connections between rounds
for round_num in range(1, num_rounds):
    if round_num in matches_by_round and (round_num + 1) in matches_by_round:
        current_matches = matches_by_round[round_num]
        next_matches = matches_by_round[round_num + 1]
        
        for idx, match in enumerate(current_matches):
            next_match_idx = idx // 2
            if next_match_idx < len(next_matches):
                next_match = next_matches[next_match_idx]
                
                if match['winner'] in [next_match['team1'], next_match['team2']]:
                    current_id = f"R{round_num}_M{idx}"
                    next_id = f"R{round_num + 1}_M{next_match_idx}"
                    dot.edge(current_id, next_id, color='blue', penwidth='2', arrowsize='0.7')

# Add round labels
round_names = {
    1: "First Round",
    2: "Second Round",
    3: "Sweet 16" if num_teams >= 16 else "Quarterfinals",
    4: "Elite 8" if num_teams >= 16 else "Semifinals",
    5: "Final Four",
    6: "Championship"
}

for round_num in range(1, num_rounds + 1):
    round_label = f"Round_{round_num}_label"
    label_text = round_names.get(round_num, f"Round {round_num}")
    
    dot.node(round_label, label_text, shape='plaintext', fontsize='14', fontname='Bold')
    if round_num in matches_by_round and matches_by_round[round_num]:
        dot.edge(round_label, f"R{round_num}_M0", style='invis')

# Add champion
champion_id = "CHAMPION"
champion_escaped = escape_team_name(champion)
champion_label = f"🏆 CHAMPION: {champion_escaped} 🏆"
dot.node(champion_id, champion_label, shape='box', style='filled', 
         fillcolor='gold', fontsize='16', fontname='Bold')

# Connect final match to champion
final_match_id = f"R{num_rounds}_M0"
dot.edge(final_match_id, champion_id, color='gold', penwidth='3', arrowsize='1')

# Add title
dot.node("title", "TOURNAMENT BRACKET", shape='plaintext', fontsize='20', fontname='Bold')
if 1 in matches_by_round and matches_by_round[1]:
    dot.edge("title", "Round_1_label", style='invis')

# Save and display
try:
    output_file = dot.render('tournament_bracket', view=True)
    print(f"\n✅ Tournament bracket saved to: {output_file}.png")
except Exception as e:
    print(f"\n⚠️ Could not generate PNG, but here's the text version:")
    print(f"Error: {e}")
    print(dot.source)

# Print text bracket as backup
print("\n" + "="*60)
print("TOURNAMENT RESULTS (TEXT VERSION):")
print("="*60)
for round_num in range(1, num_rounds + 1):
    if round_num in matches_by_round:
        print(f"\n{'='*20} ROUND {round_num} {'='*20}")
        for match in matches_by_round[round_num]:
            print(f"{match['team1']} vs {match['team2']}")
            print(f"  → Winner: {match['winner']}")
            print()

print(f"\n🏆 FINAL CHAMPION: {champion} 🏆")
import pandas as pd
from graphviz import Digraph

# Load your CSV
df = pd.read_csv('tournament.csv')

dot = Digraph(comment='Tournament Bracket', graph_attr={'rankdir': 'LR'})

# Create nodes for teams and matches
for index, row in df.iterrows():
    match_id = f"R{row['Round']}M{row['Match']}"
    label = f"{row['Team1']} vs {row['Team2']}\nWinner: {row['Winner']}"
    dot.node(match_id, label)
    
    # Logic to link rounds (e.g., Round 1 winners move to Round 2)
    # This requires your CSV to have a 'NextMatch' ID for easy linking
    if 'NextMatch' in row:
        dot.edge(match_id, row['NextMatch'])

dot.render('bracket_output', format='png', cleanup=True)
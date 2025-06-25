import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm

# Load the data
df = pd.read_csv('../data/2022-10.csv')

# Filter for laundering connections
laundering_df = df[df['Is_laundering'] == 1]

# Get all accounts involved in laundering
laundering_accounts = set(laundering_df['Sender_account']).union(set(laundering_df['Receiver_account']))

# Include connections involving laundering accounts, even if Is_laundering == 0
df = df[(df['Sender_account'].isin(laundering_accounts)) | (df['Receiver_account'].isin(laundering_accounts))]

# Filter data to exclude one-to-one connections
connection_counts = df.groupby(['Sender_account', 'Receiver_account']).size().reset_index(name='count')
filtered_df = connection_counts[connection_counts['Sender_account'].isin(connection_counts['Receiver_account']) |
                                connection_counts['Receiver_account'].isin(connection_counts['Sender_account'])]

print("Exclude one-to-one connection filter done")
print(len(filtered_df))

# filtered_df = df

# Create the graph
G = nx.DiGraph()
for _, row in tqdm(filtered_df.iterrows(), total=filtered_df.shape[0], desc="Adding edges to the graph"):
    sender = row['Sender_account']
    receiver = row['Receiver_account']
    laundering_status = 1 if sender in laundering_accounts or receiver in laundering_accounts else 0
    G.add_edge(sender, receiver, is_laundering=laundering_status)

#  only for view  #

# Define node colors based on laundering status
node_colors = []
for node in G.nodes():
    if node in laundering_accounts:
        node_colors.append('red')  # Laundering-related nodes in red
    else:
        node_colors.append('blue')  # Non-laundering nodes in blue

print("node colored")

# Visualize the graph
pos = nx.spring_layout(G, iterations=2)

print("pos iterated")

# nx.draw(G, pos, with_labels=False, node_size=2)
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=4)
# Draw edges with thinner lines
nx.draw_networkx_edges(G, pos, width=0.5)

# plt.show()
plt.title("Connect Network")
plt.savefig("graph_colored.png", dpi=300)

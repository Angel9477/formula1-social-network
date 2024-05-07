#!/usr/bin/env python
# coding: utf-8

# In[8]:


pip install networkx


# In[9]:


pip install matplotlib


# In[10]:


import networkx as nx


# In[11]:


import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Load race data
circuits_data = pd.read_csv('/Users/angelsheu/Desktop/UCI/Q3/Customer& Social/project/drive-download-20240302T222425Z-001/circuits.csv') 
constructor_results_data = pd.read_csv('/Users/angelsheu/Desktop/UCI/Q3/Customer& Social/project/drive-download-20240302T222425Z-001/constructor_results.csv')  
constructor_standings_data = pd.read_csv('/Users/angelsheu/Desktop/UCI/Q3/Customer& Social/project/drive-download-20240302T222425Z-001/constructor_standings.csv')
constructor_data = pd.read_csv('/Users/angelsheu/Desktop/UCI/Q3/Customer& Social/project/drive-download-20240302T222425Z-001/constructors.csv')  
driver_standings_data = pd.read_csv('/Users/angelsheu/Desktop/UCI/Q3/Customer& Social/project/drive-download-20240302T222425Z-001/driver_standings.csv')  
driver_data = pd.read_csv('/Users/angelsheu/Desktop/UCI/Q3/Customer& Social/project/drive-download-20240302T222425Z-001/drivers.csv')  
race_data = pd.read_csv('/Users/angelsheu/Desktop/UCI/Q3/Customer& Social/project/drive-download-20240302T222425Z-001/races.csv')  
results_data = pd.read_csv('/Users/angelsheu/Desktop/UCI/Q3/Customer& Social/project/drive-download-20240302T222425Z-001/results.csv')  


















# In[12]:


circuits_data.head() 


# In[13]:


constructor_results_data.head()


# In[14]:


constructor_standings_data.head()


# In[15]:


constructor_data.head() #merge with results to get constructor name for easy viz


# In[16]:


driver_standings_data.head() 


# In[17]:


driver_data.head() #merge driverRef (?)


# In[18]:


race_data.head()


# In[19]:


results_data


# In[20]:


# Merge driver data with results data based on driver ID
results_driver_merged = pd.merge(results_data, driver_data, how='left', left_on='driverId', right_on='driverId')

# Merge constructor data with merged driver-results data based on constructor ID
results_constructor_driver_merged = pd.merge(results_driver_merged, constructor_data, how='left', left_on='constructorId', right_on='constructorId')

# Merge constructor data with merged driver-results data based on constructor ID
results_constructor_driver_merged = pd.merge(results_constructor_driver_merged, race_data, how='left', left_on='raceId', right_on='raceId')


# Now you have the results data merged with driver data and constructor data based on driver ID and constructor ID
results_constructor_driver_merged.info()

# know the result, driver, construtor 


# In[21]:


# Filter the race data to include only the last 13 years (2010-2023)
results_constructor_driver_merged = results_constructor_driver_merged[results_constructor_driver_merged['year'] >= results_constructor_driver_merged['year'].max() - 13]

results_constructor_driver_merged


# In[22]:


import networkx as nx
import matplotlib.pyplot as plt

# Create empty directed graph
G = nx.DiGraph()

# Add driver nodes with attributes
for _, driver in results_constructor_driver_merged.iterrows():
    G.add_node(driver['driverRef'], node_type='driver', nationality=driver['nationality_x'])

# Add team nodes with attributes
for _, team in results_constructor_driver_merged.iterrows():
    G.add_node(team['constructorRef'], node_type='team', nationality=team['nationality_y'])

# Adjusting edge addition to include dynamic weights based on points
for _, race in results_constructor_driver_merged.iterrows():
    driver = race['driverRef']
    team = race['constructorRef']
    points = race['points']  # Points scored in this race
    
    # Determine if this race's points qualify for edge weight adjustment
    if points >= 10:
        if G.has_edge(driver, team):
            # Update the weight based on additional points
            G[driver][team]['weight'] += points
        else:
            # Add new edge with initial weight based on the points
            G.add_edge(driver, team, weight=points, race_ids=[race['raceId']])

# Now, G contains edges weighted by the points scored in qualifying races

# Visualize the network
plt.figure(figsize=(20, 20))
pos = nx.spring_layout(G, seed=42)  # Using a fixed seed for reproducibility
# Adjusting node sizes for better visualization
node_sizes = [G.degree(node) * 200 for node in G.nodes()]
nx.draw_networkx(G, pos, with_labels=True, node_size=node_sizes, node_color='lightblue', font_size=10, font_weight='bold')
plt.title('F1 Teams and Drivers Network Based on Points')
plt.axis('off')  # Turn off the axis for better aesthetics
plt.show()


# In[23]:


# Remove 'race_ids' attributes before saving
for u, v, data in G.edges(data=True):
    if 'race_ids' in data:
        del data['race_ids']  # Remove the 'race_ids' attribute

# Save the graph to a GraphML file
nx.write_graphml(G, "2weight_top5win_network_graph.graphml")


# In[24]:


import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# Assuming G has already been constructed with nodes and edges, and edge weights represent the total points

# Calculate the degree of each node for node size adjustment
degrees = dict(G.degree())

# Adjust node sizes more dramatically based on degrees
node_sizes = [degrees[node] * 300 for node in G.nodes()]  # Adjust this formula as needed for visibility

# Define node colors by type: 'skyblue' for drivers, 'lightgreen' for teams
node_colors = ['skyblue' if G.nodes[node]['node_type'] == 'driver' else 'lightgreen' for node in G.nodes()]

# Create a layout for our nodes using the spring layout
pos = nx.spring_layout(G, k=0.75, iterations=50, seed=42)  # Adjust layout spacing and seed for reproducibility

# Calculate edge widths based on points earned (weights)
edge_widths = [G[u][v]['weight'] / 50 for u, v in G.edges()]  # Divide by 10 to scale edge width for visibility

# Draw the graph with adjusted node sizes and edge widths
plt.figure(figsize=(30, 30))
nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.8)
nx.draw_networkx_edges(G, pos, edge_color='gray', width=edge_widths, alpha=0.7)
labels = nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold', bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))

# Customize the plot with a title and by turning off the axis
plt.title('F1 Teams and Drivers Network Based on Total Points Earned', fontsize=16)
plt.axis('off')  # Turn off the axis for better aesthetics

# Show the plot
plt.show()


# In[25]:


# Basic network analysis
print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())
print("Nodes:", G.nodes())
print("Edges:", G.edges())

# Calculate degree centrality
degree_centrality = nx.degree_centrality(G)
print("\nDegree Centrality:")


sorted_degree_centrality = sorted(degree_centrality.items(), key=lambda item: item[1], reverse=True)

# Print the sorted degree centrality values
print("Degree Centrality in descending order:")
for node, centrality in sorted_degree_centrality:
    print(f"{node}: {centrality}")


# Identify driver with highest degree centrality (most connections)
max_degree_driver = max(degree_centrality, key=degree_centrality.get)
print("\nDriver with highest degree centrality:", max_degree_driver)


# In[26]:


#ignore this part, we can just analyze the nationality based on the previous network 

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Assuming results_constructor_driver_merged is your DataFrame

# Initialize a directed graph
G = nx.DiGraph()

# Step 1: Create nodes for all unique nationalities
nationalities = set(results_constructor_driver_merged['nationality_x']).union(set(results_constructor_driver_merged['nationality_y']))
for nat in nationalities:
    G.add_node(nat, type='nationality', color='lightcoral')

# Step 2: Add nodes and edges for drivers and teams
for _, row in results_constructor_driver_merged.iterrows():
    # Add driver node if it doesn't already exist
    if row['driverRef'] not in G:
        G.add_node(row['driverRef'], type='driver', color='skyblue')
    # Add team node if it doesn't already exist
    if row['constructorRef'] not in G:
        G.add_node(row['constructorRef'], type='team', color='lightgreen')
    
    # Add edges
    G.add_edge(row['driverRef'], row['constructorRef'])
    G.add_edge(row['driverRef'], row['nationality_x'])
    G.add_edge(row['constructorRef'], row['nationality_y'])

# Generate positions for nodes using a layout that spreads nodes
pos = nx.spring_layout(G,seed=0)

# Calculate the degree for each node and create a size list
degrees = dict(G.degree())
node_sizes = [degrees[node] * 500 for node in G.nodes()]  # Adjust scale as needed

#To save your graph G to a GraphML file
nx.write_graphml(G, "Driver,team,nationality_network_graph2.graphml")

# Draw the network
plt.figure(figsize=(40, 50))  # Increase figure size
nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=[G.nodes[n]['color'] for n in G.nodes()], alpha=0.8)
nx.draw_networkx_edges(G, pos, alpha=0.5)
nx.draw_networkx_labels(G, pos, font_size=20, bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

# Set plot title and remove axes
plt.title('F1 Driver, Team, and Nationality Network', fontsize=20)
plt.axis('off')

# Show the plot
plt.show()


# In[27]:


#ignore this part

# Basic network analysis
print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())
print("Nodes:", G.nodes())
print("Edges:", G.edges())

# Calculate degree centrality
degree_centrality = nx.degree_centrality(G)
print("\nDegree Centrality:")


sorted_degree_centrality = sorted(degree_centrality.items(), key=lambda item: item[1], reverse=True)

# Print the sorted degree centrality values
print("Degree Centrality in descending order:")
for node, centrality in sorted_degree_centrality:
    print(f"{node}: {centrality}")


# Identify driver with highest degree centrality (most connections)
max_degree_driver = max(degree_centrality, key=degree_centrality.get)
print("\nDriver with highest degree centrality:", max_degree_driver)


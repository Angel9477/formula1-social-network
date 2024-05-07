#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install networkx


# In[2]:


pip install matplotlib


# In[3]:


import networkx as nx


# In[4]:


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
pitstop_data=pd.read_csv('/Users/angelsheu/Desktop/UCI/Q3/Customer& Social/project/drive-download-20240302T222425Z-001/pit_stops.csv')
qualifying_data=pd.read_csv('/Users/angelsheu/Desktop/UCI/Q3/Customer& Social/project/drive-download-20240302T222425Z-001/qualifying.csv')
















# In[5]:


constructor_data.head() #merge with results to get constructor name for easy viz


# In[6]:


# Merge driver data with results data based on driver ID
results_driver_merged = pd.merge(results_data, driver_data, how='left', left_on='driverId', right_on='driverId')

# Merge constructor data with merged driver-results data based on constructor ID
results_constructor_driver_merged = pd.merge(results_driver_merged, constructor_data, how='left', left_on='constructorId', right_on='constructorId')

# Merge constructor data with merged driver-results data based on constructor ID
results_constructor_driver_merged = pd.merge(results_constructor_driver_merged, race_data, how='left', left_on='raceId', right_on='raceId')


# Now you have the results data merged with driver data and constructor data based on driver ID and constructor ID
results_constructor_driver_merged.info()

# know the result, driver, construtor 


# In[7]:


# Filter the race data to include only the last 13 years (2010-2023)
results_constructor_driver_merged = results_constructor_driver_merged[results_constructor_driver_merged['year'] >= results_constructor_driver_merged['year'].max() - 13]

results_constructor_driver_merged


# ## Exploratory Data Analysis

# In[8]:


# First, make sure 'points' column is in numeric format
results_constructor_driver_merged['points'] = pd.to_numeric(results_constructor_driver_merged['points'], errors='coerce')

# Group by 'year', 'raceId', 'forename', and 'surname' to find the total points earned by each driver in each race
total_points_per_race = results_constructor_driver_merged.groupby(['year', 'raceId', 'forename', 'surname'])['points'].sum().reset_index()

# Group by 'year', 'forename', and 'surname' to find the average points earned by each driver per year
average_points_per_year = total_points_per_race.groupby(['year', 'forename', 'surname'])['points'].mean().reset_index()

# Find the driver with the highest average points in each year
driver_with_highest_avg_points = average_points_per_year.loc[average_points_per_year.groupby('year')['points'].idxmax()]

# Create a line plot
plt.figure(figsize=(10, 6))
plt.plot(driver_with_highest_avg_points['year'], driver_with_highest_avg_points['points'], marker='D', linestyle='-', color='#FF0000')
plt.title('Driver with Most Points')
plt.xlabel('Year')
plt.ylabel('Points per GP')

# Label the data points with the corresponding driver names
for i, row in driver_with_highest_avg_points.iterrows():
    plt.text(row['year'], row['points'], f"{row['forename']} \n{row['surname']}", ha='left', va='bottom', fontsize=8, color='black')

plt.ylim(12,26)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_facecolor('#D3D3D3')

plt.show()


# In[9]:


# Group by 'year', 'forename', and 'surname', and calculate the number of wins and total races for each person in each year
grouped_wins =results_constructor_driver_merged.groupby(['year', 'forename', 'surname']).agg({'positionOrder': lambda x: (x == 1).sum(), 'raceId': 'nunique'}).reset_index()

# Calculate the average win for each person in each year
grouped_wins['average_win'] = grouped_wins['positionOrder'] / grouped_wins['raceId']

# Display the resulting DataFrame
print(grouped_wins)


# In[10]:


# Find the row indices corresponding to the highest average_win for each year
idx_max_average_win = grouped_wins.groupby('year')['average_win'].idxmax()

# Select the rows with the highest average_win for each year
drivers_with_highest_average_win = grouped_wins.loc[idx_max_average_win]

# Display the resulting DataFrame
print(drivers_with_highest_average_win)


# In[11]:


# Create a line plot
plt.figure(figsize=(10, 6))
plt.plot(drivers_with_highest_average_win['year'], drivers_with_highest_average_win['average_win'], marker='D', linestyle='-', color='#FF0000')
plt.title('Driver with Most Wins')
plt.xlabel('Year')
plt.ylabel('Wins per GP')

# Label the data points with the corresponding driver names
for i, row in drivers_with_highest_average_win.iterrows():
    plt.text(row['year'], row['average_win'], f"{row['forename']} \n{row['surname']}", ha='left', va='bottom', fontsize=8, color='black')

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_facecolor('#D3D3D3')

plt.ylim(0.2,0.9)
plt.show()


# ## Social network analysis

# In[12]:


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


# In[13]:


# Remove 'race_ids' attributes before saving
for u, v, data in G.edges(data=True):
    if 'race_ids' in data:
        del data['race_ids']  # Remove the 'race_ids' attribute

# Save the graph to a GraphML file
nx.write_graphml(G, "2weight_top5win_network_graph.graphml")


# In[14]:


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


# In[15]:


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


# In[16]:


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


# In[20]:


import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Merge driver data with results data based on driver ID
results_driver_merged = pd.merge(results_data, driver_data, how='left', left_on='driverId', right_on='driverId')

# Merge constructor data with merged driver-results data based on constructor ID
results_constructor_driver_merged = pd.merge(results_driver_merged, constructor_data, how='left', left_on='constructorId', right_on='constructorId')

# Merge race data with merged driver-results data based on race ID
results_constructor_driver_merged = pd.merge(results_constructor_driver_merged, race_data, how='left', left_on='raceId', right_on='raceId')

# Merge pit stop data with merged driver-results data based on race ID
results_constructor_driver_merged = pd.merge(results_constructor_driver_merged, pitstop_data, how='left', left_on='raceId', right_on='raceId')

# Filter the race data to include only the last 13 years (2010-2023)
results_constructor_driver_merged = race_data[(race_data['year'] >= 2010) & (race_data['year'] <= 2023)]

# Create empty directed graph
G = nx.DiGraph()

# Add driver nodes with attributes 
for _, driver in results_constructor_driver_merged.iterrows():
    G.add_node(driver['driverRef'], node_type='driver', nationality=driver['nationality_x'])

# Add team nodes with attributes
for _, team in results_constructor_driver_merged.iterrows():
    G.add_node(team['constructorRef'], node_type='team', nationality=team['nationality_y'])

# Add edges for driver-team top 5 win relationships only
for _, race in results_constructor_driver_merged.iterrows():
    # Only add an edge if the driver won the race (position 1)
    #if race['points'] >= 10:
        G.add_edge(race['driverRef'], race['constructorRef'], race_id=race['raceId'])

# Calculate average pit stop time for each driver-team relationship
average_pitstop_times = {}
for edge in G.edges():
    driver, team = edge
    pitstop_times = results_constructor_driver_merged[(results_constructor_driver_merged['driverRef'] == driver) & (results_constructor_driver_merged['constructorRef'] == team)]['milliseconds_y']
    average_pitstop_time = pitstop_times.mean()
    average_pitstop_times[edge] = average_pitstop_time

# Define edge thicknesses based on average pit stop time
max_pitstop_time = max(average_pitstop_times.values())
min_pitstop_time = min(average_pitstop_times.values())
edge_thicknesses = [1 + 9 * (time - min_pitstop_time) / (max_pitstop_time - min_pitstop_time) for time in average_pitstop_times.values()]

# Assuming G has already been constructed with nodes and edges as previously described

# Define node colors by type: 'skyblue' for drivers, 'lightgreen' for teams
node_colors = ['skyblue' if G.nodes[node]['node_type'] == 'driver' else 'lightgreen' for node in G.nodes()]

# Create a layout for our nodes using the spring layout
pos = nx.spring_layout(G, k=0.9, iterations=50, seed=42)  # Adjusted for a more spread out layout



# Draw the graph
plt.figure(figsize=(40, 40))  # Increase figure size for better visibility
nx.draw_networkx_nodes(G, pos, node_color=node_colors, alpha=0.8,node_size=5000)
nx.draw_networkx_edges(G, pos, edge_color='gray', width=edge_thicknesses, alpha=0.5)

# Draw labels with increased font size and a white background for clarity
labels = nx.draw_networkx_labels(G, pos, font_size=14, font_weight='bold', bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))
# Add node labels as attributes to the graph
labels = {node: node for node in G.nodes()}
nx.set_node_attributes(G, labels, 'label')
# Add edge thickness attribute to the graph
nx.set_edge_attributes(G, {edge: {'edge_thickness': thickness} for edge, thickness in zip(G.edges(), edge_thicknesses)})
# To save your graph G to a GraphML file for further analysis or visualization in other tools
nx.write_graphml(G, "network_graph.graphml")


# In[18]:


# Merge driver data with results data based on driver ID
results_driver_merged = pd.merge(results_data, driver_data, how='left', left_on='driverId', right_on='driverId')
# Merge race data with merged driver-results data based on race ID
results_driver_merged = pd.merge(results_driver_merged, race_data, how='left', left_on='raceId', right_on='raceId')
# Merge race data with merged driver-results data based on race ID
results_driver_merged_constructor = pd.merge(results_driver_merged, constructor_data, how='left', left_on='constructorId', right_on='constructorId')
# Merge qualifying data with merged results-driver data based on driver ID and race ID
qualifying_results_merged = pd.merge(qualifying_data, results_driver_merged_constructor, how='inner', on=['driverId', 'raceId'])

# Filter the data to include only the last 13 years
qualifying_results_merged = qualifying_results_merged[(qualifying_results_merged['year'] >=2010) &(qualifying_results_merged['year']<=2023)]



import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
# Function to convert qualifying time to milliseconds
def time_to_milliseconds(time_str):
    parts = time_str.split(':')
    mins = int(parts[0])
    secs, millisecs = map(int, parts[1].split('.'))
    return mins * 60 * 1000 + secs * 1000 + millisecs


# Convert qualifying times to milliseconds
qualifying_columns = ['q1', 'q2', 'q3']
for col in qualifying_columns:
    qualifying_results_merged[col] = qualifying_results_merged[col].replace('\\N', np.nan)
    qualifying_results_merged[col] = qualifying_results_merged[col].fillna(method='ffill')
    qualifying_results_merged[col] = qualifying_results_merged[col].apply(time_to_milliseconds)

# Create empty directed graph
G = nx.DiGraph()

# Add driver nodes with attributes
for _, driver in qualifying_results_merged.iterrows():
    G.add_node(driver['driverRef'], node_type='driver', nationality=driver['nationality_x'])

# Add team nodes with attributes
for _, team in qualifying_results_merged.iterrows():
    G.add_node(team['constructorRef'], node_type='team', nationality=team['nationality_y'])

# Add edges for average qualifying time between drivers and teams
for _, row in qualifying_results_merged.iterrows():
    driver = row['driverRef']
    team = row['constructorRef']
    qualifying_times = [row[col] for col in qualifying_columns if not pd.isnull(row[col])]
    if qualifying_times:
        average_qualifying_time = sum(qualifying_times) /len(qualifying_times)
        G.add_edge(driver, team, avg_qualifying_time=average_qualifying_time)

# Get edge data and handle case where there are no qualifying times
edge_data = [data['avg_qualifying_time'] for _, _, data in G.edges(data=True) if 'avg_qualifying_time' in data]
if edge_data:
    # Define edge thicknesses based on average qualifying time
    max_qualifying_time = max(edge_data)
    min_qualifying_time = min(edge_data)
    edge_thicknesses = [1 + 9 * (data - min_qualifying_time) / (max_qualifying_time - min_qualifying_time) for data in edge_data]
    #(data - min_qualifying_time) / (max_qualifying_time - min_qualifying_time)
else:
    max_qualifying_time = max(edge_data)
    min_qualifying_time = min(edge_data)
    edge_thicknesses = [1 + 9 * (data - min_qualifying_time) / (max_qualifying_time - min_qualifying_time) for data in edge_data]

# Define node colors by type: 'skyblue' for drivers, 'lightgreen' for teams
node_colors = ['skyblue' if G.nodes[node]['node_type'] == 'driver' else 'lightgreen' for node in G.nodes()]

# Create a layout for our nodes using the spring layout
pos = nx.spring_layout(G, k=0.9, iterations=50, seed=42)  # Adjusted for a more spread out layout


# Draw the graph
plt.figure(figsize=(40, 40))  # Increase figure size for better visibility
nx.draw_networkx_nodes(G, pos, node_color=node_colors, alpha=0.8,node_size=5000)
nx.draw_networkx_edges(G, pos, edge_color='gray', width=edge_thicknesses, alpha=0.5)

# Draw labels with increased font size and a white background for clarity
labels = nx.draw_networkx_labels(G, pos, font_size=14, font_weight='bold', bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))

plt.title('Driver-Team Network with Edge Thickness Representing Average Qualifying Time')
plt.axis('off')  # Turn off the axis for better aesthetics
plt.show()
# Add node labels as attributes to the graph
labels = {node: node for node in G.nodes()}
nx.set_node_attributes(G, labels, 'label')
# Add edge thickness attribute to the graph
nx.set_edge_attributes(G, {edge: {'edge_thickness': thickness} for edge, thickness in zip(G.edges(), edge_thicknesses)})
# To save your graph G to a GraphML file for further analysis or visualization in other tools
nx.write_graphml(G, "network_graph.graphml")


# In[ ]:


# # 导入所需的模块
# from nbconvert import PythonExporter
# import nbformat

# # 将Jupyter笔记本转换为Python脚本的函数
# def convert_ipynb_to_py(ipynb_file):
#     # 读取.ipynb文件
#     with open(ipynb_file, 'r', encoding='utf-8') as f:
#         nb = nbformat.read(f, as_version=4)

#     # 创建PythonExporter实例
#     exporter = PythonExporter()

#     # 将.ipynb文件转换为.py格式
#     (python_code, _) = exporter.from_notebook_node(nb)

#     # 创建输出的.py文件
#     py_file = ipynb_file.replace('.ipynb', '.py')
#     with open(py_file, 'w', encoding='utf-8') as f:
#         f.write(python_code)

#     print(f"成功将.ipynb文件转换为.py文件: {py_file}")

# # 使用.ipynb文件转换为.py文件
# convert_ipynb_to_py('/Users/angelsheu/Desktop/UCI/Q3/Customer& Social/project/CSA Project4.ipynb')


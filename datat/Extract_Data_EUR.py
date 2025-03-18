path  = 'Data.xlsx'

import pandas as pd
import numpy as np

df = pd.read_excel(path)

print(df.head())
#%%
# These are the individuals and markers that we are interested in
ind = [2] + list(range(0, 109))
df_interesting = df.iloc[10:,ind]
new_column_names = df.iloc[4,ind]
new_column_names[0] = "Population"
df_interesting.columns = new_column_names
unique_populations = df_interesting['Population'].unique()
#%%
# These are the individuals that we are interested in
df_filtered_eur = df_interesting[df_interesting.iloc[:, 2] == "EUROPEAN"]
unique_populations = df_filtered_eur['Population'].unique()
print("Unique Populations:", unique_populations)
#%%
# Count the numver of individuals
population_counts = df_filtered_eur['Population'].value_counts()

populations_with_more_than_20 = population_counts[population_counts > 20].index

# Only keep the classes with more than 20 individuals 
df_filtered_eur_new = df_filtered_eur[df_filtered_eur['Population'].isin(populations_with_more_than_20)]
#%%
# Save the results
output_file = 'filtered_population_eur_update.xlsx'
df_filtered_eur_new.to_excel(output_file, index=False)

#%%
import matplotlib.pyplot as plt
# Data for the plots
data = {
    "TSI": 107,
    "IBS": 107,
    "CEU": 99,
    "FIN": 99,
    "GBR": 91,
    "SAR": 28,
    "FRA": 28,
    "TUR": 28,
    "RUS": 25,
    "BAS": 23
}
labels = list(data.keys())
sizes = list(data.values())
# Plot in the Paper
colors = plt.cm.YlOrRd(np.linspace(0.3, 1, len(labels)))  

plt.figure(figsize=(8, 8))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors,  textprops={'fontsize': 12} )

plt.show()
#%% Barplot

# Sort data in descending order
sorted_data = dict(sorted(data.items(), key=lambda item: item[1], reverse=True))

# Extract categories and values
categories = list(sorted_data.keys())
values = list(sorted_data.values())

# Plotting a vertical bar chart
plt.figure(figsize=(12, 6))
plt.bar(categories, values, color='skyblue')
plt.xticks(rotation=0, ha='center', fontsize=20)
plt.yticks(rotation=0, ha='center', fontsize=20)
plt.gca().yaxis.set_tick_params(pad=20)  # Moves y-tick labels further from the axis

plt.xlabel("Populations", fontsize=20)
plt.ylabel("Number of Individuals", fontsize=20)
plt.tight_layout()
plt.show()

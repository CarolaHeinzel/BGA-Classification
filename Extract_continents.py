pfad = 'C:\\Users\\carol\\OneDrive\\Desktop\\Promotion\\Kooperationen\\Leannart\\Data_Lennart.xlsx'


import pandas as pd

df = pd.read_excel(pfad)

print(df.head())
#%%
# These are the individuals and markers that we are interested in
ind = [2] + list(range(1, 109))
df_interesting = df.iloc[10:,ind]
new_column_names = df.iloc[4,ind]
new_column_names[0] = "Population"
df_interesting.columns = new_column_names

unique_populations = df_interesting.iloc[:,1].unique()
print("Einzigartige Populationen:", unique_populations)
#%%
desired_categories = [
    'AFRICAN', 'EUROPEAN', 'EAST ASIAN', 'SOUTH ASIAN', 
    'MIDDLE EAST', 'OCEANIAN', 'AMERICAN'
]

# Filtern des DataFrames, um nur die gewünschten Kategorien zu behalten
filtered_df = df_interesting[df_interesting.iloc[:,1].isin(desired_categories)]

population_counts =filtered_df.iloc[:,1].value_counts()

import matplotlib.pyplot as plt
# Data for the bar chart
data = {
   "EAS":     935,
"EUR" :      827,
"SAS" :   730,
"AFR"     :   623,
"MEA" :   240,
"AMR"    :   109,
"OCE"    :    77
}

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



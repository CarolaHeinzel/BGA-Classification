path  = 'C:\\Users\\carol\\OneDrive\\Desktop\\Promotion\\Kooperationen\\Leannart\\Data_Lennart.xlsx'


import pandas as pd

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

unique_populations = df_filtered_eur_new['Population'].unique()
#%%
#  Substitue France and Italy
pd.options.mode.chained_assignment = None  
df_filtered_eur_new.loc[:,'Population']= df_filtered_eur_new['Population'].replace({
    '10. France - French Basque': 'France',
    '11. France - French': 'France',
    'Toscani in Italia': 'Italy',
    '13. Italy - Sardinian': 'Italy'
})

population_counts = df_filtered_eur_new['Population'].value_counts()

#%%
# Save the results
output_file = 'filtered_population_eur.xlsx'
df_filtered_eur_new.to_excel(output_file, index=False)
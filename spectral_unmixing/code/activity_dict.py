import os
import pandas as pd
import json


calendar_dir = os.path.expanduser('~/mnt/Data-Labo-RE/27_Natural_Resources-RE/321.4_WAUM_protected/Daten/Feldkalender_AUI')

all_DE = set()
for yr in range(2016,2023):
  calendar_file = os.path.expanduser(os.path.join(calendar_dir, 'Feldkalender_2021.txt'))
  df = pd.read_csv(calendar_file, encoding='latin1', delimiter='\t')
  activity_DE = df['massnahme'].dropna().unique()
  all_DE.update(activity_DE)

all_DE = sorted(all_DE) # this will make it reproducible, since a set is unirdered and changes every time code is rerun

# Write to text file line by line
with open('activities_DE.txt', 'w') as f:
    for item in list(all_DE):
        f.write(f"{item}\n")

# Translated activities
with open('activities_EN.txt', 'r', encoding='latin1') as f:
    values_list = [line.strip() for line in f if line.strip()]

# Create dict
activity_dict = {}

for i in range(len(values_list)):
    activity_dict.update({all_DE[i]: values_list[i]})

# Save for future use
with open('activities_dict.json', 'w', encoding='utf-8') as f:
    json.dump(activity_dict, f, ensure_ascii=False, indent=4)


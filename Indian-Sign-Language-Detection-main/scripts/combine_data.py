import pandas as pd
import os

data_dir = 'C:/Users/abc/Desktop/ISL/Sign-Language-Detection/data/alphabet'  # change if your folder is named differently
combined_data = []

for file in os.listdir(data_dir):
    if file.endswith('.csv'):
        df = pd.read_csv(os.path.join(data_dir, file))
        combined_data.append(df)

final_df = pd.concat(combined_data, ignore_index=True)
final_df.to_csv('combined_data.csv', index=False)

print("Combined all gesture CSVs into one file: combined_data.csv")

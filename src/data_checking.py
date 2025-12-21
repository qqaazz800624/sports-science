#%%
import pandas as pd
import os

base_dir = r"C:\Users\User\Desktop\Baseball\factor-and-defense-factor"
file_path = os.path.join(base_dir, "statcast_2023.csv")

data = pd.read_csv(file_path)
print(len(data))
#%%

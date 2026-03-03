#%%



import pandas as pd
import os
import numpy as np

data_dir = '/neodata/open_dataset/mlb_data/'
filename = 'preprocessed/truncated_data_with_rtheta_team.parquet'
data_pth = os.path.join(data_dir, filename)
df = pd.read_parquet(data_pth)


def get_direction(angle):
    if angle < 0:
        return 'left'
    else:
        return 'right'


filtered_df = df[(df['description'] == 'hit_into_play') 
                 & (df['bb_type'].isin(['fly_ball', 'line_drive']))
                 ]
team_mapping = {'ATH': 'OAK'}
filtered_df['home_team'] = filtered_df['home_team'].replace(team_mapping)
filtered_df['spray_angle'] = np.arctan((filtered_df['hc_x'] - 125.42) / (198.27 - filtered_df['hc_y'])) * 180 / np.pi
filtered_df['direction'] = filtered_df['spray_angle'].apply(get_direction)

conditions = [
    (filtered_df['direction'] == 'left') & (filtered_df['hit_location'].isin([7, 8])) & (filtered_df['events'].isin(['single', 'double', 'triple', 'home_run', 'sac_fly'])),
    (filtered_df['direction'] == 'right') & (filtered_df['hit_location'].isin([8, 9])) & (filtered_df['events'].isin(['single', 'double', 'triple', 'home_run', 'sac_fly']))
]

choices = [
    'left_fly_ball',
    'right_fly_ball'
]

filtered_df['FLG_flyball'] = np.select(conditions, choices, default='others')

filtered_df['is_left_fly'] = (filtered_df['FLG_flyball'] == 'left_fly_ball')
filtered_df['is_right_fly'] = (filtered_df['FLG_flyball'] == 'right_fly_ball')


#%%

summary_df = filtered_df.groupby(['game_year', 'home_team']).agg(
    rate_left_flyball=('is_left_fly', 'mean'),   
    rate_right_flyball=('is_right_fly', 'mean'), 
    total_count=('events', 'size')               
).reset_index()


summary_df.to_csv('flyball_direction_summary_by_year_and_team.csv', index=False)

#%%

group_by_columns = ['game_year', 'home_team', 'direction', 'hit_location', 'events']
agg_df = filtered_df.groupby(group_by_columns).size().reset_index(name='count')

#%%


agg_df.to_csv('aggregate_hit_direction_by_park.csv', index=False)



#%%






#%%








#%%







#%%
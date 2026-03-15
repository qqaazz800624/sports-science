#%%

import pandas as pd
import numpy as np
import os
from utils import (get_expected_bases_map, 
                   Config, 
                   prepare_regression_data,
                   run_year_regression)
import argparse
from tqdm import tqdm

current_weights = {
    'single': 1,
    'double': 2,
    'triple': 3,
    'home_run': 4
}

data_dir = "/neodata/open_dataset/mlb_data/preprocessed"
prob_table = "rtheta_prob_tbl.parquet"
input_filename = "truncated_data_with_rtheta_team.parquet"
truncated_file_path = os.path.join(data_dir, input_filename)

config = Config(
    weights=current_weights,
    data_dir=data_dir,
    filename=prob_table
)

exp_map = get_expected_bases_map(config=config)
df = pd.read_parquet(truncated_file_path)
reg_df = prepare_regression_data(df, exp_map, config)

#%%

target_df = reg_df[(reg_df['game_year'] == 2024) & 
                   (reg_df['home_team'] != 'AZ')]


year = 2024
result = run_year_regression(target_df, year)

result['defense_factors']



#%%
import numpy as np
import statsmodels.formula.api as smf

year_to_check = 2024
data_2024 = reg_df[reg_df['game_year'] == year_to_check].copy()

model = smf.wls("avg_residual ~ C(park) + C(defense)", data=data_2024, weights=data_2024['weight'])
res = model.fit()


# 1. 抓出底層的 X (Design Matrix), y, w
X_matrix = model.exog        # 老師說的 x_i 組成的矩陣
y_vector = model.endog       # 老師說的 y_i
w_vector = model.weights     # 老師說的 w_i
X_names = model.exog_names   # 這 59 個變數的名字

print("=== 資料結構驗證 ===")
print(f"X 矩陣形狀: {X_matrix.shape}")  # 預期會印出 (N, 59)
print(f"y 向量形狀: {y_vector.shape}")
print(f"w 向量形狀: {w_vector.shape}")

print("\n=== 隨機印出前 n 筆資料的 x_i (1x59 向量) ===")
for i in range(60):
    print(f"--- 第 {i+1} 筆資料 ---")
    x_i = X_matrix[i]
    for name, val in zip(X_names, x_i):
        if val != 0: # 只印出有值 (1.0) 的地方
            print(f"{name}: {val}")


#%%

# 1. 篩選出 AZ 擔任防守方的所有資料
az_def_data = data_2024[data_2024['defense'] == 'AZ'].copy()

# 把模型預測的 y_hat (即 beta * x_i) 抓出來
az_def_data['predicted'] = res.fittedvalues[az_def_data.index]

# 計算模型殘差: (y_i - beta * x_i)
az_def_data['model_resid'] = az_def_data['avg_residual'] - az_def_data['predicted']
w = az_def_data['weight']

# ==========================================
# 測試一：計算 \sum (y_i - beta*x_i) * w_i
# ==========================================
sum_weighted_resid = np.sum(az_def_data['model_resid'] * w)
print("=== 驗算一 ===")
print(f"AZ 防守時的加權殘差和: {sum_weighted_resid:.10f}")

#%%

# ==========================================
# 測試二：微調 Beta，檢查 Objective Function 是否下降
# ==========================================
current_obj = np.sum((az_def_data['model_resid'] ** 2) * w)
perturbation = 0.001
az_def_data['new_predicted'] = az_def_data['predicted'] + perturbation
az_def_data['new_model_resid'] = az_def_data['avg_residual'] - az_def_data['new_predicted']
new_obj = np.sum((az_def_data['new_model_resid'] ** 2) * w)

print("=== 驗算二 ===")
print(f"原本的加權誤差平方和: {current_obj:.8f}")
print(f"Beta 調高 {perturbation} 後的平方和: {new_obj:.8f}")
print(f"變化量: {new_obj - current_obj:.8f}")






#%%









#%%
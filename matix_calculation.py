#%%
import pandas as pd
import os

from IPython.display import display as dp
import pandas as pd
import numpy as np
import pickle

from expect_score import get_truncated_dataset_with_team, get_rtheta_prob_tbl
from team_park_metrics import get_team_score
from league_score_tbl import get_league_tbl
from generate_matrix import collect_eqns

from scipy.linalg import lstsq
from sympy import symbols, Eq, linear_eq_to_matrix
import re

df = get_truncated_dataset_with_team().copy()
dist_df = get_rtheta_prob_tbl()

bat_df = get_team_score("bat")
pitch_df = get_team_score("pitch")
park_df = get_team_score("park")
league_summary_tbl = get_league_tbl()


batter_tm_col = df.pop('batter_team')
pitcher_tm_col = df.pop('pitcher_team')

new_batter_tm_col = df.columns.get_loc('batter') + 1 #type: ignore
new_pitcher_tm_col = df.columns.get_loc('pitcher') + 1 #type: ignore

df.insert(new_batter_tm_col, 'batter_team', batter_tm_col) #type: ignore
df.insert(new_pitcher_tm_col, 'pitcher_team', pitcher_tm_col) #type: ignore


#儲存所有年度的矩陣資料
all_years_matrix = {}

for year in range(2015, 2025):
    if year == 2020:
        continue
    print(f"正在建立 {year} 年方程式矩陣...")

    # 收集三組方程式
    year_eqs = collect_eqns(
        data=df,
        park_data=park_df,
        pitch_data=pitch_df,
        batter_data=bat_df,
        league_tbl=league_summary_tbl,
        metric='SLG',
        yr=year
    )

    # 合併成單一 dict（park + home + away）
    merged_eqs = {}
    for group_name, eq_dict in year_eqs.items():
        for tm, eq in eq_dict.items():
            merged_eqs[f"{group_name}_{tm}"] = eq

    # ---- 把所有方程式轉成符號形式 ----
    equations = []
    symbols_set = set()

    for tm, eq_str in merged_eqs.items():
        # 移除空白再拆成左右兩邊
        lhs, rhs = eq_str.split("=")
        lhs = lhs.strip()
        rhs = rhs.strip()

        # 提取所有變數名稱（允許底線與數字）
        var_names = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", rhs)

        # 建立符號字典
        symbol_dict = {name: symbols(name) for name in var_names}

        # 把rhs的字串轉成 sympy expression
        expr = eval(rhs, symbol_dict)

        # lhs是常數 (y_value)
        y_val = float(lhs)
        equations.append(Eq(expr, y_val))

        # 收集符號
        symbols_set.update(expr.free_symbols)

    # 建立矩陣形式
    A, b = linear_eq_to_matrix(equations, list(symbols_set))
    all_years_matrix[year] = {"A": A, "b": b, "symbols": list(symbols_set)}

    print(f"完成 {year} 年矩陣生成，共 {len(equations)} 條方程式，變數數量：{len(symbols_set)}")
    
#  ----- 儲存矩陣 -----
matrix_save_path = "/Users/yantianli/factor-and-defense-factor/all_years_matrix.pkl"
with open(matrix_save_path, "wb") as f:
    pickle.dump(all_years_matrix, f)
print(f"將所有年度矩陣儲存至：{matrix_save_path}")

# 讀取矩陣
with open(matrix_save_path, "rb") as f:
    all_years_matrix = pickle.load(f)
print("成功載入矩陣資料！")



# ----- 併所有年度矩陣 -----
# 建立變數空間
symbol_names = sorted(set(
    str(s) for y in all_years_matrix for s in all_years_matrix[y]["symbols"]
))

symbol_index = {name: i for i, name in enumerate(symbol_names)}
num_vars = len(symbol_names)

A_list = []
b_list = []

# 將每年的 A 矩陣嵌入完整空間
for y in sorted(all_years_matrix):
    A_y = np.array(all_years_matrix[y]["A"]).astype(float)
    b_y = np.array(all_years_matrix[y]["b"]).astype(float)
    symbols_y = [str(s) for s in all_years_matrix[y]["symbols"]]
    
    A_full = np.zeros((A_y.shape[0], num_vars))
    for i, sym in enumerate(symbols_y):
        j = symbol_index[sym]
        A_full[:, j] = A_y[:, i]
    
    A_list.append(A_full)
    b_list.append(b_y)

# 合併所有年度矩陣
A_all = np.vstack(A_list)
b_all = np.vstack(b_list)
print(f"A_all shape = {A_all.shape}, b_all shape = {b_all.shape}")


# 用 least square 計算
x, res, rank, s = lstsq(A_all, b_all, lapack_driver='gelsy')
#print(f"rank:{rank}")


# --- 組合結果 ---
solution_df = pd.DataFrame({
    "variable": symbol_names,
    "value": x.flatten()[:len(symbol_names)]  # 防止多餘元素
})

solution_df = solution_df.sort_values(by="variable").reset_index(drop=True)
solution_df.to_csv("/Users/yantianli/factor-and-defense-factor/solution.csv")
#%%
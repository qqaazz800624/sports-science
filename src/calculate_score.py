#%%
import pandas as pd
import numpy as np
from pathlib import Path
from pybaseball import chadwick_register

# 假設這些是你自己寫的模組，且放在同一個資料夾下
from expect_score import get_whole_dataset, get_truncated_dataset, get_rtheta_prob_tbl

# --- 1. 設定路徑管理 ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
DATA_REF = BASE_DIR / "data" / "reference"

# --- 2. 輔助函數：讀取 Fangraphs 資料 ---
def load_fangraphs_data():
    """
    讀取 Fangraphs 的投打資料。
    """
    bat_path = DATA_RAW / "fg_batting.csv"
    pitch_path = DATA_RAW / "fg_pitching.csv"

    if not bat_path.exists() or not pitch_path.exists():
        print(f"找不到 Fangraphs 資料，請確認 {DATA_RAW} 內是否有檔案。")
        return pd.DataFrame(), pd.DataFrame()

    # print("正在載入 Fangraphs 資料...")
    batter_df = pd.read_csv(bat_path)
    pitcher_df = pd.read_csv(pitch_path)
    return batter_df, pitcher_df

# --- 3. 輔助函數：取得球員 ID 對照表 (Local Cache) ---
def get_player_id_map():
    """
    取得球員 ID 對照表 (包含 MLBAM ID 和 Fangraphs ID)。
    如果本地沒有檔案，會嘗試下載並存檔。
    """
    map_path = DATA_RAW / "player_id_map.csv"
    
    if map_path.exists():
        # 如果檔案存在，直接讀取
        return pd.read_csv(map_path)
    
    print("⚠️ 未發現 ID 對照表，正在下載完整球員名單 (chadwick_register)...")
    try:
        # 下載完整的球員 ID 表 (只需要做一次)
        df_map = chadwick_register()
        
        # 建立目錄並存檔
        DATA_REF.mkdir(parents=True, exist_ok=True)
        df_map.to_csv(map_path, index=False)
        print(f"✅ ID 對照表已儲存至：{map_path}")
        return df_map
    except Exception as e:
        print(f"❌ 下載失敗：{e}")
        return pd.DataFrame()


def hip_score_tbl(data: pd.DataFrame,
                  dist_df: pd.DataFrame, 
                  year: int, 
                  player_mlbid, 
                  player_type: str,
                  method: str = 'expectation',
                  n_simulations: int = 1000,
                  random_seed: int = 42):
    """
    計算球員打進場內 (HIP) 的 Expected Score。
    """
    # 篩選資料
    mask = (
        (data['game_year'] == year) & 
        (data['game_type'] == 'R') & 
        (data['description'] == 'hit_into_play')
    )
    df = data[mask].copy()
    df = df.dropna(subset=['events'])

    # 篩選球員
    if player_mlbid is not None:
        df = df[df[player_type] == player_mlbid]
    
    if df.empty:
        print(f"找不到球員 {player_mlbid} 在 {year} 年的 HIP 資料")
        return pd.DataFrame(columns=['events', 'sum_real_count', 'sum_expected_count'])

    # 1. 計算真實事件次數
    real_df = df['events'].value_counts().reset_index()
    real_df.columns = ['events', 'sum_real_count']
    
    # 2. 準備計算 Expected Score
    counts_dict = df['r_theta'].value_counts().to_dict()
    new_df = dist_df.copy()
    new_df["total_count"] = new_df["r_theta"].map(lambda x: counts_dict.get(x, 0))


    if method == 'expectation':
        # --- 方法 A: 期望值法 ---
        new_df["expected_count"] = new_df["probability"] * new_df["total_count"]

        # Normalize (讓總期望次數 = 總真實打席數)
        real_event_count = real_df['sum_real_count'].sum()
        expected_total = new_df['expected_count'].sum()
        
        if expected_total > 0:
            new_df['expected_count'] *= (real_event_count / expected_total)

        new_df['expected_count'] = new_df['expected_count'].round(4)

    elif method == 'sampling':
        # --- 方法 B: 蒙地卡羅模擬法 ---
        np.random.seed(random_seed)
        simulation_results = []
        
        # 優化：預先分組以加快查找速度
        dist_dict = {
            rtheta: (sub['events'].values, sub['probability'].values / sub['probability'].sum())
            for rtheta, sub in new_df.groupby('r_theta')
        }

        # 這裡的模擬邏輯比較重，建議只在必要時使用
        for sim in range(n_simulations):
            simulated_events = []
            for rtheta in df['r_theta']: # 直接 iterate column 比較快
                if rtheta in dist_dict:
                    events, probs = dist_dict[rtheta]
                    simulated_events.append(np.random.choice(events, p=probs))
                else:
                    simulated_events.append(np.nan)

            sim_counts = pd.Series(simulated_events).value_counts().rename(f'sim_{sim+1}')
            simulation_results.append(sim_counts)

        sim_df = pd.concat(simulation_results, axis=1).fillna(0)
        sim_df['expected_count'] = sim_df.mean(axis=1)
        new_df = sim_df.reset_index().rename(columns={'index': 'events'})

    else:
        raise ValueError("method 必須是 'expectation' 或 'sampling'")

    # 3. 合併並匯總結果
    expected_df = (
        new_df.groupby('events')['expected_count']
        .sum()
        .reset_index()
        .rename(columns={'expected_count': 'sum_expected_count'})
    )

    combined = pd.merge(expected_df, real_df, on='events', how='outer').fillna(0)
    combined = combined.sort_values('sum_real_count', ascending=False).reset_index(drop=True)

    return combined


def nonhip_score_tbl(data: pd.DataFrame, 
                     year: int, 
                     player_mlbid, 
                     player_type: str):
    """
    計算非打進場內 (Non-HIP) 的真實事件 (如 K, BB)。
    對於這些事件，Expected Count = Real Count。
    """
    mask = (
        (data['game_year'] == year) & 
        (data['game_type'] == 'R') & 
        (data['description'] != 'hit_into_play')
    )
    df = data[mask].copy()
    df = df.dropna(subset=['events'])

    if player_mlbid is not None:
        df = df[df[player_type] == player_mlbid]

    real_df = df['events'].value_counts().reset_index()
    real_df.columns = ['events', 'sum_real_count']
    
    # 假設：K 和 BB 的預期值就是真實值 (如果不考慮好球帶模型)
    real_df['sum_expected_count'] = real_df['sum_real_count'] 

    real_df = real_df.sort_values('sum_real_count', ascending=False).reset_index(drop=True)
    return real_df

def ibb_score_tbl(year: int, 
                  player_mlbid: int, 
                  player_type: str,
                  pitcher_data: pd.DataFrame,
                  batter_data: pd.DataFrame) -> int:
    """
    取得故意四壞球 (IBB) 數量。
    會先嘗試從本地的 ID 對照表查找 Fangraphs ID。
    """
    if player_type == 'pitcher':
        df = pitcher_data
    elif player_type == 'batter':
        df = batter_data
    else:
        raise ValueError("player_type 必須是 'pitcher' 或 'batter'")
    
    if df.empty:
        return 0

    player_mlbid = int(player_mlbid)

    # 1. 取得 ID Mapping (MLBAM -> Fangraphs)
    id_map = get_player_id_map()
    
    if id_map.empty or 'key_mlbam' not in id_map.columns or 'key_fangraphs' not in id_map.columns:
        print("⚠️ ID 對照表無法使用，無法計算 IBB (回傳 0)")
        return 0

    # 2. 查找對應的 Fangraphs ID
    # 注意：Fangraphs ID 在表中有時候是 float 或 object，建議轉型處理
    try:
        mapping_row = id_map[id_map['key_mlbam'] == player_mlbid]
        if mapping_row.empty:
            print(f"⚠️ 在對照表中找不到 MLB ID: {player_mlbid}，IBB 設為 0")
            return 0
        
        fg_id = mapping_row['key_fangraphs'].values[0]
    except Exception as e:
        print(f"⚠️ ID 查找過程發生錯誤: {e}")
        return 0

    # 3. 篩選 Fangraphs 資料
    # IDfg 在某些 csv 可能會讀成字串或數字，這邊統一檢查
    # 假設 csv 裡的 IDfg 和 map 裡的 key_fangraphs 格式一致 (如果不一致可能要都轉 int)
    
    # 這裡做一個簡單的型別轉換嘗試，確保能對上
    try:
        # 確保比較雙方都是 int
        player_data = df[
            (df['game_year'] == year) &
            (df['IDfg'].astype(str).str.split('.').str[0] == str(fg_id).split('.')[0])
        ]
    except Exception:
        # Fallback 到原始比較
        player_data = df[(df['game_year'] == year) & (df['IDfg'] == fg_id)]

    # 4. 回傳 IBB 總數
    if not player_data.empty and 'Intent_walk' in player_data.columns:
        # 注意: 你的 csv 標頭看起來是 'Intent_walk' (根據之前的 command output)
        return int(player_data['Intent_walk'].sum())
    elif not player_data.empty and 'IBB' in player_data.columns:
        # 相容舊版或不同格式
        return int(player_data['IBB'].sum())
    else:
        return 0
    

def combined_score_tbl(data: pd.DataFrame,
                    dist_df: pd.DataFrame,
                    year: int,
                    player_mlbid: int,
                    player_type: str,
                    method: str = 'expectation'):
    """
    結合：
    - hip_score_tbl（打進場預期與實際）
    - nonhip_score_tbl（非打進場事件）
    - ibb_value_tbl（Fangraphs 的 IBB 資料）

    回傳：包含所有事件的完整統計表。
    """
    # 載入 FG 資料 (只在需要時載入)
    batter_data_fg, pitcher_data_fg = load_fangraphs_data()

    # calculate hip events
    hip_df = hip_score_tbl(
        data=data,
        dist_df=dist_df,
        year=year,
        player_mlbid=player_mlbid,
        player_type=player_type,
        method=method
    )

    # calculte nonhip events
    nonhip_df = nonhip_score_tbl(
        data=data,
        year=year,
        player_mlbid=player_mlbid,
        player_type=player_type
    )
    
    # add up IBB
    ibb_value = ibb_score_tbl(year, 
                            player_mlbid, 
                            player_type,
                            pitcher_data = pitcher_data_fg,
                            batter_data = batter_data_fg)
    ibb_value_df = pd.DataFrame(
        [
            {
                'events': 'intent_walk',
                'sum_real_count': ibb_value,
                'sum_expected_count': ibb_value
            }
        ]
    )
    # combin whole score
    
    combined_df = pd.concat([hip_df, nonhip_df, ibb_value_df], ignore_index=True)
    combined_df = combined_df.groupby("events", as_index=False)\
        [["sum_real_count", "sum_expected_count"]].sum()
    combined_df = combined_df.sort_values('events', ascending=True).reset_index(drop=True)
    return combined_df

if __name__ == "__main__":
    try:
        print("開始測試 combined_score_tbl ...")
        cole_score = combined_score_tbl(
            data=get_truncated_dataset(),
            dist_df=get_rtheta_prob_tbl(),
            year=2022,
            player_mlbid=592450,  # Judge
            player_type='batter',
            method='expectation'
        )
        print("\n結果預覽：")
        from IPython.display import display
        display(cole_score)
    except Exception as e:
        print(f"\n❌ 程式執行發生錯誤: {e}")
#%%
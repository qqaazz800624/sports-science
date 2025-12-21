#%%
import os
import requests


mlb_colors = {
# 美國聯盟 (American League)
    'BAL': '#DF4601',  # Baltimore Orioles (Orange)
    'BOS': '#BD3039',  # Boston Red Sox (Red)
    'CWS': '#27251F',  # Chicago White Sox (Black)
    'CLE': '#00385D',  # Cleveland Guardians (Navy)
    'DET': '#0C2340',  # Detroit Tigers (Navy)
    'HOU': '#002D62',  # Houston Astros (Navy)
    'KC':  '#004687',  # Kansas City Royals (Royal Blue)
    'LAA': '#BA0021',  # Los Angeles Angels (Red)
    'MIN': '#002B5C',  # Minnesota Twins (Navy)
    'NYY': '#003087',  # New York Yankees (Navy)
    'OAK': '#003831',  # Oakland Athletics (Forest Green)
    'SEA': '#005C5C',  # Seattle Mariners (Northwest Green/Teal)
    'TB':  '#092C5C',  # Tampa Bay Rays (Navy)
    'TEX': '#003278',  # Texas Rangers (Blue)
    'TOR': '#134A8E',  # Toronto Blue Jays (Blue)

    # 國家聯盟 (National League)
    'AZ': '#A71930',  # Arizona Diamondbacks (Sedona Red)
    'ATL': '#CE1141',  # Atlanta Braves (Scarlet)
    'CHC': '#0E3386',  # Chicago Cubs (Royal Blue)
    'CIN': '#C6011F',  # Cincinnati Reds (Red)
    'COL': '#333366',  # Colorado Rockies (Purple)
    'LAD': '#005A9C',  # Los Angeles Dodgers (Dodger Blue)
    'MIA': '#00A3E0',  # Miami Marlins (Miami Blue)
    'MIL': '#12284B',  # Milwaukee Brewers (Navy)
    'NYM': '#FF5910',  # New York Mets (Blue)
    'PHI': '#BA0C2F',  # Philadelphia Phillies (Red)
    'PIT': '#FDB827',  # Pittsburgh Pirates (Gold) *註：底色通常是黑，但黃色為主視覺亮點
    'SD':  '#FFC425',  # San Diego Padres (Brown)
    'SF':  '#FD5A1E',  # San Francisco Giants (Orange)
    'STL': '#C41E3A',  # St. Louis Cardinals (Red)
    'WSH': '#AB0003',  # Washington Nationals (Red)
}

mlb_logos = {
    'BAL': 'https://raw.githubusercontent.com/jjAnder90/baseball/main/1.png',
    'TOR': 'https://raw.githubusercontent.com/jjAnder90/baseball/main/2.png',
    'BOS': 'https://raw.githubusercontent.com/jjAnder90/baseball/main/3.png',
    'TB':  'https://raw.githubusercontent.com/jjAnder90/baseball/main/4.png',
    'NYY': 'https://raw.githubusercontent.com/jjAnder90/baseball/main/5.png',
    'CWS': 'https://raw.githubusercontent.com/jjAnder90/baseball/main/6.png',
    'CLE': 'https://raw.githubusercontent.com/jjAnder90/baseball/main/7.png',
    'KC':  'https://raw.githubusercontent.com/jjAnder90/baseball/main/8.png',
    'DET': 'https://raw.githubusercontent.com/jjAnder90/baseball/main/9.png',
    'MIN': 'https://raw.githubusercontent.com/jjAnder90/baseball/main/10.png',
    'HOU': 'https://raw.githubusercontent.com/jjAnder90/baseball/main/11.png',
    'OAK': 'https://raw.githubusercontent.com/jjAnder90/baseball/main/12.png',
    'SEA': 'https://raw.githubusercontent.com/jjAnder90/baseball/main/13.png',
    'LAA': 'https://raw.githubusercontent.com/jjAnder90/baseball/main/14.png',
    'TEX': 'https://raw.githubusercontent.com/jjAnder90/baseball/main/15.png',
    'NYM': 'https://raw.githubusercontent.com/jjAnder90/baseball/main/16.png',
    'WSH': 'https://raw.githubusercontent.com/jjAnder90/baseball/main/17.png',
    'ATL': 'https://raw.githubusercontent.com/jjAnder90/baseball/main/18.png',
    'PHI': 'https://raw.githubusercontent.com/jjAnder90/baseball/main/19.png',
    'MIA': 'https://raw.githubusercontent.com/jjAnder90/baseball/main/20.png',
    'MIL': 'https://raw.githubusercontent.com/jjAnder90/baseball/main/21.png',
    'CHC': 'https://raw.githubusercontent.com/jjAnder90/baseball/main/22.png',
    'CIN': 'https://raw.githubusercontent.com/jjAnder90/baseball/main/23.png',
    'STL': 'https://raw.githubusercontent.com/jjAnder90/baseball/main/24.png',
    'PIT': 'https://raw.githubusercontent.com/jjAnder90/baseball/main/25.png',
    'SF':  'https://raw.githubusercontent.com/jjAnder90/baseball/main/26.png',
    'COL': 'https://raw.githubusercontent.com/jjAnder90/baseball/main/27.png',
    'SD':  'https://raw.githubusercontent.com/jjAnder90/baseball/main/28.png',
    'LAD': 'https://raw.githubusercontent.com/jjAnder90/baseball/main/29.png',
    'AZ':  'https://raw.githubusercontent.com/jjAnder90/baseball/main/30.png', # Maps to ARI
}



# 1. 建立一個叫做 logos 的資料夾
folder_name = "logos"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)
    print(f"建立資料夾: {folder_name}")

print("開始下載隊徽...")

# 2. 遍歷字典下載圖片
for team, url in mlb_logos.items():
    try:
        # 取得圖片內容
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            # 存檔路徑：logos/NYY.png
            file_path = os.path.join(folder_name, f"{team}.png")
            
            with open(file_path, "wb") as f:
                f.write(response.content)
            print(f"✅ 下載成功: {team}")
        else:
            print(f"❌ 下載失敗 (狀態碼錯誤): {team}")
            
    except Exception as e:
        print(f"⚠️ 下載錯誤: {team} - {e}")

print("所有圖片處理完成！")
#%%
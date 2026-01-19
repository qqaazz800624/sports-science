#%%
import requests
import pandas as pd
import json
import re
import time
from tqdm import tqdm

def get_savant_park_factors(start_year, end_year):
    """
    Scrapes Park Factors from Baseball Savant by extracting the JSON payload 
    embedded in the page's JavaScript.
    
    Args:
        start_year (int): The starting year (e.g., 2015).
        end_year (int): The ending year (e.g., 2024).
        
    Returns:
        pd.DataFrame: A combined DataFrame containing Statcast park factors.
    """
    
    all_dfs = []
    
    # Headers to mimic a real browser visit
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    print(f"Starting Baseball Savant scrape from {start_year} to {end_year}...")
    
    for year in tqdm(range(start_year, end_year + 1), desc="Years"):
        try:
            # The URL for the specific year view
            url = f"https://baseballsavant.mlb.com/leaderboard/statcast-park-factors?type=year&year={year}&rolling=1"
            
            print(f"Fetching data for {year}...", end=" ")
            
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            # --- The Magic Step ---
            # Baseball Savant stores the data in a JavaScript variable named 'data'.
            # We use Regex to find "var data = [...];" or "data = [...];" and extract the [...] part.
            
            # This regex looks for: data = [ ... ];
            match = re.search(r'data\s*=\s*(\[.*?\]);', response.text, re.DOTALL)
            
            if match:
                json_str = match.group(1)
                data_list = json.loads(json_str)
                
                if data_list:
                    df = pd.DataFrame(data_list)
                    
                    # Add Year column for reference
                    df['year'] = year
                    
                    # Optional: Clean up column names (Savant uses snake_case usually)
                    # Keep valid columns, drop internal ID columns if needed
                    
                    all_dfs.append(df)
                    print(f"Success. ({len(df)} rows)")
                else:
                    print("Found variable, but data was empty.")
            else:
                print("Could not find data variable in page source.")

            # Sleep to be polite to the server
            time.sleep(2)
            
        except Exception as e:
            print(f"Error fetching {year}: {e}")
            continue

    # Combine all years
    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
        
        # Reorder to put year and team first if they exist
        cols = list(final_df.columns)
        priorities = ['year', 'venue_name', 'team_id']
        for p in reversed(priorities):
            if p in cols:
                cols.insert(0, cols.pop(cols.index(p)))
        
        final_df = final_df[cols]
        
        return final_df
    else:
        return pd.DataFrame()

# --- Main Execution ---

if __name__ == "__main__":
    start = 2015
    end = 2024
    
    savant_data = get_savant_park_factors(start, end)
    
    if not savant_data.empty:
        print("\n--- Scrape Complete ---")
        print(f"Total rows: {len(savant_data)}")
        print(savant_data.head())
        
        # Save to CSV
        savant_data.to_csv("savant_park_factors.csv", index=False)
        print("Saved to savant_park_factors.csv")
    else:
        print("No data found.")

#%%





#%%







#%%




#%%





#%%





#%%
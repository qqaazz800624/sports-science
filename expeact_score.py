#%%
import pandas as pd
import os

from IPython.display import display
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

from create_frame import savant_data_14_24

pd.set_option("display.max_rows", None)      # 列不要省略
pd.set_option("display.max_columns", None)   # 欄不要省略
pd.set_option("display.width", None)         # 不限制總寬度
pd.set_option("display.max_colwidth", None)  # 每欄完整顯示


savant_data_14_24 = pd.read_parquet("/Users/yantianli/factor-and-defense-factor/savant_data_14_24_with_rtheta.parquet")
print(savant_data_14_24.sample(5))
#%%
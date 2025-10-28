# __init__.py
"""
factor_and_defense_factor package

提供：
- calculate_score：用於打擊 / 投球的實際與預期結果計算
- expect_score：建立 expected_event 及 event_distribution
"""

from .calculate_score import hip_score_tbl, nonhip_score_tbl, ibb_score_tbl, combined_score_tbl
from .expect_score import get_expected_dataset, get_event_distribution, get_whole_dataset
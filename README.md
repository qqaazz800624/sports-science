# Factor and Defense Factor Analysis

## Project Overview
This repository is dedicated to the quantitative analysis of financial factors, specifically focusing on the construction, evaluation, and optimization of "Defense Factors" (Low Volatility, Quality, etc.) within a portfolio management context. The project provides a systematic framework for backtesting investment strategies, calculating factor exposures, and analyzing risk-adjusted returns.

## Key Features
- **Factor Construction**: Tools for generating various quantitative factors (Value, Momentum, Quality, and Low Volatility).
- **Defense Factor Analysis**: Specialized modules to evaluate factors that provide downside protection during market turbulence.
- **Backtesting Engine**: A robust framework to simulate historical performance with customizable rebalancing rules.
- **Performance Metrics**: Automated calculation of Sharpe Ratio, Maximum Drawdown, Information Ratio, and Alpha/Beta decomposition.
- **Visualization**: Integrated plotting utilities for cumulative returns, drawdown curves, and factor correlations.

## Tech Stack
- **Language**: Python
- **Data Analysis**: `pandas`, `numpy`
- **Scientific Computing**: `scipy`, `statsmodels`
- **Visualization**: `matplotlib`, `seaborn`
- **Financial Library**: `yfinance` (or custom data loaders for financial APIs)

## Repository Structure
```text
├── data/                  # Sample datasets and data fetching scripts
├── src/
│   ├── factor_gen.py      # Core logic for factor calculation
│   ├── backtester.py      # Backtesting engine implementation
│   └── utils.py           # Helper functions for data cleaning
├── notebooks/             # Jupyter notebooks for exploratory analysis
├── results/               # Exported charts and performance reports
├── requirements.txt       # Project dependencies
└── main.py                # Entry point for running the full pipeline
```

##  Getting Started

Follow these instructions to set up the project on your local machine.

### Prerequisites

Before you begin, ensure you have the following installed:
* **Python 3.11 or higher**: [Download Python](https://www.python.org/downloads/)
* **Git**: [Download Git](https://git-scm.com/downloads)
* **Pip**: Usually comes installed with Python.

### Installation

1. **Clone the repository:**
   Open your terminal or command prompt and run:
   ```bash
   git clone https://github.com/DanielYan0224/factor-and-defense-factor.git
   cd factor-and-defense-factor
   ```

2. **Create the project environment:**
    ```bash
   conda create -n [your_project_name] python=3.11 -y
   ```

3. **Install the required packages:**
    ```bash
   pip install -r requirements.txt
   ```
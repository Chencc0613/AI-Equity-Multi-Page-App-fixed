# AI Equity Multi-Page V5 Fusion Terminal

Build: `AI_EQUITY_MULTIPAGE_V5_FUSION_TERMINAL_2026_06_15`

## What this version is

V5 combines three previously separated layers:

1. **V4 robust data engine**: Manual override > yfinance AUTO > V5 SEED > DERIVED > MISSING.
2. **Network Control Tower**: network graph, scenario impact, role pie charts, relationship table, quick navigation buttons.
3. **Detailed single-stock pages**: each stock keeps its own driver model, key metrics, valuation sources, risk dashboard, price/trend chart, JSON/CSV export.

## Files

```text
streamlit_app.py
 data_engine.py
 requirements.txt
 .streamlit/config.toml
 pages/
   00_TSM_2330_Foundry_Toll_Road.py
   01_NVDA_AI_Infrastructure.py
   02_MSFT_Cloud_AI.py
   03_AVGO_Custom_Silicon.py
   04_GOOGL_Search_Cloud_TPU.py
   05_AMZN_AWS_AI.py
   06_META_AI_Ads.py
   07_MU_HBM_Memory_Cycle.py
   08_AMD_AI_GPU_EPYC.py
```

## Run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Streamlit Cloud

Main file path:

```text
streamlit_app.py
```

Upload the whole folder contents to the repo root. The `pages/` folder must be beside `streamlit_app.py`.

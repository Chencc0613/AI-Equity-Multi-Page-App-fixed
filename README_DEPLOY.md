# AI Equity Multi-Page App V4

## Build
`AI_EQUITY_MULTIPAGE_V4_DATA_ENGINE_2026_06_15`

## What changed in V4

- Unified Fundamental Data Engine
- Manual override for PE / EPS / margins / normalized growth on every company page
- Fallback seed values when yfinance returns missing fundamentals
- Transparent field-level data source: MANUAL / AUTO / SEED / DERIVED / MISSING
- Home page does not load all fundamental data; each stock page loads details only when opened
- No pandas_datareader / no FRED dependency
- JSON download is safe-cleaned

## Deploy

Main file path on Streamlit Cloud:

```text
streamlit_app.py
```

Local:

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## GitHub structure

```text
streamlit_app.py
requirements.txt
data_engine.py
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

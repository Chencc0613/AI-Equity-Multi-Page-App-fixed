# ============================================================
# AI Equity Multi-Page V5 Fusion Terminal
# Unified robust data engine + network control tower + detailed single-stock pages
# ============================================================
from __future__ import annotations

import json
import math
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import yfinance as yf

APP_BUILD_VERSION = "AI_EQUITY_MULTIPAGE_V5_FUSION_TERMINAL_2026_06_15"

# ============================================================
# Stable fallback fundamentals
# Manual override > yfinance AUTO > V5 seed > derived > missing.
# Seeds are deliberately conservative working values so the app never becomes useless
# just because Yahoo fundamental endpoints return empty values on Streamlit Cloud.
# ============================================================
SEED_FUNDAMENTALS: Dict[str, Dict[str, float]] = {
    "2330.TW": {"forward_pe": 22.0, "trailing_pe": 25.0, "forward_eps": 60.0, "trailing_eps": 53.0, "gross_margin": 58.0, "operating_margin": 48.0, "fcf_margin": 23.0, "growth": 0.24},
    "TSM":     {"forward_pe": 22.0, "trailing_pe": 25.0, "forward_eps": 10.8, "trailing_eps": 9.4, "gross_margin": 58.0, "operating_margin": 48.0, "fcf_margin": 23.0, "growth": 0.24},
    "NVDA":    {"forward_pe": 32.0, "trailing_pe": 45.0, "forward_eps": 5.8, "trailing_eps": 4.1, "gross_margin": 75.0, "operating_margin": 60.0, "fcf_margin": 43.0, "growth": 0.38},
    "MSFT":    {"forward_pe": 30.0, "trailing_pe": 34.0, "forward_eps": 15.0, "trailing_eps": 13.3, "gross_margin": 70.0, "operating_margin": 45.0, "fcf_margin": 32.0, "growth": 0.12},
    "AVGO":    {"forward_pe": 33.0, "trailing_pe": 63.57, "forward_eps": 11.58, "trailing_eps": 6.01, "gross_margin": 77.11, "operating_margin": 67.28, "fcf_margin": 46.25, "growth": 0.16},
    "GOOGL":   {"forward_pe": 18.0, "trailing_pe": 23.0, "forward_eps": 11.0, "trailing_eps": 8.6, "gross_margin": 58.0, "operating_margin": 32.0, "fcf_margin": 23.0, "growth": 0.16},
    "AMZN":    {"forward_pe": 31.0, "trailing_pe": 35.0, "forward_eps": 7.2, "trailing_eps": 6.4, "gross_margin": 49.0, "operating_margin": 11.0, "fcf_margin": 8.0, "growth": 0.18},
    "META":    {"forward_pe": 22.0, "trailing_pe": 26.0, "forward_eps": 32.0, "trailing_eps": 27.0, "gross_margin": 82.0, "operating_margin": 40.0, "fcf_margin": 28.0, "growth": 0.17},
    "MU":      {"forward_pe": 12.82, "trailing_pe": 46.35, "forward_eps": 76.60, "trailing_eps": 21.18, "gross_margin": 74.90, "operating_margin": 69.00, "fcf_margin": 28.90, "growth": 0.42},
    "AMD":     {"forward_pe": 27.0, "trailing_pe": 85.0, "forward_eps": 8.0, "trailing_eps": 2.5, "gross_margin": 50.0, "operating_margin": 12.0, "fcf_margin": 9.0, "growth": 0.22},
}

# ============================================================
# Company config: this is the product layer.
# Each stock has its own drivers, default scores, thresholds, watchlist and manual nodes.
# ============================================================
COMPANIES: Dict[str, Dict[str, Any]] = {
    "TSM2330": {
        "label": "TSM / 2330", "ticker": "2330.TW", "alt_ticker": "TSM", "currency": "TWD", "icon": "🏭",
        "layer": "Foundry", "page": "pages/00_TSM_2330_Foundry_Toll_Road.py",
        "theme": "Foundry Toll Road / 先進製程 + CoWoS",
        "role": "AI supply chain toll road。核心不是短線股價，而是先進製程、CoWoS、CapEx、客戶結構與地緣風險。",
        "drivers": {"Foundry Demand": 0.24, "Advanced Node / N2 / A16": 0.18, "CoWoS / Advanced Packaging": 0.18, "Customer CapEx": 0.14, "Margin / Pricing": 0.10, "CapEx Discipline": 0.08, "Geo / Export Risk": 0.08},
        "default_driver_scores": {"Foundry Demand": 0.65, "Advanced Node / N2 / A16": 0.65, "CoWoS / Advanced Packaging": 0.70, "Customer CapEx": 0.55, "Margin / Pricing": 0.45, "CapEx Discipline": 0.35, "Geo / Export Risk": -0.25},
        "pe_thresholds": [18, 24, 30, 38], "growth_default": 0.24,
        "watch": ["Monthly revenue / 月營收", "CapEx plan / 資本支出計畫", "CoWoS capacity", "NVDA/AMD/AVGO/Apple orders", "Taiwan geopolitical risk"],
        "key_metrics": [("Monthly revenue YoY %", 40.0), ("CapEx YoY %", 25.0), ("CoWoS capacity tightness", 0.75), ("Geo risk score", -0.25)],
    },
    "NVDA": {
        "label": "NVDA", "ticker": "NVDA", "currency": "USD", "icon": "🟢",
        "layer": "GPU Compute", "page": "pages/01_NVDA_AI_Infrastructure.py",
        "theme": "GPU / AI Infrastructure Platform",
        "role": "AI compute platform leader。核心看 hyperscaler CapEx、Data Center guidance、Blackwell/Rubin ramp、毛利率與出口管制。",
        "drivers": {"AI CapEx Demand": 0.30, "Data Center Guide": 0.25, "Product Cycle": 0.15, "Gross Margin / Pricing": 0.10, "Supply Chain": 0.08, "China / Export Risk": 0.07, "Valuation / Macro": 0.05},
        "default_driver_scores": {"AI CapEx Demand": 0.70, "Data Center Guide": 0.65, "Product Cycle": 0.70, "Gross Margin / Pricing": 0.70, "Supply Chain": 0.35, "China / Export Risk": -0.30, "Valuation / Macro": 0.0},
        "pe_thresholds": [30, 42, 58, 75], "growth_default": 0.38,
        "watch": ["Data Center QoQ/YoY", "Revenue guidance surprise", "Blackwell / Rubin ramp", "Gross margin > 70%", "CoWoS/HBM supply", "China export controls"],
        "key_metrics": [("Data Center QoQ %", 12.0), ("Data Center YoY %", 90.0), ("Guidance surprise %", 2.0), ("Gross margin %", 75.0), ("Export risk score", -0.30)],
    },
    "MSFT": {
        "label": "MSFT", "ticker": "MSFT", "currency": "USD", "icon": "🔷",
        "layer": "Cloud/App", "page": "pages/02_MSFT_Cloud_AI.py",
        "theme": "Azure + Copilot + Enterprise AI",
        "role": "Cloud + AI monetization。MSFT 的重點不是 CapEx 多，而是 AI CapEx 能否轉成 Azure、Copilot、margin 與 FCF。",
        "drivers": {"Azure / Cloud AI": 0.30, "Copilot Monetization": 0.20, "Margin / FCF": 0.15, "AI CapEx Efficiency": 0.12, "Enterprise Durability": 0.10, "Platform Risk": 0.05, "Valuation / Macro": 0.08},
        "default_driver_scores": {"Azure / Cloud AI": 0.35, "Copilot Monetization": 0.15, "Margin / FCF": 0.55, "AI CapEx Efficiency": 0.05, "Enterprise Durability": 0.55, "Platform Risk": -0.10, "Valuation / Macro": 0.10},
        "pe_thresholds": [22, 28, 35, 45], "growth_default": 0.12,
        "watch": ["Azure growth", "Cloud gross margin", "Copilot ARPU / seat expansion", "AI depreciation", "OpenAI dependence", "Enterprise durability"],
        "key_metrics": [("Azure growth YoY %", 31.0), ("Cloud guide surprise %", 1.5), ("Copilot monetization score", 0.20), ("AI CapEx efficiency score", 0.05), ("FCF margin %", 32.0)],
    },
    "AVGO": {
        "label": "AVGO", "ticker": "AVGO", "currency": "USD", "icon": "🔌",
        "layer": "ASIC/Network", "page": "pages/03_AVGO_Custom_Silicon.py",
        "theme": "Custom ASIC + AI Networking + VMware FCF",
        "role": "NVDA 替代路線監控器。看 custom AI ASIC、networking、VMware cash flow 與客戶集中風險。",
        "drivers": {"Custom ASIC Demand": 0.30, "AI Networking": 0.20, "Hyperscaler CapEx": 0.15, "VMware / Software FCF": 0.12, "Margin / FCF": 0.10, "Customer Concentration Risk": 0.08, "Valuation / Macro": 0.05},
        "default_driver_scores": {"Custom ASIC Demand": 0.75, "AI Networking": 0.70, "Hyperscaler CapEx": 0.55, "VMware / Software FCF": 0.45, "Margin / FCF": 0.65, "Customer Concentration Risk": -0.20, "Valuation / Macro": 0.0},
        "pe_thresholds": [24, 34, 46, 60], "growth_default": 0.16,
        "watch": ["AI semiconductor revenue", "Custom ASIC customers", "Networking backlog", "VMware margin", "Customer concentration", "ASIC replacing GPU?"],
        "key_metrics": [("AI semiconductor revenue growth %", 140.0), ("AI networking strength", 0.70), ("Custom ASIC wins", 0.75), ("VMware synergy score", 0.45), ("Customer concentration risk", -0.20)],
    },
    "GOOGL": {
        "label": "GOOGL", "ticker": "GOOGL", "currency": "USD", "icon": "🔎",
        "layer": "Cloud/App", "page": "pages/04_GOOGL_Search_Cloud_TPU.py",
        "theme": "Search AI + Cloud + TPU",
        "role": "Search monetization + GCP + Gemini + TPU。Google 同時是雲端、AI 應用與自研 ASIC 公司。",
        "drivers": {"Google Cloud AI": 0.30, "Search AI Monetization": 0.25, "Gemini Subscription": 0.12, "TPU / CapEx Efficiency": 0.12, "Margin / FCF": 0.10, "YouTube / Ads": 0.06, "Regulatory / TPU Risk": 0.03, "Valuation / Macro": 0.02},
        "default_driver_scores": {"Google Cloud AI": 0.45, "Search AI Monetization": 0.10, "Gemini Subscription": 0.10, "TPU / CapEx Efficiency": 0.30, "Margin / FCF": 0.35, "YouTube / Ads": 0.25, "Regulatory / TPU Risk": -0.25, "Valuation / Macro": 0.20},
        "pe_thresholds": [18, 24, 32, 42], "growth_default": 0.16,
        "watch": ["Search revenue resilience", "Google Cloud growth", "TPU adoption", "Gemini monetization", "Antitrust risk"],
        "key_metrics": [("Google Cloud growth %", 28.0), ("Search monetization score", 0.10), ("Gemini subscription score", 0.10), ("TPU adoption score", 0.30), ("Regulatory risk score", -0.25)],
    },
    "AMZN": {
        "label": "AMZN", "ticker": "AMZN", "currency": "USD", "icon": "📦",
        "layer": "Cloud/App", "page": "pages/05_AMZN_AWS_AI.py",
        "theme": "AWS + Trainium + Retail/Ads FCF",
        "role": "AWS AI CapEx 能不能變成 high margin revenue。看 AWS、Trainium、Bedrock、retail/ads cash flow。",
        "drivers": {"AWS Growth": 0.30, "AI CapEx Efficiency": 0.20, "Trainium / Bedrock": 0.12, "Retail / Ads FCF": 0.12, "Operating Margin": 0.10, "Debt / CapEx Pressure": 0.08, "Regulatory Risk": 0.03, "Valuation / Macro": 0.05},
        "default_driver_scores": {"AWS Growth": 0.35, "AI CapEx Efficiency": 0.05, "Trainium / Bedrock": 0.20, "Retail / Ads FCF": 0.35, "Operating Margin": 0.30, "Debt / CapEx Pressure": -0.15, "Regulatory Risk": -0.10, "Valuation / Macro": 0.0},
        "pe_thresholds": [28, 38, 55, 75], "growth_default": 0.18,
        "watch": ["AWS growth", "AWS operating income", "Trainium adoption", "CapEx/debt", "Retail ad margin"],
        "key_metrics": [("AWS growth %", 17.0), ("AWS operating margin %", 35.0), ("Trainium / Bedrock score", 0.20), ("AI CapEx efficiency", 0.05), ("Debt / CapEx pressure", -0.15)],
    },
    "META": {
        "label": "META", "ticker": "META", "currency": "USD", "icon": "🧠",
        "layer": "Ads/App", "page": "pages/06_META_AI_Ads.py",
        "theme": "AI Ads + Social Platform",
        "role": "AI 直接拉廣告效率。看 recommendation、engagement、Reels、AI agents、CapEx burden。",
        "drivers": {"AI Ad Efficiency": 0.30, "Engagement / Reels": 0.20, "AI CapEx Burden": 0.15, "Llama / AI Agents": 0.10, "Margin / FCF": 0.10, "Regulatory / Safety Risk": 0.10, "Valuation / Macro": 0.05},
        "default_driver_scores": {"AI Ad Efficiency": 0.60, "Engagement / Reels": 0.45, "AI CapEx Burden": -0.10, "Llama / AI Agents": 0.35, "Margin / FCF": 0.45, "Regulatory / Safety Risk": -0.20, "Valuation / Macro": 0.10},
        "pe_thresholds": [18, 24, 32, 42], "growth_default": 0.17,
        "watch": ["Ad impressions/pricing", "Reels engagement", "AI CapEx guide", "Family Apps margin", "Regulatory risk"],
        "key_metrics": [("Ad efficiency score", 0.60), ("Engagement / Reels score", 0.45), ("AI CapEx burden score", -0.10), ("Llama / AI agents score", 0.35), ("Regulatory risk", -0.20)],
    },
    "MU": {
        "label": "MU", "ticker": "MU", "currency": "USD", "icon": "💾",
        "layer": "Memory", "page": "pages/07_MU_HBM_Memory_Cycle.py",
        "theme": "HBM + Memory Cycle",
        "role": "HBM 結構成長 + DRAM/NAND 週期槓桿。PE 低不等於安全，要看供需反轉。",
        "drivers": {"HBM Demand": 0.30, "DRAM / NAND Pricing": 0.22, "Data Center / AI Mix": 0.15, "Gross Margin Recovery": 0.12, "Inventory / Supply Discipline": 0.08, "CapEx / Oversupply Risk": 0.08, "Valuation / Macro": 0.05},
        "default_driver_scores": {"HBM Demand": 0.85, "DRAM / NAND Pricing": 0.90, "Data Center / AI Mix": 0.90, "Gross Margin Recovery": 1.00, "Inventory / Supply Discipline": 0.65, "CapEx / Oversupply Risk": -0.35, "Valuation / Macro": 0.0},
        "pe_thresholds": [10, 16, 24, 36], "growth_default": 0.42,
        "watch": ["HBM sold-out status", "DRAM/NAND ASP", "Data center mix", "Inventory days", "CapEx oversupply risk"],
        "key_metrics": [("HBM revenue growth %", 150.0), ("HBM revenue mix %", 15.0), ("Data center revenue growth %", 181.3), ("Gross margin guide surprise pp", 6.1), ("CapEx oversupply risk", -0.35)],
    },
    "AMD": {
        "label": "AMD", "ticker": "AMD", "currency": "USD", "icon": "🔴",
        "layer": "GPU/CPU", "page": "pages/08_AMD_AI_GPU_EPYC.py",
        "theme": "AI GPU + EPYC + ROCm",
        "role": "NVDA 替代 GPU + EPYC server CPU。看 MI adoption、ROCm gap、customer breadth。",
        "drivers": {"AI GPU Adoption": 0.30, "ROCm / Software Ecosystem": 0.15, "EPYC Data Center CPU": 0.15, "Customer Breadth": 0.12, "Margin / FCF": 0.10, "Client / Gaming Drag": 0.08, "Supply / Competition": 0.05, "Valuation / Macro": 0.05},
        "default_driver_scores": {"AI GPU Adoption": 0.35, "ROCm / Software Ecosystem": -0.15, "EPYC Data Center CPU": 0.45, "Customer Breadth": 0.25, "Margin / FCF": 0.05, "Client / Gaming Drag": -0.10, "Supply / Competition": -0.25, "Valuation / Macro": 0.0},
        "pe_thresholds": [24, 34, 50, 70], "growth_default": 0.22,
        "watch": ["MI series revenue", "ROCm gap vs CUDA", "EPYC share", "Cloud customers", "Gaming/client drag"],
        "key_metrics": [("AI GPU revenue growth %", 60.0), ("AI GPU revenue mix %", 18.0), ("Data Center growth %", 35.0), ("ROCm vs CUDA gap", -0.15), ("EPYC server CPU score", 0.45)],
    },
}

NETWORK_EDGES = [
    ("MSFT", "NVDA", "Azure AI CapEx → GPU demand", 0.75), ("AMZN", "NVDA", "AWS AI CapEx → GPU demand", 0.70), ("GOOGL", "NVDA", "Cloud/AI infra → GPU demand", 0.55), ("META", "NVDA", "AI Ads/LLM CapEx → GPU demand", 0.60),
    ("NVDA", "TSM2330", "GPU orders → foundry/CoWoS", 0.75), ("AMD", "TSM2330", "MI/EPYC orders → foundry", 0.45), ("AVGO", "TSM2330", "ASIC/networking → foundry", 0.55),
    ("NVDA", "MU", "GPU systems → HBM demand", 0.70), ("AMD", "MU", "MI systems → HBM demand", 0.45), ("AVGO", "MU", "ASIC systems → HBM/network memory", 0.30),
    ("GOOGL", "AVGO", "TPU/custom silicon ecosystem", 0.50), ("AMZN", "AVGO", "Trainium/custom ASIC pressure", 0.40), ("META", "AVGO", "Custom ASIC possibility", 0.30),
    ("AVGO", "NVDA", "ASIC shift pressures GPU multiple", -0.45), ("AMD", "NVDA", "Alternative GPU competition", -0.30),
    ("MSFT", "AMZN", "Cloud AI monetization peer", 0.35), ("MSFT", "GOOGL", "Cloud/Search AI competition", 0.30), ("GOOGL", "META", "AI ads/search attention", 0.25),
]

NETWORK_POS = {
    "MSFT": (0.05, 0.86), "GOOGL": (0.34, 0.94), "AMZN": (0.64, 0.86), "META": (0.92, 0.78),
    "NVDA": (0.25, 0.52), "AMD": (0.49, 0.46), "AVGO": (0.72, 0.54),
    "TSM2330": (0.31, 0.18), "MU": (0.70, 0.18),
}

SCENARIO_EXPOSURES = {
    "TSM2330": [0.55, 0.45, 0.35, 0.20, 0.25, 0.05, 0.20, 0.35],
    "NVDA": [0.75, 0.95, -0.35, 0.25, 0.35, 0.05, 0.25, 0.45],
    "AMD": [0.50, 0.55, -0.10, 0.15, 0.25, 0.00, 0.20, 0.35],
    "AVGO": [0.55, 0.20, 0.85, 0.20, 0.20, 0.00, 0.20, 0.25],
    "MU": [0.55, 0.60, 0.25, 0.10, 0.95, 0.00, 0.15, 0.35],
    "MSFT": [0.35, 0.05, 0.10, 0.80, 0.00, 0.10, 0.35, 0.10],
    "GOOGL": [0.35, 0.05, 0.35, 0.50, 0.00, 0.60, 0.30, 0.10],
    "AMZN": [0.45, 0.10, 0.30, 0.65, 0.00, 0.15, 0.25, 0.10],
    "META": [0.40, 0.15, 0.10, 0.20, 0.00, 0.85, 0.25, 0.10],
}
SCENARIO_KEYS = ["AI CapEx", "GPU Demand", "ASIC Shift", "Cloud Monetization", "HBM / Memory", "Search / Ads AI", "Macro Liquidity", "Export Risk Ease"]

# ============================================================
# Helpers
# ============================================================
def safe_float(x: Any, default: float = np.nan) -> float:
    try:
        if x is None:
            return default
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def fmt_num(x: Any, digits: int = 2) -> str:
    v = safe_float(x)
    return "NA" if pd.isna(v) else f"{v:,.{digits}f}"


def fmt_pct_fraction(x: Any, digits: int = 2) -> str:
    v = safe_float(x)
    return "NA" if pd.isna(v) else f"{v * 100:,.{digits}f}%"


def fmt_pct_percent(x: Any, digits: int = 2) -> str:
    v = safe_float(x)
    return "NA" if pd.isna(v) else f"{v:,.{digits}f}%"


def fmt_money(x: Any, currency: str = "USD", digits: int = 2) -> str:
    v = safe_float(x)
    if pd.isna(v):
        return "NA"
    prefix = "NT$" if currency == "TWD" else "$"
    return f"{prefix}{v:,.{digits}f}"


def clamp(x: Any, lo: float = -1.0, hi: float = 1.0) -> float:
    return float(np.clip(safe_float(x, 0.0), lo, hi))


def action_color(action: str) -> str:
    return "#22c55e" if action == "BUY" else "#f59e0b"


def clean_for_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): clean_for_json(v) for k, v in obj.items() if k not in ["price_df"]}
    if isinstance(obj, (list, tuple, set)):
        return [clean_for_json(v) for v in obj]
    if isinstance(obj, pd.DataFrame):
        return obj.reset_index().astype(str).to_dict(orient="records")
    if isinstance(obj, pd.Series):
        return obj.astype(str).to_dict()
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj); return None if not np.isfinite(v) else v
    if isinstance(obj, float):
        return None if not math.isfinite(obj) else obj
    if isinstance(obj, np.bool_):
        return bool(obj)
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return str(obj)


def safe_json_dumps(obj: Any) -> str:
    return json.dumps(clean_for_json(obj), ensure_ascii=False, indent=2)

# ============================================================
# Data fetchers
# ============================================================
@st.cache_data(ttl=900, show_spinner=False)
def fetch_price_history(ticker: str, period: str = "3y") -> pd.DataFrame:
    try:
        df = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=True, threads=False)
        if df is None or df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df.dropna(how="all")
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_auto_fundamentals(ticker: str) -> Dict[str, Any]:
    out = {"source": "AUTO", "ok": False, "error": ""}
    try:
        tk = yf.Ticker(ticker)
        info = tk.info or {}
        fields = {
            "forward_pe": "forwardPE", "trailing_pe": "trailingPE", "forward_eps": "forwardEps", "trailing_eps": "trailingEps",
            "gross_margin": "grossMargins", "operating_margin": "operatingMargins", "revenue_growth": "revenueGrowth", "earnings_growth": "earningsGrowth",
            "free_cashflow": "freeCashflow", "market_cap": "marketCap",
        }
        for target, key in fields.items():
            val = safe_float(info.get(key))
            if target in ["gross_margin", "operating_margin"] and not pd.isna(val) and abs(val) <= 1.5:
                val *= 100.0
            out[target] = val
        try:
            fast = tk.fast_info
            out["last_price_fast"] = safe_float(getattr(fast, "last_price", np.nan))
        except Exception:
            out["last_price_fast"] = np.nan
        if any(not pd.isna(safe_float(out.get(k))) for k in ["forward_pe", "trailing_pe", "forward_eps", "trailing_eps", "gross_margin"]):
            out["ok"] = True
    except Exception as e:
        out["error"] = str(e)
    return out


def _resolve_field(field: str, manual: Dict[str, float], auto: Dict[str, Any], seed: Dict[str, float]) -> Tuple[float, str]:
    m = safe_float(manual.get(field))
    if not pd.isna(m) and m != 0:
        return m, "MANUAL"
    a = safe_float(auto.get(field))
    if not pd.isna(a) and a != 0:
        return a, "AUTO"
    if field == "growth":
        eg = safe_float(auto.get("earnings_growth"))
        if not pd.isna(eg) and eg > 0:
            return eg, "AUTO_EARNINGS_GROWTH"
    s = safe_float(seed.get(field))
    if not pd.isna(s) and s != 0:
        return s, "SEED"
    return np.nan, "MISSING"


def resolve_fundamentals(ticker: str, manual: Dict[str, float] | None = None) -> Dict[str, Any]:
    manual = manual or {}
    auto = fetch_auto_fundamentals(ticker)
    seed = SEED_FUNDAMENTALS.get(ticker, SEED_FUNDAMENTALS.get(ticker.replace(".TW", ""), {}))
    fields = ["forward_pe", "trailing_pe", "forward_eps", "trailing_eps", "gross_margin", "operating_margin", "fcf_margin", "growth"]
    res: Dict[str, Any] = {"ticker": ticker, "auto_ok": bool(auto.get("ok")), "auto_error": auto.get("error", ""), "source_by_field": {}}
    for f in fields:
        val, src = _resolve_field(f, manual, auto, seed)
        res[f] = val
        res["source_by_field"][f] = src
    return res


def compute_price_pack(df: pd.DataFrame) -> Dict[str, Any]:
    if df is None or df.empty or "Close" not in df:
        return {"last_price": np.nan, "ma50": np.nan, "ma60": np.nan, "ma200": np.nan, "dd_120": np.nan, "atr_pct": np.nan, "sigma_60": np.nan, "trend_state": "PRICE_MISSING", "trend_scale": 0.0}
    close = df["Close"].dropna()
    high = df.get("High", close).dropna()
    low = df.get("Low", close).dropna()
    last = safe_float(close.iloc[-1])
    ma50 = safe_float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else np.nan
    ma60 = safe_float(close.rolling(60).mean().iloc[-1]) if len(close) >= 60 else np.nan
    ma200 = safe_float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else np.nan
    roll_high = close.rolling(120, min_periods=min(60, len(close))).max()
    peak = safe_float(roll_high.iloc[-1]) if len(roll_high) else np.nan
    dd_120 = last / peak - 1.0 if not pd.isna(last) and not pd.isna(peak) and peak > 0 else np.nan
    prev_close = close.shift(1)
    tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14, min_periods=10).mean()
    atr_pct = safe_float(atr.iloc[-1]) / last if not pd.isna(last) and last > 0 and len(atr.dropna()) else np.nan
    ret = close.pct_change().dropna()
    sigma_60 = safe_float(ret.tail(60).std(ddof=1) * np.sqrt(252)) if len(ret) >= 30 else np.nan
    if not pd.isna(ma50) and not pd.isna(ma200):
        if last > ma50 and ma50 > ma200:
            trend_state, trend_scale = "STRONG", 1.0
        elif last > ma50:
            trend_state, trend_scale = "RECOVERY", 0.55
        elif last > ma200:
            trend_state, trend_scale = "WEAK_PULLBACK", 0.30
        else:
            trend_state, trend_scale = "WEAK", 0.0
    else:
        trend_state, trend_scale = "TREND_DATA_MISSING", 0.3
    return {"last_price": last, "ma50": ma50, "ma60": ma60, "ma200": ma200, "dd_120": dd_120, "atr_pct": atr_pct, "sigma_60": sigma_60, "trend_state": trend_state, "trend_scale": trend_scale}

# ============================================================
# Scoring
# ============================================================
def pe_scale(selected_pe: float, thresholds: List[float]) -> Tuple[float, str]:
    pe = safe_float(selected_pe)
    if pd.isna(pe) or pe <= 0:
        return 1.0, "PE_MISSING"
    cheap, fair, warm, expensive = thresholds
    if pe <= cheap: return 1.20, "CHEAP"
    if pe <= fair: return 1.00, "FAIR"
    if pe <= warm: return 0.70, "WARM"
    if pe <= expensive: return 0.40, "EXPENSIVE"
    return 0.15, "VERY_EXPENSIVE"


def peg_scale(peg: float) -> Tuple[float, str]:
    p = safe_float(peg)
    if pd.isna(p) or p <= 0:
        return 1.0, "PEG_MISSING"
    if p <= 0.8: return 1.20, "CHEAP"
    if p <= 1.2: return 1.00, "FAIR"
    if p <= 1.8: return 0.70, "WARM"
    if p <= 2.5: return 0.40, "EXPENSIVE"
    return 0.15, "VERY_EXPENSIVE"


def weighted_driver_score(scores: Dict[str, float], weights: Dict[str, float]) -> float:
    total_w = sum(weights.values()) or 1.0
    return float(sum(safe_float(scores.get(k), 0.0) * w for k, w in weights.items()) / total_w)


def driver_scale_from_score(score: float) -> Tuple[float, str]:
    s = safe_float(score, 0.0)
    if s >= 0.45: return 1.20, "STRONG"
    if s >= -0.15: return 1.00, "STABLE"
    if s >= -0.45: return 0.60, "WEAK"
    return 0.25, "BAD"


@st.cache_data(ttl=900, show_spinner=False)
def macro_pack() -> Dict[str, Any]:
    tickers = ["QQQ", "SMH", "SPY", "HYG", "LQD", "^VIX", "TLT", "UUP", "GLD"]
    data = {}
    for t in tickers:
        df = fetch_price_history(t, period="1y")
        if not df.empty and "Close" in df:
            data[t] = df["Close"].dropna()
    risk = 0.0
    notes = []
    def ret(t, n=21):
        s = data.get(t, pd.Series(dtype=float))
        return safe_float(s.iloc[-1] / s.iloc[-min(len(s), n+1)] - 1.0) if len(s) > n else np.nan
    qqq = ret("QQQ", 21); smh = ret("SMH", 21); spy = ret("SPY", 21); hyg_lqd = np.nan
    if "HYG" in data and "LQD" in data:
        ratio = (data["HYG"] / data["LQD"]).dropna()
        if len(ratio) > 21:
            hyg_lqd = safe_float(ratio.iloc[-1] / ratio.iloc[-22] - 1.0)
    vix = safe_float(data.get("^VIX", pd.Series([np.nan])).iloc[-1]) if "^VIX" in data else np.nan
    if not pd.isna(qqq): risk += -qqq * 2.0; notes.append(f"QQQ 1M {qqq:+.1%}")
    if not pd.isna(smh): risk += -smh * 1.5; notes.append(f"SMH 1M {smh:+.1%}")
    if not pd.isna(spy): risk += -spy * 1.0; notes.append(f"SPY 1M {spy:+.1%}")
    if not pd.isna(hyg_lqd): risk += -hyg_lqd * 4.0; notes.append(f"Credit appetite {hyg_lqd:+.1%}")
    if not pd.isna(vix): risk += max(0, (vix - 20) / 20); notes.append(f"VIX {vix:.1f}")
    risk = float(np.clip(risk, -1.5, 2.0))
    if risk >= 1.0: regime = "RISK-OFF"
    elif risk <= -0.5: regime = "RISK-ON"
    else: regime = "NEUTRAL"
    return {"risk": risk, "regime": regime, "notes": notes, "vix": vix}


def compute_analysis(company_key: str, account: Dict[str, float] | None = None, manual_fin: Dict[str, float] | None = None, driver_scores: Dict[str, float] | None = None, use_alt_ticker: bool = False) -> Dict[str, Any]:
    cfg = COMPANIES[company_key]
    ticker = cfg.get("alt_ticker") if use_alt_ticker and cfg.get("alt_ticker") else cfg["ticker"]
    account = account or {"equity": 0.0, "cash": 0.0, "shares": 0.0, "avg_cost": 0.0, "cash_use_frac": 1.0}
    df = fetch_price_history(ticker, period="3y")
    price = compute_price_pack(df)
    fin = resolve_fundamentals(ticker, manual_fin or {})
    last = price.get("last_price")
    if not pd.isna(last):
        if (pd.isna(fin.get("forward_eps")) or fin.get("forward_eps") == 0) and not pd.isna(fin.get("forward_pe")) and fin["forward_pe"] > 0:
            fin["forward_eps"] = last / fin["forward_pe"]; fin["source_by_field"]["forward_eps"] = "DERIVED"
        if (pd.isna(fin.get("trailing_eps")) or fin.get("trailing_eps") == 0) and not pd.isna(fin.get("trailing_pe")) and fin["trailing_pe"] > 0:
            fin["trailing_eps"] = last / fin["trailing_pe"]; fin["source_by_field"]["trailing_eps"] = "DERIVED"
    selected_pe = fin.get("forward_pe") if not pd.isna(fin.get("forward_pe")) else fin.get("trailing_pe")
    growth = safe_float(fin.get("growth"), cfg.get("growth_default", 0.12))
    merged_driver_scores = dict(cfg.get("default_driver_scores", {}))
    if driver_scores:
        merged_driver_scores.update(driver_scores)
    d_score = weighted_driver_score(merged_driver_scores, cfg["drivers"])
    d_scale, d_state = driver_scale_from_score(d_score)
    effective_growth = max(0.03, growth * (1 + 0.45 * d_score))
    peg = selected_pe / (effective_growth * 100.0) if not pd.isna(selected_pe) and effective_growth > 0 else np.nan
    p_scale, p_state = pe_scale(selected_pe, cfg["pe_thresholds"])
    g_scale, g_state = peg_scale(peg)
    valuation_scale = min(p_scale, g_scale)
    macro = macro_pack()
    macro_scale = 0.6 if macro["regime"] == "RISK-OFF" else 1.0
    trend_scale = price.get("trend_scale", 0.0)
    buy_scale = trend_scale * valuation_scale * d_scale * macro_scale
    buy_scale = float(np.clip(buy_scale, 0.0, 1.2))
    cash = safe_float(account.get("cash"), 0.0)
    cash_use_frac = safe_float(account.get("cash_use_frac"), 1.0)
    buy_budget = max(0.0, cash * cash_use_frac * buy_scale)
    if pd.isna(last) or last <= 0:
        est_shares = np.nan
    else:
        est_shares = math.floor(buy_budget / last)
    action = "BUY" if not pd.isna(est_shares) and est_shares > 0 and buy_scale > 0.05 else "HOLD"
    reasons = []
    if trend_scale <= 0: reasons.append("Trend scale = 0：趨勢不允許買進")
    if valuation_scale < 0.5: reasons.append("Valuation scale 偏低：估值偏熱或 PEG 偏貴")
    if d_state in ["WEAK", "BAD"]: reasons.append(f"Driver state = {d_state}：產業驅動轉弱")
    if macro["regime"] == "RISK-OFF": reasons.append("Macro = RISK-OFF：總體風險折扣")
    if cash <= 0: reasons.append("Cash = 0：沒有可用現金")
    if action == "BUY": reasons.append("價格 / 估值 / 驅動 / 現金條件通過，允許分批買進")
    holdings_value = safe_float(account.get("shares"), 0.0) * (last if not pd.isna(last) else 0.0)
    cost_value = safe_float(account.get("shares"), 0.0) * safe_float(account.get("avg_cost"), 0.0)
    pnl = holdings_value - cost_value if cost_value > 0 else np.nan
    pnl_pct = pnl / cost_value if cost_value > 0 else np.nan
    driver_df = pd.DataFrame([{"driver": k, "weight": cfg["drivers"].get(k, 0.0), "score": merged_driver_scores.get(k, 0.0), "weighted": cfg["drivers"].get(k, 0.0) * merged_driver_scores.get(k, 0.0)} for k in cfg["drivers"]])
    return {
        "company_key": company_key, "config": cfg, "ticker": ticker, "price_df": df, "price": price,
        "fundamentals": fin, "selected_pe": selected_pe, "growth": growth, "effective_growth": effective_growth, "peg": peg,
        "pe_scale": p_scale, "pe_state": p_state, "peg_scale": g_scale, "peg_state": g_state, "valuation_scale": valuation_scale,
        "driver_score": d_score, "driver_scale": d_scale, "driver_state": d_state, "driver_scores": merged_driver_scores, "driver_df": driver_df,
        "macro": macro, "macro_scale": macro_scale, "buy_scale": buy_scale, "buy_budget": buy_budget, "est_shares": est_shares, "action": action,
        "account": account, "holdings_value": holdings_value, "pnl": pnl, "pnl_pct": pnl_pct, "reasons": reasons,
    }

# ============================================================
# Charts
# ============================================================
def make_price_chart(res: Dict[str, Any]) -> go.Figure:
    df = res.get("price_df", pd.DataFrame())
    cfg = res["config"]
    fig = go.Figure()
    if df is None or df.empty or "Close" not in df:
        fig.update_layout(title="Price data missing", height=380, template="plotly_dark")
        return fig
    close = df["Close"].dropna()
    fig.add_trace(go.Scatter(x=close.index, y=close.values, name=cfg["label"], mode="lines", line=dict(width=2.4)))
    if len(close) >= 50:
        fig.add_trace(go.Scatter(x=close.index, y=close.rolling(50).mean(), name="MA50", mode="lines", line=dict(width=1.5, dash="dot")))
    if len(close) >= 200:
        fig.add_trace(go.Scatter(x=close.index, y=close.rolling(200).mean(), name="MA200", mode="lines", line=dict(width=1.5, dash="dash")))
    fig.update_layout(height=410, margin=dict(l=10, r=10, t=30, b=10), template="plotly_dark", legend=dict(orientation="h"), title="Price / Trend")
    return fig


def make_driver_chart(res: Dict[str, Any]) -> go.Figure:
    df = res["driver_df"].copy()
    fig = px.bar(df, x="driver", y="score", text=df["score"].map(lambda x: f"{x:+.2f}"), title="Company-specific Driver Score")
    fig.update_layout(height=410, margin=dict(l=10, r=10, t=45, b=90), template="plotly_dark", xaxis_tickangle=-35, yaxis_range=[-1, 1])
    return fig


def make_component_chart(res: Dict[str, Any]) -> go.Figure:
    data = pd.DataFrame([
        {"component": "Trend", "value": res["price"].get("trend_scale", 0.0)}, {"component": "PE", "value": res["pe_scale"]},
        {"component": "PEG", "value": res["peg_scale"]}, {"component": "Valuation", "value": res["valuation_scale"]},
        {"component": "Driver", "value": res["driver_scale"]}, {"component": "Macro", "value": res["macro_scale"]}, {"component": "Final", "value": res["buy_scale"]},
    ])
    fig = px.bar(data, x="component", y="value", text=data["value"].map(lambda x: f"{x:.2f}"), title="Buy Scale Components")
    fig.update_layout(height=410, margin=dict(l=10, r=10, t=45, b=50), template="plotly_dark", yaxis_range=[0, 1.25])
    return fig


def make_metric_gauge(res: Dict[str, Any]) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=float(res["driver_score"]), number={"prefix": "", "valueformat": "+.2f"},
        gauge={"axis": {"range": [-1, 1]}, "bar": {"color": action_color(res["action"])}, "steps": [{"range": [-1, -0.45], "color": "rgba(239,68,68,.25)"}, {"range": [-0.45, -0.15], "color": "rgba(245,158,11,.25)"}, {"range": [-0.15, .45], "color": "rgba(59,130,246,.18)"}, {"range": [.45, 1], "color": "rgba(34,197,94,.22)"}]},
        title={"text": "Driver Score"}
    ))
    fig.update_layout(height=300, template="plotly_dark", margin=dict(l=10, r=10, t=40, b=10))
    return fig


def make_network_graph(score_df: pd.DataFrame | None = None) -> go.Figure:
    fig = go.Figure()
    score_map = {}
    if score_df is not None and not score_df.empty:
        score_map = dict(zip(score_df["key"], score_df["network_score"]))
    # edges
    for src, dst, label, w in NETWORK_EDGES:
        x0, y0 = NETWORK_POS[src]; x1, y1 = NETWORK_POS[dst]
        color = "rgba(34,197,94,.55)" if w > 0 else "rgba(239,68,68,.55)"
        fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode="lines", line=dict(width=max(1, abs(w)*4), color=color), hoverinfo="text", text=[label, label], showlegend=False))
    # nodes
    node_x, node_y, text, size, colors = [], [], [], [], []
    for key, cfg in COMPANIES.items():
        x, y = NETWORK_POS.get(key, (0.5, 0.5))
        s = safe_float(score_map.get(key), 0.0)
        node_x.append(x); node_y.append(y)
        text.append(f"{cfg['label']}<br>{cfg['theme']}<br>Network score {s:+.2f}")
        size.append(26 + 22*abs(s))
        colors.append(s)
    fig.add_trace(go.Scatter(x=node_x, y=node_y, mode="markers+text", text=[COMPANIES[k]["label"] for k in COMPANIES], textposition="middle center", marker=dict(size=size, color=colors, colorscale="RdYlGn", cmin=-1, cmax=1, line=dict(width=2, color="white")), hovertext=text, hoverinfo="text", showlegend=False))
    fig.update_layout(template="plotly_dark", height=560, title="AI supply chain relationship map", margin=dict(l=10, r=10, t=45, b=10), xaxis=dict(visible=False, range=[-0.05,1.02]), yaxis=dict(visible=False, range=[0.05,1.02]))
    return fig

# ============================================================
# Styling
# ============================================================
def inject_css():
    st.markdown("""
    <style>
    .block-container { padding-top: 1rem; max-width: 1400px; }
    div[data-testid="stMetric"] { background: rgba(15,23,42,.92); border: 1px solid rgba(148,163,184,.20); padding: 14px; border-radius: 18px; box-shadow: 0 10px 25px rgba(0,0,0,.12); }
    div[data-testid="stMetricValue"] { font-size: 1.45rem; }
    .hero { padding: 22px; border-radius: 24px; border: 1px solid rgba(148,163,184,.22); background: linear-gradient(135deg, rgba(59,130,246,.18), rgba(34,197,94,.08)); margin-bottom: 16px; }
    .hero.hold { background: linear-gradient(135deg, rgba(148,163,184,.16), rgba(59,130,246,.05)); }
    .hero.buy { background: linear-gradient(135deg, rgba(34,197,94,.18), rgba(59,130,246,.08)); }
    .hero .action { font-size: 3rem; font-weight: 950; line-height: 1; letter-spacing: -0.06em; margin: 4px 0; }
    .pill { display: inline-block; padding: 6px 10px; margin: 4px 5px 4px 0; border-radius: 999px; font-size: .82rem; font-weight: 850; border: 1px solid rgba(148,163,184,.25); background: rgba(15,23,42,.72); }
    .pill.green { color: #86efac; border-color: rgba(34,197,94,.4); }
    .pill.red { color: #fca5a5; border-color: rgba(239,68,68,.4); }
    .pill.yellow { color: #fde68a; border-color: rgba(245,158,11,.4); }
    .small-muted { color: #94a3b8; font-size: .88rem; line-height: 1.55; }
    .section-card { padding: 16px; border-radius: 20px; border: 1px solid rgba(148,163,184,.18); background: rgba(15,23,42,.62); }
    div[data-testid="stButton"] > button { border-radius: 14px; font-weight: 850; min-height: 42px; }
    @media (max-width: 760px) { .block-container { padding-left: .8rem; padding-right: .8rem; } .hero .action { font-size: 2.3rem; } div[data-testid="stMetricValue"] { font-size: 1.15rem; } }
    </style>
    """, unsafe_allow_html=True)


def pill(text: str, kind: str = "") -> str:
    cls = "green" if kind == "green" else "red" if kind == "red" else "yellow" if kind == "yellow" else ""
    return f"<span class='pill {cls}'>{text}</span>"

# ============================================================
# Render single company page
# ============================================================
def render_company_page(company_key: str):
    cfg = COMPANIES[company_key]
    st.set_page_config(page_title=f"{cfg['label']} Buy System", page_icon=cfg.get("icon", "📈"), layout="wide", initial_sidebar_state="expanded")
    inject_css()

    st.sidebar.title(f"{cfg['icon']} {cfg['label']} 設定")
    st.sidebar.caption(APP_BUILD_VERSION)
    if st.sidebar.button("🏠 回總控台", use_container_width=True):
        st.switch_page("streamlit_app.py")

    use_alt = False
    if cfg.get("alt_ticker"):
        use_alt = st.sidebar.toggle(f"改用 {cfg['alt_ticker']} ADR", value=False)
    ticker = cfg.get("alt_ticker") if use_alt and cfg.get("alt_ticker") else cfg["ticker"]
    seed = SEED_FUNDAMENTALS.get(ticker, SEED_FUNDAMENTALS.get(cfg["ticker"], {}))
    auto = fetch_auto_fundamentals(ticker)

    with st.sidebar.expander("帳戶 / 持股", expanded=True):
        equity = st.number_input("Equity / 總權益", min_value=0.0, value=30000.0 if cfg["currency"] == "USD" else 1000000.0, step=1000.0, key=f"{company_key}_equity")
        cash = st.number_input("Free Cash / 可用現金", min_value=0.0, value=500.0 if cfg["currency"] == "USD" else 100000.0, step=100.0, key=f"{company_key}_cash")
        shares = st.number_input("Shares / 持股", min_value=0.0, value=0.0, step=1.0, key=f"{company_key}_shares")
        avg_cost = st.number_input("Avg Cost / 均價", min_value=0.0, value=0.0, step=1.0, key=f"{company_key}_avg")
        cash_use_frac = st.slider("單次最多使用現金比例", 0.0, 1.0, 1.0, 0.05, key=f"{company_key}_cash_frac")

    with st.sidebar.expander("估值 / 財務資料覆蓋 V5", expanded=True):
        st.caption("手動非 0 > yfinance AUTO > V5 SEED > DERIVED。來源會在資料頁顯示。")
        manual_fin = {}
        labels = [("forward_pe", "Forward PE"), ("trailing_pe", "Trailing PE"), ("forward_eps", "Forward EPS"), ("trailing_eps", "Trailing EPS"), ("gross_margin", "Gross Margin %"), ("operating_margin", "Operating Margin %"), ("fcf_margin", "FCF Margin %"), ("growth", "Normalized Growth, fraction")]
        for field, label in labels:
            default_val = safe_float(auto.get(field), safe_float(seed.get(field), 0.0))
            manual_fin[field] = st.number_input(f"{label} override", value=0.0, step=0.1 if field != "growth" else 0.01, format="%.4f", help=f"fallback 參考值：{default_val:.4f}；留 0 = 不覆蓋", key=f"{company_key}_{field}")

    with st.sidebar.expander("財報關鍵數值 / 觀察節點", expanded=True):
        metric_values = {}
        for name, default in cfg.get("key_metrics", []):
            step = 0.05 if abs(default) <= 1.0 else 1.0
            metric_values[name] = st.number_input(name, value=float(default), step=step, format="%.4f", key=f"{company_key}_metric_{name}")

    with st.sidebar.expander("核心 Driver 分數", expanded=True):
        st.caption("+1 順風，0 中性，-1 逆風。這是每家公司不同的股價驅動權重。")
        driver_scores = {}
        for name in cfg["drivers"]:
            default = float(cfg.get("default_driver_scores", {}).get(name, 0.0))
            driver_scores[name] = st.slider(name, -1.0, 1.0, default, 0.05, key=f"{company_key}_driver_{name}")

    res = compute_analysis(company_key, {"equity": equity, "cash": cash, "shares": shares, "avg_cost": avg_cost, "cash_use_frac": cash_use_frac}, manual_fin, driver_scores, use_alt_ticker=use_alt)
    action_cls = "buy" if res["action"] == "BUY" else "hold"
    badge_kind = "green" if res["action"] == "BUY" else "yellow"
    st.markdown(f"""
    <div class='hero {action_cls}'>
      <div class='small-muted'>{cfg['theme']}</div>
      <div class='action'>{res['action']}</div>
      <div>{pill('Ticker: ' + ticker)} {pill('Driver ' + res['driver_state'], 'green' if res['driver_state'] in ['STRONG','STABLE'] else 'red')} {pill('Trend ' + res['price'].get('trend_state','NA'))} {pill('Macro ' + res['macro']['regime'])} {pill('V5 data source safe', 'green')}</div>
      <div class='small-muted'>{cfg['role']}</div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"{cfg['label']} Price", fmt_money(res['price'].get('last_price'), cfg['currency']), f"DD120 {fmt_pct_fraction(res['price'].get('dd_120'))}")
    c2.metric("Est Buy", fmt_money(res['buy_budget'], cfg['currency'], 0), f"{res['est_shares']} shares" if not pd.isna(res['est_shares']) else "shares NA")
    c3.metric("Driver Score", fmt_num(res['driver_score']), res['driver_state'])
    c4.metric("Final Buy Scale", fmt_num(res['buy_scale']), f"PE {res['pe_state']} / PEG {res['peg_state']}")
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Selected PE", fmt_num(res['selected_pe']), res['fundamentals']['source_by_field'].get('forward_pe'))
    c6.metric("AI / Cycle PEG", fmt_num(res['peg']), f"Growth {fmt_pct_fraction(res['effective_growth'])}")
    c7.metric("Margin / FCF", fmt_pct_percent(res['fundamentals'].get('fcf_margin')), f"GM {fmt_pct_percent(res['fundamentals'].get('gross_margin'))}")
    c8.metric("Macro", res['macro']['regime'], f"Risk {res['macro']['risk']:+.2f}")

    tabs = st.tabs(["決策總覽", "公司專屬 Driver", "估值 / 資料來源", "價格 / 風險", "網絡位置", "JSON / 匯出"])
    with tabs[0]:
        l, r = st.columns([1.15, .85])
        with l:
            st.plotly_chart(make_price_chart(res), use_container_width=True)
        with r:
            st.plotly_chart(make_component_chart(res), use_container_width=True)
        st.subheader("決策原因")
        for x in res["reasons"]:
            st.write("- " + x)
        st.info("這不是自動下單系統，只輸出 BUY / HOLD 與分批買進估算。")
    with tabs[1]:
        l, r = st.columns([1.0, .82])
        with l:
            st.plotly_chart(make_driver_chart(res), use_container_width=True)
            st.dataframe(res["driver_df"], use_container_width=True, hide_index=True)
        with r:
            st.plotly_chart(make_metric_gauge(res), use_container_width=True)
            st.subheader("財報關鍵數值")
            metric_df = pd.DataFrame([{"metric": k, "value": v} for k, v in metric_values.items()])
            st.dataframe(metric_df, use_container_width=True, hide_index=True)
    with tabs[2]:
        f = res["fundamentals"]
        fields_table = pd.DataFrame([{"field": k, "value": f.get(k), "source": f.get("source_by_field", {}).get(k)} for k in ["forward_pe", "trailing_pe", "forward_eps", "trailing_eps", "gross_margin", "operating_margin", "fcf_margin", "growth"]])
        st.dataframe(fields_table, use_container_width=True, hide_index=True)
        st.caption("AUTO = yfinance 成功；SEED = V5 內建 fallback；MANUAL = 你手動覆蓋；DERIVED = 用價格/PE 反推。")
        st.write("Auto data ok:", f.get("auto_ok"), "Auto error:", f.get("auto_error", ""))
    with tabs[3]:
        risk_df = pd.DataFrame([
            {"risk_item": "ATR %", "value": res['price'].get('atr_pct')}, {"risk_item": "Sigma 60 annualized", "value": res['price'].get('sigma_60')},
            {"risk_item": "DD 120", "value": res['price'].get('dd_120')}, {"risk_item": "Macro risk", "value": res['macro'].get('risk')},
        ])
        st.dataframe(risk_df, use_container_width=True, hide_index=True)
        st.subheader("關鍵監控節點")
        for w in cfg.get("watch", []):
            st.write("- " + w)
        st.subheader("Macro proxy")
        for n in res["macro"].get("notes", []):
            st.write("- " + n)
    with tabs[4]:
        st.plotly_chart(make_network_graph(), use_container_width=True)
        related = []
        for src, dst, rel, w in NETWORK_EDGES:
            if src == company_key or dst == company_key:
                related.append({"from": COMPANIES[src]["label"], "to": COMPANIES[dst]["label"], "relationship": rel, "weight": w})
        st.dataframe(pd.DataFrame(related), use_container_width=True, hide_index=True)
    with tabs[5]:
        export = dict(res)
        export["manual_key_metrics"] = metric_values
        safe = safe_json_dumps(export)
        st.download_button("下載 JSON", data=safe, file_name=f"{company_key}_analysis_v5.json", mime="application/json", use_container_width=True)
        st.download_button("下載 Driver CSV", data=res["driver_df"].to_csv(index=False).encode("utf-8-sig"), file_name=f"{company_key}_drivers.csv", mime="text/csv", use_container_width=True)
        st.code(safe[:10000], language="json")

# ============================================================
# Network Home
# ============================================================
def compute_scenario_scores(scenario: Dict[str, float]) -> pd.DataFrame:
    vals = [scenario[k] for k in SCENARIO_KEYS]
    rows = []
    for k, exp in SCENARIO_EXPOSURES.items():
        raw = sum(e*v for e, v in zip(exp, vals)) / max(1e-9, sum(abs(e) for e in exp))
        cfg = COMPANIES[k]
        rows.append({"key": k, "ticker": cfg["label"], "layer": cfg["layer"], "role": cfg["theme"], "network_score": raw, "decision": "FAVOR" if raw > .25 else "WATCH" if raw > -.15 else "PRESSURE"})
    return pd.DataFrame(rows).sort_values("network_score", ascending=False)


def render_network_home():
    st.set_page_config(page_title="AI Equity V5 Fusion Terminal", page_icon="🧬", layout="wide", initial_sidebar_state="expanded")
    inject_css()
    st.sidebar.title("AI Equity V5")
    st.sidebar.caption(APP_BUILD_VERSION)
    st.sidebar.write("V5 = V4 資料穩定性 + V3 網絡總控 + 單股頁精緻細節。")

    st.title("🧬 AI Equity Network Control Tower")
    st.caption("總控台看 AI 鏈互相影響；單股頁保留細節、估值 fallback、手動覆蓋與下載功能。")

    st.subheader("快速打開單股頁面")
    keys = list(COMPANIES.keys())
    for row in [keys[i:i+5] for i in range(0, len(keys), 5)]:
        cols = st.columns(len(row))
        for col, k in zip(cols, row):
            cfg = COMPANIES[k]
            if col.button(f"{cfg['icon']} {cfg['label']}", use_container_width=True):
                try:
                    st.switch_page(cfg["page"])
                except Exception as e:
                    st.error(f"跳頁失敗：{e}")

    st.sidebar.subheader("情境模擬")
    defaults = {"AI CapEx": 0.40, "GPU Demand": 0.45, "ASIC Shift": 0.30, "Cloud Monetization": 0.25, "HBM / Memory": 0.55, "Search / Ads AI": 0.20, "Macro Liquidity": 0.0, "Export Risk Ease": -0.10}
    scenario = {}
    for k in SCENARIO_KEYS:
        scenario[k] = st.sidebar.slider(k, -1.0, 1.0, defaults[k], 0.05)
    score_df = compute_scenario_scores(scenario)

    tab1, tab2, tab3, tab4 = st.tabs(["網絡圖", "排名 / 情境", "曝險圓餅", "關係表 / 匯出"])
    with tab1:
        st.plotly_chart(make_network_graph(score_df), use_container_width=True)
    with tab2:
        c1, c2 = st.columns([1.0, 1.0])
        with c1:
            st.subheader("Network Ranking")
            st.dataframe(score_df, use_container_width=True, hide_index=True)
        with c2:
            fig = px.bar(score_df, x="ticker", y="network_score", color="decision", text=score_df["network_score"].map(lambda x: f"{x:+.2f}"), title="Scenario Impact")
            fig.update_layout(template="plotly_dark", height=460, yaxis_range=[-1, 1], xaxis_tickangle=-25)
            st.plotly_chart(fig, use_container_width=True)
    with tab3:
        c1, c2 = st.columns([.95, 1.05])
        with c1:
            layer_df = score_df.groupby("layer", as_index=False)["network_score"].apply(lambda s: float(np.maximum(s, 0).sum()))
            if layer_df["network_score"].sum() <= 0:
                layer_df["network_score"] = 1
            fig = px.pie(layer_df, names="layer", values="network_score", title="Positive scenario exposure by AI layer")
            fig.update_layout(template="plotly_dark", height=430)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            count_df = pd.DataFrame([{"layer": cfg["layer"], "count": 1} for cfg in COMPANIES.values()]).groupby("layer", as_index=False).sum()
            fig = px.pie(count_df, names="layer", values="count", title="Universe composition by role")
            fig.update_layout(template="plotly_dark", height=430)
            st.plotly_chart(fig, use_container_width=True)
    with tab4:
        edge_df = pd.DataFrame(NETWORK_EDGES, columns=["from", "to", "relationship", "weight"])
        edge_df["from"] = edge_df["from"].map(lambda k: COMPANIES[k]["label"])
        edge_df["to"] = edge_df["to"].map(lambda k: COMPANIES[k]["label"])
        st.dataframe(edge_df, use_container_width=True, hide_index=True)
        st.download_button("下載 Network Scores CSV", data=score_df.to_csv(index=False).encode("utf-8-sig"), file_name="ai_network_scores_v5.csv", mime="text/csv", use_container_width=True)
        st.download_button("下載 Network JSON", data=safe_json_dumps({"scenario": scenario, "scores": score_df.to_dict(orient="records"), "edges": edge_df.to_dict(orient="records")}), file_name="ai_network_v5.json", mime="application/json", use_container_width=True)

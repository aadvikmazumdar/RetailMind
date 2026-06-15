# RetailMind
### AI-powered retail intelligence for dynamic pricing, hyperlocal assortment, and margin management.

---

> *Mid-tier retail chains lose margin every week to three decisions they make manually — when to discount, what to stock, and whether a product is actually profitable after operational costs. RetailMind automates all three.*

---

## The Problem

Enterprise solutions like Blue Yonder exist but cost millions and are built for Walmart-scale operations. Smaller chains use Excel and intuition.

The result:

- Products marked down too late become deadstock
- Every branch stocks the same inventory regardless of local demand
- Products that look profitable on paper silently erode margin after rent, electricity, and staffing

---

## What RetailMind Does

### 🏷️ Dynamic Pricing Engine
Predicts weekly demand per product per store, simulates revenue at candidate price points, and recommends the optimal price within margin constraints. Markdown urgency is flagged based on demand trajectory and peak seasonality.

### 🗺️ Hyperlocal Assortment Optimization
Models the store network as a heterogeneous graph. Identifies which stores serve genuinely different customer bases and which unnecessarily duplicate inventory. Outputs differentiated assortment recommendations per store — not chain-wide planograms.

### 📊 Margin Health Monitor
Computes true contribution margin per product after allocating operational costs using NRF retail benchmarks. Flags products priced below viable margin before they silently erode profitability.

### 🤖 Multi-Agent Orchestration
A LangGraph-based agent layer routes natural language queries from store managers to specialist agents, synthesizes their outputs, and returns structured recommendations.

> *"Why is my Koramangala store losing to DMart?"*
> → Pricing Agent + Assortment Agent called in parallel → synthesized recommendation returned.

---

## System Architecture

```
M5 Walmart Transaction Data
            ↓
   ETL Pipeline (DVC versioned)
            ↓
     Feature Store (Parquet)
            ↓
┌───────────────┬────────────────┬──────────────────┐
│ Pricing       │ Assortment     │ Margin           │
│ Engine        │ GNN            │ Monitor          │
│ (LightGBM)    │ (PyTorch Geo)  │ (Cost Model)     │
└───────────────┴────────────────┴──────────────────┘
            ↓
  Multi-Agent Layer (LangGraph)
            ↓
  FastAPI  +  Plotly Dash Dashboard
```

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data & ETL | Python · Pandas · PyArrow · DVC |
| ML Models | LightGBM · PyTorch · PyTorch Geometric |
| Agent Layer | LangGraph · LangChain |
| API | FastAPI · Uvicorn |
| Dashboard | Plotly Dash |
| Experiment Tracking | MLflow |

---

## Build Progress

| Phase | Description | Status |
|---|---|---|
| 0 | ETL Pipeline & Feature Store | ✅ Complete |
| 1 | Demand Forecasting Model | ✅ Complete |
| 2 | Hyperlocal Assortment GNN | 🔄 In Progress |
| 3 | Margin Optimization | ⏳ Pending |
| 4 | Multi-Agent Layer | ⏳ Pending |
| 5 | FastAPI + Dashboard | ⏳ Pending |

---

## Dataset

Built and validated on the [M5 Walmart Forecasting Competition](https://www.kaggle.com/competitions/m5-forecasting-accuracy) dataset — 3,049 products across 10 stores over 5 years of daily sales.

The architecture is **dataset-agnostic**. Any retailer can replace M5 with their own transaction data without modifying the pipeline.

> **Note on elasticity:** M5 reflects Walmart's Every Day Low Prices strategy — price changes occur in under 2% of observations. Elasticity estimates are therefore directional rather than precise. A promotional retail dataset would yield stronger price sensitivity signal in a production deployment.

---

## Setup

```bash
git clone https://github.com/yourusername/RetailMind.git
cd RetailMind
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Download M5 data from Kaggle and place CSVs at `data/raw/m5/`.

```bash
# Phase 0 — ETL
python src/etl/etl_m5.py
python src/etl/etl_model_features.py

# Phase 1 — Demand Model
python src/models/demand_model.py
```

---

## Limitations & Roadmap

**Current limitations:**
- Elasticity signal is limited by Walmart's stable pricing strategy. A v2 using a promotional retailer dataset would significantly improve the pricing component.
- Margin benchmarks use NRF industry averages. Production deployment would ingest actual store P&L data.
- Assortment GNN currently trained on transaction patterns only. Adding demographic catchment data would improve hyperlocal differentiation.

**Planned:**
- SNAP flag integration for demand sensitivity modeling
- Per-category LightGBM models for higher-volume product groups
- Real-time inference pipeline with model monitoring

---

## Author

**Aadvik Mazumdar**
B.Tech CSE (AI-ML) · SRM Institute of Science and Technology
Research Intern · IIT Roorkee

[LinkedIn](https://linkedin.com/in/aadvikmazumdar) · [GitHub](https://github.com/aadvikmazumdar)

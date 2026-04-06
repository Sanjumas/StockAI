# StockAI — AI Inventory Management System

No Docker, no API keys, no demo data. You bring the CSVs — the model predicts.

## How to Run

### Windows
```
Double-click run.bat
```

### Mac / Linux
```bash
bash run.sh
```

### Manual (any OS)
```bash
pip install -r requirements.txt
python backend/app.py
```

Then open **http://localhost:5000** in your browser.

---

## Workflow (4 steps)

### Step 1 — Upload Inventory CSV
Your current stock. Required columns:
| Column | Notes |
|--------|-------|
| `SKU` | Unique product ID |
| `Product` or `Name` | Product name |
| `Stock` | Current units in stock |
| `Category` | *(optional)* |
| `Price` | *(optional)* |
| `Threshold` or `Reorder_Threshold` | *(optional, defaults to Stock/5)* |

Example (`inventory.csv`):
```
SKU,Product,Stock,Category,Price
SKU101,Laptop,25,Electronics,899.99
SKU204,Wireless Mouse,60,Electronics,29.99
```

### Step 2 — Upload Sales History CSV
Past sales the model learns from. Required columns:
| Column | Notes |
|--------|-------|
| `sku` or `product_id` | Must match inventory SKU |
| `quantity` or `units_sold` | Units sold per row |
| `date` or `sold_at` | *(optional)* |

Accepts: `fake_sales_data.csv`, `test_sales.csv`, or any CSV in this format.

> Sales records are only imported for SKUs that exist in your inventory.  
> Upload inventory first, then sales.

### Step 3 — Train the Model
Choose one of 4 algorithms:
- **Random Forest** — best general-purpose, handles non-linear patterns
- **Gradient Boosting** — highest accuracy on structured data
- **Linear Regression** — fast, interpretable, good for stable demand
- **Ridge Regression** — linear with regularisation, avoids overfitting

### Step 4 — View Predictions
The model predicts **7-day demand** for every SKU in your current inventory.
- Items where `predicted_demand > current_stock` are flagged for reorder
- Reorder quantity = `predicted_demand - current_stock + threshold`
- Days until stockout = `current_stock / avg_daily_sales`

---

## File Structure
```
inventory_ai/
├── backend/
│   ├── app.py          ← Flask server (run this)
│   ├── database.py     ← SQLite layer
│   ├── ml_engine.py    ← All ML logic (scikit-learn)
│   └── models/         ← Saved trained models (.pkl)
├── frontend/
│   └── index.html      ← Single-file React-free UI
├── requirements.txt
├── run.bat             ← Windows launcher
├── run.sh              ← Mac/Linux launcher
└── README.md
```

## ML Features Engineered
For each SKU, the engine computes from sales history:
- `avg_daily_7d` — 7-day rolling average
- `avg_daily_14d` — 14-day rolling average
- `avg_daily_30d` — 30-day rolling average
- `avg_daily_overall` — all-time average
- `max_daily` — peak daily sales
- `std_daily` — variability
- `trend` — slope (recent vs earlier periods)
- `acceleration` — 7d avg vs 14d avg
- `data_days` — how many days of data exist

These 9 features are fed into the chosen sklearn model to predict total 7-day demand per SKU.
"# StockAI" 

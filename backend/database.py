"""
database.py — SQLite persistence layer.
No demo data, no seeding. All data comes from user-uploaded CSVs.
"""

import sqlite3
import os
import io
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "inventory.db")


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    with get_conn() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sku TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            category TEXT DEFAULT 'General',
            current_stock INTEGER DEFAULT 0,
            reorder_threshold INTEGER DEFAULT 20,
            price REAL DEFAULT 0.0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS sales (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sku TEXT NOT NULL,
            quantity INTEGER NOT NULL,
            sold_at TEXT NOT NULL,
            source TEXT DEFAULT 'upload'
        );
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sku TEXT NOT NULL,
            message TEXT NOT NULL,
            severity TEXT DEFAULT 'warning',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            resolved INTEGER DEFAULT 0
        );
        """)


# ── Items ─────────────────────────────────────

def get_all_items():
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM items ORDER BY name").fetchall()
        return [dict(r) for r in rows]


def get_item(sku):
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM items WHERE sku=?", (sku,)).fetchone()
        return dict(row) if row else None


def upsert_item(sku, name, category="General",
                current_stock=0, reorder_threshold=20, price=0.0):
    with get_conn() as conn:
        conn.execute("""
            INSERT INTO items (sku, name, category, current_stock, reorder_threshold, price, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(sku) DO UPDATE SET
                name=excluded.name, category=excluded.category,
                current_stock=excluded.current_stock,
                reorder_threshold=excluded.reorder_threshold,
                price=excluded.price, updated_at=excluded.updated_at
        """, (sku, name, category, current_stock, reorder_threshold, price,
              datetime.utcnow().isoformat()))
        conn.commit()
    return get_item(sku)


def delete_item(sku):
    with get_conn() as conn:
        conn.execute("DELETE FROM items WHERE sku=?", (sku,))
        conn.execute("DELETE FROM sales WHERE sku=?", (sku,))
        conn.commit()


def clear_all_items():
    with get_conn() as conn:
        conn.execute("DELETE FROM items")
        conn.commit()


# ── Sales ─────────────────────────────────────

def get_sales_df():
    import pandas as pd
    with get_conn() as conn:
        rows = conn.execute("SELECT sku, quantity, sold_at FROM sales").fetchall()
    if not rows:
        return pd.DataFrame(columns=["sku", "quantity", "sold_at"])
    return pd.DataFrame([dict(r) for r in rows])


def get_recent_sales(limit=100):
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT s.sku, i.name, s.quantity, s.sold_at, s.source
            FROM sales s LEFT JOIN items i ON s.sku = i.sku
            ORDER BY s.sold_at DESC LIMIT ?
        """, (limit,)).fetchall()
        return [dict(r) for r in rows]


def clear_all_sales():
    with get_conn() as conn:
        conn.execute("DELETE FROM sales")
        conn.commit()


def import_inventory_csv(file_bytes, filename, replace=False):
    import pandas as pd
    try:
        df = pd.read_csv(io.BytesIO(file_bytes)) if filename.lower().endswith(".csv") \
             else pd.read_excel(io.BytesIO(file_bytes))
    except Exception as e:
        return {"imported": 0, "errors": [str(e)]}

    df.columns = [c.strip().lower() for c in df.columns]
    col = {}
    for c in df.columns:
        if c == "sku":                                         col["sku"] = c
        if c in ("product", "name", "item", "item_name"):     col["name"] = c
        if c in ("stock", "current_stock", "quantity", "qty"): col["stock"] = c
        if c == "category":                                    col["category"] = c
        if c == "price":                                       col["price"] = c
        if c in ("reorder_threshold", "threshold", "reorder"): col["threshold"] = c

    if "sku" not in col:
        return {"imported": 0, "errors": ["No 'SKU' column found. Got: " + str(list(df.columns))]}
    if "stock" not in col:
        return {"imported": 0, "errors": ["No stock column found. Got: " + str(list(df.columns))]}

    if replace:
        clear_all_items()

    imported, errors = 0, []
    for _, row in df.iterrows():
        try:
            sku = str(row[col["sku"]]).strip()
            if not sku or sku.lower() == "nan":
                continue
            name = str(row[col["name"]]).strip() if "name" in col else sku
            stock = int(float(row[col["stock"]]))
            category = str(row[col["category"]]).strip() if "category" in col else "General"
            price = float(row[col["price"]]) if "price" in col else 0.0
            threshold = int(float(row[col["threshold"]])) if "threshold" in col else max(5, stock // 5)
            upsert_item(sku, name, category, stock, threshold, price)
            imported += 1
        except Exception as e:
            errors.append(str(e))
    return {"imported": imported, "errors": errors[:5]}


def import_sales_csv(file_bytes, filename, replace=False):
    import pandas as pd
    try:
        df = pd.read_csv(io.BytesIO(file_bytes)) if filename.lower().endswith(".csv") \
             else pd.read_excel(io.BytesIO(file_bytes))
    except Exception as e:
        return {"imported": 0, "skipped": 0, "errors": [str(e)]}

    df.columns = [c.strip().lower() for c in df.columns]

    # Normalise to: sku, quantity, sold_at
    if "product_id" in df.columns and "sku" not in df.columns:
        df.rename(columns={"product_id": "sku"}, inplace=True)
    if "units_sold" in df.columns and "quantity" not in df.columns:
        df.rename(columns={"units_sold": "quantity"}, inplace=True)
    if "date" in df.columns and "sold_at" not in df.columns:
        df.rename(columns={"date": "sold_at"}, inplace=True)

    missing = {"sku", "quantity"} - set(df.columns)
    if missing:
        return {"imported": 0, "skipped": 0,
                "errors": [f"Missing columns: {missing}. Got: {list(df.columns)}"]}

    if "sold_at" not in df.columns:
        df["sold_at"] = datetime.utcnow().isoformat()

    if replace:
        clear_all_sales()

    known_skus = {r["sku"] for r in get_all_items()}
    imported, skipped, errors = 0, 0, []

    with get_conn() as conn:
        for _, row in df.iterrows():
            try:
                sku = str(row["sku"]).strip()
                qty = int(float(row["quantity"]))
                if qty <= 0:
                    skipped += 1
                    continue
                # Only import sales for inventory SKUs (if inventory was uploaded)
                if known_skus and sku not in known_skus:
                    skipped += 1
                    continue
                sold_at = str(row["sold_at"]).strip()
                conn.execute(
                    "INSERT INTO sales (sku, quantity, sold_at, source) VALUES (?,?,?,?)",
                    (sku, qty, sold_at, "upload")
                )
                imported += 1
            except Exception as e:
                errors.append(str(e))
                skipped += 1
        conn.commit()

    return {"imported": imported, "skipped": skipped, "errors": errors[:5]}


def get_sales_count():
    with get_conn() as conn:
        return conn.execute("SELECT COUNT(*) FROM sales").fetchone()[0]


def get_items_count():
    with get_conn() as conn:
        return conn.execute("SELECT COUNT(*) FROM items").fetchone()[0]


# ── Alerts ────────────────────────────────────

def create_alert(sku, message, severity="warning"):
    with get_conn() as conn:
        conn.execute("INSERT INTO alerts (sku, message, severity) VALUES (?,?,?)",
                     (sku, message, severity))
        conn.commit()


def get_active_alerts():
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM alerts WHERE resolved=0 ORDER BY created_at DESC LIMIT 100"
        ).fetchall()
        return [dict(r) for r in rows]


def resolve_alert(alert_id):
    with get_conn() as conn:
        conn.execute("UPDATE alerts SET resolved=1 WHERE id=?", (alert_id,))
        conn.commit()


def clear_all_alerts():
    with get_conn() as conn:
        conn.execute("DELETE FROM alerts")
        conn.commit()

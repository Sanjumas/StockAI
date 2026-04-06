"""
app.py — Flask backend. Serves the frontend and all API routes.
Run: python backend/app.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, jsonify, request, send_from_directory
import database as db
import ml_engine as ml
import pandas as pd

app = Flask(__name__, static_folder="../frontend", static_url_path="")

try:
    from flask_cors import CORS; CORS(app)
except ImportError:
    pass

db.init_db()


# ── Frontend ──────────────────────────────────

@app.route("/")
def serve_index():
    return send_from_directory(app.static_folder, "index.html")


# ── Status: what data has been uploaded ───────

@app.route("/api/status")
def status():
    items_count = db.get_items_count()
    sales_count = db.get_sales_count()
    model_info  = ml.get_model_info()
    return jsonify({
        "has_inventory": items_count > 0,
        "has_sales":     sales_count > 0,
        "inventory_count": items_count,
        "sales_count":   sales_count,
        "trained_models": list(model_info.keys()),
    })


# ── Inventory ─────────────────────────────────

@app.route("/api/items")
def list_items():
    return jsonify(db.get_all_items())


@app.route("/api/items", methods=["POST"])
def create_item():
    d = request.json or {}
    if not d.get("sku") or not d.get("name"):
        return jsonify({"error": "sku and name required"}), 400
    item = db.upsert_item(
        sku=d["sku"], name=d["name"],
        category=d.get("category", "General"),
        current_stock=int(d.get("current_stock", 0)),
        reorder_threshold=int(d.get("reorder_threshold", 20)),
        price=float(d.get("price", 0)),
    )
    return jsonify(item), 201


@app.route("/api/items/<sku>", methods=["PUT"])
def update_item(sku):
    existing = db.get_item(sku)
    if not existing:
        return jsonify({"error": "Not found"}), 404
    d = request.json or {}
    item = db.upsert_item(
        sku=sku,
        name=d.get("name", existing["name"]),
        category=d.get("category", existing["category"]),
        current_stock=int(d.get("current_stock", existing["current_stock"])),
        reorder_threshold=int(d.get("reorder_threshold", existing["reorder_threshold"])),
        price=float(d.get("price", existing["price"])),
    )
    return jsonify(item)


@app.route("/api/items/<sku>", methods=["DELETE"])
def delete_item(sku):
    db.delete_item(sku)
    return jsonify({"deleted": sku})


@app.route("/api/inventory/upload", methods=["POST"])
def upload_inventory():
    f = request.files.get("file")
    if not f:
        return jsonify({"error": "No file"}), 400
    replace = request.form.get("replace", "false").lower() == "true"
    result = db.import_inventory_csv(f.read(), f.filename, replace=replace)
    return jsonify(result)


@app.route("/api/inventory/clear", methods=["POST"])
def clear_inventory():
    db.clear_all_items()
    db.clear_all_sales()
    db.clear_all_alerts()
    return jsonify({"ok": True})


# ── Sales ─────────────────────────────────────

@app.route("/api/sales/recent")
def recent_sales():
    limit = int(request.args.get("limit", 100))
    return jsonify(db.get_recent_sales(limit))


@app.route("/api/sales/upload", methods=["POST"])
def upload_sales():
    f = request.files.get("file")
    if not f:
        return jsonify({"error": "No file"}), 400
    replace = request.form.get("replace", "false").lower() == "true"
    result = db.import_sales_csv(f.read(), f.filename, replace=replace)
    return jsonify(result)


@app.route("/api/sales/clear", methods=["POST"])
def clear_sales():
    db.clear_all_sales()
    return jsonify({"ok": True})


# ── ML: Train ─────────────────────────────────

@app.route("/api/train", methods=["POST"])
def train():
    d = request.json or {}
    model_type = d.get("model_type", "random_forest")
    sales_df = db.get_sales_df()
    if sales_df.empty:
        return jsonify({"error": "No sales data. Upload a sales CSV first."}), 400
    try:
        metrics = ml.train_model(sales_df, model_type)
        return jsonify({"ok": True, **metrics})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Training failed: {e}"}), 500


# ── ML: Predict on current inventory ──────────

@app.route("/api/predict")
def predict():
    model_type = request.args.get("model", "random_forest")
    items = db.get_all_items()
    if not items:
        return jsonify({"error": "No inventory. Upload inventory CSV first."}), 400

    sales_df = db.get_sales_df()
    inv_df = pd.DataFrame(items)

    predictions = ml.predict_demand(sales_df, inv_df, model_type)

    # Auto-create alerts
    existing_alert_skus = {a["sku"] for a in db.get_active_alerts()}
    for p in predictions:
        if p["alert"] and p["sku"] not in existing_alert_skus:
            db.create_alert(
                p["sku"],
                f"{p['name']}: stock {p['current_stock']} < predicted 7-day demand {p['predicted_demand_7d']}",
                severity="critical" if p["current_stock"] == 0 else "warning"
            )

    return jsonify(predictions)


@app.route("/api/model/info")
def model_info():
    return jsonify(ml.get_model_info())


# ── Analytics ─────────────────────────────────

@app.route("/api/analytics/summary")
def summary():
    items = db.get_all_items()
    sales_df = db.get_sales_df()
    low = [i for i in items if i["current_stock"] <= i["reorder_threshold"]]
    return jsonify({
        "total_items": len(items),
        "low_stock_count": len(low),
        "out_of_stock_count": sum(1 for i in items if i["current_stock"] == 0),
        "total_stock_value": round(sum(i["current_stock"] * i["price"] for i in items), 2),
        "total_sales_records": len(sales_df),
    })


@app.route("/api/analytics/trends")
def trends():
    days = int(request.args.get("days", 30))
    sales_df = db.get_sales_df()
    return jsonify(ml.get_sales_trends(sales_df, days))


# ── Alerts ────────────────────────────────────

@app.route("/api/alerts")
def alerts():
    return jsonify(db.get_active_alerts())


@app.route("/api/alerts/<int:aid>/resolve", methods=["POST"])
def resolve(aid):
    db.resolve_alert(aid)
    return jsonify({"resolved": aid})


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    print("\n  AI Inventory System")
    print("  Open: http://localhost:5000\n")
    app.run(debug=True, port=5000, host="0.0.0.0")

#!/bin/bash
echo "============================================"
echo " StockAI — AI Inventory System"
echo "============================================"
echo
echo "[1/2] Installing dependencies..."
pip install -r requirements.txt --quiet
echo "[OK] Dependencies installed"
echo
echo "[2/2] Starting server..."
echo "Open: http://localhost:5000"
echo "Ctrl+C to stop"
echo
python3 backend/app.py

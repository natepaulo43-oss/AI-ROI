#!/bin/bash

# AI ROI Web Tool - Stop Script
# Stops both backend and frontend servers

echo "============================================"
echo "AI ROI Prediction Tool - Stopping Services"
echo "============================================"
echo ""

# Find and kill backend process (uvicorn)
echo "Stopping Backend API..."
BACKEND_PIDS=$(pgrep -f "uvicorn app.main:app")
if [ -n "$BACKEND_PIDS" ]; then
    kill $BACKEND_PIDS 2>/dev/null
    echo "✓ Backend stopped (PIDs: $BACKEND_PIDS)"
else
    echo "  Backend not running"
fi

# Find and kill frontend process (next dev)
echo "Stopping Frontend..."
FRONTEND_PIDS=$(pgrep -f "next dev")
if [ -n "$FRONTEND_PIDS" ]; then
    kill $FRONTEND_PIDS 2>/dev/null
    echo "✓ Frontend stopped (PIDs: $FRONTEND_PIDS)"
else
    echo "  Frontend not running"
fi

# Also kill any node processes related to the project
NODE_PIDS=$(pgrep -f "node.*AI_ROI")
if [ -n "$NODE_PIDS" ]; then
    kill $NODE_PIDS 2>/dev/null
    echo "✓ Node processes stopped"
fi

echo ""
echo "============================================"
echo "✓ All Services Stopped"
echo "============================================"
echo ""

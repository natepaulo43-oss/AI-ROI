#!/bin/bash

# AI ROI Web Tool - Start Script
# Starts both backend and frontend servers

echo "============================================"
echo "AI ROI Prediction Tool - Starting Services"
echo "============================================"
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "Shutting down services..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    exit 0
}

trap cleanup SIGINT SIGTERM

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "ERROR: Virtual environment not found!"
    echo "Please create it first with: python -m venv .venv"
    exit 1
fi

# Check if models exist
if [ ! -f "backend/models/roi_model.pkl" ] || [ ! -f "backend/models/roi_classifier_best.pkl" ]; then
    echo "WARNING: Model files not found!"
    echo "Training models now..."
    source .venv/bin/activate
    python backend/train_roi_model.py
    echo ""
fi

# Start Backend
echo "Starting Backend API..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
source .venv/bin/activate
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 > ../backend.log 2>&1 &
BACKEND_PID=$!
cd ..

# Wait for backend to start
echo "Waiting for backend to initialize..."
sleep 3

# Check if backend started successfully
if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "ERROR: Backend failed to start. Check backend.log for details."
    kill $BACKEND_PID 2>/dev/null
    exit 1
fi

echo "✓ Backend running on http://localhost:8000"
echo ""

# Start Frontend
echo "Starting Frontend..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
cd frontend

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "Installing frontend dependencies..."
    npm install
fi

npm run dev > ../frontend.log 2>&1 &
FRONTEND_PID=$!
cd ..

# Wait for frontend to start
echo "Waiting for frontend to initialize..."
sleep 5

echo ""
echo "============================================"
echo "✓ All Services Running!"
echo "============================================"
echo ""
echo "Backend API:  http://localhost:8000"
echo "API Docs:     http://localhost:8000/docs"
echo "Frontend:     http://localhost:3000"
echo "Tool Page:    http://localhost:3000/tool"
echo ""
echo "Logs:"
echo "  Backend:    backend.log"
echo "  Frontend:   frontend.log"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Keep script running and wait for both processes
wait $BACKEND_PID $FRONTEND_PID

# AI ROI Project

Web deployment and research project for AI ROI prediction.

## Project Structure

```
ai-roi-project/
├── frontend/          # Next.js frontend
├── backend/           # FastAPI backend
├── data/              # Raw and processed data
├── training/          # Model training scripts
├── notebooks/         # Jupyter notebooks for research
└── README.md
```

## Setup

### Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

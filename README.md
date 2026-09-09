# AI ROI Prediction Tool

A full-stack ML tool that predicts whether an AI deployment project will hit high ROI, trained on real-world SME AI adoption case studies.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![Next.js](https://img.shields.io/badge/next.js-14+-black.svg)

## Why I built it

Most "AI ROI" claims are vibes, not numbers. This tool grounds the question in an actual dataset — 514 real AI deployment projects — and turns "will this AI initiative pay off" into a binary classification problem with a measurable, reported accuracy, instead of a consultant's slide.

## Tech Stack

Next.js + TailwindCSS (frontend), FastAPI (backend API), XGBoost/Gradient Boosting (model), Docker (backend containerization)

## Key technical decisions

- **Binary classification over regression**: predicting exact ROI percentage from 514 samples would be noisy and overconfident; framing it as High vs. Not-High ROI (≥145.5% threshold) produces a model that's honest about its precision.
- **Kept the historical model analysis instead of deleting it** ([docs/MODEL_ANALYSIS.md](docs/MODEL_ANALYSIS.md)) — documents hitting a 68.8% accuracy ceiling on an earlier 462-sample dataset and what changed to get to 76.7% on 514 samples, rather than presenting only the final number.
- **Editorial, not SaaS-dashboard, UI direction** ([docs/DESIGN.md](docs/DESIGN.md)) — designed to read like a research artifact a decision-maker would trust, not another generic analytics dashboard.

## Results

76.70% accuracy, 76.74% AUC-ROC, 75.5% average confidence on the production binary classifier (514-sample dataset).

## Live

Frontend: **https://ai-roi-eight.vercel.app/**

Backend API runs locally (Render deploy config is in the repo — `render.yaml` — but isn't currently running a live instance). See Quick Start below to run it.

##  Project Structure

```
AI_ROI/
├── frontend/              # Next.js application
│   ├── app/              # App router pages
│   ├── components/       # React components
│   └── lib/              # API utilities
├── backend/              # FastAPI application
│   ├── app/              # API endpoints
│   │   ├── main.py      # Main API server
│   │   ├── model_loader.py
│   │   └── predict.py   # Prediction logic
│   ├── models/          # Trained ML models (.gitignored)
│   └── requirements.txt
├── data/                 # Data files (.gitignored)
│   ├── raw/             # Original datasets
│   ├── processed/       # Cleaned datasets
│   └── scraper/         # Data generation scripts
├── training/            # Model training pipeline
│   ├── build_dataset.py
│   ├── train_model.py
│   └── evaluate.py
├── monitoring/          # Model monitoring tools
├── scripts/             # Utility scripts
├── docs/                # Documentation
│   ├── MODEL_ANALYSIS.md
│   ├── DESIGN.md
│   └── ...
├── CONTRIBUTING.md      # Contribution guidelines
└── LICENSE              # MIT License
```

##  Quick Start

### Prerequisites

- **Python 3.10+**
- **Node.js 18+**
- **npm or yarn**

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/AI_ROI.git
cd AI_ROI
```

### 2. Backend Setup

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r backend/requirements.txt

# Train the model (required for first-time setup)
python backend/train_roi_model.py

# Start the API server
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`  
Interactive docs at `http://localhost:8000/docs`

### 3. Frontend Setup

```bash
# In a new terminal
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

The web app will be available at `http://localhost:3000`

### 4. Test the Application

```bash
# In another terminal with venv activated
python test_api.py
```

## 📊 Model Performance

- **Algorithm**: Gradient Boosting (Binary Classification)
- **Accuracy**: 76.70%
- **AUC-ROC**: 76.74%
- **Average Confidence**: 75.5%
- **Dataset**: 514 AI deployment projects
- **Target**: High ROI (≥145.5%) vs Not-High ROI (<145.5%)

### Performance Breakdown

| Metric | Not-High ROI | High ROI |
|--------|--------------|----------|
| Precision | 85.7% | 73.5% |
| Recall | 85.7% | 62.5% |
| F1-Score | 85.7% | 67.6% |

See [`docs/MODEL_ANALYSIS.md`](docs/MODEL_ANALYSIS.md) for detailed model analysis and performance history.

## 🎨 Usage

### Web Interface

1. Navigate to `http://localhost:3000`
2. Fill out the AI project details:
   - Company information (sector, size, revenue)
   - AI use case and deployment type
   - Investment amount
   - Timeline metrics
3. Click "Calculate ROI"
4. View prediction with confidence metrics

### API Usage

```python
import requests

data = {
    "year": 2024,
    "quarter": "q1",
    "sector": "manufacturing",
    "company_size": "grande",
    "revenue_m_eur": 330.7,
    "ai_use_case": "customer service bot",
    "deployment_type": "analytics",
    "days_diagnostic": 35,
    "days_poc": 115,
    "days_to_deployment": 360,
    "investment_eur": 353519,
    "time_saved_hours_month": 0,
    "revenue_increase_percent": 0.0,
    "human_in_loop": 1
}

response = requests.post("http://localhost:8000/predict", json=data)
result = response.json()
print(f"Predicted ROI Category: {result['predicted_roi']}")
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# Backend
API_HOST=0.0.0.0
API_PORT=8000

# Frontend
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### Model Retraining

To retrain the model with new data:

```bash
# 1. Add new data to data/processed/
# 2. Run training pipeline
python training/build_dataset.py
python training/train_model.py
python training/evaluate.py

# 3. Restart backend to load new model
```

## 📦 Deployment

### Backend (FastAPI)

**Option 1: Docker**
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY backend/requirements.txt .
RUN pip install -r requirements.txt
COPY backend/ .
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Option 2: Cloud Platform (Railway, Render, Heroku)**
- Build command: `pip install -r backend/requirements.txt`
- Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`

### Frontend (Next.js)

**Vercel (Recommended)**
```bash
npm install -g vercel
cd frontend
vercel
```

**Netlify**
- Build command: `npm run build`
- Publish directory: `.next`

## 🛠️ Tech Stack

### Frontend
- **Framework**: Next.js 14 (App Router)
- **Styling**: TailwindCSS
- **Components**: shadcn/ui
- **Icons**: Lucide React
- **Charts**: Recharts

### Backend
- **Framework**: FastAPI
- **ML Library**: XGBoost, scikit-learn
- **Data Processing**: pandas, numpy
- **Validation**: Pydantic

### DevOps
- **Version Control**: Git
- **Package Management**: pip, npm
- **Testing**: pytest (backend), Jest (frontend)

## 📝 API Endpoints

### `GET /`
Health check endpoint

**Response:**
```json
{
  "message": "AI ROI Prediction API",
  "version": "3.0",
  "status": "running",
  "model_loaded": true
}
```

### `POST /predict`
Predict ROI category for an AI project

**Request Body:**
```json
{
  "year": 2024,
  "quarter": "q1",
  "sector": "manufacturing",
  "company_size": "grande",
  "revenue_m_eur": 330.7,
  "ai_use_case": "customer service bot",
  "deployment_type": "analytics",
  "days_diagnostic": 35,
  "days_poc": 115,
  "days_to_deployment": 360,
  "investment_eur": 353519,
  "time_saved_hours_month": 0,
  "revenue_increase_percent": 0.0,
  "human_in_loop": 1
}
```

**Response:**
```json
{
  "prediction": "Not-High",
  "probability_high": 0.27,
  "probability_not_high": 0.73,
  "confidence": 0.46,
  "threshold": 145.5,
  "interpretation": "Not-High ROI Expected (<145.5%). Probability: 73.0% | Confidence: 46.0%",
  "predicted_roi": 112.5,
  "roi_lower_bound": 49.83,
  "roi_upper_bound": 175.17
}
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Model trained on synthetic AI deployment case studies
- UI components from [shadcn/ui](https://ui.shadcn.com/)
- Icons from [Lucide](https://lucide.dev/)

## 📞 Support

For questions or issues:
- Open an issue on GitHub
- Check the API documentation at `http://localhost:8000/docs`
- Review [`docs/MODEL_ANALYSIS.md`](docs/MODEL_ANALYSIS.md) for model details
- Browse all documentation in the [`docs/`](docs/) folder

## 🔮 Roadmap

- [ ] Add confidence intervals to predictions
- [ ] Implement A/B testing for model versions
- [ ] Add user authentication
- [ ] Create mobile app version
- [ ] Expand dataset to 1000+ samples
- [ ] Multi-language support

---

**Built with ❤️ for better AI investment decisions**

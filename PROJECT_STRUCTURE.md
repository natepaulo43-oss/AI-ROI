# AI ROI Project Structure

## Production Models (Active)

### Binary Classifier: `backend/models/roi_classifier_best.pkl`
- **Algorithm:** Gradient Boosting Classifier
- **Accuracy:** 76.70%
- **AUC-ROC:** 76.74%
- **Average Confidence:** 75.5%
- **Purpose:** Predicts High ROI (≥145.5%) vs Not-High ROI (<145.5%)
- **Status:** ✅ Active in production

### Regression Model: `backend/models/roi_model.pkl`
- **Algorithm:** Gradient Boosting Regressor
- **R² Score:** 0.42
- **MAE:** ±62.67%
- **Purpose:** Predicts continuous ROI percentage
- **Status:** ✅ Active in production

## Project Directory Structure

```
AI_ROI/
├── backend/                      # FastAPI backend
│   ├── app/                      # Main application
│   │   ├── main.py              # FastAPI app & endpoints
│   │   ├── model_loader.py      # Model loading logic
│   │   ├── predict.py           # Prediction logic
│   │   └── schemas.py           # Pydantic schemas
│   ├── models/                   # ML models (production only)
│   │   ├── roi_classifier_best.pkl
│   │   ├── roi_classifier_best_metadata.pkl
│   │   ├── roi_model.pkl
│   │   └── README.md
│   ├── train_roi_model.py       # Training script for regression model
│   ├── roi_api.py               # Legacy API (deprecated)
│   └── requirements.txt         # Python dependencies
│
├── frontend/                     # Next.js frontend
│   ├── app/                     # App router pages
│   │   ├── page.tsx            # Landing page
│   │   ├── about/              # About page
│   │   ├── hypothesis/         # Hypothesis page
│   │   └── insights/           # Insights page
│   ├── components/              # React components
│   │   ├── landing/            # Landing page components
│   │   ├── layout/             # Layout components
│   │   ├── insights/           # Insights components
│   │   └── tool/               # ROI tool components
│   ├── lib/                    # Utilities
│   │   └── api.ts             # API client
│   ├── public/                 # Static assets
│   ├── package.json
│   └── next.config.js
│
├── data/                        # Data files (.gitignored)
│   ├── raw/                    # Original datasets
│   ├── processed/              # Cleaned datasets
│   │   ├── 515.csv            # Main training dataset
│   │   ├── ai_roi_modeling_dataset.csv
│   │   └── ai_roi_training_dataset_enhanced.csv
│   └── scraper/                # Data generation scripts
│       ├── ai_roi_data_generator.py
│       ├── generate_training_data.py
│       └── web_scraper.py
│
├── monitoring/                  # Model monitoring (optional)
│   ├── alert_system.py
│   └── model_monitor.py
│
├── docs/                        # Documentation
│   ├── MODEL_ANALYSIS.md       # Model performance analysis
│   ├── DESIGN.md              # System design
│   └── README.md              # Documentation index
│
├── .github/                     # GitHub configuration
│   └── pull_request_template.md
│
├── test_api.py                 # API testing script
├── start.ps1                   # Windows startup script
├── start.sh                    # Unix startup script
├── start_all.ps1              # Start both backend & frontend
├── start_backend.ps1          # Start backend only
├── stop.ps1                   # Windows stop script
├── stop.sh                    # Unix stop script
├── README.md                  # Main project README
├── CONTRIBUTING.md            # Contribution guidelines
├── LICENSE                    # MIT License
└── .gitignore                 # Git ignore rules
```

## Key Files Explained

### Backend (FastAPI)
- **`backend/app/main.py`** - Main API server with `/predict` endpoint
- **`backend/app/model_loader.py`** - Loads both classifier and regression models
- **`backend/app/predict.py`** - Prediction logic with feature engineering
- **`backend/train_roi_model.py`** - Script to retrain regression model

### Frontend (Next.js)
- **`frontend/app/page.tsx`** - Landing page with hero section
- **`frontend/components/tool/`** - ROI prediction tool components
- **`frontend/lib/api.ts`** - API client for backend communication

### Testing
- **`test_api.py`** - Comprehensive API testing script

### Startup Scripts
- **`start_all.ps1`** - Starts both backend and frontend (Windows)
- **`start.sh`** - Starts both backend and frontend (Unix/Mac)

## How the Models Work Together

1. **User Input** → Frontend collects project details
2. **API Request** → Sent to `/predict` endpoint
3. **Feature Engineering** → Input transformed with same features as training
4. **Dual Prediction:**
   - **Classifier** → Predicts High/Not-High ROI category with confidence
   - **Regression** → Predicts continuous ROI value with confidence intervals
5. **Forecast Generation** → Creates 12-month ROI trajectory
6. **Response** → Returns complete prediction with visualization data

## Model Training

### Classifier Training
The classifier was trained using Gradient Boosting on binary classification task:
- Threshold: 145.5% ROI
- Features: 18 engineered features including log transforms, ratios, and interactions
- Dataset: 514 AI deployment projects
- Training Script: `backend/train_roi_classifier.py`

### Regression Training
To retrain the regression model:
```bash
cd backend
python train_roi_model.py
```

## API Endpoints

### `GET /`
Health check - returns API status

### `GET /health`
Detailed health check - shows model loading status

### `POST /predict`
Main prediction endpoint
- **Input:** Project details (PredictionInput schema)
- **Output:** Complete prediction (PredictionOutput schema)

## Dependencies

### Backend
- FastAPI
- scikit-learn
- XGBoost
- pandas, numpy
- joblib

### Frontend
- Next.js 14
- React
- TailwindCSS
- Recharts (for visualizations)

## Development Workflow

1. **Start Backend:**
   ```bash
   cd backend
   uvicorn app.main:app --reload
   ```

2. **Start Frontend:**
   ```bash
   cd frontend
   npm run dev
   ```

3. **Test API:**
   ```bash
   python test_api.py
   ```

## Production Deployment

Both models are production-ready and actively used in the backend API. The classifier provides the primary prediction (High/Not-High ROI) while the regression model provides detailed ROI estimates and forecasts.

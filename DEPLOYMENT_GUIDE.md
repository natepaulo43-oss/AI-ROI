# AI ROI Web Tool - Deployment Guide

## 🚀 Quick Start

### **Step 1: Start the Backend API**

```powershell
# Navigate to project root
cd c:\Users\Nate\OneDrive\Desktop\AI_ROI

# Activate virtual environment
.venv\Scripts\Activate.ps1

# Start the FastAPI server
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Expected Output:**
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete.
Loading model from: C:\Users\Nate\...\backend\models\roi_model.pkl
✅ Model loaded successfully on startup
```

**API will be available at:**
- Local: `http://localhost:8000`
- Network: `http://YOUR_IP:8000`
- API Docs: `http://localhost:8000/docs` (interactive Swagger UI)

---

### **Step 2: Test the API**

Open a **new terminal** and run:

```powershell
# Activate venv
.venv\Scripts\Activate.ps1

# Run test suite
python test_api.py
```

**Expected Output:**
```
✅ Health check passed
✅ Prediction successful! Predicted ROI: 132.5%
✅ All tests completed
```

---

### **Step 3: Start the Frontend**

```powershell
# Navigate to frontend directory
cd frontend

# Install dependencies (first time only)
npm install

# Start development server
npm run dev
```

**Frontend will be available at:**
- `http://localhost:3000`

---

## 📡 API Endpoints

### **GET /** - Health Check
```bash
curl http://localhost:8000/
```

**Response:**
```json
{
  "message": "AI ROI Prediction API",
  "version": "2.0",
  "status": "running",
  "model_loaded": true
}
```

---

### **POST /predict** - Predict ROI

**Request:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

**Response:**
```json
{
  "predicted_roi": 132.5,
  "model_version": "v2.0_practical",
  "confidence_note": "Moderate confidence (R²=0.42). Average error ±63%."
}
```

---

## 🔧 Frontend Integration

### **Example: Calling API from React/Next.js**

```typescript
// lib/api.ts
export async function predictROI(data: PredictionInput) {
  const response = await fetch('http://localhost:8000/predict', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(data),
  });
  
  if (!response.ok) {
    throw new Error('Prediction failed');
  }
  
  return response.json();
}

// Usage in component
const handleSubmit = async (formData) => {
  try {
    const result = await predictROI({
      year: parseInt(formData.year),
      quarter: formData.quarter,
      sector: formData.sector,
      company_size: formData.companySize,
      revenue_m_eur: parseFloat(formData.revenue),
      ai_use_case: formData.useCase,
      deployment_type: formData.deploymentType,
      days_diagnostic: parseInt(formData.daysDiagnostic),
      days_poc: parseInt(formData.daysPoc),
      days_to_deployment: parseInt(formData.daysToDeployment),
      investment_eur: parseFloat(formData.investment),
      time_saved_hours_month: parseFloat(formData.timeSaved || 0),
      revenue_increase_percent: parseFloat(formData.revenueIncrease || 0),
      human_in_loop: formData.humanInLoop ? 1 : 0,
    });
    
    console.log(`Predicted ROI: ${result.predicted_roi}%`);
    setResult(result);
  } catch (error) {
    console.error('Prediction error:', error);
  }
};
```

---

## 📋 Required Input Fields

### **Mandatory Fields**

| Field | Type | Validation | Example |
|-------|------|------------|---------|
| `year` | integer | 2020-2030 | 2024 |
| `quarter` | string | q1, q2, q3, q4 | "q1" |
| `sector` | string | Any sector name | "manufacturing" |
| `company_size` | string | pme, eti, grande | "grande" |
| `revenue_m_eur` | float | > 0 | 330.7 |
| `ai_use_case` | string | Any use case | "customer service bot" |
| `deployment_type` | string | analytics, nlp, hybrid, automation, vision | "analytics" |
| `days_diagnostic` | integer | ≥ 0 | 35 |
| `days_poc` | integer | ≥ 0 | 115 |
| `days_to_deployment` | integer | ≥ 1 | 360 |
| `investment_eur` | float | > 0 | 353519 |
| `human_in_loop` | integer | 0 or 1 | 1 |

### **Optional Fields (Early Signals)**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `time_saved_hours_month` | float | 0.0 | Time saved per month (hours) |
| `revenue_increase_percent` | float | 0.0 | Revenue increase percentage |

**Note:** Including early signals significantly improves prediction accuracy.

---

## 🎨 Form Field Options

### **Sector Options**
- manufacturing
- finance
- retail
- logistique (logistics)
- services pro (professional services)
- sante (healthcare)
- energie (energy)
- telecom
- construction
- agroalimentaire (food/agriculture)

### **Company Size Options**
- **pme**: Small/Medium Enterprise
- **eti**: Mid-sized Enterprise
- **grande**: Large Enterprise

### **AI Use Case Options**
- customer service bot
- predictive analytics
- pricing optimization
- process automation
- quality control vision
- fraud detection
- sales automation
- document processing

### **Deployment Type Options**
- **analytics**: Analytics-focused
- **nlp**: Natural Language Processing
- **hybrid**: Hybrid approach
- **automation**: Automation-focused
- **vision**: Computer Vision

---

## 🐛 Troubleshooting

### **Issue: Model not found**
```
FileNotFoundError: Model file not found at: backend/models/roi_model.pkl
```

**Solution:**
```powershell
# Retrain the model
.venv\Scripts\python.exe backend\train_roi_model.py
```

---

### **Issue: CORS errors in browser**
```
Access to fetch at 'http://localhost:8000' has been blocked by CORS policy
```

**Solution:** CORS is already configured in `backend/app/main.py`. If issues persist:
1. Check that backend is running on port 8000
2. Verify frontend is making requests to correct URL
3. Check browser console for specific CORS error

---

### **Issue: Validation errors**
```
422 Unprocessable Entity
```

**Solution:** Check that all required fields are provided and match the validation rules. Use the API docs at `http://localhost:8000/docs` to see exact requirements.

---

## 📦 Production Deployment

### **Backend (FastAPI)**

**Option 1: Docker**
```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY backend/requirements.txt .
RUN pip install -r requirements.txt

COPY backend/ .
COPY data/processed/ai_roi_modeling_dataset.csv data/processed/

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Option 2: Cloud Platform (e.g., Railway, Render)**
1. Push code to GitHub
2. Connect repository to platform
3. Set build command: `pip install -r backend/requirements.txt`
4. Set start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`

---

### **Frontend (Next.js)**

**Build for production:**
```powershell
cd frontend
npm run build
npm start
```

**Deploy to Vercel:**
```powershell
npm install -g vercel
vercel
```

---

## 🔐 Security Considerations

### **For Production:**

1. **API Authentication:** Add API key authentication
2. **Rate Limiting:** Implement rate limiting to prevent abuse
3. **CORS:** Restrict `allow_origins` to your frontend domain only
4. **HTTPS:** Use HTTPS in production
5. **Environment Variables:** Store sensitive config in `.env` files

---

## 📊 Monitoring

### **Check API Status**
```powershell
curl http://localhost:8000/health
```

### **View API Logs**
Backend logs will show in the terminal where uvicorn is running.

### **Interactive API Documentation**
Visit `http://localhost:8000/docs` for:
- Interactive API testing
- Request/response schemas
- Example payloads

---

## 🔄 Updating the Model

When you retrain the model with new data:

```powershell
# 1. Retrain model
.venv\Scripts\python.exe backend\train_roi_model.py

# 2. Restart backend server
# Press Ctrl+C in the uvicorn terminal, then restart:
uvicorn app.main:app --reload
```

The new model will be automatically loaded on startup.

---

## 📞 Support

**Common Issues:**
- Model not loading → Retrain model
- CORS errors → Check backend is running
- Validation errors → Check input format
- Connection refused → Verify backend is on port 8000

**API Documentation:** `http://localhost:8000/docs`

---

**Last Updated:** February 10, 2026  
**API Version:** 2.0  
**Model Version:** v2.0_practical

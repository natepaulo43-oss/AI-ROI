# AI ROI Web Tool - Quick Start Guide

## 🚀 Get Your Website Running in 3 Steps

### **Step 1: Start the Backend API**

Open a terminal and run:

```powershell
cd c:\Users\Nate\OneDrive\Desktop\AI_ROI
.\start_backend.ps1
```

**Wait for this message:**
```
✅ Model loaded successfully on startup
INFO: Uvicorn running on http://0.0.0.0:8000
```

**Keep this terminal open!** The API needs to stay running.

---

### **Step 2: Start the Frontend**

Open a **NEW terminal** and run:

```powershell
cd c:\Users\Nate\OneDrive\Desktop\AI_ROI\frontend
npm run dev
```

**Wait for:**
```
✓ Ready in 2.5s
○ Local: http://localhost:3000
```

---

### **Step 3: Test the Tool**

1. **Open your browser:** `http://localhost:3000/tool`

2. **Fill out the form:**
   - **Sector:** Manufacturing
   - **Company Size:** Large Enterprise
   - **Revenue:** 330.7 million USD
   - **AI Use Case:** Customer Service Bot
   - **Deployment Type:** Analytics
   - **Investment:** 353,519 USD
   - **Year:** 2024
   - **Quarter:** Q1
   - **Days Diagnostic:** 35
   - **Days POC:** 115
   - **Days to Deployment:** 360
   - **Time Saved:** 0 (optional)
   - **Revenue Increase:** 0 (optional)

3. **Click "Calculate ROI"**

4. **See the ML prediction!** Should show ~-11.7% ROI (this is a real example from the training data)

---

## ✅ What's Happening Behind the Scenes

1. User fills out form in **USD** (frontend)
2. Frontend converts USD → EUR (backend expects EUR)
3. API calls ML model at `http://localhost:8000/predict`
4. Model uses **GradientBoosting** (R²=0.42) to predict ROI
5. Result displayed with confidence note

---

## 💰 Currency Handling

- **Frontend displays:** USD (user-friendly)
- **Backend processes:** EUR (model trained on EUR data)
- **Conversion rate:** 1 USD = 0.91 EUR (automatic)

---

## 📊 Form Fields Explained

### **Required Fields:**

| Field | Description | Example |
|-------|-------------|---------|
| Sector | Industry type | Manufacturing, Finance, Retail |
| Company Size | PME, ETI, or Grande | Large Enterprise |
| Revenue (M USD) | Annual revenue in millions | 330.7 |
| AI Use Case | Type of AI application | Customer Service Bot |
| Deployment Type | Technical approach | Analytics, NLP, Hybrid |
| Investment (USD) | Total AI investment | 353,519 |
| Year | Deployment year | 2024 |
| Quarter | Deployment quarter | Q1 |
| Days Diagnostic | Diagnostic phase duration | 35 |
| Days POC | Proof-of-concept duration | 115 |
| Days to Deployment | Total deployment time | 360 |

### **Optional Fields (Improve Accuracy):**

| Field | Description | Impact |
|-------|-------------|--------|
| Time Saved (hrs/month) | Early time savings | +26.6% importance |
| Revenue Increase (%) | Early revenue impact | +1.5% importance |

**Note:** Including early signals significantly improves prediction accuracy!

---

## 🎯 Expected Results

### **Example 1: No Early Signals**
- Investment: $100,000
- No time savings yet
- **Predicted ROI:** ~50-150% (moderate confidence)

### **Example 2: With Early Signals**
- Investment: $100,000
- Time saved: 500 hrs/month
- Revenue increase: 20%
- **Predicted ROI:** ~200-300% (higher confidence)

---

## 🐛 Troubleshooting

### **"Failed to get prediction" Error**

**Cause:** Backend API not running

**Fix:**
```powershell
# Check if backend is running
curl http://localhost:8000/health

# If not, start it:
.\start_backend.ps1
```

---

### **Form Fields Not Showing**

**Cause:** Frontend not started or wrong URL

**Fix:**
- Make sure you're at `http://localhost:3000/tool` (not just `/`)
- Restart frontend: `npm run dev`

---

### **Prediction Takes Too Long**

**Normal:** First prediction loads the model (~2-3 seconds)
**After that:** Should be instant (<500ms)

---

## 📱 Using on the Website

Once both servers are running:

1. **Backend API:** `http://localhost:8000` (stays in background)
2. **Website:** `http://localhost:3000/tool` (user-facing)

Users only see the website - they never interact with the API directly!

---

## 🔄 Making Changes

### **Retrained the Model?**

```powershell
# 1. Retrain
python backend\train_roi_model.py

# 2. Restart backend (Ctrl+C in backend terminal, then)
.\start_backend.ps1
```

### **Changed Frontend Code?**

Next.js auto-reloads - just save the file!

### **Changed Backend Code?**

Backend auto-reloads with `--reload` flag (already enabled)

---

## 🌐 Deploying to Production

See `DEPLOYMENT_GUIDE.md` for full production deployment instructions.

**Quick summary:**
- Backend: Deploy to Railway/Render/Heroku
- Frontend: Deploy to Vercel/Netlify
- Update `NEXT_PUBLIC_API_URL` in frontend to point to production backend

---

## 📞 Need Help?

**Backend not working?**
- Check: `http://localhost:8000/docs` for API documentation
- Run: `python test_api.py` to test backend

**Frontend not working?**
- Check browser console (F12) for errors
- Verify backend is running first

**Model predictions seem wrong?**
- Remember: Model has R²=0.42 (moderate accuracy)
- Average error: ±63%
- Best with early deployment signals

---

**🎉 You're all set! Your AI ROI prediction tool is now live and using machine learning!**

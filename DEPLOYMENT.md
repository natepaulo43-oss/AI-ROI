# Deployment Guide

This guide explains how to deploy the AI ROI Calculator with the frontend on Vercel and the ML API on Render (free tier).

## Architecture

- **Frontend**: Next.js app deployed on Vercel
- **Backend API**: FastAPI with ML models deployed on Render (free tier)
- **Communication**: Frontend calls backend via environment variable `NEXT_PUBLIC_API_URL`

## Why This Setup?

The ML models (~3 MB total) with sklearn/xgboost dependencies are too heavy for Vercel's serverless functions. By hosting the API separately on Render's free tier, we get:
- Persistent container that keeps models loaded in memory
- No cold start issues after first wake-up
- Free hosting with 512 MB RAM (sufficient for our models)

## Step 1: Deploy Backend API to Render

### Option A: Using render.yaml (Recommended)

1. **Push your code to GitHub** (if not already done)
   ```bash
   git add .
   git commit -m "Add deployment configuration"
   git push origin main
   ```

2. **Create a Render account** at https://render.com

3. **Create a new Web Service**:
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Render will auto-detect the `render.yaml` file
   - Click "Apply" to use the configuration

4. **Wait for deployment** (5-10 minutes):
   - Render will build the Docker image
   - Install dependencies
   - Load the ML models
   - The service will be available at: `https://your-service-name.onrender.com`

5. **Test the API**:
   ```bash
   curl https://your-service-name.onrender.com/health
   ```
   Should return: `{"status":"healthy","classifier_loaded":true,"regression_loaded":true}`

### Option B: Manual Setup

1. Go to Render Dashboard → New Web Service
2. Connect your repository
3. Configure:
   - **Name**: `ai-roi-api` (or your choice)
   - **Region**: Oregon (or closest to your users)
   - **Branch**: `main`
   - **Root Directory**: `backend`
   - **Runtime**: Docker
   - **Dockerfile Path**: `./Dockerfile`
   - **Plan**: Free
4. Add environment variable:
   - `PORT` = `8000`
5. Click "Create Web Service"

### Important Notes for Free Tier

- **Sleep after 15 minutes of inactivity**: The service will spin down after 15 min without requests
- **Cold start time**: 30-60 seconds to wake up and load models
- **Monthly limit**: 750 hours/month (sufficient for demos and light usage)
- **Memory**: 512 MB RAM (our models use ~200 MB)

## Step 2: Deploy Frontend to Vercel

1. **Install Vercel CLI** (optional):
   ```bash
   npm i -g vercel
   ```

2. **Configure environment variable**:
   
   Create `.env.production` in the `frontend` directory:
   ```bash
   NEXT_PUBLIC_API_URL=https://your-service-name.onrender.com
   ```
   
   Replace `your-service-name` with your actual Render service URL.

3. **Deploy to Vercel**:

   **Option A: Using Vercel Dashboard**
   - Go to https://vercel.com
   - Click "Add New" → "Project"
   - Import your GitHub repository
   - Configure:
     - **Framework Preset**: Next.js
     - **Root Directory**: `frontend`
     - **Build Command**: `npm run build`
     - **Output Directory**: `.next`
   - Add environment variable:
     - `NEXT_PUBLIC_API_URL` = `https://your-service-name.onrender.com`
   - Click "Deploy"

   **Option B: Using Vercel CLI**
   ```bash
   cd frontend
   vercel --prod
   ```
   When prompted, add the environment variable.

4. **Verify deployment**:
   - Visit your Vercel URL
   - Fill out the form and submit
   - First request may take 30-60s (API cold start)
   - Subsequent requests should be fast

## Step 3: Configure CORS (if needed)

The backend is already configured to allow all origins:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    ...
)
```

For production, you may want to restrict this to your Vercel domain:
```python
allow_origins=["https://your-app.vercel.app"],
```

## Step 4: Keep API Warm (Optional)

To prevent cold starts during demos, you can ping the API periodically:

### Option A: Vercel Cron Job

Create `frontend/app/api/cron/route.ts`:
```typescript
export async function GET() {
  const apiUrl = process.env.NEXT_PUBLIC_API_URL;
  
  try {
    await fetch(`${apiUrl}/health`);
    return Response.json({ success: true });
  } catch (error) {
    return Response.json({ success: false, error: String(error) });
  }
}
```

Add to `vercel.json`:
```json
{
  "crons": [{
    "path": "/api/cron",
    "schedule": "*/10 * * * *"
  }]
}
```

### Option B: External Uptime Monitor

Use a free service like:
- UptimeRobot (https://uptimerobot.com)
- Cron-job.org (https://cron-job.org)

Configure to ping `https://your-service-name.onrender.com/health` every 10 minutes.

## Troubleshooting

### API Returns 503 or Times Out
- **Cause**: Free tier is sleeping
- **Solution**: Wait 30-60s and retry. The frontend has automatic retry logic.

### CORS Errors
- **Cause**: Backend not allowing frontend origin
- **Solution**: Check CORS configuration in `backend/app/main.py`

### Models Not Loading
- **Cause**: Model files missing or path incorrect
- **Solution**: Ensure `.pkl` files are in `backend/models/` and committed to git

### Frontend Shows "localhost:8000" Error
- **Cause**: Environment variable not set
- **Solution**: Add `NEXT_PUBLIC_API_URL` to Vercel environment variables and redeploy

## Environment Variables Summary

### Frontend (Vercel)
```bash
NEXT_PUBLIC_API_URL=https://your-service-name.onrender.com
```

### Backend (Render)
```bash
PORT=8000  # Auto-set by Render
PYTHON_VERSION=3.11.0  # Optional
```

## Cost Breakdown

- **Vercel**: Free tier (100 GB bandwidth, unlimited requests)
- **Render**: Free tier (750 hours/month, 512 MB RAM)
- **Total**: $0/month for light usage

## Upgrading for Production

If you need better performance:

### Render Paid Plans
- **Starter ($7/month)**: No sleep, 512 MB RAM
- **Standard ($25/month)**: 2 GB RAM, faster CPU

### Alternative: Railway
- **Hobby ($5/month)**: 512 MB RAM, 500 hours/month
- **Developer ($20/month)**: 8 GB RAM, unlimited hours

## Model Integrity

All deployment files preserve model integrity:
- ✅ Models are loaded from the same `.pkl` files
- ✅ No model conversion or modification
- ✅ Same feature engineering pipeline
- ✅ Identical prediction logic
- ✅ Data remains in original format

The Docker container simply packages the existing Python environment without changing any ML code.

## Testing Checklist

Before sharing your link:

1. ✅ Backend health check returns `{"status":"healthy"}`
2. ✅ Frontend loads without errors
3. ✅ Can submit a prediction and get results
4. ✅ Charts and visualizations render correctly
5. ✅ Error messages are user-friendly
6. ✅ First request completes (even if slow)
7. ✅ Subsequent requests are fast

## Support

If you encounter issues:
1. Check Render logs: Dashboard → Your Service → Logs
2. Check Vercel logs: Dashboard → Your Project → Deployments → Logs
3. Test API directly: `curl https://your-api.onrender.com/health`
4. Verify environment variables are set correctly

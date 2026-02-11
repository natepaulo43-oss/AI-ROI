# Web Scraping Workflow for AI ROI Dataset

## Overview
This workflow uses **Claude Code for VS Code** (the extension you installed) to manually research and curate AI ROI case studies from the web, then integrate them into your training pipeline.

---

## 🎯 Goal
Collect real-world AI deployment case studies with documented ROI to increase dataset variance and improve model performance.

---

## 📋 Step-by-Step Process

### 1. **Open the Template CSV**
- File location: `data/raw/web_research_ai_roi.csv`
- This file has the correct column headers matching your training data schema

### 2. **Use Claude Code Extension to Research**

Open the Claude Code chat panel in VS Code and ask research questions like:

**Example prompts:**
```
"Find 5 healthcare AI deployments from 2023-2024 with documented ROI. 
Extract: company size, investment amount, ROI percentage, and time to deployment."

"Search for manufacturing companies that deployed predictive maintenance AI. 
I need: sector, investment cost, deployment timeline, and ROI results."

"Look for failed AI projects in retail with negative ROI. 
I need diverse examples including failures for my training data."
```

### 3. **Approve Web Requests**
- Claude Code will ask permission to browse specific URLs
- Review each URL and approve if it looks like a credible source:
  - ✅ Industry reports (McKinsey, Gartner, Forrester)
  - ✅ Company case studies (vendor sites, tech blogs)
  - ✅ Academic papers with real deployments
  - ✅ News articles about AI implementations
  - ❌ Marketing fluff without data
  - ❌ Hypothetical scenarios

### 4. **Extract Data from Each Case Study**

For each finding, ask Claude Code to extract these fields:

| Field | What to Look For |
|-------|------------------|
| `year` | Deployment year (2020-2026) |
| `quarter` | Quarter (q1-q4) - estimate from dates |
| `sector` | Industry (healthcare, finance, retail, manufacturing, etc.) |
| `company_size` | pme (<250 employees), eti (250-5000), grande (>5000) |
| `revenue_m_eur` | Company revenue in millions EUR (convert if needed) |
| `ai_use_case` | Description (e.g., "customer service bot", "predictive maintenance") |
| `deployment_type` | analytics, nlp, hybrid, automation, or vision |
| `days_diagnostic` | Days in diagnostic phase (estimate 20-60 if not stated) |
| `days_poc` | Days in POC (estimate 60-120 if not stated) |
| `days_to_deployment` | Total days to deployment |
| `investment_eur` | Total investment in EUR (convert currencies if needed) |
| `roi` | ROI percentage (this is the target variable!) |
| `time_saved_hours_month` | Time savings per month in hours |
| `revenue_increase_percent` | Revenue increase percentage |
| `human_in_loop` | 0 or 1 (whether humans oversee the AI) |

### 5. **Add Rows to CSV**

Open `data/raw/web_research_ai_roi.csv` and manually add each case study as a new row.

**Example row:**
```csv
2024,q2,healthcare,grande,450.0,patient triage chatbot,nlp,45,90,210,380000,52.3,1200,8.5,1
```

**Tips:**
- Use Excel, VS Code, or any CSV editor
- Ensure no missing commas
- Use `0` for missing numeric values (not blank)
- Use lowercase for categorical fields

### 6. **Aim for Diversity**

Target distribution:
- **Sectors**: 10+ different industries
- **Company sizes**: 30% pme, 30% eti, 40% grande
- **ROI range**: Include successes (>50%), moderate (0-50%), and failures (<0%)
- **Deployment types**: Mix across all 5 types

---

## 🔄 After Data Collection

### Run the Full Pipeline

Once you've added 20+ rows to `web_research_ai_roi.csv`:

```powershell
# Option 1: Run the automated pipeline script
.\scripts\run_full_pipeline.ps1

# Option 2: Run steps manually
cd data
python integrate_datasets.py

cd ..\backend
python train_roi_model_improved.py
```

This will:
1. ✅ Integrate your web research data with existing datasets
2. ✅ Retrain the model with the new data
3. ✅ Save the updated model to `backend/models/`

### Restart the API

```powershell
cd backend
uvicorn app.main:app --reload
```

### Test the Updated Model

```powershell
python test_api.py
```

---

## 📊 Quality Checklist

Before running the pipeline, verify:
- [ ] At least 20 new rows added
- [ ] All rows have values in every column (use 0 for missing numerics)
- [ ] ROI values are realistic (-100% to 500%)
- [ ] Company sizes use only: pme, eti, grande
- [ ] Deployment types use only: analytics, nlp, hybrid, automation, vision
- [ ] Quarters use only: q1, q2, q3, q4
- [ ] No duplicate case studies
- [ ] CSV format is valid (no extra commas, proper quotes)

---

## 💡 Example Claude Code Conversation

**You:** "Find recent AI chatbot deployments in customer service with documented ROI"

**Claude Code:** [Requests to browse 3 URLs from Gartner, a vendor case study, and a tech blog]

**You:** [Approve the requests]

**Claude Code:** "I found 3 relevant case studies:
1. Large retail company (5000+ employees) deployed NLP chatbot in Q2 2024..."

**You:** "Extract all the data fields I need for my CSV: year, quarter, sector, company_size, revenue, use case, deployment type, timeline, investment, ROI, time savings, revenue increase, human oversight"

**Claude Code:** "Here's the extracted data:
- Year: 2024
- Quarter: q2
- Sector: retail
- Company size: grande (8000 employees)
- Revenue: €320M
- AI use case: customer service chatbot
- Deployment type: nlp
- Days diagnostic: 30 (estimated)
- Days POC: 90
- Days to deployment: 180
- Investment: €250,000
- ROI: 45%
- Time saved: 800 hours/month
- Revenue increase: 5%
- Human in loop: 1"

**You:** [Copy this data into the CSV as a new row]

---

## 🚀 Expected Outcome

After collecting 50-100 diverse case studies:
- **Higher model accuracy** (more training examples)
- **Better generalization** (diverse sectors and scenarios)
- **Improved edge case handling** (failures and unusual deployments)
- **More reliable predictions** across different company sizes and use cases

---

## 📝 Notes

- **Claude Code cannot automate this** - it's a manual, curated process
- **Quality over quantity** - 50 well-researched cases > 200 low-quality ones
- **Document sources** - Keep notes on where data came from (optional but helpful)
- **Iterate** - Start with 20 rows, test the pipeline, then add more

---

## ❓ Troubleshooting

**Q: Claude Code won't browse certain sites**  
A: Some sites block automated access. Try different sources or manually extract data.

**Q: I can't find exact values for all fields**  
A: Use reasonable estimates based on industry averages. Document assumptions.

**Q: The pipeline fails after adding new data**  
A: Check CSV format - common issues are missing commas, extra quotes, or wrong data types.

**Q: How do I convert USD to EUR?**  
A: Use current exchange rates (~0.92 EUR per USD as of 2024).

---

**Ready to start?** Open Claude Code and begin researching! 🎉

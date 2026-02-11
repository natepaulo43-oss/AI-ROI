# Web Research Data Collection Guide

## Purpose
This guide explains how to use Claude Code (VS Code extension) to research AI ROI case studies and populate the training dataset with high-variance, real-world data.

## Target Data Schema
Each row should contain the following fields (matching `ai_roi_modeling_dataset.csv`):

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `year` | int | Deployment year | 2024 |
| `quarter` | str | Deployment quarter (q1-q4) | q2 |
| `sector` | str | Industry sector | healthcare, finance, retail, manufacturing |
| `company_size` | str | Company size (pme/eti/grande) | grande |
| `revenue_m_eur` | float | Company revenue in millions EUR | 150.5 |
| `ai_use_case` | str | AI use case description | customer service bot, predictive maintenance |
| `deployment_type` | str | Type (analytics/nlp/hybrid/automation/vision) | nlp |
| `days_diagnostic` | int | Days in diagnostic phase | 30 |
| `days_poc` | int | Days in proof-of-concept | 90 |
| `days_to_deployment` | int | Total days to deployment | 180 |
| `investment_eur` | float | Total investment in EUR | 250000 |
| `roi` | float | Return on investment percentage | 45.2 |
| `time_saved_hours_month` | float | Time saved per month (hours) | 500 |
| `revenue_increase_percent` | float | Revenue increase percentage | 12.5 |
| `human_in_loop` | int | Human oversight (0 or 1) | 1 |

## Research Strategy

### 1. **Use Claude Code to Find Case Studies**
Ask Claude Code questions like:
- "Find recent AI ROI case studies in the healthcare sector from 2023-2024"
- "Search for manufacturing companies that deployed predictive maintenance AI and reported ROI"
- "Look for enterprise AI automation projects with documented time savings"

### 2. **Approve Web Requests**
Claude Code will ask permission to browse specific URLs. Review and approve relevant sources:
- Industry reports (Gartner, McKinsey, Forrester)
- Company case studies (vendor sites, tech blogs)
- Academic papers with real-world deployments
- News articles about AI implementations

### 3. **Extract Data Points**
For each case study, ask Claude Code to extract:
- Company details (sector, size, revenue if available)
- AI project specifics (use case, type, timeline)
- Financial metrics (investment, ROI, revenue impact)
- Operational metrics (time savings, efficiency gains)

### 4. **Fill in Missing Data with Reasonable Estimates**
If exact values aren't available:
- **Company size**: Infer from employee count (pme: <250, eti: 250-5000, grande: >5000)
- **Timeline**: Use industry averages (diagnostic: 20-60 days, POC: 60-120 days)
- **Investment**: Estimate based on project scope and company size
- **Quarter**: Use publication date or deployment announcement date

### 5. **Add Rows to CSV**
Open `data/raw/web_research_ai_roi.csv` and add each case study as a new row.

**Example row:**
```csv
2024,q2,healthcare,grande,450.0,patient triage chatbot,nlp,45,90,210,380000,52.3,1200,8.5,1
```

## Quality Guidelines

### ✅ Good Sources
- Named companies with verifiable deployments
- Quantified ROI or impact metrics
- Recent data (2020-2026)
- Diverse sectors and company sizes

### ❌ Avoid
- Hypothetical scenarios or projections
- Marketing claims without data
- Duplicate case studies
- Outdated data (pre-2020)

## Target: 50-100 New Rows
Aim for high variance:
- **Sectors**: Mix of healthcare, finance, retail, manufacturing, logistics, energy, etc.
- **Company sizes**: Balance across pme, eti, grande
- **ROI range**: Include both successes (>50% ROI) and failures (<0% ROI)
- **Deployment types**: Spread across analytics, nlp, hybrid, automation, vision

## After Data Collection

1. **Validate CSV format**: Ensure no missing commas, proper data types
2. **Run integration**: `python data/integrate_datasets.py`
3. **Retrain model**: `python training/train_model.py` (or your training script)
4. **Test API**: Verify predictions with new model

## Example Claude Code Prompts

```
"Find 5 healthcare AI deployments from 2023-2024 with documented ROI. 
Extract: company size, investment amount, ROI percentage, and time to deployment."

"Search for failed AI projects in retail sector. I need cases where ROI was negative 
to add variance to my training dataset."

"Look for manufacturing predictive maintenance AI case studies. 
Focus on: days to deployment, investment cost, and operational savings."
```

---

**Note**: Claude Code cannot automate this process. You must manually review each finding, approve web requests, and copy data into the CSV. This is a curated, quality-over-quantity approach.

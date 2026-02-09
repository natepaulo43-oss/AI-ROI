# Product Requirements Document (PRD)

## Project Title
**AI Adoption ROI Prediction Tool for SMEs**

## Context & Goal
This product is a web-based interactive tool that operationalizes the research project *“AI Adoption in Small and Medium Enterprises: Predicting ROI.”* The website will be the **primary deliverable presented at the SEDSI conference**, demonstrating how the underlying machine learning model can be used as a decision-support system for small and medium-sized enterprises (SMEs).

The goal is **not** to build a startup-ready SaaS, but a **clean, credible, academically grounded application** that:
- Translates the paper into a usable tool
- Demonstrates methodological rigor
- Is intuitive to demo live at a conference
- Clearly communicates assumptions, limitations, and insights

The UI should remain clean, minimal, and dashboard-inspired, similar in visual tone to the provided reference, but fully tailored to the research purpose.

---

## Target Audience
- Academic reviewers and conference attendees (SEDSI)
- Faculty and students interested in applied AI and analytics
- Practitioners curious about AI ROI decision-making

Technical sophistication of the user should **not** be assumed.

---

## Core User Flow (Conference Demo)
1. User lands on homepage
2. User understands the purpose of the tool in under 10 seconds
3. User navigates to the ROI Tool
4. User inputs firm characteristics and AI adoption variables
5. User receives:
   - Predicted ROI outcome
   - High-level interpretation
   - Model transparency elements
6. User can explore methodology and limitations

---

## Information Architecture

### 1. Landing / Overview Page
**Purpose:** Establish credibility and context immediately.

**Required Components:**
- Project title and subtitle
- One-paragraph abstract-style summary
- Visual callout highlighting:
  - Data-driven
  - Machine learning-based
  - SME-focused
- Primary CTA: “Explore the ROI Tool”

**Optional Enhancements:**
- Conference badge or note (e.g., “Presented at SEDSI”)

---

### 2. ROI Prediction Tool Page (Core Feature)
**Purpose:** Live demonstration of the model.

**Layout:**
- Card-based layout inspired by dashboard UI
- Inputs on the left or top
- Results prominently displayed

#### Inputs (Initial Version)
Inputs should map **directly** to variables discussed in the paper. Examples include:
- Firm size (employees)
- Industry category
- Current AI usage level
- AI investment level (budget or relative scale)
- Operational maturity
- Primary AI use case (finance, ops, marketing, etc.)

Inputs should:
- Be clearly labeled
- Use dropdowns or sliders where possible
- Avoid free-text when unnecessary

#### Outputs
- **Predicted ROI** (primary metric)
- Directional indicator (positive / neutral / negative)
- Optional confidence band or disclaimer

#### Interpretation Panel
- Plain-language explanation of what the result means
- Emphasis that this is an estimate, not a guarantee

---

### 3. Model Insights Page
**Purpose:** Academic transparency and rigor.

**Required Components:**
- Model type explanation (e.g., gradient boosting regression)
- Feature importance visualization
- Short explanation of why the model was chosen

This page is critical for reviewer confidence.

---

### 4. Methodology & Data Page
**Purpose:** Translate the paper’s methods into web form.

**Required Components:**
- Data sources overview
- Data preprocessing summary
- Training and validation approach
- Evaluation metrics (e.g., RMSE, R²)

This page should mirror the paper’s methods section.

---

### 5. Limitations & Ethics Page
**Purpose:** Address bias, assumptions, and responsible AI use.

**Required Components:**
- Known data limitations
- Model generalizability concerns
- Bias considerations
- Statement on human-in-the-loop decision making

This aligns directly with the abstract’s focus on bias and collaboration.

---

## UI / Design Principles
- Clean, neutral color palette
- High contrast for readability
- Dashboard-style cards and charts
- Minimal animation
- No unnecessary interactivity

The UI should feel:
- Professional
- Academic
- Calm

---

## Technical Requirements (High-Level)

### Frontend
- Next.js (TypeScript)
- Modular components
- Responsive but desktop-first (conference demo)

### Backend
- Python FastAPI
- Pre-trained ML model loaded at runtime
- REST endpoint for predictions

### Model
- Gradient Boosted Regression (XGBoost)
- Saved preprocessing pipeline
- Reproducible training workflow

---

## Non-Goals
- User authentication
- Data persistence
- Monetization
- Production-scale optimization

---

## Success Criteria (SEDSI)
- Tool runs reliably during live demo
- Reviewers understand model purpose quickly
- Clear linkage between paper and application
- Strong visual clarity
- Transparent methodology

---

## Data & Model Specifications

**Dataset Overview**
- Source: Aggregated public and survey-based SME datasets related to AI adoption and firm performance
- Format: CSV files
- Approximate size: To be finalized (expected hundreds to low-thousands of observations)

**Feature Set (Final, Locked for v1)**
- firm_size (numeric, integer): Number of employees
- industry (categorical): SME industry classification
- operational_maturity (numeric, ordinal 1–5)
- ai_usage_level (categorical): None / Low / Moderate / High
- ai_investment (numeric, float): Annual AI-related spend in USD
- primary_use_case (categorical): Finance / Operations / Marketing / HR / Other

**Target Variable**
- Predicted ROI from AI adoption
- Defined as a percentage-based performance improvement proxy derived from financial and operational indicators

**Model Choice**
- XGBoost Regression (static, pre-trained)
- Rationale: Strong performance on tabular data, interpretability via feature importance, academic credibility

**Training & Validation**
- Train/test split: 80/20
- Cross-validation: 5-fold during development
- Baseline metrics to report: R², RMSE, MAE

---

## Deployment Strategy

**Frontend**
- Framework: Next.js (TypeScript)
- Hosting: Vercel
- Mode: Static build with client-side API calls

**Backend**
- Framework: FastAPI (Python)
- Hosting: Render or Railway (free tier acceptable for demo)
- Model loaded once at startup (no per-request loading)

**Environment Management**
- Environment variables stored via hosting provider dashboard
- No secrets exposed client-side

---

## Conference Demo Script (Outline)

**Estimated Duration:** 3–5 minutes

1. Landing Page
   - Brief explanation of the research question
   - Motivation for predicting AI ROI in SMEs

2. ROI Tool Page
   - Walk through one pre-filled example scenario
   - Submit inputs and display predicted ROI
   - Explain result in plain language

3. Model Insights Page
   - Highlight key drivers via feature importance
   - Brief justification of model choice

4. Limitations & Ethics Page
   - Address bias, assumptions, and appropriate use

**Prepared Demo Scenarios**
- Small Retail SME (low AI usage)
- Mid-size Manufacturing Firm (moderate AI investment)
- Professional Services Firm (high AI maturity)

---

## Pre-Conference Checklist

- Backend deployed and reachable via stable public URL (48 hours before conference)
- Frontend deployed and tested on conference network (24 hours before)
- Input validation and graceful error handling verified
- Offline backup prepared (screenshots or short screen recording)
- QR code generated linking to live site

---

## Next Steps

1. Confirm final UI layout and navigation
2. Freeze API request/response schema
3. Train baseline model and record performance metrics
4. Begin phased implementation per build plan
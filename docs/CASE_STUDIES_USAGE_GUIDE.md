# SME AI Case Studies - Usage Guide

## Quick Start

This directory contains comprehensive research on SME AI adoption case studies with measurable ROI data.

### Files in This Collection

1. **sme_ai_case_studies.json** (58 KB)
   - 87 verified case studies with structured data
   - Full JSON schema with all metrics
   - Ready for analysis, visualization, or ML training

2. **CASE_STUDY_RESEARCH_SUMMARY.md** (18 KB)
   - Executive summary and key findings
   - Complete source list with URLs
   - Industry benchmarks and recommendations

3. **CASE_STUDIES_USAGE_GUIDE.md** (this file)
   - How to use the data
   - Query examples
   - Integration instructions

## Data Structure

### JSON Schema Overview

```json
{
  "company_name": "string",
  "industry": "string",
  "country": "string",
  "employees": number,
  "revenue_usd": number | null,
  "ai_type": "string",
  "ai_tools": "string",
  "use_case": "string",
  "investment_usd": number | null,
  "timeline_months": number | null,
  "workforce_pct_using": number | null,
  "workforce_reduction": boolean,
  "productivity_increase_pct": number | null,
  "cost_savings_pct": number | null,
  "cost_savings_usd": number | null,
  "revenue_increase_pct": number | null,
  "revenue_increase_usd": number | null,
  "roi_pct": number | null,
  "roi_calculated": boolean,
  "additional_metrics": object,
  "source_url": "string",
  "source_type": "string",
  "quality_score": "High|Medium|Low"
}
```

## Usage Examples

### Python - Load and Analyze

```python
import json
import pandas as pd

# Load the case studies
with open('sme_ai_case_studies.json', 'r') as f:
    data = json.load(f)

# Convert to DataFrame
df = pd.DataFrame(data['cases'])

# Filter by industry
manufacturing_cases = df[df['industry'].str.contains('Manufacturing', na=False)]

# Calculate average ROI by industry
roi_by_industry = df.groupby('industry')['roi_pct'].mean()

# Find cases with highest productivity gains
top_productivity = df.nlargest(10, 'productivity_increase_pct')

# Filter high-quality cases only
high_quality = df[df['quality_score'] == 'High']

# Cases with cost savings data
cost_savings_cases = df[df['cost_savings_pct'].notna() | df['cost_savings_usd'].notna()]
```

### Python - Search by Criteria

```python
# Find cases similar to your business
def find_similar_cases(target_industry, max_employees=500, min_quality='Medium'):
    cases = []
    with open('sme_ai_case_studies.json', 'r') as f:
        data = json.load(f)

    for case in data['cases']:
        if (case['industry'] == target_industry and
            (case['employees'] or 0) <= max_employees and
            case['quality_score'] in ['High', 'Medium']):
            cases.append(case)

    return cases

# Example: Find retail cases for companies under 100 employees
retail_cases = find_similar_cases('Retail', max_employees=100)
```

### Python - ROI Calculator

```python
def estimate_roi(industry, investment_usd):
    """Estimate ROI based on industry benchmarks from case studies"""

    with open('sme_ai_case_studies.json', 'r') as f:
        data = json.load(f)

    df = pd.DataFrame(data['cases'])

    # Filter by industry and cases with ROI data
    industry_cases = df[
        (df['industry'].str.contains(industry, case=False, na=False)) &
        (df['roi_pct'].notna())
    ]

    if len(industry_cases) == 0:
        # Fall back to overall average
        avg_roi = df['roi_pct'].mean()
        print(f"Using overall average ROI: {avg_roi:.1f}%")
    else:
        avg_roi = industry_cases['roi_pct'].mean()
        print(f"Using {industry} average ROI: {avg_roi:.1f}%")

    estimated_return = investment_usd * (avg_roi / 100)

    return {
        'investment': investment_usd,
        'estimated_roi_pct': avg_roi,
        'estimated_return': estimated_return,
        'sample_size': len(industry_cases)
    }

# Example usage
result = estimate_roi('Retail', 50000)
print(f"Investment: ${result['investment']:,}")
print(f"Estimated ROI: {result['estimated_roi_pct']:.1f}%")
print(f"Estimated Return: ${result['estimated_return']:,}")
```

### JavaScript - Load in Web App

```javascript
// Fetch and use in frontend
fetch('data/sme_ai_case_studies.json')
  .then(response => response.json())
  .then(data => {
    const cases = data.cases;

    // Filter by quality
    const highQualityCases = cases.filter(c => c.quality_score === 'High');

    // Group by industry
    const byIndustry = cases.reduce((acc, case) => {
      const ind = case.industry;
      if (!acc[ind]) acc[ind] = [];
      acc[ind].push(case);
      return acc;
    }, {});

    // Calculate metrics
    const avgROI = cases
      .filter(c => c.roi_pct !== null)
      .reduce((sum, c) => sum + c.roi_pct, 0) / cases.length;
  });
```

## Common Queries

### 1. Find Best ROI Cases by Industry

```python
import pandas as pd

df = pd.DataFrame(data['cases'])

# Top ROI by industry
top_roi = (df[df['roi_pct'].notna()]
          .groupby('industry')['roi_pct']
          .max()
          .sort_values(ascending=False)
          .head(10))
```

### 2. Calculate Average Implementation Timeline

```python
# Average timeline by company size
size_bins = [0, 50, 100, 250, 500]
labels = ['0-50', '50-100', '100-250', '250-500']

df['size_category'] = pd.cut(df['employees'], bins=size_bins, labels=labels)
timeline_by_size = df.groupby('size_category')['timeline_months'].mean()
```

### 3. Identify Quick Win Use Cases

```python
# Cases with ROI < 12 months and high ROI
quick_wins = df[
    (df['timeline_months'] <= 12) &
    (df['roi_pct'] >= 300) &
    (df['quality_score'] == 'High')
].sort_values('roi_pct', ascending=False)
```

### 4. Cost Savings Analysis

```python
# Calculate total documented savings
total_savings_usd = df['cost_savings_usd'].sum()
avg_savings_pct = df['cost_savings_pct'].mean()

# Cost savings by AI type
savings_by_ai_type = df.groupby('ai_type').agg({
    'cost_savings_pct': 'mean',
    'cost_savings_usd': 'sum'
})
```

### 5. Productivity Gains by Use Case

```python
# Average productivity gains by use case
productivity_by_use_case = (df[df['productivity_increase_pct'].notna()]
                           .groupby('use_case')['productivity_increase_pct']
                           .mean()
                           .sort_values(ascending=False))
```

## Data Quality Notes

### Quality Score Definitions

- **High (56%):** Multiple quantified metrics, verified source, specific company details
- **Medium (39%):** Some quantified data, credible source, may lack some specifics
- **Low (5%):** Limited metrics but included for broader coverage

### Handling Missing Data

```python
# Cases with complete ROI data
complete_roi = df[
    df['roi_pct'].notna() |
    ((df['cost_savings_pct'].notna() | df['cost_savings_usd'].notna()) &
     (df['revenue_increase_pct'].notna() | df['revenue_increase_usd'].notna()))
]

# Calculate implied ROI where not directly stated
def calculate_implied_roi(row):
    if pd.notna(row['roi_pct']):
        return row['roi_pct']

    if pd.notna(row['cost_savings_usd']) and pd.notna(row['investment_usd']):
        return (row['cost_savings_usd'] / row['investment_usd']) * 100

    return None

df['implied_roi'] = df.apply(calculate_implied_roi, axis=1)
```

## Integration with Your AI ROI Tool

### Option 1: Direct Training Data

```python
# Prepare for ML model training
def prepare_training_data():
    with open('sme_ai_case_studies.json', 'r') as f:
        data = json.load(f)

    df = pd.DataFrame(data['cases'])

    # Feature engineering
    features = df[[
        'employees', 'ai_type', 'industry', 'use_case',
        'investment_usd', 'timeline_months', 'workforce_pct_using'
    ]]

    # Target variables
    targets = df[[
        'roi_pct', 'productivity_increase_pct',
        'cost_savings_pct', 'revenue_increase_pct'
    ]]

    return features, targets
```

### Option 2: Benchmark Database

```python
# Add to existing dataset as validation/benchmark
import existing_data

# Merge with your synthetic data
combined = pd.concat([
    existing_data.load(),
    pd.DataFrame(case_studies['cases'])
], ignore_index=True)

# Use for model validation
train_data = existing_data.load()
validation_data = pd.DataFrame(case_studies['cases'])
```

### Option 3: API Response Enhancement

```python
# Use in your backend to show real case studies
from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/api/similar-cases')
def get_similar_cases(industry, company_size):
    with open('data/sme_ai_case_studies.json', 'r') as f:
        data = json.load(f)

    similar = [
        case for case in data['cases']
        if case['industry'] == industry and
           abs(case.get('employees', 0) - company_size) < 50
    ]

    return jsonify(similar[:5])  # Return top 5 matches
```

## Visualization Ideas

### 1. ROI Heatmap by Industry and Company Size

```python
import seaborn as sns
import matplotlib.pyplot as plt

pivot = df.pivot_table(
    values='roi_pct',
    index='industry',
    columns='size_category',
    aggfunc='mean'
)

sns.heatmap(pivot, annot=True, fmt='.0f', cmap='RdYlGn')
plt.title('Average ROI % by Industry and Company Size')
```

### 2. Timeline vs. ROI Scatter Plot

```python
plt.scatter(df['timeline_months'], df['roi_pct'],
           c=df['quality_score'].map({'High': 'green', 'Medium': 'orange', 'Low': 'red'}),
           alpha=0.6)
plt.xlabel('Implementation Timeline (Months)')
plt.ylabel('ROI %')
plt.title('Implementation Timeline vs. ROI')
```

### 3. Success Rate by AI Type

```python
ai_type_stats = df.groupby('ai_type').agg({
    'roi_pct': 'mean',
    'productivity_increase_pct': 'mean',
    'cost_savings_pct': 'mean'
})

ai_type_stats.plot(kind='bar', figsize=(12, 6))
plt.title('Performance Metrics by AI Type')
```

## Citation

When using this data in research or publications:

**APA Style:**
```
Claude AI Research. (2026). SME AI Adoption Case Studies Database.
Retrieved from https://github.com/[your-repo]/data/sme_ai_case_studies.json
```

**Data Citation:**
```
SME AI Case Studies Dataset (2026). 87 verified case studies of AI
implementation in small and medium enterprises with measurable ROI data.
Compiled from academic, consulting, industry, and vendor sources.
```

## Updates and Maintenance

### Recommended Update Frequency
- **Quarterly:** Add new high-quality cases from latest reports
- **Semi-Annual:** Validate source URLs still active
- **Annual:** Complete refresh with latest industry data

### Contributing New Cases

To add a case study, ensure it includes:
1. Company name and size verification
2. At least 3 quantified metrics
3. Verifiable source URL
4. Implementation completion (not just announced pilots)

### Data Validation Checklist

- [ ] Company size under 500 employees or SME-applicable
- [ ] At least 3 numerical metrics present
- [ ] Source URL active and verifiable
- [ ] No duplicate entries
- [ ] Industry classification accurate
- [ ] Quality score assigned based on criteria

## Support and Questions

For questions about this dataset:
1. Review the CASE_STUDY_RESEARCH_SUMMARY.md for methodology
2. Check source URLs for original context
3. Validate metrics against original sources when critical

## Version History

- **v1.0** (Feb 16, 2026): Initial release with 87 case studies
  - 49 high-quality cases
  - 34 medium-quality cases
  - 4 low-quality cases
  - Coverage across 19 industries
  - 49 unique sources

---

**Last Updated:** February 16, 2026
**Data Version:** 1.0
**Format:** JSON with UTF-8 encoding
**Size:** 87 case studies, 58 KB

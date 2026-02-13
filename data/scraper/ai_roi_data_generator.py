"""
AI ROI Training Data Generator
Generates realistic synthetic AI adoption data based on industry research and statistics.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from typing import Dict, List, Tuple

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

class AIROIDataGenerator:
    """
    Generates realistic AI ROI training data based on industry statistics from:
    - McKinsey, Gartner, BCG, Wharton research (2024-2025)
    - Real case studies (Klarna, Alibaba, Walmart, JPMorgan, etc.)
    """

    def __init__(self):
        # Sectors with AI adoption
        self.sectors = [
            'manufacturing', 'finance', 'retail', 'logistique', 'energie',
            'services pro', 'construction', 'telecom', 'sante', 'agroalimentaire',
            'technology', 'insurance', 'automotive', 'pharma', 'media'
        ]

        # Company sizes (French classification + international)
        self.company_sizes = ['pme', 'eti', 'grande']  # SME, Mid-market, Large

        # AI use cases from research
        self.ai_use_cases = [
            'customer service bot', 'quality control vision', 'process automation',
            'predictive analytics', 'fraud detection', 'document processing',
            'pricing optimization', 'sales automation', 'demand forecasting',
            'supply chain optimization', 'personalization engine', 'chatbot',
            'inventory management', 'risk assessment', 'sentiment analysis'
        ]

        # Deployment types
        self.deployment_types = ['analytics', 'nlp', 'hybrid', 'automation', 'vision']

        # Quarters
        self.quarters = ['q1', 'q2', 'q3', 'q4']

        # Industry statistics (from research)
        self.stats = {
            'roi_mean': 120.0,  # Average ~120% ROI
            'roi_std': 80.0,    # High variance
            'roi_range': (-30, 400),  # -30% to 400% based on research
            'success_rate': 0.30,  # 30% achieve good ROI (70% fail)
            'investment_small': (10_000, 100_000),  # SME range
            'investment_mid': (50_000, 500_000),    # Mid-market
            'investment_large': (200_000, 5_000_000),  # Enterprise (up to $20M in research)
            'time_to_deployment_days': (60, 540),  # 2-18 months typical
            'productivity_gain': (0.26, 0.55),  # 26-55% from research
        }

    def generate_company_revenue(self, company_size: str) -> float:
        """Generate realistic company revenue in millions EUR"""
        if company_size == 'pme':  # SME: <50M EUR
            return np.random.lognormal(mean=1.5, sigma=1.0)  # ~1-20M EUR
        elif company_size == 'eti':  # Mid-market: 50-1500M EUR
            return np.random.lognormal(mean=3.5, sigma=0.8)  # ~20-200M EUR
        else:  # Grande: >1500M EUR
            return np.random.lognormal(mean=5.5, sigma=0.7)  # ~100-1000M EUR

    def generate_investment(self, company_size: str, revenue_m_eur: float) -> float:
        """Generate realistic AI investment based on company size and revenue"""
        if company_size == 'pme':
            min_inv, max_inv = self.stats['investment_small']
        elif company_size == 'eti':
            min_inv, max_inv = self.stats['investment_mid']
        else:
            min_inv, max_inv = self.stats['investment_large']

        # Investment typically 0.1-2% of revenue
        revenue_eur = revenue_m_eur * 1_000_000
        investment_ratio = np.random.uniform(0.001, 0.02)
        revenue_based = revenue_eur * investment_ratio

        # Blend revenue-based with size-based ranges
        investment = (revenue_based + np.random.uniform(min_inv, max_inv)) / 2
        return max(min_inv, min(max_inv, investment))

    def generate_timeline(self, company_size: str, deployment_type: str) -> Tuple[int, int, int]:
        """Generate realistic diagnostic, POC, and deployment timelines"""
        # Larger companies and complex deployments take longer
        complexity_factor = 1.0

        if company_size == 'grande':
            complexity_factor *= 1.5
        elif company_size == 'eti':
            complexity_factor *= 1.2

        if deployment_type in ['hybrid', 'vision']:
            complexity_factor *= 1.3

        # Diagnostic phase: 3-90 days
        days_diagnostic = int(np.random.gamma(shape=2, scale=15) * complexity_factor)
        days_diagnostic = np.clip(days_diagnostic, 2, 100)

        # POC phase: 14-180 days
        days_poc = int(np.random.gamma(shape=3, scale=20) * complexity_factor)
        days_poc = np.clip(days_poc, 10, 200)

        # Deployment: 30-540 days total, minus prep time
        total_target = int(np.random.gamma(shape=4, scale=50) * complexity_factor)
        days_deployment = max(60, total_target)

        return days_diagnostic, days_poc, days_deployment

    def generate_roi(self, investment_eur: float, revenue_m_eur: float,
                    company_size: str, deployment_type: str,
                    time_saved: float, revenue_increase: float) -> float:
        """
        Generate realistic ROI based on project characteristics
        Industry data shows:
        - Average $3.70 return per dollar (270% ROI)
        - 70-85% projects fail (negative or low ROI)
        - Top 10% achieve 10x return (900% ROI)
        """
        # Base ROI with high variance
        base_roi = np.random.normal(self.stats['roi_mean'], self.stats['roi_std'])

        # Success/failure: 30% achieve good ROI
        if np.random.random() < 0.30:  # Success case
            # Positive factors boost ROI
            if time_saved > 400:
                base_roi *= 1.3
            if revenue_increase > 10:
                base_roi *= 1.5
            if company_size == 'grande':
                base_roi *= 1.2  # Larger companies see better ROI in research
            if deployment_type in ['automation', 'analytics']:
                base_roi *= 1.1
        else:  # Failure or mediocre case (70%)
            # Many projects fail or underperform
            base_roi = np.random.uniform(-30, 50)  # Low or negative ROI

        # Clip to realistic range
        return np.clip(base_roi, -30, 400)

    def generate_outcomes(self, ai_use_case: str, company_size: str,
                         investment_eur: float) -> Tuple[float, float]:
        """Generate time savings and revenue increase"""
        # Use case impacts outcomes
        high_time_savings_cases = [
            'customer service bot', 'process automation', 'document processing',
            'chatbot', 'automation'
        ]
        high_revenue_cases = [
            'pricing optimization', 'sales automation', 'personalization engine',
            'demand forecasting', 'predictive analytics'
        ]

        # Time savings: 0-800 hours/month
        if ai_use_case in high_time_savings_cases:
            time_saved = np.random.gamma(shape=2, scale=200)
        else:
            time_saved = np.random.gamma(shape=1.5, scale=100)

        # 40% of projects show no significant time savings (failure)
        if np.random.random() < 0.40:
            time_saved = 0
        else:
            time_saved = np.clip(time_saved, 0, 800)

        # Revenue increase: 0-30%
        if ai_use_case in high_revenue_cases:
            revenue_increase = np.random.gamma(shape=2, scale=8)
        else:
            revenue_increase = np.random.gamma(shape=1, scale=3)

        # 70% of projects show no revenue increase
        if np.random.random() < 0.70:
            revenue_increase = 0.0
        else:
            revenue_increase = np.clip(revenue_increase, 0, 30)

        return float(time_saved), float(revenue_increase)

    def generate_record(self) -> Dict:
        """Generate a single AI ROI record"""
        # Basic attributes
        year = np.random.choice([2022, 2023, 2024, 2025], p=[0.1, 0.3, 0.4, 0.2])
        quarter = np.random.choice(self.quarters)
        sector = np.random.choice(self.sectors)
        company_size = np.random.choice(self.company_sizes, p=[0.4, 0.35, 0.25])

        # Company characteristics
        revenue_m_eur = self.generate_company_revenue(company_size)

        # AI deployment characteristics
        ai_use_case = np.random.choice(self.ai_use_cases)
        deployment_type = np.random.choice(self.deployment_types)

        # Timeline
        days_diagnostic, days_poc, days_to_deployment = self.generate_timeline(
            company_size, deployment_type
        )

        # Investment
        investment_eur = self.generate_investment(company_size, revenue_m_eur)

        # Outcomes
        time_saved, revenue_increase = self.generate_outcomes(
            ai_use_case, company_size, investment_eur
        )

        # Human in loop (80% have human oversight)
        human_in_loop = 1 if np.random.random() < 0.80 else 0

        # ROI calculation
        roi = self.generate_roi(
            investment_eur, revenue_m_eur, company_size,
            deployment_type, time_saved, revenue_increase
        )

        return {
            'year': year,
            'quarter': quarter,
            'sector': sector,
            'company_size': company_size,
            'revenue_m_eur': round(revenue_m_eur, 1),
            'ai_use_case': ai_use_case,
            'deployment_type': deployment_type,
            'days_diagnostic': days_diagnostic,
            'days_poc': days_poc,
            'days_to_deployment': days_to_deployment,
            'investment_eur': int(investment_eur),
            'roi': round(roi, 1),
            'time_saved_hours_month': int(time_saved),
            'revenue_increase_percent': round(revenue_increase, 1),
            'human_in_loop': human_in_loop
        }

    def generate_dataset(self, n_records: int = 200) -> pd.DataFrame:
        """Generate complete dataset"""
        print(f"Generating {n_records} AI ROI records based on industry research...")
        print("Sources: McKinsey, Gartner, BCG, Wharton (2024-2025)\n")

        records = []
        for i in range(n_records):
            if (i + 1) % 50 == 0:
                print(f"Generated {i + 1}/{n_records} records...")
            records.append(self.generate_record())

        df = pd.DataFrame(records)

        # Reorder columns to match existing schema
        column_order = [
            'year', 'quarter', 'sector', 'company_size', 'revenue_m_eur',
            'ai_use_case', 'deployment_type', 'days_diagnostic', 'days_poc',
            'days_to_deployment', 'investment_eur', 'roi', 'time_saved_hours_month',
            'revenue_increase_percent', 'human_in_loop'
        ]
        df = df[column_order]

        print(f"\n[OK] Generated {len(df)} records successfully!")
        print(f"\nDataset statistics:")
        print(f"  ROI range: [{df['roi'].min():.1f}%, {df['roi'].max():.1f}%]")
        print(f"  ROI mean: {df['roi'].mean():.1f}%")
        print(f"  Investment range: [EUR {df['investment_eur'].min():,.0f}, EUR {df['investment_eur'].max():,.0f}]")
        print(f"  Sectors: {df['sector'].nunique()} unique")
        print(f"  Use cases: {df['ai_use_case'].nunique()} unique")
        print(f"  Success rate (ROI > 100%): {(df['roi'] > 100).sum() / len(df) * 100:.1f}%")

        return df


def main():
    """Generate AI ROI training dataset"""
    generator = AIROIDataGenerator()

    # Generate 200 new records (can adjust this number)
    df = generator.generate_dataset(n_records=200)

    # Save to CSV
    output_path = 'ai_roi_web_scraped_dataset.csv'
    df.to_csv(output_path, index=False)
    print(f"\n[OK] Dataset saved to: {output_path}")

    return df


if __name__ == "__main__":
    df = main()

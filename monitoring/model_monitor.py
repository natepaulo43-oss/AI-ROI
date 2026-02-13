"""
Automated Model Monitoring and Drift Detection
- Tracks prediction accuracy over time
- Detects model drift
- Alerts when retraining needed
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
import json
import os
from pathlib import Path

class ModelMonitor:
    """
    Monitor ML model performance and detect drift
    """

    def __init__(self, model_path, log_path='monitoring/prediction_log.csv'):
        self.model_path = model_path
        self.log_path = log_path
        self.thresholds = {
            'mae_alert': 100.0,  # Alert if MAE > 100%
            'r2_alert': 0.0,     # Alert if R² < 0
            'drift_alert': 0.15,  # Alert if R² drops > 15%
            'min_samples': 10     # Need 10+ predictions to calculate metrics
        }

    def log_prediction(self, input_data, prediction, actual_roi=None, metadata=None):
        """
        Log a prediction for monitoring
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'predicted_roi': prediction,
            'actual_roi': actual_roi,
            'year': input_data.get('year'),
            'quarter': input_data.get('quarter'),
            'sector': input_data.get('sector'),
            'company_size': input_data.get('company_size'),
            'investment_eur': input_data.get('investment_eur'),
            'model_version': metadata.get('model_version', 'unknown') if metadata else 'unknown'
        }

        # Create log file if doesn't exist
        log_dir = Path(self.log_path).parent
        log_dir.mkdir(parents=True, exist_ok=True)

        if not os.path.exists(self.log_path):
            df = pd.DataFrame([log_entry])
            df.to_csv(self.log_path, index=False)
        else:
            df = pd.DataFrame([log_entry])
            df.to_csv(self.log_path, mode='a', header=False, index=False)

        print(f"[LOG] Prediction logged: {prediction:.2f}%")

    def load_predictions(self, days_back=30):
        """Load recent predictions"""
        if not os.path.exists(self.log_path):
            print("[INFO] No prediction log found")
            return None

        df = pd.read_csv(self.log_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Filter recent
        cutoff = datetime.now() - timedelta(days=days_back)
        recent = df[df['timestamp'] >= cutoff].copy()

        return recent

    def calculate_metrics(self, df):
        """Calculate performance metrics"""
        if df is None or len(df) < self.thresholds['min_samples']:
            return None

        # Only calculate on records with actual ROI
        df_complete = df.dropna(subset=['actual_roi'])

        if len(df_complete) < self.thresholds['min_samples']:
            return {
                'status': 'insufficient_data',
                'samples': len(df_complete),
                'required': self.thresholds['min_samples']
            }

        mae = np.mean(np.abs(df_complete['predicted_roi'] - df_complete['actual_roi']))
        rmse = np.sqrt(np.mean((df_complete['predicted_roi'] - df_complete['actual_roi'])**2))

        # R² score
        ss_res = np.sum((df_complete['actual_roi'] - df_complete['predicted_roi'])**2)
        ss_tot = np.sum((df_complete['actual_roi'] - df_complete['actual_roi'].mean())**2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else float('-inf')

        return {
            'status': 'calculated',
            'samples': len(df_complete),
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mean_error': np.mean(df_complete['predicted_roi'] - df_complete['actual_roi']),
            'median_error': np.median(df_complete['predicted_roi'] - df_complete['actual_roi'])
        }

    def detect_drift(self, baseline_r2=0.30):
        """
        Detect model drift by comparing recent performance to baseline
        """
        recent = self.load_predictions(days_back=30)
        metrics = self.calculate_metrics(recent)

        if metrics is None or metrics['status'] != 'calculated':
            return {
                'drift_detected': False,
                'reason': 'insufficient_data',
                'action': 'collect_more_data'
            }

        # Check for drift
        r2_drop = baseline_r2 - metrics['r2']
        mae_high = metrics['mae'] > self.thresholds['mae_alert']
        r2_negative = metrics['r2'] < self.thresholds['r2_alert']

        drift_detected = (
            r2_drop > self.thresholds['drift_alert'] or
            mae_high or
            r2_negative
        )

        return {
            'drift_detected': drift_detected,
            'current_r2': metrics['r2'],
            'baseline_r2': baseline_r2,
            'r2_drop': r2_drop,
            'current_mae': metrics['mae'],
            'reason': self._get_drift_reason(r2_drop, mae_high, r2_negative),
            'action': 'retrain_model' if drift_detected else 'continue_monitoring',
            'metrics': metrics
        }

    def _get_drift_reason(self, r2_drop, mae_high, r2_negative):
        """Determine reason for drift"""
        reasons = []
        if r2_drop > self.thresholds['drift_alert']:
            reasons.append(f"R² dropped by {r2_drop:.3f}")
        if mae_high:
            reasons.append("MAE exceeds threshold")
        if r2_negative:
            reasons.append("R² is negative")

        return " | ".join(reasons) if reasons else "No drift"

    def generate_report(self, days_back=30):
        """Generate monitoring report"""
        print("=" * 80)
        print("MODEL MONITORING REPORT")
        print("=" * 80)
        print(f"\nReport Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Period: Last {days_back} days")

        recent = self.load_predictions(days_back=days_back)

        if recent is None or len(recent) == 0:
            print("\n[INFO] No predictions in log")
            return

        print(f"\nTotal Predictions: {len(recent)}")
        print(f"With Actual ROI: {recent['actual_roi'].notna().sum()}")

        # Calculate metrics
        metrics = self.calculate_metrics(recent)

        if metrics is None or metrics['status'] != 'calculated':
            print(f"\n[WARNING] Insufficient data for metrics calculation")
            print(f"           Need {self.thresholds['min_samples']} samples with actual ROI")
            return

        print("\n" + "-" * 80)
        print("PERFORMANCE METRICS")
        print("-" * 80)
        print(f"\nSamples: {metrics['samples']}")
        print(f"MAE:     {metrics['mae']:.2f}%")
        print(f"RMSE:    {metrics['rmse']:.2f}%")
        print(f"R² Score: {metrics['r2']:.4f}")
        print(f"Mean Error:   {metrics['mean_error']:.2f}% (bias)")
        print(f"Median Error: {metrics['median_error']:.2f}%")

        # Drift detection
        print("\n" + "-" * 80)
        print("DRIFT DETECTION")
        print("-" * 80)

        drift_result = self.detect_drift(baseline_r2=0.30)

        if drift_result['drift_detected']:
            print("\n[ALERT] Model drift detected!")
            print(f"Reason: {drift_result['reason']}")
            print(f"Action: {drift_result['action']}")
            print(f"\nRecommendation: Retrain model with recent data")
        else:
            print("\n[OK] No significant drift detected")
            print(f"     Model performance is stable")

        # Error distribution
        print("\n" + "-" * 80)
        print("ERROR DISTRIBUTION")
        print("-" * 80)

        df_complete = recent.dropna(subset=['actual_roi'])
        errors = np.abs(df_complete['predicted_roi'] - df_complete['actual_roi'])

        print(f"\n25th percentile: {np.percentile(errors, 25):.2f}%")
        print(f"Median:          {np.percentile(errors, 50):.2f}%")
        print(f"75th percentile: {np.percentile(errors, 75):.2f}%")
        print(f"Max error:       {errors.max():.2f}%")
        print(f"Errors > 100%:   {(errors > 100).sum()}/{len(errors)} ({(errors > 100).sum()/len(errors)*100:.1f}%)")

        # Predictions by sector/size
        print("\n" + "-" * 80)
        print("PREDICTIONS BY SEGMENT")
        print("-" * 80)

        print("\nBy Sector:")
        sector_stats = df_complete.groupby('sector').agg({
            'predicted_roi': 'count',
            'actual_roi': 'mean'
        }).round(2)
        print(sector_stats)

        print("\nBy Company Size:")
        size_stats = df_complete.groupby('company_size').agg({
            'predicted_roi': 'count',
            'actual_roi': 'mean'
        }).round(2)
        print(size_stats)

        print("\n" + "=" * 80)

        return metrics, drift_result

def monitor_model(model_path='backend/models/roi_model_temporal.pkl', days_back=30):
    """Convenience function to run monitoring"""
    monitor = ModelMonitor(model_path)
    metrics, drift = monitor.generate_report(days_back=days_back)
    return monitor, metrics, drift

if __name__ == "__main__":
    # Run monitoring
    monitor, metrics, drift = monitor_model(days_back=30)

    # Example: Log a new prediction
    # monitor.log_prediction(
    #     input_data={'year': 2025, 'quarter': 'q4', 'sector': 'finance',
    #                 'company_size': 'grande', 'investment_eur': 1000000},
    #     prediction=150.5,
    #     actual_roi=None,  # Will be filled in later
    #     metadata={'model_version': 'temporal_v1'}
    # )

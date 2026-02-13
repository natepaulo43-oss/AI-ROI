"""
Alert System for Model Monitoring
- Sends alerts when issues detected
- Logs all alerts
- Configurable thresholds
"""

import json
import os
from datetime import datetime
from pathlib import Path

class AlertSystem:
    """
    Alert system for model monitoring
    """

    def __init__(self, alert_log_path='monitoring/alerts.json'):
        self.alert_log_path = alert_log_path
        Path(alert_log_path).parent.mkdir(parents=True, exist_ok=True)

    def send_alert(self, alert_type, severity, message, details=None):
        """
        Log an alert
        severity: 'low', 'medium', 'high', 'critical'
        """
        alert = {
            'timestamp': datetime.now().isoformat(),
            'alert_type': alert_type,
            'severity': severity,
            'message': message,
            'details': details or {}
        }

        # Log to file
        alerts = self._load_alerts()
        alerts.append(alert)
        self._save_alerts(alerts)

        # Print to console
        symbol = self._get_severity_symbol(severity)
        print(f"\n{symbol} [{severity.upper()}] {alert_type}")
        print(f"   {message}")
        if details:
            for key, value in details.items():
                print(f"   - {key}: {value}")

        return alert

    def _get_severity_symbol(self, severity):
        """Get emoji for severity level"""
        return {
            'low': '[INFO]',
            'medium': '[WARNING]',
            'high': '[ALERT]',
            'critical': '[CRITICAL]'
        }.get(severity, '[INFO]')

    def _load_alerts(self):
        """Load existing alerts"""
        if not os.path.exists(self.alert_log_path):
            return []

        try:
            with open(self.alert_log_path, 'r') as f:
                return json.load(f)
        except:
            return []

    def _save_alerts(self, alerts):
        """Save alerts to file"""
        with open(self.alert_log_path, 'w') as f:
            json.dump(alerts, f, indent=2)

    def get_recent_alerts(self, hours=24):
        """Get alerts from last N hours"""
        alerts = self._load_alerts()
        cutoff = datetime.now().timestamp() - (hours * 3600)

        recent = []
        for alert in alerts:
            alert_time = datetime.fromisoformat(alert['timestamp']).timestamp()
            if alert_time >= cutoff:
                recent.append(alert)

        return recent

    def check_and_alert(self, drift_result, metrics):
        """
        Check monitoring results and send alerts if needed
        """
        alerts_sent = []

        # Critical: Model drift detected
        if drift_result['drift_detected']:
            alert = self.send_alert(
                alert_type='model_drift',
                severity='high',
                message='Model drift detected - retraining recommended',
                details={
                    'reason': drift_result['reason'],
                    'current_r2': f"{drift_result['current_r2']:.4f}",
                    'baseline_r2': f"{drift_result['baseline_r2']:.4f}",
                    'action': drift_result['action']
                }
            )
            alerts_sent.append(alert)

        # High: MAE exceeds threshold
        if metrics and metrics.get('mae', 0) > 100:
            alert = self.send_alert(
                alert_type='high_mae',
                severity='medium',
                message='Model accuracy degraded',
                details={
                    'mae': f"{metrics['mae']:.2f}%",
                    'threshold': '100%',
                    'samples': metrics['samples']
                }
            )
            alerts_sent.append(alert)

        # Medium: Negative R²
        if metrics and metrics.get('r2', 1) < 0:
            alert = self.send_alert(
                alert_type='negative_r2',
                severity='medium',
                message='Model R² is negative - predictions worse than mean',
                details={
                    'r2': f"{metrics['r2']:.4f}",
                    'samples': metrics['samples']
                }
            )
            alerts_sent.append(alert)

        # Low: Insufficient data
        if metrics and metrics.get('status') == 'insufficient_data':
            alert = self.send_alert(
                alert_type='insufficient_data',
                severity='low',
                message='Not enough data to calculate metrics',
                details={
                    'current_samples': metrics.get('samples', 0),
                    'required_samples': metrics.get('required', 10)
                }
            )
            alerts_sent.append(alert)

        return alerts_sent

if __name__ == "__main__":
    # Example usage
    alert_system = AlertSystem()

    # Simulate drift detection
    drift_result = {
        'drift_detected': True,
        'reason': 'R² dropped by 0.20',
        'current_r2': 0.10,
        'baseline_r2': 0.30,
        'action': 'retrain_model'
    }

    metrics = {
        'status': 'calculated',
        'samples': 50,
        'mae': 85.5,
        'r2': 0.10
    }

    alerts = alert_system.check_and_alert(drift_result, metrics)
    print(f"\n{len(alerts)} alert(s) sent")

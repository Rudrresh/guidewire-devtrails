import joblib
import pandas as pd
import numpy as np
import subprocess
import time
from datetime import datetime

model = joblib.load("models/kubernetes_issue_classifier.pkl")
scaler = joblib.load("models/scaler.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")

feature_columns = [
    "cpu_usage", "memory_usage", "disk_usage", "network_io",
    "pod_status", "node_status", "latency_ms", "error_rate", "logs_anomalies"
]


def predict_issue(metrics_df):
    scaled = scaler.transform(metrics_df)
    prediction = model.predict(scaled)
    decoded = label_encoder.inverse_transform(prediction)
    return decoded[0]

def get_deployments():
    try:
        result = subprocess.run(["kubectl", "get", "deployments", "-o", "jsonpath={.items[*].metadata.name}"], capture_output=True, text=True)
        deployments = result.stdout.strip().split()
        return deployments
    except Exception as e:
        print(f"[!] Failed to fetch deployments: {e}")
        return []

def get_pods():
    try:
        result = subprocess.run(["kubectl", "get", "pods", "-o", "jsonpath={.items[*].metadata.name}"], capture_output=True, text=True)
        pods = result.stdout.strip().split()
        return pods
    except Exception as e:
        print(f"[!] Failed to fetch pods: {e}")
        return []

def remediate(issue):
    print(f"[{datetime.now()}] Detected issue: {issue}")

    deployments = get_deployments()
    pods = get_pods()

    if issue == "Pod Failure":
        for pod in pods:
            subprocess.run(["kubectl", "delete", "pod", pod], check=False)
    elif issue == "Resource Exhaustion":
        for deployment in deployments:
            subprocess.run(["kubectl", "scale", "--replicas=5", f"deployment/{deployment}"], check=False)
    elif issue == "Network Issue":
        for pod in pods:
            subprocess.run(["kubectl", "delete", "pod", pod], check=False)
    elif issue == "Service Disruption":
        for deployment in deployments:
            subprocess.run(["kubectl", "rollout", "restart", f"deployment/{deployment}"], check=False)
    elif issue == "No Issue":
        print("✅ No remediation needed.")

while True:
    metrics = get_simulated_metrics()
    issue = predict_issue(metrics)
    remediate(issue)
    time.sleep(60)
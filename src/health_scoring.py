
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PREDICTIONS_PATH = ROOT / "models" / "cmapss_predictions.csv"

# Load predicted Remaining Useful Life
df = pd.read_csv(PREDICTIONS_PATH)

# Convert predicted RUL into a health score (0–100).
# This is a project-defined scoring rule, not a NASA standard.
df["health_score"] = (
 df["predicted_RUL"].clip(lower=0, upper=125) / 125 * 100
).round(2)

def classify_health(score):
    if score >= 80:
        return "Healthy"
    elif score >= 50:
        return "Warning"
    return "Critical"

df["status"] = df["health_score"].apply(classify_health)

def recommendation(status):
    if status == "Healthy":
        return "Continue regular monitoring"
    elif status == "Warning":
        return "Schedule preventive maintenance"
    return "Prioritize maintenance inspection"

df["recommendation"] = df["status"].apply(recommendation)

OUTPUT_PATH = ROOT / "models" / "engine_health_scores.csv"
df.to_csv(OUTPUT_PATH, index=False)

print("\nENGINE HEALTH SCORING RESULTS")
print(df.head(10).to_string(index=False))

print("\nHealth Status Summary")
print(df["status"].value_counts())

print(f"\nHealth scores saved to: {OUTPUT_PATH}")

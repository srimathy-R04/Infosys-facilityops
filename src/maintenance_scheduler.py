
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = ROOT / "models" / "engine_health_scores.csv"
OUTPUT_PATH = ROOT / "models" / "maintenance_schedule.csv"

# Load engine health information
df = pd.read_csv(INPUT_PATH)

# Set maintenance priority
def get_priority(status):
    if status == "Critical":
        return "High"
    elif status == "Warning":
        return "Medium"
    return "Low"

df["priority"] = df["status"].apply(get_priority)

# Recommend maintenance action
def get_action(status):
    if status == "Critical":
        return "Immediate inspection and maintenance"
    elif status == "Warning":
        return "Schedule preventive maintenance"
    return "Routine monitoring"

df["maintenance_action"] = df["status"].apply(get_action)

# Suggested maintenance timeline (project-defined)
def get_schedule(status):
    if status == "Critical":
        return "Within 1 day"
    elif status == "Warning":
        return "Within 7 days"
    return "Within 30 days"

df["recommended_schedule"] = df["status"].apply(get_schedule)

# Generate alerts
def generate_alert(row):
    if row["status"] == "Critical":
        return f"URGENT: Engine {row['engine']} needs immediate inspection."
    elif row["status"] == "Warning":
        return f"WARNING: Engine {row['engine']} needs preventive maintenance."
    return f"INFO: Engine {row['engine']} is healthy. Continue monitoring."

df["maintenance_alert"] = df.apply(generate_alert, axis=1)

# Sort by priority
priority_order = {"High": 0, "Medium": 1, "Low": 2}
df["priority_order"] = df["priority"].map(priority_order)

df = df.sort_values(
    by=["priority_order", "health_score"],
    ascending=[True, True]
).drop(columns=["priority_order"])

# Save schedule
df.to_csv(OUTPUT_PATH, index=False)

print("\nMAINTENANCE SCHEDULE")
print(df[
    ["engine", "health_score", "status", "priority",
     "maintenance_action", "recommended_schedule"]
].head(15).to_string(index=False))

print("\nMAINTENANCE PRIORITY SUMMARY")
print(df["priority"].value_counts())

print("\nALERT SUMMARY")
print(df["maintenance_alert"].head(10).to_string(index=False))

print(f"\nSchedule saved to: {OUTPUT_PATH}")

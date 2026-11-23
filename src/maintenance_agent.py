
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "models" / "maintenance_schedule.csv"

df = pd.read_csv(DATA_PATH)


def show_summary():
    print("\nMAINTENANCE SUMMARY")
    print("-------------------")
    print("Total engines:", len(df))
    print("\nStatus:")
    print(df["status"].value_counts().to_string())
    print("\nPriority:")
    print(df["priority"].value_counts().to_string())


def show_critical():
    critical = df[df["status"] == "Critical"]

    print("\nCRITICAL ENGINES")
    print(critical[
        ["engine", "health_score", "predicted_RUL",
         "maintenance_action", "recommended_schedule"]
    ].to_string(index=False))


def show_high_priority():
    high = df[df["priority"] == "High"]

    print("\nHIGH PRIORITY MAINTENANCE")
    print(high[
        ["engine", "health_score", "maintenance_alert"]
    ].to_string(index=False))


def show_engine(engine_id):
    result = df[df["engine"] == engine_id]

    if result.empty:
        print("Engine not found.")
        return

    print("\nENGINE DETAILS")
    print(result[
        ["engine", "health_score", "predicted_RUL",
         "status", "priority", "maintenance_action",
         "recommended_schedule", "maintenance_alert"]
    ].to_string(index=False))


def process_query(query):
    query = query.lower().strip()

    if "summary" in query or "how many" in query or "total" in query:
        show_summary()

    elif "critical" in query:
        show_critical()

    elif "high priority" in query or "urgent" in query:
        show_high_priority()

    elif "engine" in query:
        import re
        match = re.search(r"engine\s*(\d+)", query)

        if match:
            show_engine(int(match.group(1)))
        else:
            print("Please mention an engine number. Example: Engine 34")

    elif "warning" in query:
        print(df[df["status"] == "Warning"].to_string(index=False))

    elif "healthy" in query:
        print(df[df["status"] == "Healthy"].to_string(index=False))

    else:
        print("I couldn't understand that question.")
        print("Try: summary, critical engines, high priority, or Engine 34.")


if __name__ == "__main__":
    print("====================================")
    print("   SMART MAINTENANCE AGENT")
    print("====================================")
    print("Ask about engine health and maintenance.")
    print("Type 'exit' to stop.")

    while True:
        user_query = input("\nYou: ")

        if user_query.lower().strip() == "exit":
            print("Maintenance Agent stopped.")
            break

        process_query(user_query)


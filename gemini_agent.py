"""Gemini-powered agent using plain HTTPS requests (no google-genai library needed).
Falls back to the offline agent automatically if the key, internet or quota fails.
Forecasting is shared with SmartEnergyAgent: clean-history median by weekday/hour
with stuck/anomaly exclusion and a four-week rolling backtest.

Key:    GEMINI_API_KEY in .streamlit/secrets.toml (loaded by app.py) or  set GEMINI_API_KEY=...
Model:  optional  set GEMINI_MODEL=gemini-3-flash-preview  (otherwise several are tried)
"""
import json
import os
import urllib.error
import urllib.request

from llm_agent import SYSTEM, TOOLS, MAX_STEPS
from offline_agent import OfflineAgent

BASE = "https://generativelanguage.googleapis.com/v1beta/models/"
CANDIDATE_MODELS = [m for m in [os.environ.get("GEMINI_MODEL"), "gemini-3.1-flash-lite",
                                "gemini-3.5-flash-lite", "gemini-3.5-flash"] if m]

_WORKING = {}   # remembers the model that worked


def _declarations():
    out = []
    for t in TOOLS:
        d = {"name": t["name"], "description": t["description"]}
        if t["input_schema"].get("properties"):      # Gemini rejects empty parameter objects
            d["parameters"] = t["input_schema"]
        out.append(d)
    
   
    out.extend([
        {
            "name": "get_engine_health",
            "description": "Get the predicted remaining useful life and health status of a NASA engine.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "engine_id": {
                        "type": "INTEGER",
                        "description": "Engine number"
                    }
                },
                "required": ["engine_id"]
            }
        },
        {
            "name": "list_engines_by_status",
            "description": "List NASA engines with a requested health status.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "status": {
                        "type": "STRING",
                        "enum": ["Healthy", "Warning", "Critical"]
                    }
                },
                "required": ["status"]
            }
        },
        {
            "name": "maintenance_schedule_summary",
            "description": "Summarize NASA engine maintenance priorities and recommended schedules."
        }
    ])
    return out

def _plain(x):
    return json.loads(json.dumps(x, default=str))   # numpy/pandas values -> plain JSON


class GeminiAgent(OfflineAgent):
    def __init__(self, df, tariff=8.0, co2=0.727,
             maintenance_data=None, predictive_maintenance_data=None):

       super().__init__(df, tariff, co2, maintenance_data=maintenance_data)
       self.predictive_maintenance_data = predictive_maintenance_data
       self.key = (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "").strip()
       self.last_error = None
    def _call(self, model, contents):
        body = {"systemInstruction": {"parts": [{"text": SYSTEM}]},
                "contents": contents,
                "tools": [{"functionDeclarations": _declarations()}],
                "generationConfig": {"temperature": 0.2}}
        req = urllib.request.Request(BASE + f"{model}:generateContent", json.dumps(body).encode(),
                                     {"Content-Type": "application/json", "x-goog-api-key": self.key})
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"HTTP {e.code}: {e.read().decode()[:200]}")

    def _chat(self, model, question, history):
        contents = [{"role": "user" if m["role"] == "user" else "model",
                     "parts": [{"text": m["content"]}]} for m in (history or [])][-12:]
        contents.append({"role": "user", "parts": [{"text": question}]})
        for _ in range(MAX_STEPS):
            data = self._call(model, contents)
            cands = data.get("candidates") or []
            if not cands or "content" not in cands[0]:
                raise RuntimeError("empty answer: " + str(data.get("promptFeedback", ""))[:150])
            content = cands[0]["content"]
            parts = content.get("parts", [])
            calls = [p["functionCall"] for p in parts if "functionCall" in p]
            if not calls:
                text = "".join(p.get("text", "") for p in parts).strip()
                if not text:
                    raise RuntimeError("empty text answer")
                return text
            contents.append(content)   # keep the model turn exactly as returned (Gemini 3 needs its signatures)
            contents.append({"role": "user", "parts": [
                {"functionResponse": {"name": c["name"],
                                      "response": {"result": _plain(self._dispatch(c["name"], dict(c.get("args") or {})))}}}
                for c in calls]})
        return "That needed too many steps. Could you narrow the question?"

    def _offline_nasa_answer(self, question):
        import re

        q = question.lower()
        data = getattr(self, "predictive_maintenance_data", None)

        # Leave normal energy/campus questions to the existing offline agent.
        if not any(word in q for word in ["engine", "nasa", "turbofan"]):
            return None

        if data is None:
            return "NASA predictive-maintenance data is unavailable."

        health = data["health"]
        schedule = data["schedule"]

        # 1. Individual engine health
        match = re.search(r"\bengine\s*#?\s*(\d+)\b", q)
        if match:
            engine_id = int(match.group(1))
            rows = health[health["engine"] == engine_id]

            if rows.empty:
                return f"Engine {engine_id} was not found."

            row = rows.iloc[0]
            answer = (
                f"**Engine {engine_id} Health**\n\n"
                f"- Status: {row['status']}\n"
                f"- Predicted RUL: {row['predicted_RUL']:.2f} cycles\n"
                f"- Health score: {row['health_score']:.2f}\n"
            )

            if "actual_RUL" in row.index:
                answer += f"- Actual RUL: {row['actual_RUL']} cycles\n"

            if "recommendation" in row.index:
                answer += f"- Recommendation: {row['recommendation']}\n"

            return answer

        # 2. List engines by health status
        status = next(
            (s for s in ["critical", "warning", "healthy"] if s in q),
            None
        )

        if status and any(word in q for word in ["list", "show", "which", "all", "engines"]):
            result = health[health["status"].str.lower() == status]

            lines = [
                "| Engine | Predicted RUL | Health Score | Status |",
                "|---|---:|---:|---|"
            ]

            for _, row in result.iterrows():
                lines.append(
                    f"| {row['engine']} | {row['predicted_RUL']:.2f} | "
                    f"{row['health_score']:.2f} | {row['status']} |"
                )

            return (
                f"**{len(result)} {status.title()} engines**\n\n"
                + "\n".join(lines)
            )

        # 3. Maintenance schedule summary
        if any(word in q for word in ["schedule", "priority", "maintenance", "summary"]):
            counts = schedule["priority"].value_counts()

            answer = f"**NASA Maintenance Schedule — {len(schedule)} engines**\n\n"
            for priority in ["High", "Medium", "Low"]:
                answer += f"- {priority}: {counts.get(priority, 0)} engines\n"

            urgent = schedule[schedule["priority"] == "High"].copy()
            urgent = urgent.sort_values("health_score").head(5)

            answer += "\n**Five most urgent engines**\n\n"
            answer += "| Engine | Health Score | Recommended Schedule |\n"
            answer += "|---:|---:|---|\n"

            for _, row in urgent.iterrows():
                answer += (
                    f"| {row['engine']} | {row['health_score']:.2f} | "
                    f"{row['recommended_schedule']} |\n"
                )

            return answer

        return "Please ask about a specific engine, engine status, or the NASA maintenance schedule."

    
    def ask(self, question, history=None):
        if not question or not question.strip():
            return "Please enter a question."

        if not self.key:
            nasa_answer = self._offline_nasa_answer(question)
            if nasa_answer is not None:
                return nasa_answer
            return super().ask(question, history) + "\n\n_(Offline mode: no GEMINI_API_KEY found.)_"

        models = [_WORKING["m"]] if "m" in _WORKING else CANDIDATE_MODELS

        for model in models:
            try:
                answer = self._chat(model, question.strip(), history)
                _WORKING["m"] = model
                return answer
            except Exception as e:
                self.last_error = f"{model}: {str(e)[:200]}"
                _WORKING.pop("m", None)

        nasa_answer = self._offline_nasa_answer(question)
        if nasa_answer is not None:
            return nasa_answer + "\n\n_(NASA offline mode: Gemini unavailable.)_"

        return super().ask(question, history) + f"\n\n_(Gemini unavailable, answered offline. {self.last_error})_"

    def _dispatch(self, name, args):
        if name == "get_engine_health":
            return self._get_engine_health(args)

        if name == "list_engines_by_status":
            return self._list_engines_by_status(args)

        if name == "maintenance_schedule_summary":
            return self._maintenance_schedule_summary()

        return super()._dispatch(name, args)

    def _get_engine_health(self, args):
        if self.predictive_maintenance_data is None:
            return {"error": "NASA maintenance data is unavailable."}

        health = self.predictive_maintenance_data["health"]
        engine_id = int(args["engine_id"])

        result = health[health["engine"] == engine_id]

        if result.empty:
            return {"error": f"Engine {engine_id} not found."}

        return result.iloc[0].to_dict()

    def _list_engines_by_status(self, args):
        if self.predictive_maintenance_data is None:
            return {"error": "NASA maintenance data is unavailable."}

        health = self.predictive_maintenance_data["health"]
        status = args["status"]

        result = health[health["status"].str.lower() == status.lower()]

        return {
            "status": status,
            "engine_count": len(result),
            "engines": result[["engine", "predicted_RUL", "health_score", "status"]].to_dict("records")
        }

    def _maintenance_schedule_summary(self):
        if self.predictive_maintenance_data is None:
            return {"error": "NASA maintenance data is unavailable."}

        schedule = self.predictive_maintenance_data["schedule"]

        return {
            "total_engines": len(schedule),
            "priority_summary": schedule["priority"].value_counts().to_dict(),
            "urgent_engines": schedule[
                schedule["priority"] == "High"
            ][["engine", "health_score", "maintenance_action",
               "recommended_schedule"]].to_dict("records")
        }
    if __name__ == "__main__":
         print("Offline NASA test ready")
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
    return out


def _plain(x):
    return json.loads(json.dumps(x, default=str))   # numpy/pandas values -> plain JSON


class GeminiAgent(OfflineAgent):

    def __init__(self, df, tariff=8.0, co2=0.727, maintenance_data=None):
        super().__init__(df, tariff, co2, maintenance_data=maintenance_data)
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
            with urllib.request.urlopen(req, timeout=60) as r:
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

    def ask(self, question, history=None):
        if not question or not question.strip():
            return "Please enter a question."
        if not self.key:
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
        return super().ask(question, history) + f"\n\n_(Gemini unavailable, answered offline. {self.last_error})_"
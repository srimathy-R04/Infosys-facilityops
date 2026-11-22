"""Elite FacilityOps agent: Claude decides which of your Python tools to call.
Falls back to the old keyword agent if there is no API key or the API fails.

Needs:  pip install anthropic     and     set ANTHROPIC_API_KEY=your_key
Optional: set FACILITYOPS_MODEL=claude-sonnet-5-5   (default)
"""
import json
import os

import pandas as pd

from agent import EnergyAgent

try:
    import anthropic
except ImportError:  # keyword mode still works
    anthropic = None

MODEL = os.environ.get("FACILITYOPS_MODEL", "claude-sonnet-5-5")
MAX_STEPS = 6

SYSTEM = """You are FacilityOps Copilot, an expert energy and maintenance analyst for a campus
with four blocks (Block A to D) and simulated lifts, generators and HVAC units.

Rules:
- Never state a number you did not get from a tool. If you need data, call a tool. You may call several.
- Answer the actual question first in one sentence, then give supporting numbers, then (when useful)
  one practical recommendation or next step. Be concise; use a small table for comparisons.
- Always use the user's current tariff (Rs/kWh) and CO2 factor, which the tools already apply.
- Be honest about data: energy readings are real Kaggle ASHRAE data (hourly, one year);
  the anomaly ACCURACY SCORE was measured by planting artificial spikes, but the anomalies the detector flags in the data are real hours, never planted; maintenance data is SIMULATED. Mention this
  when it affects how much the user should trust the answer.
- Blocks listed in anomaly_detection_not_scored_for have very low usage, so anomaly detection is not reliable for
  them. Say that instead of reporting zero anomalies. For forecasts, quote the backtest accuracy and the stated dates.
- stuck_meter_hours_pct is the share of hours a meter repeated one reading (bad data). Mention it when it
  affects trust, and never present stuck stretches as real consumption.
- For night-waste questions rank blocks by night_vs_day_ratio, not by off-hours share alone, and say plainly when a
  block is not reliable (reliable=false) because its meter was stuck.
- For data quality or reliability questions call data_quality and quote its percentages and labels exactly. The
  whole-year and last-8-weeks stuck percentages are different numbers; never mix them up.
- Flagged anomalies are always in working-meter hours (stuck hours are never scored), so do not call them artifacts of a faulty meter.
- When describing a forecast (method, dates, accuracy), copy it from the tool result.
- If the user's message is a statement and not a question, check it with a tool and answer neutrally; do not just agree.
- If a question is ambiguous (e.g. 'last month' when data ends earlier), say which dates you used.
- If the question is unrelated to energy or maintenance, say so briefly and steer back.
- Use the conversation history to resolve follow-ups like 'and for Block C?'."""

TOOLS = [
    {"name": "data_overview",
     "description": "Date range, blocks, assets, tariff and CO2 factor in use. Call first if unsure what data exists.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "energy_summary",
     "description": "Electricity totals (kWh, cost, CO2) optionally filtered by block and date range, "
                    "optionally grouped by day/week/month/hour_of_day/weekday. hour_of_day and weekday return AVERAGE kWh per hour.",
     "input_schema": {"type": "object", "properties": {
         "block": {"type": "string", "description": "e.g. 'Block A'. Omit for all blocks."},
         "start": {"type": "string", "description": "YYYY-MM-DD inclusive"},
         "end": {"type": "string", "description": "YYYY-MM-DD inclusive"},
         "group_by": {"type": "string", "enum": ["none", "day", "week", "month", "hour_of_day", "weekday"]}}}},
    {"name": "peak_and_waste",
     "description": "Peak hour, peak day and off-hours (7pm-7am and weekends) share of consumption, per block. "
                    "Use for 'where are we wasting energy' questions.",
     "input_schema": {"type": "object", "properties": {"block": {"type": "string"}}}},
    {"name": "compare_blocks",
     "description": "Compare total consumption of two blocks.",
     "input_schema": {"type": "object", "properties": {
         "block1": {"type": "string"}, "block2": {"type": "string"}}, "required": ["block1", "block2"]}},
    {"name": "savings_scenario",
     "description": "What-if: savings (kWh, Rs, CO2) if consumption drops by a percentage.",
     "input_schema": {"type": "object", "properties": {
         "percentage": {"type": "number"}, "block": {"type": "string"}}, "required": ["percentage"]}},
    {"name": "list_anomalies",
     "description": "Top flagged anomalous hours (largest spike ratio first), optionally per block, plus counts per block.",
     "input_schema": {"type": "object", "properties": {
         "block": {"type": "string"}, "limit": {"type": "integer"}}}},
    {"name": "asset_health",
     "description": "Health, risk and recommendation for maintenance assets. Give asset for one (e.g. 'Lift 3'), "
                    "kind ('Lift','Generator','HVAC') to filter, or only_high_risk=true.",
     "input_schema": {"type": "object", "properties": {
         "asset": {"type": "string"}, "kind": {"type": "string"}, "only_high_risk": {"type": "boolean"}}}},
    {"name": "asset_trend",
     "description": "Last N days of temperature and vibration for one asset.",
     "input_schema": {"type": "object", "properties": {
         "asset": {"type": "string"}, "days": {"type": "integer"}}, "required": ["asset"]}},
    {"name": "data_quality",
     "description": "Data reliability per block: share of hours the meter was stuck on one repeated reading (whole year and last "
                    "8 weeks separately), longest stuck run, a reliability label, and whether anomaly scoring and forecasting apply. "
                    "Use for any question about data quality, reliability or trust.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "forecast_next_week",
     "description": "Forecast hourly/daily electricity demand for the 7 days after the data ends, using the average "
                    "weekday-and-hour pattern of the last 8 weeks. Also returns a backtest accuracy on the last 4 weeks.",
     "input_schema": {"type": "object", "properties": {"block": {"type": "string", "description": "Omit for all blocks."}}}},
    {"name": "maintenance_orders",
     "description": "All open maintenance work orders with priority.",
     "input_schema": {"type": "object", "properties": {}}},
]


class SmartEnergyAgent(EnergyAgent):
    """Same data and tools as EnergyAgent, plus an LLM brain and memory-aware ask()."""

    def __init__(self, df, tariff=8.0, co2=0.727, maintenance_data=None):
        super().__init__(df, tariff, co2)
        # Pass the cached maintenance tuple from Streamlit so the model is NOT retrained on every question.
        self._maintenance_cache = maintenance_data
        key = os.environ.get("ANTHROPIC_API_KEY")
        self.client = anthropic.Anthropic(api_key=key) if (anthropic and key) else None

    # ---------- new tools ----------
    def _block_df(self, block):
        d = self.df
        if block:
            d = d[d.block.str.lower() == block.lower()]
            if d.empty:
                return None
        return d

    def data_overview(self):
        status, _, _, _ = self._maintenance_data()
        return {"start": str(self.df.timestamp.min()), "end": str(self.df.timestamp.max()),
                "blocks": sorted(self.df.block.unique()), "hourly_rows": len(self.df),
                "assets": status.groupby("kind").size().to_dict(),
                "tariff_rs_per_kwh": self.tariff, "co2_kg_per_kwh": self.co2,
                "stuck_meter_hours_pct": ({b: round(100 * v, 1) for b, v in self.df.groupby("block").stuck.mean().items()}
                                          if "stuck" in self.df.columns else {}),
                "anomaly_detection_not_scored_for": (sorted(self.df[~self.df.scored].block.unique())
                                                     if "scored" in self.df.columns else [])}

    def energy_summary(self, block=None, start=None, end=None, group_by="none"):
        d = self._block_df(block)
        if d is None:
            return {"error": f"{block} was not found. Blocks: {sorted(self.df.block.unique())}"}
        if start:
            d = d[d.timestamp >= pd.Timestamp(start)]
        if end:
            d = d[d.timestamp < pd.Timestamp(end) + pd.Timedelta(days=1)]
        if d.empty:
            return {"error": "No data in that range.", **{k: v for k, v in self.data_overview().items() if k in ("start", "end")}}
        total = d.kwh.sum()
        out = {"scope": block or "All blocks", "total_kwh": round(total, 1),
               "cost_rs": round(total * self.tariff, 0), "co2_kg": round(total * self.co2, 0),
               "from": str(d.timestamp.min()), "to": str(d.timestamp.max())}
        if not block:
            out["by_block_kwh"] = d.groupby("block").kwh.sum().round(1).to_dict()
        if group_by in ("day", "week", "month"):
            s = d.set_index("timestamp").kwh.resample({"day": "D", "week": "W", "month": "MS"}[group_by]).sum().round(1)
            out["series_kwh"] = {str(k.date()): v for k, v in s.tail(40).items()}
        elif group_by == "hour_of_day":
            out["avg_kwh_per_hour_by_hour"] = d.groupby(d.timestamp.dt.hour).kwh.mean().round(1).to_dict()
        elif group_by == "weekday":
            names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            g = d.groupby(d.timestamp.dt.dayofweek).kwh.mean().round(1)
            out["avg_kwh_per_hour_by_weekday"] = {names[i]: v for i, v in g.items()}
        return out

    def peak_and_waste(self, block=None):
        d = self._block_df(block)
        if d is None:
            return {"error": f"{block} was not found."}
        res = {}
        for b, g in d.groupby("block"):
            stuck_pct = round(100 * g.stuck.mean(), 1) if "stuck" in g.columns else 0.0
            c = g[~g.stuck] if "stuck" in g.columns else g          # shares and ratios use clean readings only
            enough = len(c) >= 500
            c = c if enough else g
            hour, dow = c.timestamp.dt.hour, c.timestamp.dt.dayofweek
            off_c = (hour >= 19) | (hour < 7) | (dow >= 5)
            on_mean, off_mean = c.kwh[~off_c].mean(), c.kwh[off_c].mean()
            by_hour = c.groupby(hour).kwh.mean()
            gh, gd = g.timestamp.dt.hour, g.timestamp.dt.dayofweek
            off_g = (gh >= 19) | (gh < 7) | (gd >= 5)                # metered totals use all hours
            daily = g.set_index("timestamp").kwh.resample("D").sum()
            res[b] = {"peak_hour": int(by_hour.idxmax()), "peak_day": str(daily.idxmax().date()),
                      "peak_day_kwh": round(daily.max(), 1),
                      "off_hours_share_pct": round(100 * c.kwh[off_c].sum() / c.kwh.sum(), 1),
                      "night_vs_day_ratio": round(float(off_mean / on_mean), 2) if on_mean else None,
                      "off_hours_kwh": round(g.kwh[off_g].sum(), 1),
                      "off_hours_cost_rs": round(g.kwh[off_g].sum() * self.tariff, 0),
                      "stuck_hours_pct": stuck_pct,
                      "reliable": bool(enough and stuck_pct < 30),
                      "how_to_read": "Off-hours (7pm-7am + weekends) is 64.3% of all hours, so a perfectly flat load "
                                     "shows 64.3%. night_vs_day_ratio near 1.0 means the block runs as much at night as "
                                     "in the day (waste); about 0.5 or below is healthy."}
        return res

    def list_anomalies(self, block=None, limit=5):
        return {"summary": self.anomaly_summary(), "top": self.anomalies(block, int(limit or 5))}

    def asset_health_tool(self, asset=None, kind=None, only_high_risk=False):
        status, _, _, _ = self._maintenance_data()
        s = status
        if asset:
            s = s[s.asset.str.lower() == asset.lower()]   # exact match, so 'HVAC 1' never hits 'HVAC 10'
        if kind:
            s = s[s.kind.str.lower() == kind.lower()]
        if only_high_risk:
            s = s[s.risk >= 0.5]
        if s.empty:
            return {"error": "No matching asset (it may have already failed and been taken out of service)."}
        return s.head(30).to_dict("records")

    def asset_trend(self, asset, days=14):
        _, _, hist, _ = self._maintenance_data()
        h = hist[hist.asset.str.lower() == asset.lower()].sort_values("day")
        if h.empty:
            return {"error": f"{asset} was not found."}
        h = h.tail(int(days or 14))[["day", "temperature", "vibration", "risk"]].round(2)
        return h.to_dict("records")

    def data_quality(self):
        df = self.df
        if "stuck" not in df.columns:
            return {"error": "Stuck-meter information is not available in this data."}
        end = df.timestamp.max()
        out = {}
        for b, g in df.sort_values("timestamp").groupby("block"):
            run_id = (g.kwh != g.kwh.shift()).cumsum()
            year = 100 * g.stuck.mean()
            recent = 100 * g[g.timestamp > end - pd.Timedelta(weeks=8)].stuck.mean()
            out[b] = {"stuck_hours_pct_whole_year": round(year, 1),
                      "stuck_hours_pct_last_8_weeks": round(recent, 1),
                      "longest_identical_run_hours": int(g.groupby(run_id).kwh.size().max()),
                      "reliability": "High" if year < 5 else "Moderate" if year <= 30 else "Low",
                      "anomaly_scored": bool(g.scored.iloc[0]) if "scored" in g.columns else None,
                      "forecast_available": bool(recent <= 50)}
        out["notes"] = ("Reliability: High under 5% stuck, Moderate 5-30%, Low over 30% (whole year). "
                        "Stuck hours are never scored for anomalies, so flagged anomalies are always in working-meter hours.")
        return out

    def forecast_next_week(self, block=None, weeks=4):
        d = self._block_df(block)
        if d is None:
            return {"error": f"{block} was not found. Blocks: {sorted(self.df.block.unique())}"}
        end_all = d.timestamp.max()
        excluded = {}
        if "stuck" in d.columns:                       # a meter stuck on one value cannot be forecast
            recent = d[d.timestamp > end_all - pd.Timedelta(weeks=8)]
            share = recent.groupby("block").stuck.mean()
            excluded = {b: round(100 * v) for b, v in share.items() if v > 0.5}
            d = d[~d.block.isin(excluded)]
        if d.empty:
            return {"scope": block or "All blocks", "forecast_not_reliable": True,
                    "reason": "The meter has been stuck on one repeated reading for most of the last 8 weeks "
                              f"({excluded}), so a forecast would be meaningless."}
        hourly = d.groupby("timestamp").kwh.sum().asfreq("h").interpolate(limit=6)

        def predict(history, idx):
            h = history.dropna().tail(weeks * 168)
            prof = h.groupby([h.index.dayofweek, h.index.hour]).median()      # median: robust to odd hours
            return pd.Series([prof.get((t.dayofweek, t.hour), float("nan")) for t in idx], index=idx)

        end = hourly.index.max()
        abs_d = tot_d = abs_h = tot_h = 0.0
        for k in range(4, 0, -1):                       # backtest: predict each of the last 4 weeks from the weeks before it
            t_end = end - pd.Timedelta(hours=168 * (k - 1))
            t_start = t_end - pd.Timedelta(hours=167)
            test = hourly[t_start:t_end].dropna()
            pred = predict(hourly[:t_start - pd.Timedelta(hours=1)], test.index)
            ok = pred.notna()
            abs_h += (test[ok] - pred[ok]).abs().sum()
            tot_h += test[ok].sum()
            da, dp = test[ok].resample("D").sum(), pred[ok].resample("D").sum()
            abs_d += (da - dp).abs().sum()
            tot_d += da.sum()
        idx = pd.date_range(end + pd.Timedelta(hours=1), periods=168, freq="h")
        pred = predict(hourly, idx)
        daily = pred.resample("D").sum().round(1)
        kwh = float(pred.sum())
        out = {"scope": block or "All blocks",
               "forecast_from": str(idx[0]), "forecast_to": str(idx[-1]),
               "daily_kwh": {f"{k.date()} ({k.day_name()[:3]})": v for k, v in daily.items()},
               "total_kwh": round(kwh, 1), "cost_rs": round(kwh * self.tariff, 0), "co2_kg": round(kwh * self.co2, 0),
               "peak_hour_of_day": int(pred.groupby(pred.index.hour).mean().idxmax()),
               "backtest_daily_accuracy_pct": round(max(0.0, 100 * (1 - abs_d / max(tot_d, 1e-6))), 1),
               "backtest_hourly_accuracy_pct": round(max(0.0, 100 * (1 - abs_h / max(tot_h, 1e-6))), 1),
               "method": f"median of the same weekday and hour over the last {weeks} weeks",
               "note": "The data ends on " + str(end.date()) + ", so this is the week after that date, not the real next week."}
        if excluded:
            out["excluded_blocks_stuck_meter_pct_last_8_weeks"] = excluded
        return out

    # ---------- LLM loop ----------
    def _dispatch(self, name, args):
        table = {"data_overview": self.data_overview, "energy_summary": self.energy_summary,
                 "peak_and_waste": self.peak_and_waste, "compare_blocks": self.compare_blocks,
                 "savings_scenario": lambda **a: self.savings(a.get("percentage", 10), a.get("block")),
                 "list_anomalies": self.list_anomalies, "asset_health": self.asset_health_tool,
                 "asset_trend": self.asset_trend, "maintenance_orders": self.maintenance_orders,
                 "forecast_next_week": self.forecast_next_week,
                 "data_quality": self.data_quality}
        try:
            return table[name](**args)
        except Exception as e:  # tell the model what went wrong so it can retry
            return {"error": f"{type(e).__name__}: {e}"}

    def ask(self, question, history=None):
        """history: list of {'role': 'user'|'assistant', 'content': str} from earlier turns."""
        if not question or not question.strip():
            return "Please enter a question."
        if not self.client:
            return super().ask(question) + "\n\n_(Keyword mode: set ANTHROPIC_API_KEY for the full assistant.)_"
        msgs = [{"role": m["role"], "content": m["content"]} for m in (history or [])][-12:]
        msgs.append({"role": "user", "content": question})
        try:
            for _ in range(MAX_STEPS):
                r = self.client.messages.create(model=MODEL, max_tokens=1200, system=SYSTEM,
                                                tools=TOOLS, messages=msgs)
                if r.stop_reason != "tool_use":
                    return "".join(b.text for b in r.content if b.type == "text").strip()
                msgs.append({"role": "assistant", "content": r.content})
                results = [{"type": "tool_result", "tool_use_id": b.id,
                            "content": json.dumps(self._dispatch(b.name, b.input), default=str)}
                           for b in r.content if b.type == "tool_use"]
                msgs.append({"role": "user", "content": results})
            return "I needed too many steps for that. Could you narrow the question?"
        except Exception as e:
            return (super().ask(question)
                    + f"\n\n_(AI service unavailable: {type(e).__name__}: {str(e)[:300]})_")
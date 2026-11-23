"""Smart OFFLINE agent: no API key, no internet.
Understands blocks, assets, dates, percentages and follow-ups ("and Block C?").
Reuses the tools in llm_agent.SmartEnergyAgent, so numbers come straight from your data.
"""
import re

import pandas as pd

from llm_agent import SmartEnergyAgent

MONTH_NAMES = ["january", "february", "march", "april", "may", "june", "july",
               "august", "september", "october", "november", "december"]
MONTH_RE = r"\b(?:in|during|for|of|from)\s+(" + "|".join(MONTH_NAMES + [m[:3] for m in MONTH_NAMES] + ["sept"]) + r")\b"

PATTERNS = {
    "help": r"\b(hello|hi|hey|help|what can you|what do you do|capabilit\w+)\b",
    "quality": r"reliab|data quality|\bstuck|trustworth|how good is the data|missing data",
    "forecast": r"forecast|next (week|7 days)|coming week|upcoming week|tomorrow|projection|projected|expected usage",
    "compare": r"compar|versus|\bvs\b|difference|differ\b|between",
    "waste": r"wast|night|off[- ]?hours?|after[- ]?hours?|weekend|idle|standby|overnight|inefficien",
    "peak": r"\bpeak|busiest|rush|maximum demand",
    "saving": r"\bsav(e|ing|ings)\b|reduc\w*|\bcut\b|lower|decrease",
    "orders": r"work\s?orders?|maintenance orders?|\borders?\b|to-?do",
    "risk": r"high[- ]?risk|critical|urgent|immediate|needs? maintenance|attention|at risk|priorit\w+|fail\w*|breakdown|inspect\w*|danger|worst",
    "equipment": r"equipment|\basset|maintenance|health|\blifts?\b|generators?|hvac|elevator|machine",
    "anomaly": r"anomal|unusual|spike|abnormal|outlier|\bodd\b|weird|suspicious|irregular",
    "carbon": r"carbon|co2|emission|footprint|green ?house",
    "cost": r"\bcost|\bbill|rupee|\brs\b|₹|price|expens|spend|\bpay\b",
    "trend": r"trend|daily|weekly|monthly|per (day|week|month)|by (day|week|month)|over time|month[- ]?wise|day[- ]?wise|each month|hour of (the )?day|weekday",
    "usage": r"usage|consum|electric|energy|kwh|power|total|how much",
}
RANK_HIGH = r"highest|most|top|maximum|\bmax\b|biggest|largest|worst|\bmore\b"
RANK_LOW = r"lowest|least|minimum|\bmin\b|smallest|\bless\b|best"
FOLLOWUP = r"^(and|also|what about|how about|then|now|same|ok(ay)?,? and)\b"


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


class OfflineAgent(SmartEnergyAgent):

    # ---------------- parsing ----------------
    def _blocks(self, q):
        avail = sorted(self.df.block.unique())
        found, unknown = [], []
        for m in re.finditer(r"\bblock\b\s*-?\s*([a-z])\b", q.lower()):
            name = f"Block {m.group(1).upper()}"
            target = found if name in avail else unknown
            if name not in target:
                target.append(name)
        return found, unknown

    def _asset(self, q):
        m = re.search(r"\b(lift|generator|hvac)\s*-?\s*(\d+)\b", q.lower())
        if not m:
            return None
        kind = {"lift": "Lift", "generator": "Generator", "hvac": "HVAC"}[m.group(1)]
        return f"{kind} {int(m.group(2))}"

    def _kind(self, q):
        m = re.search(r"\b(lift|elevator|generator|hvac)s?\b", q.lower())
        if not m:
            return None
        return {"lift": "Lift", "elevator": "Lift", "generator": "Generator", "hvac": "HVAC"}[m.group(1)]

    def _period(self, q):
        ql = q.lower()
        end_data = self.df.timestamp.max().normalize()
        dates = re.findall(r"\d{4}-\d{2}-\d{2}", ql)
        if dates:
            s, e = dates[0], dates[-1]
            return s, e, f"from {s} to {e}"
        m = re.search(MONTH_RE, ql)
        if m:
            num = [mn[:3] for mn in MONTH_NAMES].index(m.group(1)[:3]) + 1
            year = end_data.year
            start = pd.Timestamp(year=year, month=num, day=1)
            end = start + pd.offsets.MonthEnd(0)
            return str(start.date()), str(end.date()), f"in {start.strftime('%B %Y')}"
        m = re.search(r"(?:last|past)\s+(\d+)\s+days?", ql)
        n = int(m.group(1)) if m else 7 if re.search(r"last week|past week", ql) else 30 if re.search(r"last month|past month", ql) else None
        if n:
            start = end_data - pd.Timedelta(days=n - 1)
            return str(start.date()), str(end_data.date()), f"in the last {n} days of data (ending {end_data.date()})"
        return None, None, "over the whole dataset"

    def _pct(self, q):
        m = re.search(r"(\d+(?:\.\d+)?)\s*(%|percent|per cent)", q.lower())
        return float(m.group(1)) if m else None

    def _intent(self, q, blocks):
        ql = q.lower()
        has = lambda k: re.search(PATTERNS[k], ql) is not None
        if has("help") and len(ql.split()) <= 6:
            return "help"
        if has("quality"):
            return "quality"
        if has("forecast"):
            return "forecast"
        if has("compare") and len(blocks) >= 2:
            return "compare"
        pct = self._pct(q) is not None
        if has("waste") and not pct:
            return "waste"
        if has("peak"):
            return "waste"
        if has("saving") and (pct or not has("waste")):
            return "saving"
        if has("orders"):
            return "orders"
        asset = self._asset(q)
        if has("risk"):
            return "asset" if asset else "risk"
        if asset:
            return "asset"
        if has("anomaly"):
            return "anomaly"
        if has("equipment"):
            return "equipment"
        if has("carbon"):
            return "carbon"
        if has("cost"):
            return "cost"
        if has("trend"):
            return "trend"
        if has("usage") or re.search(RANK_HIGH + "|" + RANK_LOW, ql):
            return "usage"
        return None

    # ---------------- main entry ----------------
    def ask(self, question, history=None):
        if not question or not question.strip():
            return "Please enter a question."
        try:
            return self._answer(question.strip(), history or [])
        except Exception as e:
            return f"Sorry, I couldn't process that ({type(e).__name__}: {e}). Try rephrasing."

    def _answer(self, q, history):
        blocks, unknown = self._blocks(q)
        ctx = dict(blocks=blocks, asset=self._asset(q), kind=self._kind(q),
                   period=self._period(q), pct=self._pct(q), q=q)
        intent = self._intent(q, blocks)

        # follow-up: "and Block C?" re-runs the previous question with the new details
        prev = [m["content"] for m in history if m["role"] == "user"]
        if prev and (intent is None or re.match(FOLLOWUP, q.lower())) and (blocks or ctx["asset"] or ctx["period"][0]):
            pq, pb, pi = None, [], None
            for cand in reversed(prev):
                cb, _ = self._blocks(cand)
                ci = self._intent(cand, cb)
                if ci is not None:
                    pq, pb, pi = cand, cb, ci
                    break
            if pq:
                if intent is None or intent == "usage":
                    intent = pi
                    ctx["q"] = pq + " " + q
                ctx["blocks"] = blocks or pb
                ctx["asset"] = ctx["asset"] or self._asset(pq)
                if not ctx["period"][0]:
                    ctx["period"] = self._period(pq)
                ctx["pct"] = ctx["pct"] or self._pct(pq)
                ctx["kind"] = ctx["kind"] or self._kind(pq)

        if unknown and not ctx["blocks"]:
            return f"I don't have data for {', '.join(unknown)}. Available blocks: {', '.join(sorted(self.df.block.unique()))}."
        if intent is None:
            if ctx["blocks"]:
                intent = "usage"
            else:
                return self._help(prefix="I'm not sure what you want to know. ")
        return getattr(self, "_h_" + intent)(ctx)

    # ---------------- helpers ----------------
    def _tot(self, block, ctx):
        s, e, _ = ctx["period"]
        return self.energy_summary(block=block, start=s, end=e)

    def _rank_words(self, q):
        ql = q.lower()
        if re.search(RANK_LOW, ql) and not re.search(RANK_HIGH, ql):
            return "low"
        if re.search(RANK_HIGH, ql):
            return "high"
        return None

    # ---------------- handlers ----------------
    def _help(self, prefix=""):
        return (prefix + "I can answer questions about your campus data. Examples:\n"
                "- *Which block uses the most electricity?* / *Compare Block A and Block C*\n"
                "- *What is the cost / carbon of Block B in March?*\n"
                "- *Where are we wasting energy at night?* / *When is the peak hour?*\n"
                "- *How much can we save if Block D cuts 15%?*\n"
                "- *Show unusual consumption in Block B*\n"
                "- *Which equipment needs attention?* / *Tell me about HVAC 5* / *Show work orders*\n"
                "- *Show the monthly trend for Block A*, then follow up with *and Block C?*")

    def _h_help(self, ctx):
        return self._help("Hi! ")

    def _h_usage(self, ctx):
        block = ctx["blocks"][0] if ctx["blocks"] else None
        r = self._tot(block, ctx)
        if "error" in r:
            return r["error"]
        label = ctx["period"][2]
        if block:
            return (f"**{block}** used **{r['total_kwh']:,.0f} kWh** {label}, costing about "
                    f"**Rs {r['cost_rs']:,.0f}** and emitting **{r['co2_kg']:,.0f} kg CO2**.")
        by = r["by_block_kwh"]
        total = r["total_kwh"]
        words = self._rank_words(ctx["q"])
        top = (min if words == "low" else max)(by, key=by.get)
        head = (f"**{top}** has the {'lowest' if words == 'low' else 'highest'} consumption {label}: "
                f"**{by[top]:,.0f} kWh** ({by[top] / total:.0%} of the total).")
        rows = [[b, f"{v:,.0f}", f"{v / total:.0%}"] for b, v in sorted(by.items(), key=lambda x: -x[1])]
        return head + "\n\n" + md_table(["Block", "kWh", "Share"], rows) + f"\n\nTotal: **{total:,.0f} kWh**."

    def _money_table(self, ctx, key, unit, rate_name):
        r = self._tot(None, ctx)
        if "error" in r:
            return r["error"]
        if ctx["blocks"]:
            blocks = ctx["blocks"]
            by = {b: self._tot(b, ctx)[key] for b in blocks}
        else:
            by = {b: round(v * (self.tariff if key == "cost_rs" else self.co2), 0) for b, v in r["by_block_kwh"].items()}
        top = max(by, key=by.get)
        rows = [[b, f"{v:,.0f}"] for b, v in sorted(by.items(), key=lambda x: -x[1])]
        total = sum(by.values())
        unit_txt = "Rs" if key == "cost_rs" else "kg CO2"
        tail = f"\n\nTotal **{'Rs ' + format(total, ',.0f') if key == 'cost_rs' else format(total, ',.0f') + ' kg CO2'}**." + (f" **{top}** is the largest contributor." if len(by) > 1 else "")
        return (f"{rate_name} {ctx['period'][2]}:\n\n" + md_table(["Block", unit], rows) + tail)

    def _h_cost(self, ctx):
        return self._money_table(ctx, "cost_rs", "Cost (Rs)", f"Estimated electricity cost (Rs {self.tariff:g}/kWh)")

    def _h_carbon(self, ctx):
        return self._money_table(ctx, "co2_kg", "CO2 (kg)", f"Estimated CO2 emissions ({self.co2:g} kg/kWh)")

    def _h_compare(self, ctx):
        b1, b2 = ctx["blocks"][:2]
        r1, r2 = self._tot(b1, ctx), self._tot(b2, ctx)
        for r in (r1, r2):
            if "error" in r:
                return r["error"]
        hi, lo = (b1, b2) if r1["total_kwh"] >= r2["total_kwh"] else (b2, b1)
        rh, rl = (r1, r2) if hi == b1 else (r2, r1)
        diff = rh["total_kwh"] - rl["total_kwh"]
        pct = diff / max(rl["total_kwh"], 1) * 100
        rows = [[b, f"{r['total_kwh']:,.0f}", f"{r['cost_rs']:,.0f}", f"{r['co2_kg']:,.0f}"] for b, r in ((b1, r1), (b2, r2))]
        return (f"**{hi}** uses **{diff:,.0f} kWh ({pct:.0f}%) more** than **{lo}** {ctx['period'][2]}, "
                f"which is about Rs {diff * self.tariff:,.0f} extra.\n\n"
                + md_table(["Block", "kWh", "Cost (Rs)", "CO2 (kg)"], rows))

    def _h_saving(self, ctx):
        pct = ctx["pct"] if ctx["pct"] is not None else 10.0
        if not 0 <= pct <= 100:
            return "Please give a reduction between 0 and 100%."
        block = ctx["blocks"][0] if ctx["blocks"] else None
        r = self._tot(block, ctx)
        if "error" in r:
            return r["error"]
        kwh = r["total_kwh"] * pct / 100
        out = (f"If **{block or 'all blocks'}** cut consumption by **{pct:g}%** {ctx['period'][2]}, you would save "
               f"**{kwh:,.0f} kWh**, **Rs {kwh * self.tariff:,.0f}** and **{kwh * self.co2:,.0f} kg CO2**.")
        w = self.peak_and_waste(block)
        if "error" not in w:
            off = sum(v["off_hours_kwh"] for v in w.values())
            out += (f"\n\nWhere to find it: off-hours (7pm-7am and weekends) use about **{off:,.0f} kWh** "
                    f"(Rs {off * self.tariff:,.0f}). Cutting just 30% of that saves Rs {off * 0.3 * self.tariff:,.0f}.")
        return out

    def _h_waste(self, ctx):
        block = ctx["blocks"][0] if ctx["blocks"] else None
        w = self.peak_and_waste(block)
        if "error" in w:
            return w["error"]
        rows = [[b, f"{v['peak_hour']}:00", f"{v['off_hours_share_pct']}%",
                 v["night_vs_day_ratio"] if v["night_vs_day_ratio"] is not None else "n/a",
                 f"{v['off_hours_kwh']:,.0f}", f"{v['off_hours_cost_rs']:,.0f}",
                 f"{v['stuck_hours_pct']}% stuck" + ("" if v["reliable"] else " (not reliable)")] for b, v in w.items()]
        ok = {b: v for b, v in w.items() if v["reliable"] and v["night_vs_day_ratio"] is not None}
        txt = ("Off-hours means 7pm-7am plus weekends, which is 64% of all hours, so a perfectly flat load would show 64%. "
               "The night/day ratio is the better signal: about 1.0 means the block runs as much at night as in the day.\n\n"
               + md_table(["Block", "Peak hour", "Off-hours share", "Night/day ratio", "Off-hours kWh",
                           "Off-hours cost (Rs)", "Data quality"], rows))
        if ok:
            worst = max(ok, key=lambda b: ok[b]["night_vs_day_ratio"])
            r = ok[worst]["night_vs_day_ratio"]
            txt += (f"\n\n**{worst}** is the clearest case: at night it uses {r:.0%} of its daytime level "
                    f"(Rs {ok[worst]['off_hours_cost_rs']:,.0f} off-hours cost).")
            if r > 0.8:
                txt += " That is high for a building that should be mostly idle then. Check HVAC schedules, lighting and standby equipment."
        bad = [b for b, v in w.items() if not v["reliable"]]
        if bad:
            txt += "\n\n_Not reliable because many readings are stuck on one value: " + ", ".join(bad) + "._"
        return txt

    def _h_trend(self, ctx):
        ql = ctx["q"].lower()
        block = ctx["blocks"][0] if ctx["blocks"] else None
        g = ("month" if re.search(r"month", ql) else "week" if "week" in ql and "weekday" not in ql else
             "weekday" if "weekday" in ql else "hour_of_day" if "hour" in ql else "day")
        s, e, label = ctx["period"]
        r = self.energy_summary(block=block, start=s, end=e, group_by=g)
        if "error" in r:
            return r["error"]
        key = {"day": "series_kwh", "week": "series_kwh", "month": "series_kwh",
               "hour_of_day": "avg_kwh_per_hour_by_hour", "weekday": "avg_kwh_per_hour_by_weekday"}[g]
        series = {(pd.Timestamp(k).strftime("%b %Y") if g == "month" else str(k)): v for k, v in r[key].items()}
        if len(series) < 2:
            return self._h_usage(ctx)
        hi, lo = max(series, key=series.get), min(series, key=series.get)
        unit = "avg kWh per hour" if g in ("hour_of_day", "weekday") else "kWh"
        out = (f"**{r['scope']}** by {g.replace('_', ' ')} {label}: highest is **{hi}** ({series[hi]:,.1f} {unit}), "
               f"lowest is **{lo}** ({series[lo]:,.1f} {unit}), average {sum(series.values()) / len(series):,.1f}.")
        items = list(series.items())
        shown = items if len(items) <= 12 else items[-10:]
        return out + "\n\n" + md_table(["Period", unit], [[k, f"{v:,.1f}"] for k, v in shown])

    def _h_quality(self, ctx):
        q = self.data_quality()
        if "error" in q:
            return q["error"]
        rows = [[b, f"{v['stuck_hours_pct_whole_year']}%", f"{v['stuck_hours_pct_last_8_weeks']}%",
                 f"{v['longest_identical_run_hours']} h", v["reliability"],
                 "yes" if v["anomaly_scored"] else "no", "yes" if v["forecast_available"] else "no"]
                for b, v in q.items() if b != "notes"]
        return ("A stuck meter repeats one reading for hours or weeks, so those hours are not real consumption.\n\n"
                + md_table(["Block", "Stuck (whole year)", "Stuck (last 8 weeks)", "Longest identical run",
                            "Reliability", "Anomalies scored", "Forecast available"], rows)
                + "\n\n" + q["notes"])

    def _h_forecast(self, ctx):
        block = ctx["blocks"][0] if ctx["blocks"] else None
        r = self.forecast_next_week(block)
        if "error" in r:
            return r["error"]
        if r.get("forecast_not_reliable"):
            return "No forecast: " + r["reason"]
        rows = [[k, f"{v:,.0f}"] for k, v in r["daily_kwh"].items()]
        return (f"**{r['scope']}** forecast for {r['forecast_from'][:10]} to {r['forecast_to'][:10]}: "
                f"**{r['total_kwh']:,.0f} kWh**, about Rs {r['cost_rs']:,.0f} and {r['co2_kg']:,.0f} kg CO2. "
                f"Busiest hour of the day: {r['peak_hour_of_day']}:00.\n\n"
                + md_table(["Day", "Forecast kWh"], rows)
                + ("\n\n_Left out because their meters were stuck on one value: "
                   + ", ".join(f"{k} ({v}% of the last 8 weeks)" for k, v in r["excluded_blocks_stuck_meter_pct_last_8_weeks"].items()) + "._"
                   if r.get("excluded_blocks_stuck_meter_pct_last_8_weeks") else "")
                + f"\n\nBacktest on the last 4 weeks: daily accuracy **{r['backtest_daily_accuracy_pct']}%**, "
                  f"hourly accuracy {r['backtest_hourly_accuracy_pct']}%. Method: {r['method']}. {r['note']}")

    def _h_anomaly(self, ctx):
        block = ctx["blocks"][0] if ctx["blocks"] else None
        if block and "scored" in self.df.columns and not self.df[self.df.block == block].scored.iloc[0]:
            return (f"Anomaly detection is not scored for **{block}**: its usage is too low for the detector to be reliable, "
                    "so no flags are shown.")
        r = self.list_anomalies(block, 5)
        summ, top = r["summary"], r["top"]
        if summ["total_flagged_hours"] == 0:
            return "No anomalies were flagged in this data."
        if not block and self._rank_words(ctx["q"]) == "high" and re.search(r"which|most|highest", ctx["q"].lower()):
            counts = summ["block_counts"]
            b = max(counts, key=counts.get)
            return (f"**{b}** has the most flagged hours (**{counts[b]}**). Total flagged: "
                    f"{summ['total_flagged_hours']} hours across {summ['blocks_affected']} blocks.\n\n"
                    + md_table(["Block", "Flagged hours"], [[k, v] for k, v in sorted(counts.items(), key=lambda x: -x[1])]))
        if not top:
            return f"No anomalies were flagged for {block}."
        rows = [[str(a["timestamp"]), a["block"], f"{a['kwh']:,.1f}", a["reason"]] for a in top]
        n = summ["block_counts"].get(block, summ["total_flagged_hours"]) if block else summ["total_flagged_hours"]
        return (f"**{n} flagged hour{'s' if n != 1 else ''}** for {block or 'all blocks'}. "
                f"{'The biggest spike' if len(top) == 1 else f'The {len(top)} biggest spikes'}:\n\n"
                + md_table(["When", "Block", "kWh", "Why flagged"], rows)
                + "\n\nA spike is flagged when both the Isolation Forest and the per-hour baseline rule agree.")

    def _h_asset(self, ctx):
        name = ctx["asset"]
        d = self.asset_health_tool(asset=name)
        if isinstance(d, dict) and "error" in d:
            return f"{name}: {d['error']}"
        a = d[0]
        t = self.asset_trend(name, 7)
        trend = ""
        if isinstance(t, list) and len(t) >= 2:
            trend = (f"\n\nOver the last {len(t)} days temperature moved {t[0]['temperature']} to {t[-1]['temperature']} C "
                     f"and vibration {t[0]['vibration']} to {t[-1]['vibration']} mm/s.")
        return (f"**{a['asset']}**: status **{a['status']}**, health {a['health_score']}/100, "
                f"failure risk within 7 days **{a['risk']:.0%}**.\n\n"
                f"Temperature {a['temperature']} C, vibration {a['vibration']} mm/s.\n"
                f"Recommendation: **{a['recommendation']}**." + trend + "\n\n_Based on simulated maintenance data._")

    def _h_risk(self, ctx):
        d = self.asset_health_tool(kind=ctx["kind"], only_high_risk=True)

        if isinstance(d, dict):
            allr = self.asset_health_tool(kind=ctx["kind"])
            watch = sorted(allr, key=lambda x: -x["risk"])[:3]

            return (
                "No assets are above the 50% risk threshold.\n\n"
                "Assets closest to the threshold:\n\n"
                + md_table(
                    ["Asset", "Risk", "Health", "Recommendation"],
                    [[a["asset"], f"{a['risk']:.0%}",
                    a["health_score"], a["recommendation"]]
                    for a in watch]
                )
            )

        rows = [
            [
                a["asset"],
                f"{a['risk']:.0%}",
                a["health_score"],
                f"{a['temperature']} C",
                f"{a['vibration']} mm/s",
                a["recommendation"]
            ]
            for a in sorted(d, key=lambda x: -x["risk"])
        ]

        answer = (
            f"**{len(d)} asset(s) need maintenance attention.**\n\n"
            + md_table(
                ["Asset", "Failure risk", "Health",
                "Temperature", "Vibration", "Recommendation"],
                rows
            )
        )

        answer += (
            "\n\n**Why are these assets flagged?**\n"
            "The Random Forest model analyses temperature and vibration "
            "deviations from each asset's normal baseline, their recent "
            "moving averages and vibration trends. It uses these patterns "
            "to estimate failure risk within the next 7 days.\n\n"
            "Assets with risk >= 50% receive High priority and an inspection "
            "recommendation within 3 days. This is a model-based warning, "
            "not confirmation of an actual equipment fault.\n\n"
            "_Based on simulated campus sensor data._"
        )

        return answer


    def _h_equipment(self, ctx):
        d = self.asset_health_tool(kind=ctx["kind"])
        if isinstance(d, dict):
            return d["error"]
        counts = pd.Series([a["status"] for a in d]).value_counts().to_dict()
        worst = sorted(d, key=lambda x: x["health_score"])[:5]
        return (f"**{len(d)} {(ctx['kind'].lower() + 's') if ctx['kind'] and ctx['kind'] != 'HVAC' else (ctx['kind'] + ' units') if ctx['kind'] else 'assets'}** monitored: "
                    + ", ".join(f"{v} {k}" for k, v in counts.items()) + ".\n\nLowest health:\n\n"
                    + md_table(["Asset", "Status", "Health", "Risk"], [[a["asset"], a["status"], a["health_score"], f"{a['risk']:.0%}"] for a in worst])
                    + "\n\n_Based on simulated maintenance data._")

    def _h_orders(self, ctx):
        o = self.maintenance_orders()
        if not o:
            return "There are no open work orders."
        return f"**{len(o)} open work orders:**\n\n" + md_table(
                ["Asset", "Priority", "Work order"], [[x["asset"], x["priority"], x["work_order"]] for x in o])
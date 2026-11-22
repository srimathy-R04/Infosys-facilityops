
"""Energy Intelligence Agent - Hybrid-ready Python fallback."""

import re
import pandas as pd
import maintenance


class EnergyAgent:

    def __init__(self, df, tariff=8.0, co2=0.727):
        self.df = df.copy()
        self.tariff = float(tariff)
        self.co2 = float(co2)
        self._maintenance_cache = None

    # ==========================================
    # ENERGY ANALYSIS TOOLS
    # ==========================================

    def usage_by_block(self):
        return self.df.groupby("block")["kwh"].sum().round(2).to_dict()

    def top_block(self):
        usage = self.usage_by_block()
        block = max(usage, key=usage.get)
        return {"block": block, "kwh": usage[block]}

    def compare_blocks(self, block1, block2):
        usage = self.usage_by_block()

        result = {}
        for block in [block1, block2]:
            if block not in usage:
                return {"error": f"{block} was not found."}
            result[block] = usage[block]

        higher = max(result, key=result.get)
        lower = min(result, key=result.get)

        return {
            "consumption": result,
            "higher_block": higher,
            "difference_kwh": round(result[higher] - result[lower], 2)
        }

    def cost(self):
        return {
            block: round(kwh * self.tariff, 2)
            for block, kwh in self.usage_by_block().items()
        }

    def carbon(self):
        return {
            block: round(kwh * self.co2, 2)
            for block, kwh in self.usage_by_block().items()
        }

    def savings(self, percentage=10, block=None):
        usage = self.usage_by_block()

        if block:
            if block not in usage:
                return {"error": f"{block} was not found."}
            total = usage[block]
        else:
            total = sum(usage.values())

        saved_energy = total * percentage / 100

        return {
            "scope": block or "All blocks",
            "reduction_percent": percentage,
            "energy_saved_kwh": round(saved_energy, 2),
            "money_saved_rs": round(saved_energy * self.tariff, 2),
            "co2_reduced_kg": round(saved_energy * self.co2, 2)
        }

    # ==========================================
    # ANOMALY ANALYSIS TOOLS
    # ==========================================

    def anomalies(self, block=None, limit=5):
        data = self.df[self.df["anomaly"] == True].copy()

        if block:
            data = data[data["block"].str.lower() == block.lower()]

        if "ratio" in data.columns:
            data = data.sort_values("ratio", ascending=False)

        columns = ["timestamp", "block", "kwh", "reason"]

        return data[columns].head(limit).to_dict("records")

    def anomaly_summary(self):
        data = self.df[self.df["anomaly"] == True]

        return {
            "total_flagged_hours": len(data),
            "blocks_affected": data["block"].nunique(),
            "block_counts": data.groupby("block").size().to_dict()
        }

    # ==========================================
    # MAINTENANCE TOOLS
    # ==========================================

    def _maintenance_data(self):
        if self._maintenance_cache is None:
            self._maintenance_cache = maintenance.run()

        return self._maintenance_cache

    def asset_health(self):
        status, orders, hist, metrics = self._maintenance_data()

        columns = [
            "asset", "kind", "temperature", "vibration",
            "risk", "health_score", "status", "recommendation"
        ]

        return status[columns].to_dict("records")

    def high_risk_assets(self):
        status, orders, hist, metrics = self._maintenance_data()

        risky = status[status["risk"] >= 0.5]

        return risky[
            ["asset", "temperature", "vibration", "risk",
             "health_score", "recommendation"]
        ].to_dict("records")

    def maintenance_orders(self):
        status, orders, hist, metrics = self._maintenance_data()

        return orders.to_dict("records")

    def asset_details(self, asset_name):
        status, orders, hist, metrics = self._maintenance_data()

        result = status[
            status["asset"].str.lower() == asset_name.lower()
        ]

        if result.empty:
            return {"error": f"{asset_name} was not found."}

        return result.iloc[0].to_dict()

    def asset_history(self, asset_name):
        status, orders, hist, metrics = self._maintenance_data()

        result = hist[
            hist["asset"].str.lower() == asset_name.lower()
        ].sort_values("day")

        if result.empty:
            return {"error": f"{asset_name} was not found."}

        return result.tail(7)[
            ["day", "temperature", "vibration"]
        ].to_dict("records")

    # ==========================================
    # QUESTION UNDERSTANDING
    # ==========================================

    def _extract_blocks(self, question):
        available = sorted(self.df["block"].unique())

        return [
            block for block in available
            if block.lower() in question.lower()
        ]

    def _extract_asset(self, question):
        status, orders, hist, metrics = self._maintenance_data()

        for asset in status["asset"].unique():
            if asset.lower() in question.lower():
                return asset

        return None

    # ==========================================
    # RESPONSE GENERATION
    # ==========================================

    def _keyword(self, question):
        q = question.lower().strip()
        blocks = self._extract_blocks(q)

        # 1. Block comparison
        if any(w in q for w in ["compare", "difference", "versus", "vs"]):
            if len(blocks) >= 2:
                result = self.compare_blocks(blocks[0], blocks[1])

                if "error" in result:
                    return result["error"]

                values = result["consumption"]

                return (
                    f"{blocks[0]} consumes {values[blocks[0]]:,.2f} kWh, "
                    f"while {blocks[1]} consumes {values[blocks[1]]:,.2f} kWh. "
                    f"{result['higher_block']} consumes more by "
                    f"{result['difference_kwh']:,.2f} kWh."
                )

            return "Please mention two blocks, for example: Compare Block A and Block C."

        # 2. Energy savings
        if any(w in q for w in ["save", "saving", "reduce", "reduction"]):
            match = re.search(r"(\d+(?:\.\d+)?)\s*%", q)
            percentage = float(match.group(1)) if match else 10

            if not 0 <= percentage <= 100:
                return "Please provide a reduction percentage between 0 and 100."

            result = self.savings(
                percentage,
                blocks[0] if blocks else None
            )

            return (
                f"If {result['scope']} reduces energy consumption by "
                f"{percentage}%, estimated savings are "
                f"{result['energy_saved_kwh']:,.2f} kWh, "
                f"Rs {result['money_saved_rs']:,.2f}, and "
                f"{result['co2_reduced_kg']:,.2f} kg of CO2."
            )

        # 3. High-risk equipment
        if any(w in q for w in ["high risk", "critical", "immediate attention",
                                 "at risk", "urgent"]):
            assets = self.high_risk_assets()

            if not assets:
                return "No high-risk assets were identified."

            answer = ["High-risk equipment:"]
            for a in assets:
                answer.append(
                    f"{a['asset']}: risk {a['risk']:.0%}, "
                    f"health {a['health_score']}/100. "
                    f"Recommendation: {a['recommendation']}."
                )

            return "\n".join(answer)

        # 4. Maintenance work orders
        if "work order" in q or "maintenance order" in q:
            orders = self.maintenance_orders()

            if not orders:
                return "There are no pending maintenance work orders."

            return "\n".join(
                f"{o['asset']} ({o['priority']}): {o['work_order']}"
                for o in orders
            )

        # 5. Specific equipment details
        asset = None
        if any(w in q for w in ["asset", "equipment", "temperature",
                                 "vibration", "lift", "generator", "hvac"]):
            asset = self._extract_asset(q)

        if asset:
            details = self.asset_details(asset)

            return (
                f"Equipment: {details['asset']}\n"
                f"Temperature: {details['temperature']} C\n"
                f"Vibration: {details['vibration']} mm/s\n"
                f"Failure risk within 7 days: {details['risk']:.1%}\n"
                f"Health score: {details['health_score']}/100\n"
                f"Status: {details['status']}\n"
                f"Recommendation: {details['recommendation']}\n"
                f"Note: Based on simulated maintenance data."
            )

        # 6. General equipment health
        if any(w in q for w in ["equipment health", "asset health",
                                 "maintenance status"]):
            assets = self.asset_health()

            return "\n".join(
                f"{a['asset']}: {a['status']} "
                f"(health {a['health_score']}/100)"
                for a in assets
            )

        # 7. Anomalies
        if any(w in q for w in ["anomal", "unusual", "spike", "abnormal"]):
            data = self.anomalies(blocks[0] if blocks else None)

            if not data:
                return "No flagged anomalies were found for this query."

            return "\n".join(
                f"{a['block']} at {a['timestamp']}: "
                f"{a['kwh']} kWh. Reason: {a['reason']}"
                for a in data
            )

        # 8. Carbon emissions
        if any(w in q for w in ["carbon", "co2", "emission"]):
            values = self.carbon()
            return "CO2 emissions by block (kg): " + ", ".join(
                f"{b}: {v:,.2f}" for b, v in values.items()
            )

        # 9. Electricity cost
        if any(w in q for w in ["cost", "bill", "rupee", "price"]):
            values = self.cost()
            return "Estimated electricity cost (Rs): " + ", ".join(
                f"{b}: {v:,.2f}" for b, v in values.items()
            )

        # 10. Highest-consuming block
        if any(w in q for w in ["highest", "maximum", "most", "top"]):
            result = self.top_block()

            return (
                f"{result['block']} has the highest electricity "
                f"consumption at {result['kwh']:,.2f} kWh."
            )

        # 11. General energy usage
        if any(w in q for w in ["usage", "consumption", "electricity", "energy"]):
            values = self.usage_by_block()

            return "Electricity consumption by block (kWh): " + ", ".join(
                f"{b}: {v:,.2f}" for b, v in values.items()
            )

        return (
            "I can help with energy consumption, block comparisons, "
            "anomalies, electricity cost, carbon emissions, energy savings, "
            "equipment health and maintenance work orders. "
            "Please try asking one of these questions."
        )

    def ask(self, question):
        if not question or not question.strip():
            return "Please enter a question."

        try:
            return self._keyword(question)
        except Exception as e:
            return f"Unable to process this question: {type(e).__name__}: {e}"

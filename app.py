"""FacilityOps dashboard.  Run:  py -m streamlit run app.py   (from the fo folder)"""
import os

import pandas as pd
import plotly.express as px
import streamlit as st

import anomaly
import gemini_agent
import maintenance

st.set_page_config(page_title="FacilityOps | Energy & Maintenance", page_icon="⚡", layout="wide")

# ---- optional Gemini key (kept in .streamlit/secrets.toml, never committed to GitHub) ----
try:
    os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"].strip()
except Exception:
    pass

# ------------------------------------------------------------------ look and feel
PALETTE = ["#2dd4bf", "#60a5fa", "#a78bfa", "#fbbf24", "#f472b6", "#34d399"]
STATUS_COLORS = {"Excellent": "#34d399", "Good": "#2dd4bf", "Warning": "#fbbf24", "Critical": "#f87171"}

st.markdown("""
<style>
.block-container {padding-top: 1.4rem; max-width: 1320px;}
#MainMenu, footer {visibility: hidden;}
.hero {background: linear-gradient(120deg, #0f766e 0%, #1d4ed8 100%); border-radius: 18px;
       padding: 22px 28px; margin-bottom: 18px;}
.hero h1 {margin: 0; font-size: 1.85rem; color: #ffffff; letter-spacing: -0.01em;}
.hero p {margin: 6px 0 0; color: #d6e4ff; font-size: 0.98rem;}
.kpi {background: #111a2e; border: 1px solid #1f2b45; border-radius: 14px; padding: 14px 16px; height: 100%;}
.kpi .l {font-size: 0.74rem; color: #8da2c0; text-transform: uppercase; letter-spacing: 0.07em;}
.kpi .v {font-size: 1.55rem; font-weight: 700; color: #e6edf7; margin: 2px 0;}
.kpi .s {font-size: 0.78rem; color: #2dd4bf;}
.kpi.warn .s {color: #fbbf24;}
.kpi.bad .s {color: #f87171;}
.stTabs [data-baseweb="tab-list"] {gap: 6px;}
.stTabs [data-baseweb="tab"] {background: #111a2e; border-radius: 10px 10px 0 0; padding: 8px 18px;}
.stTabs [aria-selected="true"] {background: #0f766e; color: #ffffff;}
.badge {display: inline-block; padding: 2px 11px; border-radius: 999px; font-size: 0.75rem;
        background: #12324a; color: #7dd3fc; border: 1px solid #1e4e73;}
.note {color: #8da2c0; font-size: 0.82rem;}
</style>
""", unsafe_allow_html=True)


def kpi(col, label, value, sub="", tone=""):
    col.markdown(f'<div class="kpi {tone}"><div class="l">{label}</div><div class="v">{value}</div>'
                 f'<div class="s">{sub}</div></div>', unsafe_allow_html=True)


def style(fig, height=360):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      height=height, margin=dict(l=10, r=10, t=50, b=10), colorway=PALETTE,
                      title_font_size=16, legend=dict(orientation="h", y=-0.18))
    fig.update_xaxes(gridcolor="#1f2b45", zeroline=False)
    fig.update_yaxes(gridcolor="#1f2b45", zeroline=False)
    return fig


def section(title, caption=""):
    st.markdown(f"#### {title}")
    if caption:
        st.markdown(f'<div class="note">{caption}</div>', unsafe_allow_html=True)


# ------------------------------------------------------------------ data
st.markdown('<div class="hero"><h1>⚡ FacilityOps</h1>'
            '<p>Energy monitoring, anomaly detection and predictive maintenance for campus buildings.</p></div>',
            unsafe_allow_html=True)

if not os.path.exists("data/energy.csv"):
    st.error("data/energy.csv not found. Run prep_kaggle.py first (see README.md).")
    st.stop()


@st.cache_data
def energy_data():
    return anomaly.detect(anomaly.load())


@st.cache_data
def maintenance_data():
    return maintenance.run()


df = energy_data()

with st.sidebar:
    st.markdown("### Settings")
    tariff = st.number_input("Tariff (Rs per kWh)", value=8.0, step=0.5)
    co2 = st.number_input("Grid CO2 (kg per kWh)", value=0.727, step=0.01)
    st.markdown("---")
    st.markdown("### About")
    st.markdown(
        f'<div class="note">Energy data: real hourly readings from the Kaggle ASHRAE dataset '
        f'({df.timestamp.min():%d %b %Y} to {df.timestamp.max():%d %b %Y}).<br><br>'
        f'Anomaly accuracy is measured on planted spikes.<br><br>'
        f'Maintenance data is simulated.</div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Energy", "Maintenance", "Ask the Agent"])

# ------------------------------------------------------------------ tab 1: energy
with tab1:
    by_block = df.groupby("block")["kwh"].sum().round(0)
    total = df.kwh.sum()
    flagged = int(df.anomaly.sum())
    top_block = by_block.idxmax()

    c1, c2, c3, c4, c5 = st.columns(5)
    kpi(c1, "Total energy", f"{total:,.0f} kWh", f"{len(df):,} hourly readings")
    kpi(c2, "Highest block", top_block, f"{by_block.max() / total:.0%} of total")
    kpi(c3, "Anomalous hours", f"{flagged:,}", f"{flagged / len(df):.2%} of all hours",
        "warn" if flagged else "")
    kpi(c4, "Estimated cost", f"Rs {total * tariff:,.0f}", f"at Rs {tariff:g} per kWh")
    kpi(c5, "CO2 emitted", f"{total * co2:,.0f} kg", f"at {co2:g} kg per kWh")
    st.write("")

    left, right = st.columns(2)
    bar_df = by_block.reset_index().sort_values("kwh")
    left.plotly_chart(style(px.bar(bar_df, x="kwh", y="block", orientation="h", text_auto=",.0f",
                                   title="Block-wise electricity (kWh)")), use_container_width=True)
    right.plotly_chart(style(px.pie(by_block.reset_index(), names="block", values="kwh", hole=0.55,
                                    title="Share of total consumption")), use_container_width=True)

    section("Block explorer", "Daily consumption with the days that contain a flagged hour highlighted.")
    block = st.radio("Choose a block", sorted(df.block.unique()), horizontal=True, label_visibility="collapsed")
    d = df[df.block == block]
    daily = d.set_index("timestamp").kwh.resample("D").sum().reset_index()
    fig = px.area(daily, x="timestamp", y="kwh", title=f"{block}: daily electricity (kWh)")
    fig.update_traces(line_color=PALETTE[0], fillcolor="rgba(45,212,191,0.15)")
    flag_days = set(d[d.anomaly].timestamp.dt.floor("D"))
    marks = daily[daily.timestamp.isin(flag_days)]
    fig.add_scatter(x=marks.timestamp, y=marks.kwh, mode="markers", name="Day with a flagged hour",
                    marker=dict(color="#f87171", size=8, line=dict(color="#0b1220", width=1)))
    st.plotly_chart(style(fig, 380), use_container_width=True)

    section("Flagged anomalies", "Largest spikes first. Flagged only when both the Isolation Forest and the baseline rule agree.")
    top = (df[df.anomaly].sort_values("ratio", ascending=False)
           [["timestamp", "block", "kwh", "ratio", "reason"]].head(20))
    st.dataframe(top, use_container_width=True, hide_index=True, column_config={
        "timestamp": "When", "block": "Block",
        "kwh": st.column_config.NumberColumn("kWh", format="%.1f"),
        "ratio": st.column_config.NumberColumn("Times normal", format="%.1fx"),
        "reason": "Why flagged"})

    with st.expander("Model evaluation (on planted spikes)"):
        if st.button("Run evaluation"):
            st.dataframe(anomaly.evaluate(anomaly.load()), use_container_width=True, hide_index=True)
            st.caption("Spikes are planted in the real readings because real anomaly labels are not available.")

# ------------------------------------------------------------------ tab 2: maintenance
with tab2:
    status, orders, hist, metrics = maintenance_data()
    high = int((status.risk >= 0.5).sum())
    m1, m2, m3, m4 = st.columns(4)
    kpi(m1, "Assets monitored", len(status), "lifts, generators, HVAC")
    kpi(m2, "High-risk assets", high, "risk of 50% or more", "bad" if high else "")
    kpi(m3, "Open work orders", len(orders), "needs inspection", "warn" if len(orders) else "")
    kpi(m4, "Model F1 score", f"{metrics['f1']:.2f}", "on unseen assets (simulated data)")
    st.write("")

    cl, cr = st.columns([1, 2])
    counts = status.status.value_counts().reset_index()
    counts.columns = ["status", "assets"]
    cl.plotly_chart(style(px.pie(counts, names="status", values="assets", hole=0.55, title="Equipment health",
                                 color="status", color_discrete_map=STATUS_COLORS), 360), use_container_width=True)
    with cr:
        section("Asset health", "Lowest health first.")
        st.dataframe(status, use_container_width=True, hide_index=True, height=320, column_config={
            "asset": "Asset", "kind": "Type",
            "temperature": st.column_config.NumberColumn("Temp (C)", format="%.1f"),
            "vibration": st.column_config.NumberColumn("Vibration (mm/s)", format="%.2f"),
            "risk": st.column_config.ProgressColumn("Failure risk (7 days)", min_value=0, max_value=1, format="%.2f"),
            "health_score": st.column_config.ProgressColumn("Health", min_value=0, max_value=100, format="%d"),
            "status": "Status", "recommendation": "Recommendation"})

    section("Maintenance work orders")
    if len(orders):
        st.dataframe(orders, use_container_width=True, hide_index=True)
    else:
        st.success("No work orders.")

    section("Asset history")
    asset = st.selectbox("Choose an asset", sorted(hist.asset.unique()), label_visibility="collapsed")
    h = hist[hist.asset == asset]
    t1, t2 = st.columns(2)
    t1.plotly_chart(style(px.line(h, x="day", y="temperature", title=f"{asset}: temperature (C)"), 300),
                    use_container_width=True)
    t2.plotly_chart(style(px.line(h, x="day", y="vibration", title=f"{asset}: vibration (mm/s)"), 300),
                    use_container_width=True)
    st.caption("Simulated lift, generator and HVAC sensor data.")

# ------------------------------------------------------------------ tab 3: agent
with tab3:
    mode = "Gemini with offline fallback" if os.environ.get("GEMINI_API_KEY") else "Offline mode"
    st.markdown(f'<span class="badge">{mode}</span>', unsafe_allow_html=True)
    st.markdown("Ask about energy use, costs, carbon, anomalies, savings or equipment health.")

    if "chat" not in st.session_state:
        st.session_state.chat = []

    suggestions = ["Which block uses the most electricity?", "Where are we wasting energy at night?",
                   "Which equipment needs attention?", "How much can we save if we cut 15%?"]
    cols = st.columns(len(suggestions))
    for col, text in zip(cols, suggestions):
        if col.button(text, use_container_width=True):
            st.session_state.pending = text

    for m in st.session_state.chat:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    prompt = st.chat_input("Type your question...") or st.session_state.pop("pending", None)
    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)
        bot = gemini_agent.GeminiAgent(df, tariff, co2, maintenance_data=maintenance_data())
        with st.chat_message("assistant"):
            with st.spinner("Analysing your data..."):
                answer = bot.ask(prompt, st.session_state.chat)
            st.markdown(answer)
        st.session_state.chat += [{"role": "user", "content": prompt},
                                  {"role": "assistant", "content": answer}]

    if st.session_state.chat and st.button("Clear conversation"):
        st.session_state.chat = []
        st.rerun()
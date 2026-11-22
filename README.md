Agentic FacilityOps AI Platform

AI-powered building operations and facility intelligence system

This repository contains Milestone 1 (Energy Intelligence & Monitoring) of the Agentic FacilityOps AI Platform, with an early preview of Milestone 2 (Predictive Maintenance).

The system loads real hourly electricity data for four campus buildings, checks the data quality, detects abnormal consumption, forecasts the next week's demand, shows everything on an interactive dashboard, and answers questions through an Energy Agent chatbot.

![Energy dashboard](Screenshots/energy.png)

![Maintenance preview](Screenshots/maintenance.png)

![Energy agent](Screenshots/Agent.png)
Milestone 1 coverage
Requirement in the project brief	What was built	Where
Integrate utility and IoT data	Real hourly electricity meter data (Kaggle ASHRAE), four buildings named Block A to D, with a data-quality check for stuck meters	prep_kaggle.py, anomaly.py
Build Energy Agent	Tool-calling agent using the Gemini API, with a rule-based offline fallback	gemini_agent.py, offline_agent.py, llm_agent.py, agent.py
Energy consumption analytics	Totals per block, share of consumption, daily trends, peak hours, night-versus-day load, cost and CO2	app.py, llm_agent.py
Energy monitoring dashboard	Streamlit dashboard with KPI cards and interactive charts	app.py
Energy efficiency recommendations	Savings what-ifs, off-hours waste findings, advice from the agent	offline_agent.py, gemini_agent.py
Anomaly detection accuracy of 85% or more	One Isolation Forest per block plus a local seasonal baseline, evaluated on planted spikes (see Evaluation)	anomaly.py
Features
Anomaly detection (anomaly.py)

Real meter data contains stuck stretches (the same reading repeated for hours) and level shifts, so the detector handles both:

Stuck meters. Six or more identical readings in a row count as a stuck stretch. These hours are reported as a data-quality finding and are never scored.
Local seasonal baseline. "Normal" for an hour is the median of the same block, weekday and hour in the up to 8 weeks before and after it, using clean readings only. The higher of the two sides is used, so a permanent level shift is not mistaken for a spike, while an isolated spike is high against both sides.
One Isolation Forest per block, trained on hour, weekday, ratio to baseline and a z-score.
Two methods must agree. An hour is flagged only when the Isolation Forest and the rule (more than 1.4 times the baseline and a z-score above 3) both fire.
Low-usage blocks are not scored. A block averaging under 5 kWh per hour is excluded, because noise looks like spikes there. The agent says so instead of reporting zero anomalies.
Every flag has a plain-language reason, for example "2.1x its normal level for Monday 14:00".
Energy forecast

The agent can forecast hourly and daily demand for the 7 days after the data ends, using the median of the same weekday and hour over the last 4 weeks. A backtest predicts each of the last 4 weeks from the weeks before it and reports daily and hourly accuracy. Blocks whose meter was stuck for most of the last 8 weeks are excluded, because a forecast from a stuck meter would be meaningless. The dataset ends in 2016, so the forecast covers the week after that date, not the real next week.

Dashboard (app.py)

Total energy, highest-consuming block, anomalous hours, estimated cost and CO2 (tariff and grid factor adjustable in the sidebar), a block comparison chart, a share-of-consumption donut, a daily trend per block with flagged days marked, a table of the largest anomalies, and a model evaluation panel. Forecasts and data-quality reports are available through the Ask the Agent tab.

Energy Agent

Ask in plain language, for example:

"Which block uses the most electricity?"
"Compare Block A and Block C in March."
"Where are we wasting energy at night?"
"How much can we save if Block D cuts 15%?"
"Forecast next week for Block B."
"How reliable is the data for each block?"
"Show unusual consumption in Block B." Then follow up with "and Block C?"

Agent tools: usage, cost, carbon, trends, peak hours, off-hours waste (ranked by night-versus-day ratio), savings what-ifs, anomalies, data quality, 7-day demand forecast with backtest, plus maintenance tools.

The agent has two layers:

Gemini mode. The Gemini model decides which Python tools to call and writes the answer. All numbers come from the tools, not from the model's memory.
Offline mode. A rule-based agent understands blocks, dates, percentages and follow-up questions. It takes over automatically when there is no key, no internet or the free quota is used up.
Maintenance preview (early Milestone 2)

Simulated temperature and vibration data for lifts, generators and HVAC units, a failure-within-7-days prediction model, health scores and work orders. This data is simulated; see Limitations.

Tools and technologies
Area	Tool
Language	Python 3.13
Data handling	pandas, NumPy
Machine learning	scikit-learn (Isolation Forest, Random Forest)
Dashboard	Streamlit, Plotly
AI agent	Google Gemini API (function calling over plain HTTPS, no extra library), rule-based offline fallback
Data source	Kaggle ASHRAE - Great Energy Predictor III
Development	VS Code, Git, GitHub
Project structure
app.py                 Streamlit dashboard (Energy, Maintenance, Ask the Agent)
prep_kaggle.py         Step 1: builds data/energy.csv from the Kaggle files
anomaly.py             Step 2: data quality, anomaly detection and evaluation
maintenance.py         Milestone 2 preview: failure prediction and health scores
llm_agent.py           Analysis tools (including forecast and data quality) and tool definitions
offline_agent.py       Rule-based agent (no key, no internet)
gemini_agent.py        Gemini-powered agent with automatic offline fallback
agent.py               Original keyword agent (base class)
fake_ashrae.py         Optional: creates a small fake dataset for testing
test_gemini.py         Optional: checks that your Gemini key works
.streamlit/config.toml Dashboard theme
Setup
Install the packages
py -m pip install -r requirements.txt
Get the data. Download the Kaggle competition files from ASHRAE - Great Energy Predictor III and put train.csv and building_metadata.csv in a folder named data/. The dataset is large and has its own terms, so it is not included in this repository. Not downloaded yet? Run py fake_ashrae.py to create a small test dataset in the same format.
Prepare the energy file (creates data/energy.csv, four buildings as Block A to D)
py prep_kaggle.py
Check the anomaly model (prints data quality, then precision, recall and F1, overall and per block)
py anomaly.py
Optional: maintenance preview (failure prediction, health scores, work orders)
py maintenance.py
Run the dashboard
py -m streamlit run app.py
Optional: enable Gemini mode

The app works without any key, in offline mode. To turn on the Gemini agent, create a free key at Google AI Studio, then create .streamlit/secrets.toml:

GEMINI_API_KEY = "your-key-here"

Restart the app. The badge in the Ask the Agent tab changes to "Gemini with offline fallback".

Never write the key in README.md, code files or screenshots. secrets.toml is listed in .gitignore, so it is not uploaded. If there is no key, no internet or the quota is used up, the offline rule-based agent answers instead.

Evaluation

Real anomaly labels do not exist in this dataset, so accuracy is measured on spikes planted into hours of working meters (about 2% of scorable hours, 1.5 to 2.5 times normal). Only hours that can be scored are evaluated. The anomalies flagged in the real data are real hours and are never planted. Run py anomaly.py to reproduce the results.

Overall (planted spikes)

Method	Accuracy	Precision	Recall	F1
Isolation Forest				
Baseline rule				
Both agree				
Either				

Per block

Block	Scorable hours	Planted spikes	Precision	Recall	F1
Block A					
Block B					
Block C					
Block D					

Data quality (share of hours stuck on a repeated reading)

Block	Stuck hours (%)
Block A	
Block B	
Block C	
Block D	
<!-- Fill the three tables with the output of: py anomaly.py -->

Only about 2% of hours carry a planted spike, so accuracy alone looks high even for a weak detector. Precision, recall and F1 are reported alongside it.

Limitations
Only electricity is covered. The brief also mentions water and other utilities, for which this dataset has no data.
Anomaly accuracy is measured on planted spikes, not on real labelled events.
Some meters are stuck on one reading for long stretches. Those hours are excluded from anomaly scoring and, for badly affected blocks, from forecasting.
The forecast uses a simple weekday-and-hour pattern, so it cannot anticipate holidays or one-off events.
The data is historical hourly readings, not a live IoT stream.
The maintenance data is simulated. Real alternatives are NASA C-MAPSS and the Kaggle "Machine Predictive Maintenance Classification" dataset.
Gemini free-tier models and limits change over time. If the AI mode stops responding, the app falls back to offline mode and shows the reason.
Roadmap
Milestone	Focus	Status
1 (Weeks 1-2)	Energy Intelligence & Monitoring	Done
2 (Weeks 3-4)	Predictive Maintenance System	Early preview (simulated data)
3 (Weeks 5-6)	Occupancy & Security Intelligence	Planned
4 (Weeks 7-8)	Cost Optimization & Enterprise Deployment	Planned
Author

Srimathy R, Infosys Springboard internship project.
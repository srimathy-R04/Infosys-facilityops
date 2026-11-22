FacilityOps: Milestone 1 (Energy) and Milestone 2 (Maintenance) code
Setup (once)
pip install -r requirements.txt
Run order (from this folder)
Get the data
Real: download the Kaggle "ASHRAE - Great Energy Predictor III" files, put train.csv and building_metadata.csv inside the data/ folder.
Not downloaded yet? Run python fake_ashrae.py to create a small test file in the same format.
python prep_kaggle.py -> creates data/energy.csv (4 buildings as Block A to D)
python anomaly.py -> trains one anomaly model per block and prints precision, recall, F1 (overall and per block)
python maintenance.py -> Milestone 2: failure prediction, health scores, work orders
streamlit run app.py -> dashboard with Energy, Maintenance and Ask-the-Agent tabs
Optional: LLM agent

Uses the Gemini API through plain HTTPS (no extra library). Put your key in .streamlit/secrets.toml:

GEMINI_API_KEY = "your-key-here"

Never write the key in README.md, code files or screenshots. Add .streamlit/ to .gitignore. If there is no key, no internet or the quota is used up, the offline rule-based agent answers instead.

Energy Agent tools

Usage, cost, carbon, trends, peak hours, off-hours waste, savings what-ifs, anomalies, 7-day demand forecast (with a backtest on the last 4 weeks), plus maintenance tools.

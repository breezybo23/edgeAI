import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import os
from datetime import datetime, timedelta

# --- 1. UI & DARK MODE CONFIG ---
st.set_page_config(page_title="EdgeLab v27.8 NBA Pro", layout="wide")

st.markdown("""
<style>
    .main { background-color: #0d1117 !important; color: #c9d1d9; }
    .stApp { background-color: #0d1117; }
    .stTabs [data-baseweb="tab-list"] { background-color: #0d1117; }
    .stTabs [data-baseweb="tab"] {
        background-color: #161b22; border: 1px solid #30363d;
        border-radius: 8px 8px 0px 0px; color: #8b949e; padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] { background-color: #1f6feb !important; color: white !important; }
    .game-card { 
        background: #161b22; border: 1px solid #30363d; border-radius: 12px; 
        padding: 15px; margin-bottom: 20px; color: #c9d1d9;
    }
    .winner-text { color: #3fb950; font-weight: 800; font-size: 1.2rem; margin: 2px 0; }
    .status-badge { font-size: 0.6rem; background: #21262d; padding: 3px 8px; border-radius: 5px; color: #8b949e; border: 1px solid #30363d; }
    .fatigue-pill { font-size: 0.55rem; background: #442d10; color: #e3b341; padding: 2px 6px; border-radius: 4px; margin-left: 5px; }
    .score-text { font-size: 1.3rem; font-weight: 900; color: #ffffff; margin-top: 5px; }
    .high-conf-border { border: 2px solid #3fb950 !important; }
    .fantasy-pill { font-size: 0.6rem; background: #23392b; color: #4ade80; padding: 2px 8px; border-radius: 10px; margin-top: 5px; display: inline-block; }
</style>
""", unsafe_allow_html=True)

# --- 2. BRAIN DATA ENGINE ---
BRAIN_FILE = "nba_neural_pro_v27.json"

def get_brain():
    if os.path.exists(BRAIN_FILE):
        try:
            with open(BRAIN_FILE, 'r') as f:
                data = json.load(f)
                # Ensure all metrics exist
                for key in ['hits', 'misses', 'streaks', 'weights', 'experience', 'last_learned_ids']:
                    if key not in data: 
                        data[key] = {} if key in ['streaks', 'weights'] else ([] if key=='last_learned_ids' else 0)
                return data
        except: pass
    return {"last_learned_ids": [], "weights": {}, "streaks": {}, "experience": 0, "hits": 0, "misses": 0}

def save_brain(brain):
    with open(BRAIN_FILE, 'w') as f: json.dump(brain, f)

# --- 3. CORE NBA LOGIC ---
def run_pro_prediction(home_name, away_name, brain, fatigue_list):
    h_w = brain['weights'].get(home_name, 50.0)
    a_w = brain['weights'].get(away_name, 50.0)
    
    h_fatigue = -3.5 if home_name in fatigue_list else 0
    a_fatigue = -3.5 if away_name in fatigue_list else 0
    
    # NBA Base Spread Advantage + Weight Delta
    diff = (h_w + h_fatigue) - (a_w + a_fatigue) + 2.85
    sims = np.random.normal(diff, 10.5, 10000)
    prob = np.mean(sims > 0) * 100
    winner = home_name if prob > 50 else away_name
    return {"winner": winner, "conf": round(prob if prob > 50 else (100-prob), 1), "h_b2b": h_fatigue < 0, "a_b2b": a_fatigue < 0}

def live_learning_engine(brain):
    """Instant Audit: Scans recent games and updates sample size/win rate immediately"""
    dates = [(datetime.now() - timedelta(1)).strftime('%Y%m%d'), datetime.now().strftime('%Y%m%d')]
    updated = False
    for d in dates:
        url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={d}"
        try:
            data = requests.get(url, timeout=5).json()
            for event in data.get('events', []):
                g_id = event['id']
                # Check if game just finished and hasn't been audited yet
                if event['status']['type']['completed'] and g_id not in brain['last_learned_ids']:
                    comp = event['competitions'][0]['competitors']
                    h, a = next(t for t in comp if t['homeAway'] == 'home'), next(t for t in comp if t['homeAway'] == 'away')
                    h_score, a_score = int(h.get('score', 0)), int(a.get('score', 0))
                    actual_winner = h['team']['name'] if h_score > a_score else a['team']['name']
                    
                    # Grade the AI's previous prediction for this game
                    pred = run_pro_prediction(h['team']['name'], a['team']['name'], brain, [])
                    if pred['winner'] == actual_winner:
                        brain['hits'] += 1
                    else:
                        brain['misses'] += 1

                    # Update Neural Weights
                    eff_mult = 1.2 if abs(h_score - a_score) > 15 else 1.0
                    for team_meta, won in [(h, h_score > a_score), (a, a_score > h_score)]:
                        name = team_meta['team']['name']
                        new_streak = (brain['streaks'].get(name, 0) + 1) if won else 0
                        brain['streaks'][name] = new_streak
                        adj = (1.25 * eff_mult) if won else (-0.75)
                        brain['weights'][name] = round(brain['weights'].get(name, 50.0) + adj + (1.0 if new_streak >= 3 else 0), 2)
                    
                    brain['last_learned_ids'].append(g_id)
                    updated = True
        except: continue
    if updated: save_brain(brain)

def get_fatigue_status(date_str):
    prev_date = (datetime.strptime(date_str, '%Y%m%d') - timedelta(1)).strftime('%Y%m%d')
    try:
        data = requests.get(f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={prev_date}", timeout=5).json()
        return [t['team']['name'] for e in data.get('events', []) for t in e['competitions'][0]['competitors']]
    except: return []

# --- 4. RENDERER ---
def draw_pro_card(event, brain, fatigue_list, threshold):
    comp = event['competitions'][0]
    teams = comp['competitors']
    h, a = next(t for t in teams if t['homeAway'] == 'home'), next(t for t in teams if t['homeAway'] == 'away')
    pred = run_pro_prediction(h['team']['name'], a['team']['name'], brain, fatigue_list)
    
    if pred['conf'] < threshold: return 
    
    # Fantasy Scraper (Top Performer)
    perf_text = ""
    try:
        perf = comp['leaders'][0]['leaders'][0]
        perf_text = f"⚡ {perf['athlete']['shortName']}: {perf['displayValue']}"
    except: pass

    is_done = event['status']['type']['completed']
    high_conf_class = "high-conf-border" if pred['conf'] >= 75 else ""
    
    html = f"""<div class="game-card {high_conf_class}">
<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
<span class="status-badge">{event['status']['type']['shortDetail']}</span>
<span style="color:{'#3fb950' if pred['conf'] >= 75 else '#58a6ff'}; font-weight:700; font-size:0.75rem;">
{'⭐ ' if pred['conf'] >= 75 else ''}{pred['conf']}% AI PROJECTION</span>
</div>
<div style="display:flex; justify-content:space-around; align-items:center; text-align:center;">
<div style="flex:1;"><img src="{a['team']['logo']}" width="40"><div style="font-size:0.75rem; font-weight:600; margin-top:5px;">{a['team']['name']}{'<span class="fatigue-pill">B2B</span>' if pred['a_b2b'] else ''}</div>{f'<div class="score-text">{a["score"]}</div>' if is_done else ''}</div>
<div style="flex:1.5;"><div style="font-size:0.55rem; opacity:0.5; text-transform:uppercase;">AI Pick</div><div class="winner-text">{pred['winner']}</div>{f'<div class="fantasy-pill">{perf_text}</div>' if perf_text else ''}</div>
<div style="flex:1;"><img src="{h['team']['logo']}" width="40"><div style="font-size:0.75rem; font-weight:600; margin-top:5px;">{h['team']['name']}{'<span class="fatigue-pill">B2B</span>' if pred['h_b2b'] else ''}</div>{f'<div class="score-text">{h["score"]}</div>' if is_done else ''}</div>
</div>
</div>"""
    st.markdown(html, unsafe_allow_html=True)

# --- 5. APP EXECUTION ---
BRAIN = get_brain()
live_learning_engine(BRAIN) # Runs every refresh to audit new results

st.title("🛡️ EdgeLab v27.8 NBA Live Audit")

with st.sidebar:
    st.header("🧠 Neural Diagnostics")
    hits = BRAIN.get('hits', 0)
    misses = BRAIN.get('misses', 0)
    total_audited = hits + misses
    acc = (hits / total_audited * 100) if total_audited > 0 else 0
    
    st.metric("Live Win Rate", f"{acc:.1f}%")
    st.metric("Sample Size", f"{total_audited} Games")
    st.caption("Updated instantly upon game completion.")
    
    st.divider()
    st.subheader("⚙️ Filter")
    conf_threshold = st.slider("Confidence Threshold", 50, 95, 50)

tabs = st.tabs(["⏪ Yesterday", "📅 Today", "⏩ Tomorrow"])
for i, tab in enumerate(tabs):
    with tab:
        date_str = (datetime.now() + timedelta(i-1)).strftime('%Y%m%d')
        fatigue_data = get_fatigue_status(date_str)
        try:
            slate = requests.get(f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={date_str}").json().get('events', [])
            if not slate: st.info("No games found.")
            else:
                cols = st.columns(2)
                for idx, event in enumerate(slate):
                    with cols[idx % 2]: draw_pro_card(event, BRAIN, fatigue_data, conf_threshold)
        except: st.error("Sync Error")
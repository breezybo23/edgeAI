import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import os
from datetime import datetime, timedelta

# --- 1. UI & DARK MODE CONFIG ---
st.set_page_config(page_title="EdgeLab v25.7", layout="wide")

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
    .score-text { font-size: 1.3rem; font-weight: 900; color: #ffffff; margin-top: 5px; }
    .trend-up { color: #3fb950; font-size: 0.7rem; }
    .trend-down { color: #f85149; font-size: 0.7rem; }
</style>
""", unsafe_allow_html=True)

# --- 2. THE EVOLVING BRAIN WITH RECENCY BIAS ---
BRAIN_FILE = "nba_neural_v25.json"

def get_brain():
    if os.path.exists(BRAIN_FILE):
        try:
            with open(BRAIN_FILE, 'r') as f: return json.load(f)
        except: pass
    return {"last_learned_ids": [], "weights": {}, "experience": 0}

def save_brain(brain):
    with open(BRAIN_FILE, 'w') as f: json.dump(brain, f)

def live_learning_engine(brain):
    """Updates weights with higher multipliers for recent games"""
    dates = [(datetime.now() - timedelta(1)).strftime('%Y%m%d'), datetime.now().strftime('%Y%m%d')]
    new_lessons = 0
    for d in dates:
        url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={d}"
        try:
            data = requests.get(url, timeout=5).json()
            for event in data.get('events', []):
                g_id = event['id']
                if event['status']['type']['completed'] and g_id not in brain['last_learned_ids']:
                    comp = event['competitions'][0]['competitors']
                    h, a = next(t for t in comp if t['homeAway'] == 'home'), next(t for t in comp if t['homeAway'] == 'away')
                    h_win = int(h.get('score', 0)) > int(a.get('score', 0))
                    
                    # RECENCY MULTIPLIER: 1.5x for games within last 48 hours
                    multiplier = 1.5 
                    
                    for team, won in [(h['team']['name'], h_win), (a['team']['name'], not h_win)]:
                        curr = brain['weights'].get(team, 50.0)
                        # Adjustment formula with recency weight
                        adjustment = (0.85 * multiplier) if won else (-0.45 * multiplier)
                        brain['weights'][team] = round(curr + adjustment, 2)
                    
                    brain['last_learned_ids'].append(g_id)
                    brain['experience'] += 1
                    new_lessons += 1
        except: continue
    if new_lessons > 0:
        brain['last_learned_ids'] = brain['last_learned_ids'][-100:]
        save_brain(brain)

# --- 3. PREDICTION ENGINE ---
def run_prediction(home_name, away_name, brain):
    h_w, a_w = brain['weights'].get(home_name, 50.0), brain['weights'].get(away_name, 50.0)
    diff = (h_w - a_w) + 2.85
    sims = np.random.normal(diff, 11.0, 5000)
    prob = np.mean(sims > 0) * 100
    winner = home_name if prob > 50 else away_name
    return {"winner": winner, "conf": round(prob if prob > 50 else (100-prob), 1), "h_w": h_w, "a_w": a_w}

# --- 4. RENDERER ---
def draw_card(event, brain):
    teams = event['competitions'][0]['competitors']
    h = next(t for t in teams if t['homeAway'] == 'home')
    a = next(t for t in teams if t['homeAway'] == 'away')
    pred = run_prediction(h['team']['name'], a['team']['name'], brain)
    is_done = event['status']['type']['completed']
    
    h_score = f'<div class="score-text">{h["score"]}</div>' if is_done else ''
    a_score = f'<div class="score-text">{a["score"]}</div>' if is_done else ''
    
    # Render with standard flex layout (v25.6 fix)
    html = f"""<div class="game-card">
<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
<span class="status-badge">{event['status']['type']['shortDetail']}</span>
<span style="color:#58a6ff; font-weight:700; font-size:0.75rem;">{pred['conf']}% AI CONFIDENCE</span>
</div>
<div style="display:flex; justify-content:space-around; align-items:center; text-align:center;">
<div style="flex:1;">
<img src="{a['team']['logo']}" width="40"><br>
<div style="font-size:0.75rem; font-weight:600; margin-top:5px;">{a['team']['name']}</div>
{a_score}
</div>
<div style="flex:1.5; padding: 0 5px;">
<div style="font-size:0.55rem; opacity:0.5; text-transform:uppercase;">Projected Winner</div>
<div class="winner-text">{pred['winner']}</div>
</div>
<div style="flex:1;">
<img src="{h['team']['logo']}" width="40"><br>
<div style="font-size:0.75rem; font-weight:600; margin-top:5px;">{h['team']['name']}</div>
{h_score}
</div>
</div>
</div>"""
    st.markdown(html, unsafe_allow_html=True)

# --- 5. APP EXECUTION ---
BRAIN = get_brain()
live_learning_engine(BRAIN)

st.title("🛡️ EdgeLab v25.7 Neural (Recency Enabled)")

tabs = st.tabs(["⏪ Yesterday", "📅 Today", "⏩ Tomorrow"])
for i, tab in enumerate(tabs):
    with tab:
        offset = i - 1
        date_str = (datetime.now() + timedelta(offset)).strftime('%Y%m%d')
        try:
            url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={date_str}"
            slate = requests.get(url, timeout=5).json().get('events', [])
        except:
            slate = []
            
        if not slate:
            st.info("No games scheduled.")
        else:
            cols = st.columns(2)
            for idx, event in enumerate(slate):
                with cols[idx % 2]:
                    draw_card(event, BRAIN)

with st.sidebar:
    st.header("🧠 Brain Status")
    st.metric("Exp Points", BRAIN['experience'])
    st.write("Top Tier (Recent Form Weighted):")
    ranked = sorted(BRAIN['weights'].items(), key=lambda x: x[1], reverse=True)[:8]
    for team, val in ranked:
        st.caption(f"**{team}**: {val}")
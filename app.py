import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import os
from datetime import datetime, timedelta

# --- 1. UI & DARK MODE CONFIG ---
st.set_page_config(page_title="EdgeLab v26.0 Pro", layout="wide")

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
</style>
""", unsafe_allow_html=True)

# --- 2. MULTI-FACTOR NEURAL BRAIN ---
BRAIN_FILE = "nba_neural_pro_v26.json"

def get_brain():
    if os.path.exists(BRAIN_FILE):
        try:
            with open(BRAIN_FILE, 'r') as f: return json.load(f)
        except: pass
    return {"last_learned_ids": [], "weights": {}, "streaks": {}, "experience": 0}

def save_brain(brain):
    with open(BRAIN_FILE, 'w') as f: json.dump(brain, f)

def live_learning_engine(brain):
    """Refined Learning: Factors in Point Differential and Streaks"""
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
                    
                    h_score, a_score = int(h.get('score', 0)), int(a.get('score', 0))
                    diff = abs(h_score - a_score)
                    h_win = h_score > a_score
                    
                    # 1. Efficiency/Blowout Multiplier (Winning by 15+ is more impressive)
                    eff_mult = 1.2 if diff > 15 else 1.0
                    
                    for team_meta, won in [(h, h_win), (a, not h_win)]:
                        name = team_meta['team']['name']
                        curr = brain['weights'].get(name, 50.0)
                        streak = brain['streaks'].get(name, 0)
                        
                        # Update Streak
                        new_streak = (streak + 1) if won else 0
                        brain['streaks'][name] = new_streak
                        
                        # 2. Streak Bonus (+1.0 for 3+ wins)
                        streak_bonus = 1.0 if new_streak >= 3 else 0
                        
                        # 3. Adjustment Logic (Recency 1.5x built-in)
                        adj = (0.85 * 1.5 * eff_mult) if won else (-0.45 * 1.5)
                        brain['weights'][name] = round(curr + adj + streak_bonus, 2)
                    
                    brain['last_learned_ids'].append(g_id)
                    brain['experience'] += 1
                    new_lessons += 1
        except: continue
    if new_lessons > 0:
        save_brain(brain)

# --- 3. FATIGUE TRACKER ---
def get_fatigue_status(date_str):
    """Checks which teams played on the previous day"""
    prev_date = (datetime.strptime(date_str, '%Y%m%d') - timedelta(1)).strftime('%Y%m%d')
    url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={prev_date}"
    try:
        data = requests.get(url, timeout=5).json()
        teams_who_played = []
        for event in data.get('events', []):
            for team in event['competitions'][0]['competitors']:
                teams_who_played.append(team['team']['name'])
        return teams_who_played
    except: return []

# --- 4. PREDICTION CORE ---
def run_pro_prediction(home_name, away_name, brain, fatigue_list):
    h_w, a_w = brain['weights'].get(home_name, 50.0), brain['weights'].get(away_name, 50.0)
    
    # Apply Fatigue Penalties (-3.5 spread points for B2B)
    h_fatigue = -3.5 if home_name in fatigue_list else 0
    a_fatigue = -3.5 if away_name in fatigue_list else 0
    
    # Calculation: (Weight + Home Advantage + Fatigue)
    diff = (h_w + h_fatigue) - (a_w + a_fatigue) + 2.85
    
    sims = np.random.normal(diff, 10.5, 10000)
    prob = np.mean(sims > 0) * 100
    winner = home_name if prob > 50 else away_name
    return {
        "winner": winner, 
        "conf": round(prob if prob > 50 else (100-prob), 1),
        "h_b2b": h_fatigue < 0, "a_b2b": a_fatigue < 0
    }

# --- 5. RENDERER ---
def draw_pro_card(event, brain, fatigue_list):
    teams = event['competitions'][0]['competitors']
    h = next(t for t in teams if t['homeAway'] == 'home')
    a = next(t for t in teams if t['homeAway'] == 'away')
    
    pred = run_pro_prediction(h['team']['name'], a['team']['name'], brain, fatigue_list)
    is_done = event['status']['type']['completed']
    
    h_b2b_tag = '<span class="fatigue-pill">B2B</span>' if pred['h_b2b'] else ''
    a_b2b_tag = '<span class="fatigue-pill">B2B</span>' if pred['a_b2b'] else ''
    
    html = f"""<div class="game-card">
<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
<span class="status-badge">{event['status']['type']['shortDetail']}</span>
<span style="color:#58a6ff; font-weight:700; font-size:0.75rem;">{pred['conf']}% AI PROJECTION</span>
</div>
<div style="display:flex; justify-content:space-around; align-items:center; text-align:center;">
<div style="flex:1;">
<img src="{a['team']['logo']}" width="40"><br>
<div style="font-size:0.75rem; font-weight:600; margin-top:5px;">{a['team']['name']}{a_b2b_tag}</div>
{f'<div class="score-text">{a["score"]}</div>' if is_done else ''}
</div>
<div style="flex:1.5;">
<div style="font-size:0.55rem; opacity:0.5; text-transform:uppercase;">AI Pick</div>
<div class="winner-text">{pred['winner']}</div>
</div>
<div style="flex:1;">
<img src="{h['team']['logo']}" width="40"><br>
<div style="font-size:0.75rem; font-weight:600; margin-top:5px;">{h['team']['name']}{h_b2b_tag}</div>
{f'<div class="score-text">{h["score"]}</div>' if is_done else ''}
</div>
</div>
</div>"""
    st.markdown(html, unsafe_allow_html=True)

# --- 6. EXECUTION ---
BRAIN = get_brain()
live_learning_engine(BRAIN)

st.title("🛡️ EdgeLab v26.0 Professional")

tabs = st.tabs(["⏪ Yesterday", "📅 Today", "⏩ Tomorrow"])
for i, tab in enumerate(tabs):
    with tab:
        offset = i - 1
        date_obj = datetime.now() + timedelta(offset)
        date_str = date_obj.strftime('%Y%m%d')
        fatigue_data = get_fatigue_status(date_str)
        
        try:
            url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={date_str}"
            slate = requests.get(url, timeout=5).json().get('events', [])
        except: slate = []
            
        if not slate:
            st.info("No games scheduled.")
        else:
            cols = st.columns(2)
            for idx, event in enumerate(slate):
                with cols[idx % 2]:
                    draw_pro_card(event, BRAIN, fatigue_data)

with st.sidebar:
    st.header("🧠 Engine Diagnostics")
    st.metric("Neural Exp", f"{BRAIN['experience']} Games")
    st.write("Current Power Rankings:")
    ranked = sorted(BRAIN['weights'].items(), key=lambda x: x[1], reverse=True)[:10]
    for team, val in ranked:
        streak = BRAIN['streaks'].get(team, 0)
        streak_text = f"🔥 {streak}" if streak >= 3 else ""
        st.caption(f"**{team}**: {val} {streak_text}")
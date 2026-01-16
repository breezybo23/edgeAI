import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import os
from datetime import datetime, timedelta

# --- 1. UI & DARK MODE CONFIG ---
st.set_page_config(page_title="EdgeLab v45.9 Omega Vault", layout="wide")

st.html("""
<style>
    .stApp { background-color: #0d1117 !important; color: #c9d1d9; }
    .stTabs [data-baseweb="tab-list"] { background-color: #0d1117; }
    .stTabs [data-baseweb="tab"] {
        background-color: #161b22; border: 1px solid #30363d;
        border-radius: 8px 8px 0px 0px; color: #8b949e; padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] { background-color: #1f6feb !important; color: white !important; }
    .game-card { 
        background: #161b22; border: 1px solid #30363d; border-radius: 12px; 
        padding: 20px; margin-bottom: 20px; color: #c9d1d9; border-left: 5px solid #238636;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .high-conf-border { border: 2px solid #3fb950 !important; background: #052111 !important; }
    .value-alert-border { border: 2px solid #d29922 !important; background: #261a02 !important; box-shadow: 0px 0px 15px rgba(210, 153, 34, 0.15); }
    .winner-text { color: #3fb950; font-weight: 800; font-size: 1.3rem; margin: 2px 0; }
    .spread-pill { background: #238636; color: white; padding: 2px 10px; border-radius: 6px; font-size: 0.85rem; font-weight: 700; margin-top: 5px; display: inline-block; }
    .status-badge { font-size: 0.65rem; background: #30363d; padding: 3px 10px; border-radius: 5px; color: #8b949e; }
    .fatigue-pill { font-size: 0.55rem; background: #442d10; color: #e3b341; padding: 2px 6px; border-radius: 4px; margin-left: 5px; font-weight: bold; }
    .road-pill { background: #4a1515; color: #ff7b72; }
    .rest-pill { background: #1f6feb; color: white; font-size: 0.55rem; padding: 2px 6px; border-radius: 4px; margin-left: 5px; font-weight: bold; }
    .projection-box { background: #0d1117; border: 1px solid #30363d; border-radius: 8px 8px 0 0; padding: 12px; margin-top: 15px; display: flex; justify-content: space-around; }
    .proj-label { font-size: 0.6rem; color: #8b949e; text-transform: uppercase; margin-bottom: 3px; }
    .proj-val { font-size: 1rem; font-weight: 800; color: #58a6ff; }
    .vegas-sub { font-size: 0.65rem; color: #d29922; font-weight: 600; margin-top: 2px; }
    .strategy-box { 
        background: #0d1117; border: 1px solid #30363d; border-top: none; border-radius: 0 0 8px 8px; 
        padding: 12px; font-size: 0.85rem;
    }
    .strategy-title { color: #8b949e; font-weight: 700; text-transform: uppercase; font-size: 0.65rem; margin-bottom: 5px; }
    .bet-advice { color: #58a6ff; font-weight: 900; font-size: 1.1rem; border-top: 1px solid #30363d; padding-top: 8px; margin-top: 10px; }
    .conf-val { font-size: 1.2rem; font-weight: 800; }
</style>
""")

# --- 2. DATA ENGINES ---
BRAIN_FILE = "nba_neural_pro_v27.json"

def get_brain():
    if os.path.exists(BRAIN_FILE):
        try:
            with open(BRAIN_FILE, 'r') as f: return json.load(f)
        except: pass
    return {"last_learned_ids": [], "weights": {}, "hits": 0, "misses": 0}

def save_brain(brain):
    with open(BRAIN_FILE, 'w') as f: json.dump(brain, f)

def get_slate_teams(date_str):
    try:
        url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={date_str}"
        data = requests.get(url, timeout=5).json()
        return [t['team']['name'] for e in data.get('events', []) for t in e['competitions'][0]['competitors']]
    except: return []

# --- 3. DUAL-CONFIDENCE ANALYTICS ---
def run_strategic_prediction(home_name, away_name, brain, fatigue_data, v_num=None, v_fav_is_home=None):
    h_w, a_w = brain['weights'].get(home_name, 54.0), brain['weights'].get(away_name, 52.0)
    h_b, a_b = home_name in fatigue_data['b2b'], away_name in fatigue_data['b2b']
    
    h_f, a_f = (-3.5 if h_b else 0), (-5.5 if a_b else 0)
    h_r = 2.0 if (home_name in fatigue_data['rested'] and a_b) else 0
    a_r = 2.0 if (away_name in fatigue_data['rested'] and h_b) else 0
    
    ai_margin = (h_w + h_f + h_r) - (a_w + a_f + a_r) + 2.85
    
    # ML Confidence
    sims = np.random.normal(ai_margin, 10.5, 10000)
    ml_prob = np.mean(sims > 0) * 100
    winner = home_name if ml_prob > 50 else away_name
    ml_conf = ml_prob if ml_prob > 50 else (100 - ml_prob)
    
    # Spread Confidence
    spread_conf = 50.0
    is_value = False
    if v_num is not None:
        v_spread_signed = v_num if v_fav_is_home else -v_num
        if winner == home_name:
            spread_conf = np.mean(sims > -v_spread_signed) * 100
        else:
            spread_conf = np.mean(sims < -v_spread_signed) * 100
        is_value = abs(ai_margin - (-v_spread_signed)) >= 3.5

    reasons = []
    if h_r > 0 or a_r > 0: reasons.append("Fresh legs advantage detected.")
    if h_b or a_b: reasons.append(f"Fatigue penalty applied to {'Home' if h_b else 'Away'} team.")
    if abs(h_w - a_w) > 5: reasons.append("Neural power disparity.")
    
    return {
        "winner": winner, "ml_conf": round(ml_conf, 1), "spread_conf": round(spread_conf, 1),
        "ai_spread": abs(round(ai_margin * 0.85, 1)), "is_value": is_value,
        "total": round(222.5 + ((h_w + a_w)/10) + (h_f + a_f), 1),
        "reasons": " ".join(reasons) if reasons else "Baseline statistical parity.",
        "h_b2b": h_b, "a_b2b": a_b, "h_rest": h_r > 0, "a_rest": a_r > 0
    }

# --- 4. RENDER ENGINE ---
def draw_strategic_card(event, brain, fatigue_data, threshold):
    try:
        comp = event['competitions'][0]
        h, a = next(t for t in comp['competitors'] if t['homeAway'] == 'home'), next(t for t in comp['competitors'] if t['homeAway'] == 'away')
        
        # Vegas Parsing
        v_line, v_ou = "N/A", "N/A"
        v_num, v_fav_is_home, v_fav_abbr = 0, True, ""
        try:
            odds = comp['odds'][0]
            v_line = odds.get('details', 'N/A')
            v_ou = odds.get('overUnder', 'N/A')
            v_parts = v_line.split(' ')
            v_fav_abbr = v_parts[0]
            v_num = float(v_parts[-1])
            v_fav_is_home = v_fav_abbr == h['team']['abbreviation']
        except: pass

        pred = run_strategic_prediction(h['team']['name'], a['team']['name'], brain, fatigue_data, v_num, v_fav_is_home)
        if pred['ml_conf'] < threshold: return

        # Advice Logic
        if pred['spread_conf'] > pred['ml_conf'] and pred['is_value']:
            best_bet_type = "SPREAD"
            display_conf = pred['spread_conf']
            pick_team_name = h['team']['name'] if pred['winner'] == h['team']['name'] else a['team']['name']
            sign = "+" if (pred['winner'] != v_fav_abbr) else ""
            advice = f"Take {pick_team_name} {sign}{v_num if pred['winner'] == v_fav_abbr else abs(v_num)} on the SPREAD"
        else:
            best_bet_type = "WINNER"
            display_conf = pred['ml_conf']
            advice = f"Take {pred['winner']} on the WINNER (Moneyline)"

        def get_team_box(team, is_home, p):
            pill = ""
            if is_home and p['h_b2b']: pill = '<span class="fatigue-pill">HOME B2B</span>'
            elif not is_home and p['a_b2b']: pill = '<span class="fatigue-pill road-pill">ROAD B2B</span>'
            if (is_home and p['h_rest']) or (not is_home and p['a_rest']): pill += '<span class="rest-pill">REST ADV</span>'
            score = f'<div style="font-size:1.4rem; font-weight:900;">{team["score"]}</div>' if event['status']['type']['completed'] else ""
            return f'<div style="flex:1;"><img src="{team["team"]["logo"]}" width="45"><div style="font-size:0.8rem; font-weight:600; margin-top:5px;">{team["team"]["name"]}{pill}</div>{score}</div>'

        card_style = f"game-card {'high-conf-border' if pred['ml_conf'] >= 75 else ''} {'value-alert-border' if pred['is_value'] else ''}"

        html = f"""
        <div class="{card_style}">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span class="status-badge">{event['status']['type']['shortDetail']}</span>
                <span style="color:{'#d29922' if pred['is_value'] else '#3fb950'}; font-weight:800; font-size:0.7rem;">
                    {'🔥 VALUE ALERT' if pred['is_value'] else '⭐ AI PROJECTION'}: {pred['ml_conf']}%
                </span>
            </div>
            <div style="display:flex; justify-content:space-around; align-items:center; text-align:center; margin-top:15px;">
                {get_team_box(a, False, pred)}
                <div style="flex:1.5;">
                    <div style="font-size:0.6rem; color:#8b949e;">PREDICTED WINNER</div>
                    <div class="winner-text">{pred['winner']}</div>
                    <div class="spread-pill">AI SPREAD: -{pred['ai_spread']}</div>
                </div>
                {get_team_box(h, True, pred)}
            </div>
            <div class="projection-box">
                <div style="text-align:center;"><div class="proj-label">AI TOTAL</div><div class="proj-val">{pred['total']}</div><div class="vegas-sub">Vegas: {v_ou}</div></div>
                <div style="text-align:center;"><div class="proj-label">MARKET LINE</div><div class="proj-val">{v_line}</div><div class="vegas-sub">Edge: {'ACTIVE' if pred['is_value'] else 'NONE'}</div></div>
            </div>
            <div class="strategy-box">
                <div style="display:flex; justify-content:space-between; text-align:center; margin-bottom:10px;">
                    <div style="flex:1;"><div class="strategy-title">Winner Conf.</div><div class="conf-val" style="color:#3fb950;">{pred['ml_conf']}%</div></div>
                    <div style="flex:1; border-left:1px solid #30363d;"><div class="strategy-title">Spread Conf.</div><div class="conf-val" style="color:#d29922;">{pred['spread_conf']}%</div></div>
                </div>
                <div class="strategy-title">AI Logic</div>
                <div style="margin-bottom:8px; color:#c9d1d9;">{pred['reasons']}</div>
                <div class="bet-advice">
                    <span style="color:#8b949e; font-size:0.6rem; display:block; text-transform:uppercase;">Recommended Choice ({display_conf}%)</span>
                    {advice}
                </div>
            </div>
        </div>
        """
        st.html(html)
    except: pass

# --- 5. EXECUTION LOOP ---
def live_learning_loop(brain):
    dates = [(datetime.now() - timedelta(1)).strftime('%Y%m%d'), datetime.now().strftime('%Y%m%d')]
    updated = False
    for d in dates:
        try:
            url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={d}"
            data = requests.get(url, timeout=5).json()
            for event in data.get('events', []):
                g_id = event['id']
                if event['status']['type']['completed'] and g_id not in brain['last_learned_ids']:
                    comp = event['competitions'][0]['competitors']
                    h, a = next(t for t in comp if t['homeAway'] == 'home'), next(t for t in comp if t['homeAway'] == 'away')
                    h_s, a_s = int(h.get('score', 0)), int(a.get('score', 0))
                    is_blowout = abs(h_s - a_s) >= 25
                    hit_val, miss_val = (0.6, -0.3) if is_blowout else (1.25, -0.75)
                    winner_actual = h['team']['name'] if h_s > a_s else a['team']['name']
                    pred = run_strategic_prediction(h['team']['name'], a['team']['name'], brain, {'b2b':[], 'rested':[]})
                    if pred['winner'] == winner_actual: brain['hits'] += 1
                    else: brain['misses'] += 1
                    for team, won in [(h, h_s > a_s), (a, a_s > h_s)]:
                        name = team['team']['name']
                        brain['weights'][name] = round(brain['weights'].get(name, 50.0) + (hit_val if won else miss_val), 2)
                    brain['last_learned_ids'].append(g_id)
                    updated = True
        except: continue
    if updated: save_brain(brain)

BRAIN = get_brain()
live_learning_loop(BRAIN)

with st.sidebar:
    st.title("🛡️ Omega Advisor")
    total_g = BRAIN['hits'] + BRAIN['misses']
    st.metric("Model Accuracy", f"{(BRAIN['hits']/total_g*100):.1f}%" if total_g > 0 else "0.0%")
    st.divider()
    conf_threshold = st.slider("Signal Sensitivity", 50, 95, 54)
    if st.button("Purge Neural Data"):
        save_brain({"last_learned_ids": [], "weights": {}, "hits": 0, "misses": 0})
        st.rerun()

st.title("EdgeLab v45.9 Strategic NBA Advisor")

tabs = st.tabs(["⏪ Yesterday", "📅 Today", "⏩ Tomorrow"])
for i, tab in enumerate(tabs):
    with tab:
        d_str = (datetime.now() + timedelta(i-1)).strftime('%Y%m%d')
        y_str = (datetime.strptime(d_str, '%Y%m%d') - timedelta(1)).strftime('%Y%m%d')
        db_str = (datetime.strptime(d_str, '%Y%m%d') - timedelta(2)).strftime('%Y%m%d')
        ctx = {"b2b": get_slate_teams(y_str), "rested": [t for t in get_slate_teams(db_str) if t not in get_slate_teams(y_str)]}
        try:
            slate = requests.get(f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={d_str}").json().get('events', [])
            if not slate: st.info("No games scheduled.")
            else:
                c1, c2 = st.columns(2)
                for idx, event in enumerate(slate):
                    with c1 if idx % 2 == 0 else c2: draw_strategic_card(event, BRAIN, ctx, conf_threshold)
        except: st.error("Neural Sync Error")
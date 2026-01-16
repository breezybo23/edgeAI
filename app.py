import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from nba_api.live.nba.endpoints import scoreboard
from datetime import datetime, timedelta

# --- 1. SYSTEM CONFIGURATION ---
st.set_page_config(page_title="EdgeLab AI v18.0 - Ultimate Suite", layout="wide")

# Custom CSS for Dark Mode and UI elements
st.markdown("""
<style>
    .main { background-color: #0E1117; color: #FFFFFF; }
    .stMetric { background: #161b22; border: 1px solid #30363d; padding: 15px; border-radius: 12px; }
    .best-bet-header { color: #00FFAA; font-size: 1.1rem; font-weight: bold; margin-bottom: 10px; }
    .injury-tag { background: #3d0000; color: #ff4b4b; padding: 4px 10px; border-radius: 6px; font-size: 0.8rem; border: 1px solid #ff4b4b; margin-right: 5px; }
    .conf-high { color: #00FFAA; font-weight: bold; }
    .conf-med { color: #FFAA00; font-weight: bold; }
    .conf-low { color: #FF4B4B; font-weight: bold; }
    .live-dot { color: #ff4b4b; font-weight: bold; animation: blinker 1.5s linear infinite; }
    .score-box { background: #1a1c23; border: 1px solid #30363d; padding: 10px; border-radius: 8px; text-align: center; font-size: 1.8rem; font-weight: bold; border-top: 3px solid #00FFAA; }
    .bet-row { background: #1a1c23; padding: 12px; border-radius: 8px; margin-bottom: 8px; border-left: 4px solid #30363d; }
    @keyframes blinker { 50% { opacity: 0; } }
</style>
""", unsafe_allow_html=True)

# --- 2. DATA REPOSITORY ---
TEAM_DB = {
    "Boston Celtics": {"off": 122.1, "def": 110.4, "pace": 98.8, "h_bias": 3.4, "sigma": 9.2},
    "New York Knicks": {"off": 118.5, "def": 112.2, "pace": 96.5, "h_bias": 2.8, "sigma": 8.5},
    "Golden State Warriors": {"off": 117.2, "def": 116.8, "pace": 100.5, "h_bias": 3.1, "sigma": 11.8},
    "Detroit Pistons": {"off": 114.8, "def": 110.2, "pace": 99.1, "h_bias": 2.5, "sigma": 9.5},
    "Phoenix Suns": {"off": 116.5, "def": 115.2, "pace": 98.5, "h_bias": 2.2, "sigma": 10.4},
    "San Antonio Spurs": {"off": 115.8, "def": 111.5, "pace": 101.2, "h_bias": 4.1, "sigma": 11.0},
    "Milwaukee Bucks": {"off": 116.2, "def": 118.4, "pace": 100.8, "h_bias": 2.1, "sigma": 10.2},
    "Dallas Mavericks": {"off": 118.5, "def": 116.2, "pace": 100.1, "h_bias": 2.5, "sigma": 11.5},
    "Utah Jazz": {"off": 112.5, "def": 119.8, "pace": 101.5, "h_bias": 3.2, "sigma": 12.1}
}

VEGAS_MARKET = {
    "Knicks@Warriors": {"spread": -7.5, "total": 226.5, "h_ml": -320, "a_ml": +260},
    "Suns@Pistons": {"spread": -4.5, "total": 218.5, "h_ml": -190, "a_ml": +160},
    "Bucks@Spurs": {"spread": -8.5, "total": 224.5, "h_ml": -380, "a_ml": +300},
    "Jazz@Mavericks": {"spread": -6.5, "total": 232.5, "h_ml": -260, "a_ml": +210}
}

STAR_MAP = {
    "Jalen Brunson": {"val": -9.5, "team": "Knicks"},
    "Devin Booker": {"val": -10.2, "team": "Suns"},
    "Giannis Antetokounmpo": {"val": -11.5, "team": "Bucks"},
    "Jayson Tatum": {"val": -10.8, "team": "Celtics"}
}

# --- 3. CORE LOGIC ---
def format_spread(val):
    return f"+{val}" if val > 0 else f"{val}"

def get_auto_injuries():
    return ["Jalen Brunson", "Devin Booker", "Jayson Tatum"]

def calculate_confidence(edge, sigma):
    z_score = abs(edge) / sigma
    conf = (norm.cdf(z_score) - 0.5) * 2 * 100 
    return round(min(conf + 50, 99.9), 1)

def run_supreme_ai(h_name, a_name, active_injuries):
    fb = {"off": 115.0, "def": 115.0, "pace": 100.0, "h_bias": 2.5, "sigma": 10.5}
    h = next((v for k, v in TEAM_DB.items() if k in h_name), fb)
    a = next((v for k, v in TEAM_DB.items() if k in a_name), fb)
    
    h_eff, a_eff = h['off'], a['off']
    h_impacts, a_impacts = [], []
    for p in active_injuries:
        if p in STAR_MAP:
            if STAR_MAP[p]["team"] in h_name: h_eff += STAR_MAP[p]["val"]; h_impacts.append(p)
            if STAR_MAP[p]["team"] in a_name: a_eff += STAR_MAP[p]["val"]; a_impacts.append(p)

    exp_pace = (h['pace'] + a['pace']) / 2
    h_proj = ((h_eff + h['h_bias'] + a['def']) / 2) * (exp_pace / 100)
    a_proj = ((a_eff + h['def']) / 2) * (exp_pace / 100)
    
    h_sims = np.random.normal(h_proj, h['sigma'], 10000)
    a_sims = np.random.normal(a_proj, a['sigma'], 10000)
    
    return {
        "h_score": round(h_proj, 1), "a_score": round(a_proj, 1),
        "ai_total": round(h_proj + a_proj, 1), "ai_spread": round(a_proj - h_proj, 1),
        "win_prob": round(np.mean(h_sims > a_sims) * 100, 1),
        "sigma": (h['sigma'] + a['sigma']) / 2, "h_out": h_impacts, "a_out": a_impacts
    }

def get_best_bets(ai, vegas, h_name, a_name):
    bets = []
    s_edge = abs(ai['ai_spread'] - vegas['spread'])
    s_conf = calculate_confidence(s_edge, ai['sigma'])
    s_pick = f"{h_name} {format_spread(vegas['spread'])}" if ai['ai_spread'] < vegas['spread'] else f"{a_name} {format_spread(-vegas['spread'])}"
    bets.append({"type": "Spread", "pick": s_pick, "conf": s_conf})

    t_edge = abs(ai['ai_total'] - vegas['total'])
    t_conf = calculate_confidence(t_edge, ai['sigma'] * 1.4)
    t_pick = f"{'OVER' if ai['ai_total'] > vegas['total'] else 'UNDER'} {vegas['total']}"
    bets.append({"type": "Total", "pick": t_pick, "conf": t_conf})

    implied_h_prob = 100 / (abs(vegas['h_ml']) + 100) if vegas['h_ml'] < 0 else 100 / (vegas['h_ml'] + 100)
    ml_edge = (ai['win_prob']/100) - implied_h_prob
    ml_conf = calculate_confidence(ml_edge * 10, 2.0)
    bets.append({"type": "Moneyline", "pick": f"{h_name if ml_edge > 0 else a_name} ML", "conf": ml_conf})

    return sorted(bets, key=lambda x: x['conf'], reverse=True)

# --- 4. DATA FETCHING (CACHED) ---

@st.cache_data(ttl=600)  # Cache for 10 minutes
def fetch_scoreboard_data(date_str=None):
    """Fetches and caches scoreboard data to prevent redundant API calls."""
    try:
        if date_str:
            sb = scoreboard.ScoreBoard(game_date=date_str)
        else:
            sb = scoreboard.ScoreBoard()
        return sb.get_dict()['scoreboard']['games']
    except Exception as e:
        st.error(f"Live Sync Error: {e}")
        return None

# --- 5. MAIN INTERFACE ---
st.title("🏀 EdgeLab Ultimate v18.0")

# Fetch initial data for today
board_data = fetch_scoreboard_data()

if board_data:
    # Check if all games for today are final (Status 3)
    all_final = all(g['gameStatus'] == 3 for g in board_data)
    
    display_date = datetime.now()
    if all_final:
        display_date = datetime.now() + timedelta(days=1)
        date_str = display_date.strftime('%Y-%m-%d')
        st.info(f"📅 Today's slate complete. Showing Tomorrow's Projections: **{display_date.strftime('%b %d, %Y')}**")
        # Re-fetch for next day
        board_data = fetch_scoreboard_data(date_str)
    else:
        st.write(f"**Live Engine:** Active | **Injuries:** Automated | **Date:** {display_date.strftime('%b %d, %Y')}")

    active_injuries = get_auto_injuries()

    if board_data:
        for idx, g in enumerate(board_data):
            h_team, a_team = g['homeTeam']['teamName'], g['awayTeam']['teamName']
            h_full, a_full = f"{g['homeTeam']['teamCity']} {h_team}", f"{g['awayTeam']['teamCity']} {a_team}"
            
            status = g['gameStatusText']
            is_live = g['gameStatus'] == 2 
            live_label = f"<span class='live-dot'>● {status}</span>" if is_live else status

            res = run_supreme_ai(h_full, a_full, active_injuries)
            m_key = f"{a_team}@{h_team}"
            v_data = VEGAS_MARKET.get(m_key, {"spread": -5.0, "total": 215.0, "h_ml": -200, "a_ml": 170})
            
            with st.container():
                st.markdown(f"### {a_full} @ {h_full} | {live_label}", unsafe_allow_html=True)
                c_score, c_proj, c_bets = st.columns([1.2, 1.5, 1.5])
                
                with c_score:
                    sc1, sc2 = st.columns(2)
                    sc1.markdown(f"<div class='score-box'>{g['awayTeam']['score']}</div>", unsafe_allow_html=True)
                    sc1.caption(f"{a_team}")
                    sc2.markdown(f"<div class='score-box'>{g['homeTeam']['score']}</div>", unsafe_allow_html=True)
                    sc2.caption(f"{h_team}")
                    curr_tot = g['awayTeam']['score'] + g['homeTeam']['score']
                    st.metric("Live Total", curr_tot, delta=f"{round(curr_tot - v_data['total'], 1)} vs Market")

                with c_proj:
                    st.write("**AI ANALYTICS**")
                    p1, p2 = st.columns(2)
                    p1.metric("AI Score", f"{res['a_score']}-{res['h_score']}")
                    p1.metric("Win Prob", f"{res['win_prob']}%")
                    p2.metric("AI Spread", format_spread(res['ai_spread']))
                    p2.metric("AI Total", res['ai_total'])

                with c_bets:
                    st.markdown("<div class='best-bet-header'>🎯 CONFIDENCE PICKS</div>", unsafe_allow_html=True)
                    for bet in get_best_bets(res, v_data, h_team, a_team):
                        b_color = "conf-high" if bet['conf'] > 80 else "conf-med" if bet['conf'] > 65 else "conf-low"
                        st.markdown(f"""
                        <div class="bet-row">
                            <small>{bet['type']}</small><br>
                            <strong>{bet['pick']}</strong> | <span class="{b_color}">{bet['conf']}% Conf.</span>
                        </div>
                        """, unsafe_allow_html=True)

                if res['h_out'] or res['a_out']:
                    inj_html = "".join([f"<span class='injury-tag'>OUT: {p}</span>" for p in res['a_out'] + res['h_out']])
                    st.markdown(inj_html, unsafe_allow_html=True)
                
                st.divider()
else:
    st.warning("No data found for the current or upcoming slate.")
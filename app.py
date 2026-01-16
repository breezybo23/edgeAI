import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
from nba_api.live.nba.endpoints import scoreboard as live_scoreboard
from nba_api.stats.endpoints import scoreboardv2 as stats_scoreboard
from datetime import datetime, timedelta

# --- 1. SYSTEM CONFIGURATION ---
st.set_page_config(page_title="EdgeLab AI v18.1 - Ultimate Suite", layout="wide")

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

# --- 2. DATA REPOSITORY & ID MAPPER ---
# Maps NBA Stats IDs to Full Names and Short Names
TEAM_MAP = {
    1610612738: {"full": "Boston Celtics", "short": "Celtics"},
    1610612752: {"full": "New York Knicks", "short": "Knicks"},
    1610612744: {"full": "Golden State Warriors", "short": "Warriors"},
    1610612765: {"full": "Detroit Pistons", "short": "Pistons"},
    1610612756: {"full": "Phoenix Suns", "short": "Suns"},
    1610612759: {"full": "San Antonio Spurs", "short": "Spurs"},
    1610612749: {"full": "Milwaukee Bucks", "short": "Bucks"},
    1610612742: {"full": "Dallas Mavericks", "short": "Mavericks"},
    1610612762: {"full": "Utah Jazz", "short": "Jazz"},
    1610612743: {"full": "Denver Nuggets", "short": "Nuggets"},
    1610612746: {"full": "LA Clippers", "short": "Clippers"},
    1610612747: {"full": "Los Angeles Lakers", "short": "Lakers"}
}

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
    "Knicks@Warriors": {"spread": -7.5, "total": 226.5},
    "Suns@Pistons": {"spread": -4.5, "total": 218.5},
    "Bucks@Spurs": {"spread": -8.5, "total": 224.5},
    "Jazz@Mavericks": {"spread": -6.5, "total": 232.5}
}

# --- 3. CORE LOGIC ---
def format_spread(val):
    return f"+{val}" if val > 0 else f"{val}"

def calculate_confidence(edge, sigma):
    z_score = abs(edge) / sigma
    conf = (norm.cdf(z_score) - 0.5) * 2 * 100 
    return round(min(conf + 50, 99.9), 1)

def run_supreme_ai(h_full, a_full):
    fb = {"off": 115.0, "def": 115.0, "pace": 100.0, "h_bias": 2.5, "sigma": 10.5}
    h = TEAM_DB.get(h_full, fb)
    a = TEAM_DB.get(a_full, fb)
    
    exp_pace = (h['pace'] + a['pace']) / 2
    h_proj = ((h['off'] + h['h_bias'] + a['def']) / 2) * (exp_pace / 100)
    a_proj = ((a['off'] + h['def']) / 2) * (exp_pace / 100)
    
    return {
        "h_score": round(h_proj, 1), "a_score": round(a_proj, 1),
        "ai_total": round(h_proj + a_proj, 1), "ai_spread": round(a_proj - h_proj, 1),
        "win_prob": round(norm.cdf((h_proj - a_proj) / h['sigma']) * 100, 1),
        "sigma": (h['sigma'] + a['sigma']) / 2
    }

def get_best_bets(ai, vegas, h_short, a_short):
    bets = []
    # Spread Pick
    s_edge = abs(ai['ai_spread'] - vegas['spread'])
    s_conf = calculate_confidence(s_edge, ai['sigma'])
    s_pick = f"{h_short} {format_spread(vegas['spread'])}" if ai['ai_spread'] < vegas['spread'] else f"{a_short} {format_spread(-vegas['spread'])}"
    bets.append({"type": "Spread", "pick": s_pick, "conf": s_conf})
    
    # Total Pick
    t_edge = abs(ai['ai_total'] - vegas['total'])
    t_conf = calculate_confidence(t_edge, ai['sigma'] * 1.4)
    t_pick = f"{'OVER' if ai['ai_total'] > vegas['total'] else 'UNDER'} {vegas['total']}"
    bets.append({"type": "Total", "pick": t_pick, "conf": t_conf})
    
    return sorted(bets, key=lambda x: x['conf'], reverse=True)

# --- 4. DATA FETCHING (HYBRID CACHED) ---
@st.cache_data(ttl=600)
def fetch_games_standardized(date_target=None):
    today_str = datetime.now().strftime('%Y-%m-%d')
    try:
        if date_target is None or date_target == today_str:
            raw = live_scoreboard.ScoreBoard().get_dict()
            return [{
                "h_name": f"{g['homeTeam']['teamCity']} {g['homeTeam']['teamName']}",
                "a_name": f"{g['awayTeam']['teamCity']} {g['awayTeam']['teamName']}",
                "h_short": g['homeTeam']['teamName'],
                "a_short": g['awayTeam']['teamName'],
                "h_score": g['homeTeam']['score'],
                "a_score": g['awayTeam']['score'],
                "status": g['gameStatusText'],
                "live": g['gameStatus'] == 2,
                "final": g['gameStatus'] == 3
            } for g in raw['scoreboard']['games']]
        else:
            raw = stats_scoreboard.ScoreBoardV2(game_date=date_target).get_dict()
            rows = raw['resultSets'][0]['rowSet']
            games = []
            for r in rows:
                h_id, a_id = r[6], r[7] # Home/Away Team IDs from Stats API
                h_info = TEAM_MAP.get(h_id, {"full": f"Team {h_id}", "short": "TBD"})
                a_info = TEAM_MAP.get(a_id, {"full": f"Team {a_id}", "short": "TBD"})
                games.append({
                    "h_name": h_info['full'], "a_name": a_info['full'],
                    "h_short": h_info['short'], "a_short": a_info['short'],
                    "h_score": 0, "a_score": 0,
                    "status": "Scheduled", "live": False, "final": False
                })
            return games
    except Exception as e:
        return []

# --- 5. MAIN INTERFACE ---
st.title("🏀 EdgeLab Ultimate v18.1")

games_list = fetch_games_standardized()
all_done = all(g['final'] for g in games_list) if games_list else False

if all_done:
    tomorrow = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    st.info(f"📅 Today's slate complete. Fetching Tomorrow: {tomorrow}")
    games_list = fetch_games_standardized(tomorrow)
else:
    st.write(f"**Live Engine:** Active | **Date:** {datetime.now().strftime('%b %d, %Y')}")

if not games_list:
    st.warning("No games found for this period.")
else:
    for g in games_list:
        res = run_supreme_ai(g['h_name'], g['a_name'])
        v_data = VEGAS_MARKET.get(f"{g['a_short']}@{g['h_short']}", {"spread": -5.5, "total": 220.5})

        with st.container():
            live_label = f"<span class='live-dot'>● {g['status']}</span>" if g['live'] else g['status']
            st.markdown(f"### {g['a_name']} @ {g['h_name']} | {live_label}", unsafe_allow_html=True)
            
            c1, c2, c3 = st.columns([1, 1.5, 1.5])
            with c1:
                sc1, sc2 = st.columns(2)
                sc1.markdown(f"<div class='score-box'>{g['a_score']}</div>", unsafe_allow_html=True)
                sc1.caption(f"{g['a_short']}")
                sc2.markdown(f"<div class='score-box'>{g['h_score']}</div>", unsafe_allow_html=True)
                sc2.caption(f"{g['h_short']}")
            with c2:
                st.write("**AI PROJECTIONS**")
                st.metric("Proj Score", f"{res['a_score']} - {res['h_score']}")
                st.metric("Win Prob", f"{res['win_prob']}%")
            with c3:
                st.markdown("<div class='best-bet-header'>🎯 BEST BETS</div>", unsafe_allow_html=True)
                for bet in get_best_bets(res, v_data, g['h_short'], g['a_short']):
                    b_color = "conf-high" if bet['conf'] > 80 else "conf-med" if bet['conf'] > 65 else "conf-low"
                    st.markdown(f"<div class='bet-row'>{bet['type']}: **{bet['pick']}** | <span class='{b_color}'>{bet['conf']}%</span></div>", unsafe_allow_html=True)
            st.divider()
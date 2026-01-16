import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
from nba_api.live.nba.endpoints import scoreboard as live_scoreboard
from nba_api.stats.endpoints import scoreboardv2 as stats_scoreboard
from datetime import datetime, timedelta

# --- 1. SYSTEM CONFIGURATION ---
st.set_page_config(page_title="EdgeLab AI v18.5", layout="wide")

# Custom CSS for Dark Mode & Betting UI
st.markdown("""
<style>
    .main { background-color: #0E1117; color: #FFFFFF; }
    .stMetric { background: #161b22; border: 1px solid #30363d; padding: 15px; border-radius: 12px; }
    .score-box { background: #1a1c23; border: 1px solid #30363d; padding: 10px; border-radius: 8px; text-align: center; font-size: 1.8rem; font-weight: bold; border-top: 3px solid #00FFAA; }
    .winner-tag { color: #00FFAA; font-weight: bold; background: #00FFAA22; padding: 4px 10px; border-radius: 6px; border: 1px solid #00FFAA; }
    .edge-box { background: #00FFAA11; border-left: 4px solid #00FFAA; padding: 10px; border-radius: 4px; }
    .market-label { color: #8b949e; font-size: 0.85rem; font-weight: bold; }
    .accuracy-header { background: linear-gradient(90deg, #00FFAA22, #0E1117); padding: 15px; border-radius: 10px; border: 1px solid #30363d; margin-bottom: 20px; }
</style>
""", unsafe_allow_html=True)

# --- 2. DATA REPOSITORY ---
TEAM_MAP = {1610612738: {"full": "Boston Celtics", "short": "Celtics"}, 1610612752: {"full": "New York Knicks", "short": "Knicks"}, 1610612744: {"full": "Golden State Warriors", "short": "Warriors"}, 1610612765: {"full": "Detroit Pistons", "short": "Pistons"}, 1610612756: {"full": "Phoenix Suns", "short": "Suns"}, 1610612759: {"full": "San Antonio Spurs", "short": "Spurs"}, 1610612749: {"full": "Milwaukee Bucks", "short": "Bucks"}, 1610612742: {"full": "Dallas Mavericks", "short": "Mavericks"}, 1610612762: {"full": "Utah Jazz", "short": "Jazz"}, 1610612743: {"full": "Denver Nuggets", "short": "Nuggets"}, 1610612746: {"full": "LA Clippers", "short": "Clippers"}, 1610612747: {"full": "Los Angeles Lakers", "short": "Lakers"}}

TEAM_DB = {"Boston Celtics": {"off": 122.1, "def": 110.4, "pace": 98.8, "h_bias": 3.4, "sigma": 9.2}, "New York Knicks": {"off": 118.5, "def": 112.2, "pace": 96.5, "h_bias": 2.8, "sigma": 8.5}, "Golden State Warriors": {"off": 117.2, "def": 116.8, "pace": 100.5, "h_bias": 3.1, "sigma": 11.8}, "Detroit Pistons": {"off": 114.8, "def": 110.2, "pace": 99.1, "h_bias": 2.5, "sigma": 9.5}, "Phoenix Suns": {"off": 116.5, "def": 115.2, "pace": 98.5, "h_bias": 2.2, "sigma": 10.4}, "San Antonio Spurs": {"off": 115.8, "def": 111.5, "pace": 101.2, "h_bias": 4.1, "sigma": 11.0}, "Milwaukee Bucks": {"off": 116.2, "def": 118.4, "pace": 100.8, "h_bias": 2.1, "sigma": 10.2}, "Dallas Mavericks": {"off": 118.5, "def": 116.2, "pace": 100.1, "h_bias": 2.5, "sigma": 11.5}, "Utah Jazz": {"off": 112.5, "def": 119.8, "pace": 101.5, "h_bias": 3.2, "sigma": 12.1}}

VEGAS_MARKET = {
    "Knicks@Warriors": {"spread": -7.5, "total": 226.5},
    "Suns@Pistons": {"spread": -4.5, "total": 218.5}
}

# --- 3. CORE ANALYTICS ---
def run_analytics(h_full, a_full, v_spread, v_total):
    fb = {"off": 115.0, "def": 115.0, "pace": 100.0, "h_bias": 2.5, "sigma": 10.5}
    h, a = TEAM_DB.get(h_full, fb), TEAM_DB.get(a_full, fb)
    
    # Calculate Scores
    pace = (h['pace'] + a['pace']) / 2
    h_proj = ((h['off'] + h['h_bias'] + a['def']) / 2) * (pace / 100)
    a_proj = ((a['off'] + h['def']) / 2) * (pace / 100)
    
    ai_total = h_proj + a_proj
    ai_spread = a_proj - h_proj # Negative = Home Fav
    
    return {
        "h_score": round(h_proj, 1), "a_score": round(a_proj, 1),
        "ai_total": round(ai_total, 1), "ai_spread": round(ai_spread, 1),
        "winner": h_full if h_proj > a_proj else a_full,
        "win_prob": round(norm.cdf((h_proj - a_proj) / h['sigma']) * 100, 1),
        "t_edge": round(abs(ai_total - v_total), 1),
        "s_edge": round(abs(ai_spread - v_spread), 1)
    }

# --- 4. DATA FETCHING ---
@st.cache_data(ttl=300)
def fetch_master_data():
    games = []
    try:
        raw = live_scoreboard.ScoreBoard().get_dict()
        for g in raw['scoreboard']['games']:
            games.append({
                "h": f"{g['homeTeam']['teamCity']} {g['homeTeam']['teamName']}", "a": f"{g['awayTeam']['teamCity']} {g['awayTeam']['teamName']}",
                "h_s": g['homeTeam']['score'], "a_s": g['awayTeam']['score'], "h_short": g['homeTeam']['teamName'], "a_short": g['awayTeam']['teamName'],
                "status": g['gameStatusText'], "final": g['gameStatus'] == 3, "day": "Today"
            })
    except: pass
    try:
        tmrw = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
        raw_tm = stats_scoreboard.ScoreBoardV2(game_date=tmrw).get_dict()
        for r in raw_tm['resultSets'][0]['rowSet']:
            h_info = TEAM_MAP.get(r[6], {"full": "Unknown", "short": "TBD"})
            a_info = TEAM_MAP.get(r[7], {"full": "Unknown", "short": "TBD"})
            games.append({"h": h_info['full'], "a": a_info['full'], "h_s": 0, "a_s": 0, "h_short": h_info['short'], "a_short": a_info['short'], "status": "Scheduled", "final": False, "day": "Tomorrow"})
    except: pass
    return games

# --- 5. INTERFACE ---
st.title("🏀 EdgeLab Ultimate v18.5")
master_list = fetch_master_data()

# Performance Tracker
finals = [g for g in master_list if g['final']]
if finals:
    wins = sum(1 for g in finals if (g['h_s'] > g['a_s'] and run_analytics(g['h'], g['a'], 0, 0)['h_score'] > run_analytics(g['h'], g['a'], 0, 0)['a_score']) or (g['a_s'] > g['h_s'] and run_analytics(g['h'], g['a'], 0, 0)['a_score'] > run_analytics(g['h'], g['a'], 0, 0)['h_score']))
    st.markdown(f"<div class='accuracy-header'>🎯 AI Accuracy: <b>{round(wins/len(finals)*100,1)}%</b> today</div>", unsafe_allow_html=True)

for day in ["Today", "Tomorrow"]:
    day_list = [g for g in master_list if g['day'] == day]
    if day_list:
        st.subheader(f"📅 {day}")
        for g in day_list:
            v_data = VEGAS_MARKET.get(f"{g['a_short']}@{g['h_short']}", {"spread": -4.0, "total": 224.0})
            res = run_analytics(g['h'], g['a'], v_data['spread'], v_data['total'])
            
            with st.container():
                st.markdown(f"### {g['a']} @ {g['h']} | <small>{g['status']}</small>", unsafe_allow_html=True)
                c1, c2, c3 = st.columns([1, 1.2, 1.5])
                
                with c1:
                    st.markdown(f"<div class='score-box'>{g['a_s']} - {g['h_s']}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div style='text-align:center; margin-top:10px;'><span class='winner-tag'>PRED: {res['winner']}</span></div>", unsafe_allow_html=True)

                with c2:
                    st.write("**AI PREDICTED SCORE**")
                    st.metric(f"{g['a_short']}", res['a_score'])
                    st.metric(f"{g['h_short']}", res['h_score'])
                
                with c3:
                    st.write("**AI EDGE ANALYSIS**")
                    st.markdown(f"""
                    <div class='edge-box'>
                        <span class='market-label'>TOTAL:</span> Vegas {v_data['total']} vs <b>AI {res['ai_total']}</b> (Edge: {res['t_edge']})<br>
                        <span class='market-label'>SPREAD:</span> Vegas {v_data['spread']} vs <b>AI {res['ai_spread']}</b> (Edge: {res['s_edge']})<br>
                        <span class='market-label'>WIN PROB:</span> <b>{res['win_prob']}%</b>
                    </div>
                    """, unsafe_allow_html=True)
                st.divider()
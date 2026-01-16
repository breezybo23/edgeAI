import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta
from nba_api.stats.endpoints import scoreboardv2

# --- 1. SYSTEM CONFIG & DARK THEME ---
st.set_page_config(page_title="EdgeLab Intelligence v20.0", layout="wide")

st.markdown("""
<style>
    .main { background-color: #0E1117; color: #FFFFFF; font-family: 'Inter', sans-serif; }
    .live-badge { 
        background-color: #FF4B4B; color: white; padding: 3px 10px; 
        border-radius: 4px; font-weight: bold; font-size: 0.75rem; 
        animation: blinker 1.5s linear infinite;
    }
    @keyframes blinker { 50% { opacity: 0; } }
    .winner-box { 
        background-color: #1a1c23; border: 1px solid #00FFAA; 
        padding: 10px; border-radius: 8px; margin-bottom: 15px;
    }
    .winner-name { color: #00FFAA; font-weight: 800; font-size: 1.2rem; }
    .accuracy-banner {
        background: linear-gradient(90deg, #00FFAA 0%, #0088ff 100%);
        color: #0E1117; padding: 15px; border-radius: 10px;
        text-align: center; font-weight: 800; margin-bottom: 20px;
    }
    [data-testid="stMetricValue"] { font-size: 1.8rem !important; color: #FFFFFF !important; font-weight: 700; }
    .stMetric { background: #161b22; padding: 15px; border-radius: 10px; border: 1px solid #30363d; }
</style>
""", unsafe_allow_html=True)

# --- 2. INTELLIGENCE REPOSITORY ---
TEAM_DB = {
    "Celtics": {"off": 122.1, "def": 110.4, "pace": 98.8, "id": 1610612738},
    "Heat": {"off": 113.2, "def": 112.5, "pace": 96.2, "id": 1610612748},
    "Knicks": {"off": 118.5, "def": 112.2, "pace": 96.5, "id": 1610612752},
    "Warriors": {"off": 117.2, "def": 116.8, "pace": 100.5, "id": 1610612744},
    "Suns": {"off": 116.5, "def": 115.2, "pace": 98.5, "id": 1610612756},
    "Pistons": {"off": 114.8, "def": 110.2, "pace": 99.1, "id": 1610612765},
    "Bucks": {"off": 116.2, "def": 118.4, "pace": 100.8, "id": 1610612749},
    "Mavericks": {"off": 118.5, "def": 116.2, "pace": 100.1, "id": 1610612742}
}

# --- 3. ANALYTICS ENGINE ---
def run_ai_prediction(h_id, a_id, h_nickname, a_nickname):
    # Lookup stats in DB using nickname; fallback to average if missing
    h_stats = TEAM_DB.get(h_nickname, {"off": 115.0, "def": 115.0, "pace": 99.0})
    a_stats = TEAM_DB.get(a_nickname, {"off": 115.0, "def": 115.0, "pace": 99.0})
    
    pace = (h_stats['pace'] + a_stats['pace']) / 2
    h_proj = ((h_stats['off'] + 2.5 + a_stats['def']) / 2) * (pace / 100)
    a_proj = ((a_stats['off'] + h_stats['def']) / 2) * (pace / 100)
    
    win_prob = norm.cdf((h_proj - a_proj) / 10.0) * 100
    winner_name = h_nickname if h_proj > a_proj else a_nickname
    
    return {
        "h_score": round(h_proj, 1), "a_score": round(a_proj, 1),
        "spread": round(a_proj - h_proj, 1),
        "winner_name": winner_name,
        "winner_id": h_id if h_proj > a_proj else a_id,
        "prob": round(win_prob if h_proj > a_proj else (100 - win_prob), 1)
    }

# --- 4. DATA FETCHING ---
@st.cache_data(ttl=60)
def fetch_daily_slate(target_date):
    try:
        f_date = target_date.strftime('%m/%d/%Y')
        sb = scoreboardv2.ScoreboardV2(game_date=f_date).get_dict()
        games_raw = sb['resultSets'][0]['rowSet']
        lines_raw = sb['resultSets'][1]['rowSet']
        
        games_list = []
        for g in games_raw:
            g_id, h_id, a_id = g[2], g[6], g[7]
            # Match LineScore entries to get Nicknames and Scores
            h_line = next((x for x in lines_raw if x[3] == h_id and x[2] == g_id), None)
            a_line = next((x for x in lines_raw if x[3] == a_id and x[2] == g_id), None)
            
            if h_line and a_line:
                games_list.append({
                    "game_id": g_id, "status": g[4], 
                    "h_id": h_id, "a_id": a_id,
                    "h_nick": h_line[6], "a_nick": a_line[6],
                    "h_score": h_line[22] or 0, "a_score": a_line[22] or 0, 
                    "period": g[9] or 0
                })
        return games_list
    except Exception as e:
        return []

# --- 5. UI RENDER ---
st.title("🛡️ EdgeLab Intelligence v20.0")

tabs = st.tabs(["Yesterday", "Today's Slate", "Tomorrow"])
dates = [datetime.now() - timedelta(1), datetime.now(), datetime.now() + timedelta(1)]

for i, tab in enumerate(tabs):
    with tab:
        current_date = dates[i]
        st.subheader(f"Games for {current_date.strftime('%A, %b %d')}")
        slate = fetch_daily_slate(current_date)
        
        if not slate:
            st.info("No games scheduled for this date.")
        else:
            correct_picks, total_final = 0, 0
            
            for g in slate:
                pred = run_ai_prediction(g['h_id'], g['a_id'], g['h_nick'], g['a_nick'])
                
                # Accuracy tracking for Yesterday
                if i == 0 and "Final" in g['status']:
                    total_final += 1
                    actual_winner_id = g['h_id'] if g['h_score'] > g['a_score'] else g['a_id']
                    if pred['winner_id'] == actual_winner_id:
                        correct_picks += 1

                with st.container():
                    c1, c2, c3 = st.columns([1, 2, 1])
                    with c1: st.image(f"https://cdn.nba.com/logos/nba/{g['a_id']}/primary/L/logo.svg", width=60)
                    with c2:
                        st.write(f"**{g['a_nick']} @ {g['h_nick']}**")
                        if "Live" in g['status'] or (g['period'] > 0 and i == 1):
                            st.markdown("<span class='live-badge'>LIVE</span>", unsafe_allow_html=True)
                        st.caption(f"Status: {g['status']}")
                    with c3: st.image(f"https://cdn.nba.com/logos/nba/{g['h_id']}/primary/L/logo.svg", width=60)

                    st.markdown(f"""
                    <div class='winner-box'>
                        <small style='color:#8b949e'>AI PREDICTION</small><br>
                        <span class='winner-name'>{pred['winner_name']}</span> <small>({pred['prob']}%)</small>
                    </div>
                    """, unsafe_allow_html=True)

                    m1, m2, m3 = st.columns(3)
                    m1.metric("SCORE", f"{g['a_score']} - {g['h_score']}")
                    m2.metric("AI SPREAD", pred['spread'])
                    m3.metric("AI PROJ", f"{pred['a_score']} - {pred['h_score']}")
                    st.divider()
            
            if i == 0 and total_final > 0:
                acc = round((correct_picks / total_final) * 100, 1)
                st.markdown(f"<div class='accuracy-banner'>YESTERDAY'S AI ACCURACY: {acc}% ({correct_picks}/{total_final})</div>", unsafe_allow_html=True)

st.caption("v20.0 | Full Nickname Mapping & Accuracy Tracker")
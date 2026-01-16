import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta
from nba_api.stats.endpoints import scoreboardv2, leaguedashteamstats

# --- 1. SYSTEM CONFIG & DARK THEME ---
st.set_page_config(page_title="EdgeLab Intelligence v21.1", layout="wide")

st.markdown("""
<style>
    .main { background-color: #0E1117; color: #FFFFFF; font-family: 'Inter', sans-serif; }
    
    .live-badge { 
        background-color: #FF4B4B; color: white; padding: 3px 10px; 
        border-radius: 4px; font-weight: bold; font-size: 0.75rem; 
        animation: blinker 1.5s linear infinite;
    }
    @keyframes blinker { 50% { opacity: 0; } }

    /* Dynamic Winner Boxes */
    .winner-box { padding: 12px; border-radius: 8px; margin-bottom: 15px; border-left: 5px solid; }
    .risk-low { background-color: #12221a; border-color: #00FFAA; box-shadow: 0 0 10px rgba(0,255,170,0.1); }
    .risk-med { background-color: #222112; border-color: #FFD700; box-shadow: 0 0 10px rgba(255,215,0,0.1); }
    .risk-high { background-color: #221212; border-color: #FF4B4B; box-shadow: 0 0 10px rgba(255,75,75,0.1); }
    
    .winner-name { font-weight: 800; font-size: 1.2rem; }
    .risk-label { font-size: 0.7rem; text-transform: uppercase; font-weight: bold; margin-bottom: 5px; display: block; }

    .accuracy-banner {
        background: linear-gradient(90deg, #00FFAA 0%, #0088ff 100%);
        color: #0E1117; padding: 15px; border-radius: 10px;
        text-align: center; font-weight: 800; margin-top: 30px;
    }

    [data-testid="stMetricValue"] { font-size: 1.6rem !important; color: #FFFFFF !important; font-weight: 700; }
    .stMetric { background: #161b22; padding: 12px; border-radius: 10px; border: 1px solid #30363d; }
    [data-testid="stMetricLabel"] { color: #8b949e !important; font-size: 0.85rem !important; }
</style>
""", unsafe_allow_html=True)

# --- 2. DYNAMIC INTELLIGENCE REPOSITORY ---
@st.cache_data(ttl=3600)
def get_live_team_stats():
    try:
        # Fetching Advanced Stats for all 30 teams
        stats_df = leaguedashteamstats.LeagueDashTeamStats(
            measure_type_detailed_defense='Advanced',
            season='2025-26' # Ensure this matches current season
        ).get_data_frames()[0]
        
        # Mapping Full Names to Nicknames (last word of name)
        stats_df['NICKNAME'] = stats_df['TEAM_NAME'].str.split().str[-1]
        
        live_db = {}
        for _, row in stats_df.iterrows():
            live_db[row['NICKNAME']] = {
                "off": row['OFF_RATING'],
                "def": row['DEF_RATING'],
                "pace": row['PACE'],
                "id": row['TEAM_ID']
            }
        return live_db
    except Exception as e:
        return {}

# Global variable for current stats
TEAM_DB = get_live_team_stats()

# --- 3. ANALYTICS ENGINE ---
def run_ai_prediction(h_id, a_id, h_nickname, a_nickname):
    # Lookup live stats; default to league median if team not found
    h_stats = TEAM_DB.get(h_nickname, {"off": 114.5, "def": 114.5, "pace": 99.1})
    a_stats = TEAM_DB.get(a_nickname, {"off": 114.5, "def": 114.5, "pace": 99.1})
    
    pace = (h_stats['pace'] + a_stats['pace']) / 2
    h_proj = ((h_stats['off'] + 2.5 + a_stats['def']) / 2) * (pace / 100)
    a_proj = ((a_stats['off'] + h_stats['def']) / 2) * (pace / 100)
    
    win_prob = norm.cdf((h_proj - a_proj) / 10.0) * 100
    winner_name = h_nickname if h_proj > a_proj else a_nickname
    prob = round(win_prob if h_proj > a_proj else (100 - win_prob), 1)
    
    # Sorting and Risk Logic
    if prob >= 75:
        risk_class, risk_text, sort_val = "risk-low", "Low Risk (Strong Play)", 3
    elif prob >= 60:
        risk_class, risk_text, sort_val = "risk-med", "Moderate Risk (Toss-up)", 2
    else:
        risk_class, risk_text, sort_val = "risk-high", "High Risk (Avoid)", 1
        
    return {
        "h_score": round(h_proj, 1), "a_score": round(a_proj, 1),
        "spread": round(a_proj - h_proj, 1), "winner_name": winner_name,
        "winner_id": h_id if h_proj > a_proj else a_id, "prob": prob,
        "risk_class": risk_class, "risk_text": risk_text, "sort_val": sort_val
    }

# --- 4. DATA FETCHING ---
@st.cache_data(ttl=30)
def fetch_daily_slate(target_date):
    try:
        f_date = target_date.strftime('%m/%d/%Y')
        sb = scoreboardv2.ScoreboardV2(game_date=f_date).get_dict()
        games_raw, lines_raw = sb['resultSets'][0]['rowSet'], sb['resultSets'][1]['rowSet']
        
        games_list = []
        for g in games_raw:
            g_id, h_id, a_id = g[2], g[6], g[7]
            h_line = next((x for x in lines_raw if x[3] == h_id and x[2] == g_id), None)
            a_line = next((x for x in lines_raw if x[3] == a_id and x[2] == g_id), None)
            if h_line and a_line:
                games_list.append({
                    "game_id": g_id, "status": g[4], "h_id": h_id, "a_id": a_id,
                    "h_nick": h_line[6], "a_nick": a_line[6],
                    "h_score": h_line[22] or 0, "a_score": a_line[22] or 0, "period": g[9] or 0
                })
        return games_list
    except: return []

# --- 5. UI RENDER ---
st.title("🛡️ EdgeLab Intelligence v21.1")
st.caption("Real-Time Analytics Dashboard | Dark Mode")

tabs = st.tabs(["Yesterday", "Today's Slate", "Tomorrow"])
dates = [datetime.now() - timedelta(1), datetime.now(), datetime.now() + timedelta(1)]

for i, tab in enumerate(tabs):
    with tab:
        current_date = dates[i]
        st.subheader(f"Schedule for {current_date.strftime('%A, %b %d')}")
        raw_slate = fetch_daily_slate(current_date)
        
        if not raw_slate:
            st.info("No games found for this date.")
        else:
            # Sort by confidence
            analyzed_slate = []
            for g in raw_slate:
                g['pred'] = run_ai_prediction(g['h_id'], g['a_id'], g['h_nick'], g['a_nick'])
                analyzed_slate.append(g)
            
            sorted_slate = sorted(analyzed_slate, key=lambda x: (x['pred']['sort_val'], x['pred']['prob']), reverse=True)

            correct_picks, completed_games = 0, 0
            for g in sorted_slate:
                pred = g['pred']
                if "Final" in g['status']:
                    completed_games += 1
                    actual_winner_id = g['h_id'] if g['h_score'] > g['a_score'] else g['a_id']
                    if pred['winner_id'] == actual_winner_id: correct_picks += 1

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
                    <div class='winner-box {pred['risk_class']}'>
                        <span class='risk-label' style='color:{'#00FFAA' if 'low' in pred['risk_class'] else '#FFD700' if 'med' in pred['risk_class'] else '#FF4B4B'}'>{pred['risk_text']}</span>
                        <span class='winner-name'>{pred['winner_name']}</span> <small>({pred['prob']}%)</small>
                    </div>
                    """, unsafe_allow_html=True)

                    m1, m2 = st.columns(2)
                    with m1: st.metric(f"LIVE: {g['a_nick']} - {g['h_nick']}", f"{g['a_score']} - {g['h_score']}")
                    with m2: st.metric(f"AI: {g['a_nick']} - {g['h_nick']}", f"{pred['a_score']} - {pred['h_score']}", delta=f"Spread: {pred['spread']}")
                    st.divider()
            
            if completed_games > 0:
                acc = round((correct_picks / completed_games) * 100, 1)
                st.markdown(f"<div class='accuracy-banner'>AI ACCURACY FOR {current_date.strftime('%b %d')}: {acc}% ({correct_picks}/{completed_games} Correct)</div>", unsafe_allow_html=True)

st.caption("v21.1 | Live API Feed | Automatic Accuracy & Sorting")
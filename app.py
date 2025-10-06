# app.py - V24 Dashboard Pro complet, stylé, pédagogique, sectoriel, multi-tickers
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
from math import isnan
import numpy as np

# ---- Page config ----
st.set_page_config(page_title="Analyse Actions - Dashboard Pro", layout="wide")

# ---- CSS ----
st.markdown("""
<style>
.stApp { background-color:#0f1720; color:#e6eef6; }
.big-title { font-size:32px; font-weight:700; color:#f8fafc; margin-bottom:0.25rem; }
.muted { color:#9aa6b2; }
.progress-bar { border-radius:8px; height:24px; margin-bottom:6px; }
.tooltip { color:#cfe7ff; font-size:13px; }
.tab-header { font-weight:700; margin-bottom:0.25rem; font-size:18px; }
</style>
""", unsafe_allow_html=True)

# ---- Sidebar ----
with st.sidebar:
    st.header("Recherche")
    ticker_input = st.text_input("Ticker principal (ex: AAPL, MSFT, EXA.PA)", value="AAPL")
    multi_tickers = st.text_area("Comparer plusieurs tickers (séparés par une virgule)", value="MSFT,GOOGL,AMZN")
    compare_sector = st.checkbox("Comparer uniquement les tickers du même secteur", value=True)
    trading_mode = st.selectbox("Profil trading", ["Équilibré", "Long terme / Value", "Court terme / Spéculatif"])
    aggressive = st.checkbox("Mode agressif", value=False)
    st.write("---")
    st.markdown("⚙️ Entrez le ticker exact, avec suffixe marché si nécessaire")

# ---- Helper functions ----
def safe(val, default=None):
    if val is None or (isinstance(val,float) and (pd.isna(val) or val!=val)):
        return default
    return val

def clip_progress(val, max_abs=1):
    try:
        val_float = float(val)
    except:
        val_float = 0
    percent = int(abs(val_float)/max_abs*100)
    return max(0, min(percent, 100))

SECTOR_SCALES = {
    "Biotechnology": {"PE":(0,40,80),"P/S":(0,6,12),"Profit Margin":(0.0,0.05,0.10)},
    "Healthcare": {"PE":(0,25,45),"P/S":(0,4,8),"Profit Margin":(0.05,0.10,0.15)},
    "Industrial": {"PE":(0,15,25),"P/S":(0,2,4),"Profit Margin":(0.05,0.10,0.15)},
    "Technology": {"PE":(0,30,60),"P/S":(0,5,12),"Profit Margin":(0.05,0.10,0.20)},
    "Financial Services": {"PE":(0,12,20),"P/S":(0,3,6),"Profit Margin":(0.05,0.12,0.18)}
}
DEFAULT_SCALES = {"PE":(0,20,40),"P/S":(0,3,6),"Profit Margin":(0.05,0.10,0.15)}

tooltips = {
    "P/E": "Price / Earnings : <15 attractif, >30 spéculatif",
    "P/S": "Price / Sales : faible = bon, élevé = spéculatif",
    "P/B": "Price / Book : <1 = sous-évalué, >3 = cher",
    "Profit Margin": "Marge nette : >10% solide, <5% faible",
}

@st.cache_data(ttl=300)
def fetch_info(ticker):
    t = yf.Ticker(ticker)
    try: info_raw = t.info
    except: info_raw={}
    info = {}
    info['ticker']=ticker
    info['longName']=info_raw.get('longName') or ticker
    info['sector']=info_raw.get('sector') or "Unknown"
    info['price']=safe(info_raw.get('currentPrice'))
    info['marketCap']=safe(info_raw.get('marketCap'))
    info['trailingPE']=safe(info_raw.get('trailingPE'))
    info['priceToSales']=safe(info_raw.get('priceToSalesTrailing12Months'))
    info['priceToBook']=safe(info_raw.get('priceToBook'))
    info['profitMargins']=safe(info_raw.get('profitMargins'))
    info['operatingMargins']=safe(info_raw.get('operatingMargins'))
    info['revenueGrowth']=safe(info_raw.get('revenueGrowth'))
    info['trailingEPS']=safe(info_raw.get('trailingEps'))
    info['totalCash']=safe(info_raw.get('totalCash'))
    info['totalDebt']=safe(info_raw.get('totalDebt'))
    info['currentRatio']=safe(info_raw.get('currentRatio'))
    info['bookValue']=safe(info_raw.get('bookValue'))
    info['sharesOutstanding']=safe(info_raw.get('sharesOutstanding'))
    try: hist = t.history(period="1y", actions=False)
    except: hist=None
    info['history']=hist
    try:
        cal = t.calendar
        info['calendar'] = cal.to_dict() if cal is not None else {}
    except:
        info['calendar'] = {}
    return info

def interpret(val, key, sector=None):
    if val is None: return ("N/A","gray","Donnée indisponible")
    sector_scales = SECTOR_SCALES.get(sector, DEFAULT_SCALES)
    if key=="PE": low,med,high = sector_scales['PE']
    elif key=="P/S": low,med,high = sector_scales['P/S']
    elif key=="Profit Margin": low,med,high = sector_scales['Profit Margin']
    else: return (str(val),"gray","")
    if key=="Profit Margin": val_display=f"{val*100:.1f}%"
    else: val_display=f"{val:.2f}"
    if key=="PE" or key=="P/S":
        if val < med: return (val_display,"green","Attractif")
        if val > high: return (val_display,"red","Élevé / spéculatif")
        return (val_display,"yellow","Dans la norme")
    if key=="Profit Margin":
        if val>0.10: return (val_display,"green","Marge solide")
        if val<0: return (val_display,"red","Perte / marge négative")
        return (val_display,"yellow","Marge modeste")
    return (val_display,"gray","")

def compute_score(info, aggressive=False, mode="Équilibré"):
    score=0
    if mode=="Long terme / Value":
        weight={'valuation':4,'growth':2,'margins':3,'balance':3}
    elif mode=="Court terme / Spéculatif":
        weight={'valuation':2,'growth':4,'margins':1,'balance':1}
    else:
        weight={'valuation':3,'growth':3,'margins':2,'balance':2}

    pe=info.get('trailingPE'); ps=info.get('priceToSales'); eps=info.get('trailingEPS'); sector=info.get('sector')
    if eps and eps>0 and pe: _,color,_=interpret(pe,"PE",sector)
    elif ps: _,color,_=interpret(ps,"P/S",sector)
    else: color="yellow"
    score += weight['valuation']*(1 if color=="green" else 0.6 if color=="yellow" else 0.2)

    growth_val=info.get('revenueGrowth');_,color_g,_=interpret(growth_val,"Profit Margin")
    score += weight['growth']*(1 if color_g=="green" else 0.6 if color_g=="yellow" else 0.2)

    pm_val=info.get('profitMargins');_,color_m,_=interpret(pm_val,"Profit Margin")
    score += weight['margins']*(1 if color_m=="green" else 0.6 if color_m=="yellow" else 0.2)

    cr_val=info.get('currentRatio');_,color_cr,_=interpret(cr_val,"PE")
    score += weight['balance']*(1 if color_cr=="green" else 0.6 if color_cr=="yellow" else 0.2)

    maxscore=sum(weight.values())
    final_score=(score/maxscore)*10
    if aggressive: final_score+=0.3
    final_score=max(0,min(10,final_score))

    if final_score>=7: verdict=("BUY","green","Acheter")
    elif final_score>=4.5: verdict=("WATCH","yellow","Surveiller")
    else: verdict=("AVOID","red","Éviter")
    return round(final_score,1), verdict

# ---- Main ----
if not ticker_input:
    st.info("Entrez un ticker dans la sidebar")
    st.stop()

with st.spinner("Récupération des données..."):
    info = fetch_info(ticker_input)

# ---- Tabs ----
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Analyse détaillée", "Graphiques", "Conseil trading", "Comparatif multi-tickers"])

# ---- Tab 1 : Overview ----
with tab1:
    st.markdown(f"### {info['longName']} ({info['ticker']})")
    st.markdown(f"**Prix actuel : {info['price']} €** | Secteur : {info['sector']}")

# ---- Tab 2 : Analyse détaillée ----
with tab2:
    st.subheader("🟢 Étape 1 : Valorisation")
    for label, val, key in [("P/E", info['trailingPE'], "PE"),
                            ("P/S", info['priceToSales'], "P/S"),
                            ("P/B", info['priceToBook'], "P/B")]:
        disp,color,exp = interpret(val,key,info['sector'])
        display_val = f"{val:.2f}" if val is not None else "N/A"
        st.markdown(f"{label} : {display_val} ({exp})")
        st.progress(clip_progress(val or 0, max_abs=val if val else 1))

    st.subheader("📈 Étape 2 : Croissance")
    rg_val, rg_color, rg_exp = interpret(info['revenueGrowth'],"Profit Margin")
    st.markdown(f"Croissance revenus : {rg_val} ({rg_exp})")
    st.progress(clip_progress(info['revenueGrowth'], max_abs=0.5))

    st.subheader("💹 Étape 3 : Marges")
    pm_val, pm_color, pm_exp = interpret(info['profitMargins'],"Profit Margin")
    om_val, om_color, om_exp = interpret(info['operatingMargins'],"Profit Margin")
    st.markdown(f"Marge nette : {pm_val} ({pm_exp})")
    st.progress(clip_progress(info['profitMargins'], max_abs=0.5))
    st.markdown(f"Marge opérationnelle : {om_val} ({om_exp})")
    st.progress(clip_progress(info['operatingMargins'], max_abs=0.5))

    st.subheader("🏦 Étape 4 : Bilan / Levier")
    cr_val, cr_color,_ = interpret(info['currentRatio'],"PE")
    debt_ratio = info['totalDebt']/info['totalCash'] if info['totalCash'] else None
    dr_color = "green" if debt_ratio and debt_ratio<1 else "yellow" if debt_ratio and debt_ratio<2 else "red"
    st.markdown(f"Current Ratio : {cr_val}")
    st.progress(clip_progress(info['currentRatio'], max_abs=3))
    st.markdown(f"Dette / Cash : {debt_ratio:.2f}" if debt_ratio else "Dette / Cash : N/A")

# ---- Tab 3 : Graphiques ----
with tab3:
    if info['history'] is not None:
        st.subheader("📊 Historique prix")
        fig = px.line(info['history'], x=info['history'].index, y='Close',
                      labels={'Close':'Prix (€)'}, title='Historique 1 an')
        st.plotly_chart(fig, use_container_width=True)

        # Volatilité
        hist = info['history']['Close']
        vol_1y = hist.pct_change().std()
        vol_1m = hist.tail(22).pct_change().std()
        st.markdown(f"Volatilité 1 an : {vol_1y*100:.2f}% | Volatilité 1 mois : {vol_1m*100:.2f}%")
        if vol_1y>0.05: st.markdown("⚠️ Volatilité annuelle élevée, risque important")
        elif vol_1y<0.02: st.markdown("📈 Volatilité faible, actif stable")
        else: st.markdown("ℹ️ Volatilité modérée")

        # Mini-prédiction tendance simple
        recent_trend = hist.tail(22).mean() - hist.tail(44).mean()
        st.markdown(f"Tendance 1 mois : {'Hausse 📈' if recent_trend>0 else 'Baisse 📉' if recent_trend<0 else 'Stable ⬌'}")

# ---- Tab 4 : Conseil trading ----
with tab4:
    score, verdict = compute_score(info, aggressive, trading_mode)
    st.markdown(f'<h2 style="color:{verdict[1]}; text-align:center;">Verdict : {verdict[0]} ({score}/10) - {verdict[2]}</h2>', unsafe_allow_html=True)
    st.markdown("### Conseils personnalisés :")
    if info['history'] is not None:
        recent_vol = info['history']['Close'].pct_change().tail(10).std()
        if recent_vol>0.03: st.markdown("- Volatilité récente élevée : prudence")
        else: st.markdown("- Volatilité récente modérée")
    if info['calendar']:
        st.markdown("- Événements à venir :")
        for k,v in info['calendar'].items():
            st.markdown(f"  - {k}: {v}")

# ---- Tab 5 : Comparatif multi-tickers ----
with tab5:
    st.subheader("📊 Comparatif multi-tickers P/E et Heatmap sectorielle")
    tickers_list = [t.strip() for t in multi_tickers.split(",") if t.strip()]
    data = []
    for t in tickers_list:
        try:
            info_t = fetch_info(t)
            data.append({
                "Ticker": t,
                "P/E": info_t.get('trailingPE'),
                "Sector": info_t.get('sector'),
                "Price": info_t.get('price')
            })
        except:
            pass
    df_multi = pd.DataFrame(data)

    if compare_sector:
        df_multi = df_multi[df_multi['Sector']==info['sector']]

    if not df_multi.empty:
        # Graphique P/E comparatif
        fig_multi = px.bar(df_multi, x='Ticker', y='P/E', color='P/E', text='P/E',
                           color_continuous_scale=px.colors.sequential.Viridis,
                           title=f"Comparatif P/E dans le secteur {info['sector']}")
        st.plotly_chart(fig_multi, use_container_width=True)
        
        # Heatmap sectorielle
        pivot = df_multi.pivot_table(index='Sector', columns='Ticker', values='P/E')
        fig_heat = px.imshow(pivot, text_auto=True, color_continuous_scale='Viridis', title="Heatmap P/E sectorielle")
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.markdown("Pas assez de données pour le comparatif multi-tickers.")

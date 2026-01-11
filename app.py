import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
from datetime import datetime, timedelta
from scipy.optimize import minimize

# --- PAGE CONFIG ---
st.set_page_config(page_title="Alpha-Gen Portfolio", layout="wide")
st.title("⚡ Alpha-Gen: AI Portfolio Architect")

# --- 1. CONFIGURATION ---
FUNDS = {
    "Small Cap": {
        "SBI Small Cap": "119598",
        "Nippon India Small Cap": "118778",
        "Bank of India Small Cap": "145287",
        "Quant Small Cap": "120847"
    },
    "Mid Cap": {
        "WhiteOak Mid Cap": "147853",
        "HDFC Mid Cap": "119339",
        "Motilal Oswal Midcap": "127599"
    },
    "Flexi/Multi Cap": {
        "Parag Parikh Flexi Cap": "122639",
        "Quant Active Fund": "120836",
        "HDFC Flexi Cap": "119325"
    },
    "Value/Contra": {
        "SBI Contra": "119717",
        "Bandhan Sterling Value": "120146"
    },
    "Large Cap/Index": {
        "UTI Nifty 50": "120716",
        "HDFC Index Sensex": "119062"
    }
}

# --- 2. ROBUST DATA ENGINE (With Sanitizer) ---
@st.cache_data(ttl=3600)
def get_data(fund_codes):
    data = {}
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0.4472.124 Safari/537.36'}
    
    for name, code in fund_codes.items():
        try:
            url = f"https://api.mfapi.in/mf/{code}"
            resp = requests.get(url, headers=headers, timeout=5)
            if resp.status_code != 200: continue
            
            resp_json = resp.json()
            if resp_json['status'] == 'FAIL': continue
            
            df = pd.DataFrame(resp_json['data'])
            df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
            df['nav'] = pd.to_numeric(df['nav'], errors='coerce')
            
            # Sort is crucial for pct_change
            series = df.set_index('date')['nav'].sort_index()
            
            # DATA SANITIZER: Ignore funds with too little history
            if len(series) < 200: continue 
            
            data[name] = series
        except:
            continue
    
    if not data: return pd.DataFrame()
    
    # Combine and Clean
    df = pd.concat(data.values(), keys=data.keys(), axis=1)
    df = df.ffill() # Fill gaps
    
    # Filter last 3 years to ensure relevance
    start_date = datetime.now() - timedelta(days=365*3)
    df = df[df.index >= start_date]
    
    # Drop "Ghost" Rows (where funds haven't started yet)
    df = df.dropna() 
    
    return df

# --- 3. MATH ENGINE (Corrected Annualization) ---
def optimize_portfolio(prices, risk_profile):
    # 1. Calculate Returns
    returns = prices.pct_change().dropna()
    
    # SANITIZER: Remove "Flat" funds (Zero Volatility)
    # If a fund has std dev = 0, it breaks the optimizer
    valid_cols = returns.std() > 0.0001
    returns = returns.loc[:, valid_cols]
    
    if returns.empty:
        return None, None, None
        
    # 2. Annualize Inputs
    # Daily Mean * 252 = Annual Mean
    mean_returns = returns.mean() * 252 
    # Daily Cov * 252 = Annual Cov
    cov_matrix = returns.cov() * 252    
    
    num_assets = len(mean_returns)
    
    # 3. Define Optimization Objectives
    def portfolio_volatility(weights):
        # Result is Annualized Standard Deviation
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    def negative_sharpe(weights):
        p_ret = np.sum(mean_returns * weights)
        p_vol = portfolio_volatility(weights)
        # Risk Free Rate = 6.5% (0.065)
        return -(p_ret - 0.065) / p_vol 
    
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0.0, 1.0) for _ in range(num_assets))
    
    # 4. Risk Profiles
    init_guess = num_assets * [1./num_assets,]
    
    if risk_profile == "Conservative":
        bounds = tuple((0.0, 0.25) for _ in range(num_assets))
        result = minimize(portfolio_volatility, init_guess, 
                         method='SLSQP', bounds=bounds, constraints=constraints)
    elif risk_profile == "Moderate":
        bounds = tuple((0.0, 0.40) for _ in range(num_assets))
        result = minimize(negative_sharpe, init_guess, 
                         method='SLSQP', bounds=bounds, constraints=constraints)
    else: # Aggressive
        bounds = tuple((0.0, 0.60) for _ in range(num_assets))
        result = minimize(negative_sharpe, init_guess, 
                         method='SLSQP', bounds=bounds, constraints=constraints)
    
    return result.x, mean_returns, cov_matrix, returns.columns

# --- 4. SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Preferences")
    risk = st.radio("Risk Appetite", ["Conservative", "Moderate", "Aggressive"], index=1)
    cats = st.multiselect("Categories", list(FUNDS.keys()), default=["Small Cap", "Mid Cap", "Flexi/Multi Cap"])
    budget = st.number_input("Investment (₹)", value=100000, step=5000)

# --- 5. EXECUTION ---
selected_funds = {}
for c in cats: selected_funds.update(FUNDS[c])

if len(selected_funds) < 2:
    st.warning("⚠️ Select at least 2 categories.")
    st.stop()

with st.spinner("Fetching market data..."):
    df_prices = get_data(selected_funds)

if df_prices.empty:
    st.error("❌ No overlapping data found. Try different funds.")
    st.stop()

# Run Math
weights_array, mu, sigma, valid_funds = optimize_portfolio(df_prices, risk)

if weights_array is None:
    st.error("❌ Optimization failed. Data was too flat/stable to calculate risk.")
    st.stop()

weights = dict(zip(valid_funds, weights_array))

# Calculate Specs (Explicitly Annualized)
final_ret = np.sum(mu * weights_array)
final_vol = np.sqrt(np.dot(weights_array.T, np.dot(sigma, weights_array)))
final_sharpe = (final_ret - 0.065) / final_vol

# --- 6. DISPLAY ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🏆 Your Portfolio")
    alloc = []
    for name, w in weights.items():
        if w > 0.01:
            alloc.append({"Fund": name, "Alloc": f"{w*100:.1f}%", "Amt": int(w * budget), "Raw": w})
    
    df_show = pd.DataFrame(alloc).sort_values("Raw", ascending=False)
    st.dataframe(df_show[["Fund", "Alloc", "Amt"]], use_container_width=True, hide_index=True)

with col2:
    st.subheader("📊 Annualized Specs")
    st.metric("Expected Return", f"{final_ret*100:.1f}%")
    st.metric("Risk (Volatility)", f"{final_vol*100:.1f}%")
    st.metric("Sharpe Ratio", f"{final_sharpe:.2f}")

st.divider()

# Charts
c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(px.pie(df_show, values='Amt', names='Fund', hole=0.4), use_container_width=True)
with c2:
    st.subheader("Historical Growth (3Y)")
    # Re-calculate daily weighted returns for the chart
    daily_returns = df_prices[valid_funds].pct_change().dropna()
    w_ret = (daily_returns * pd.Series(weights)).sum(axis=1)
    cum_ret = (1 + w_ret).cumprod()
    st.line_chart(cum_ret)

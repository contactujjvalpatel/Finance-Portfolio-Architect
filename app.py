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

# --- 2. DATA ENGINE ---
@st.cache_data(ttl=3600)
def get_data(fund_codes):
    data = {}
    for name, code in fund_codes.items():
        try:
            url = f"https://api.mfapi.in/mf/{code}"
            resp = requests.get(url, timeout=5).json()
            if resp['status'] == 'FAIL': continue
            df = pd.DataFrame(resp['data'])
            df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
            df['nav'] = pd.to_numeric(df['nav'], errors='coerce')
            data[name] = df.set_index('date')['nav'].sort_index()
        except:
            continue
    
    if not data: return pd.DataFrame()
    
    # Align and filter last 3 years
    df = pd.concat(data.values(), keys=data.keys(), axis=1).dropna()
    start_date = datetime.now() - timedelta(days=365*3)
    return df[df.index >= start_date]

# --- 3. MATH ENGINE (No-Dependency Version) ---
def optimize_portfolio(prices, risk_profile):
    # Calculate returns and covariance
    returns = prices.pct_change().dropna()
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    num_assets = len(mean_returns)
    
    # Define Objectives
    def portfolio_volatility(weights):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    def negative_sharpe(weights):
        p_ret = np.sum(returns.mean() * weights) * 252
        p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return -(p_ret - 0.065) / p_vol # Assuming 6.5% risk free
    
    # Constraints
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    
    # Risk Bounds (The Logic)
    if risk_profile == "Conservative":
        bounds = tuple((0.0, 0.20) for _ in range(num_assets)) # Max 20% per fund
        result = minimize(portfolio_volatility, num_assets*[1./num_assets,], 
                         method='SLSQP', bounds=bounds, constraints=constraints)
    elif risk_profile == "Moderate":
        bounds = tuple((0.0, 0.35) for _ in range(num_assets)) # Max 35% per fund
        result = minimize(negative_sharpe, num_assets*[1./num_assets,], 
                         method='SLSQP', bounds=bounds, constraints=constraints)
    else: # Aggressive
        bounds = tuple((0.0, 0.60) for _ in range(num_assets)) # Max 60% per fund
        result = minimize(negative_sharpe, num_assets*[1./num_assets,], 
                         method='SLSQP', bounds=bounds, constraints=constraints)
    
    return result.x, mean_returns, cov_matrix

# --- 4. SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Preferences")
    risk = st.radio("Risk Appetite", ["Conservative", "Moderate", "Aggressive"], index=2)
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
    st.error("❌ Data Fetch Error. Check internet.")
    st.stop()

# Run Math
weights_array, mu, sigma = optimize_portfolio(df_prices, risk)
weights = dict(zip(df_prices.columns, weights_array))

# Calculate Specs
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
    # THIS WAS THE LINE THAT CAUSED THE ERROR
    st.dataframe(df_show[["Fund", "Alloc", "Amt"]], use_container_width=True, hide_index=True)

with col2:
    st.subheader("📊 Specs")
    st.metric("Expected Return", f"{final_ret*100:.1f}%")
    st.metric("Risk (Vol)", f"{final_vol*100:.1f}%")
    st.metric("Sharpe Ratio", f"{final_sharpe:.2f}")

st.divider()
c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(px.pie(df_show, values='Amt', names='Fund', hole=0.4), use_container_width=True)
with c2:
    st.subheader("Historical Trajectory (3Y)")
    w_ret = (df_prices.pct_change() * pd.Series(weights)).sum(axis=1)
    st.line_chart((1 + w_ret).cumprod())
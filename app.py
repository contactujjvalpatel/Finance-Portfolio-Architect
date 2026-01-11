import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
from datetime import datetime, timedelta
from scipy.optimize import minimize

# --- PAGE CONFIG ---
st.set_page_config(page_title="Alpha-Gen Portfolio", layout="wide")
st.title("Finance Portfolio Architect")

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

import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
from datetime import datetime, timedelta
from scipy.optimize import minimize

# --- PAGE CONFIG ---
st.set_page_config(page_title="Alpha-Gen Pro", layout="wide", page_icon="⚡")
st.title("⚡ Alpha-Gen: AI Portfolio Architect")
st.markdown("### *Institutional-Grade Portfolio Optimization*")

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

@st.cache_data(ttl=3600)
def get_benchmark():
    # Fetch Nifty 50 Index Fund data as a benchmark proxy
    try:
        url = "https://api.mfapi.in/mf/120716" # UTI Nifty 50
        resp = requests.get(url).json()
        df = pd.DataFrame(resp['data'])
        df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
        df['nav'] = pd.to_numeric(df['nav'])
        return df.set_index('date')['nav'].sort_index()
    except:
        return pd.Series()

# --- 3. MATH ENGINE ---
def optimize_portfolio(prices, risk_profile):
    returns = prices.pct_change().dropna()
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    num_assets = len(mean_returns)
    
    # Risk Free Rate
    rf = 0.065 
    
    def portfolio_volatility(weights):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    def negative_sharpe(weights):
        p_ret = np.sum(returns.mean() * weights) * 252
        p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return -(p_ret - rf) / p_vol
    
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    
    # Dynamic Risk Bounds
    if risk_profile == "Conservative":
        # Max 20% per fund, Min 5% cash equivalent behavior (simulated by lower bounds)
        bounds = tuple((0.0, 0.20) for _ in range(num_assets))
        result = minimize(portfolio_volatility, num_assets*[1./num_assets,], 
                         method='SLSQP', bounds=bounds, constraints=constraints)
    elif risk_profile == "Moderate":
        # Max 35% per fund
        bounds = tuple((0.0, 0.35) for _ in range(num_assets))
        result = minimize(negative_sharpe, num_assets*[1./num_assets,], 
                         method='SLSQP', bounds=bounds, constraints=constraints)
    else: # Aggressive
        # Max 60% per fund
        bounds = tuple((0.0, 0.60) for _ in range(num_assets))
        result = minimize(negative_sharpe, num_assets*[1./num_assets,], 
                         method='SLSQP', bounds=bounds, constraints=constraints)
    
    return result.x, mean_returns, cov_matrix

# --- 4. SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Configuration")
    risk = st.radio("Risk Appetite", ["Conservative", "Moderate", "Aggressive"], index=1)
    
    st.subheader("Asset Universe")
    cats = st.multiselect("Select Categories", list(FUNDS.keys()), 
                          default=["Small Cap", "Mid Cap", "Flexi/Multi Cap"])
    
    budget = st.number_input("Capital to Deploy (₹)", value=100000, step=10000)
    
    if st.button("🚀 Architect Portfolio"):
        run_optimization = True
    else:
        run_optimization = False

# --- 5. EXECUTION LOGIC ---
if run_optimization:
    selected_funds = {}
    for c in cats: selected_funds.update(FUNDS[c])

    if len(selected_funds) < 2:
        st.error("⚠️ Please select at least 2 categories for diversification.")
        st.stop()

    with st.spinner("Crunching numbers & fetching market data..."):
        df_prices = get_data(selected_funds)
        if df_prices.empty:
            st.error("❌ Data Fetch Error. Check internet connection.")
            st.stop()
            
        weights_array, mu, sigma = optimize_portfolio(df_prices, risk)
        weights = dict(zip(df_prices.columns, weights_array))

    # --- Calculations ---
    final_ret = np.sum(mu * weights_array)
    final_vol = np.sqrt(np.dot(weights_array.T, np.dot(sigma, weights_array)))
    final_sharpe = (final_ret - 0.065) / final_vol

    # Drawdown Calculation
    cum_ret = (1 + (df_prices.pct_change() * weights_array).sum(axis=1)).cumprod()
    peak = cum_ret.expanding(min_periods=1).max()
    drawdown = (cum_ret / peak) - 1
    max_mdd = drawdown.min()

    # --- 6. DISPLAY DASHBOARD ---
    
    # Top Stats Row
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Expected CAGR", f"{final_ret*100:.1f}%", delta="Annualized")
    kpi2.metric("Risk (Volatility)", f"{final_vol*100:.1f}%", delta_color="inverse")
    kpi3.metric("Sharpe Ratio", f"{final_sharpe:.2f}")
    kpi4.metric("Max Drawdown", f"{max_mdd*100:.1f}%", help="Worst fall from peak in last 3Y")

    st.divider()

    # TABS FOR VISUALS
    tab1, tab2, tab3 = st.tabs(["📈 Market Comparison", "🧩 Correlation Map", "💰 Allocation"])

    with tab1:
        st.subheader("Portfolio vs Nifty 50")
        bench_prices = get_benchmark()
        
        if not bench_prices.empty:
            # Align dates
            common_start = max(df_prices.index.min(), bench_prices.index.min())
            df_p = df_prices[df_prices.index >= common_start]
            df_b = bench_prices[bench_prices.index >= common_start]
            
            # Normalize to 100
            port_series = (1 + (df_p.pct_change() * weights_array).sum(axis=1)).cumprod() * 100
            bench_series = (1 + df_b.pct_change()).cumprod() * 100
            
            chart_df = pd.DataFrame({
                "AI Portfolio": port_series,
                "Nifty 50 Benchmark": bench_series
            }).ffill().dropna()
            
            st.line_chart(chart_df, color=["#00FFAA", "#FF4B4B"])
            
            alpha = port_series.iloc[-1] - bench_series.iloc[-1]
            st.caption(f"💡 Result: ₹100 invested 3 years ago would be **₹{port_series.iloc[-1]:.0f}** in this portfolio vs **₹{bench_series.iloc[-1]:.0f}** in Nifty 50.")
        else:
            st.warning("Benchmark data unavailable. Showing portfolio performance only.")
            st.line_chart((1 + (df_prices.pct_change() * weights_array).sum(axis=1)).cumprod())

    with tab2:
        st.subheader("Correlation Matrix")
        st.markdown("Use this to ensure you aren't buying funds that move exactly the same way (Dark Red = High Correlation).")
        corr = df_prices.corr()
        fig_corr = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r")
        st.plotly_chart(fig_corr, use_container_width=True)

    with tab3:
        c1, c2 = st.columns([1, 2])
        
        alloc_data = []
        for f, w in weights.items():
            if w > 0.01:
                alloc_data.append({"Fund": f, "Weight": w, "Amount": int(w * budget)})
        
        df_alloc = pd.DataFrame(alloc_data).sort_values("Weight", ascending=False)
        
        with c1:
            st.dataframe(
                df_alloc.style.format({"Weight": "{:.1%}", "Amount": "₹{:,}"}), 
                hide_index=True, 
                use_container_width=True
            )
        with c2:
            fig_pie = px.pie(df_alloc, values="Amount", names="Fund", hole=0.4, 
                             color_discrete_sequence=px.colors.sequential.RdBu)
            st.plotly_chart(fig_pie, use_container_width=True)

else:
    st.info("👈 Select your preferences and click **'Architect Portfolio'** to start.")

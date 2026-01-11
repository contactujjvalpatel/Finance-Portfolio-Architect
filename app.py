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

# --- 1. CONFIGURATION (EXPANDED UNIVERSE) ---
FUNDS = {
    "Large Cap & Index": {
        "UTI Nifty 50 Index": "120716",
        "HDFC Index Sensex": "119062",
        "ICICI Pru Bluechip": "120586",
        "Nippon India Large Cap": "118742"
    },
    "Large & Mid Cap": {
        "Mirae Asset Large & Midcap": "118825",
        "SBI Large & Midcap": "103009",
        "Kotak Equity Opportunities": "100912",
        "ICICI Pru Large & Mid Cap": "100353"
    },
    "Flexi & Multi Cap": {
        "Parag Parikh Flexi Cap": "122639",
        "Quant Active Fund (Multi)": "120836",
        "HDFC Flexi Cap": "119325",
        "Nippon India Multi Cap": "118758"
    },
    "Mid Cap": {
        "HDFC Mid Cap Opportunities": "119339",
        "Motilal Oswal Midcap": "127599",
        "Kotak Emerging Equity": "119806",
        "SBI Magnum Midcap": "103028"
    },
    "Small Cap": {
        "Nippon India Small Cap": "118778",
        "SBI Small Cap": "119598",
        "Quant Small Cap": "120847",
        "Axis Small Cap": "125354"
    },
    "Focused Funds": {
        "SBI Focused Equity": "119736",
        "HDFC Focused 30": "119349",
        "Franklin India Focused": "118579",
        "IIFL Focused Equity": "128954"
    },
    "Value & Contra": {
        "SBI Contra": "119717",
        "Bandhan Sterling Value": "120146",
        "Templeton India Value": "118536"
    },
    "ELSS (Tax Saver)": {
        "Mirae Asset Tax Saver": "135767",
        "Quant Tax Plan": "120844",
        "DSP Tax Saver": "119230"
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
        # Max 20% per fund
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
                          default=["Small Cap", "Mid Cap", "Flexi & Multi Cap"])
    
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

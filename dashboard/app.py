"""Main Streamlit dashboard application."""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import time
from typing import Dict, Any, List

# Import components
from components.api_client import get_api_client
from components.charts import (
    create_price_chart, create_returns_chart, create_drawdown_chart,
    create_correlation_heatmap, create_options_iv_surface, create_options_greeks_chart
)
from components.metrics import (
    display_key_metrics, display_options_summary, display_signal_summary,
    create_risk_metrics_table, create_options_chain_table, display_portfolio_metrics
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Quant Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2E86AB;
        margin-bottom: 1rem;
    }
    
    .alert-success {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 0.75rem;
        border-radius: 0.25rem;
        margin-bottom: 1rem;
    }
    
    .alert-warning {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 0.75rem;
        border-radius: 0.25rem;
        margin-bottom: 1rem;
    }
    
    .alert-danger {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 0.75rem;
        border-radius: 0.25rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize API client
api_client = get_api_client()

# Sidebar configuration
st.sidebar.title("📈 Quant Dashboard")
st.sidebar.markdown("---")

# Check API health
health_status = api_client.get_health()
if health_status.get('status') == 'healthy':
    st.sidebar.success("✅ Analytics Service Online")
else:
    st.sidebar.error("❌ Analytics Service Offline")
    st.error("Cannot connect to analytics service. Please check if the service is running.")

# Main navigation
tab_selection = st.sidebar.selectbox(
    "Select Dashboard Tab",
    ["Markets", "Risk", "Options", "Signals", "Data"]
)

# Symbol configuration
st.sidebar.markdown("### Symbol Configuration")

# Get available symbols from health status
available_symbols = health_status.get('active_symbols', [])
if not available_symbols:
    available_symbols = ['AAPL', 'MSFT', 'TSLA', 'SPY', 'BTCUSDT', 'ETHUSDT']

selected_symbol = st.sidebar.selectbox(
    "Select Symbol",
    available_symbols,
    index=0 if available_symbols else None
)

# Time range configuration
lookback_days = st.sidebar.slider(
    "Lookback Days",
    min_value=30,
    max_value=1000,
    value=252,
    step=30
)

# Source selection
source_options = ['yfinance', 'binance']
default_source = 'binance' if selected_symbol in ['BTCUSDT', 'ETHUSDT'] else 'yfinance'
selected_source = st.sidebar.selectbox(
    "Data Source",
    source_options,
    index=source_options.index(default_source)
)

# Benchmark selection for risk metrics
benchmark_symbol = st.sidebar.selectbox(
    "Benchmark (for Beta/Alpha)",
    ['SPY', 'QQQ', 'IWM', 'VTI'],
    index=0
)

st.sidebar.markdown("---")

# Auto-refresh toggle
auto_refresh = st.sidebar.checkbox("Auto Refresh (30s)", value=False)
if auto_refresh:
    time.sleep(30)
    st.rerun()

# Main content area
st.markdown('<h1 class="main-header">Quantitative Finance Dashboard</h1>', unsafe_allow_html=True)

# Tab content
if tab_selection == "Markets":
    st.header(f"📊 Markets - {selected_symbol}")
    
    # Fetch data
    with st.spinner("Loading market data..."):
        ts_data = api_client.get_time_series_metrics(selected_symbol, lookback_days, selected_source)
        risk_data = api_client.get_risk_metrics(selected_symbol, lookback_days, selected_source, benchmark_symbol)
        tech_data = api_client.get_technical_indicators(selected_symbol, lookback_days, selected_source)
    
    if ts_data and risk_data:
        # Key metrics
        st.subheader("Key Metrics")
        display_key_metrics(risk_data, ts_data, selected_symbol)
        
        st.markdown("---")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Price & Technical Indicators")
            # Combine data for chart
            chart_data = {
                'timestamps': ts_data.get('timestamps', []),
                'close_prices': [100 * (1 + r) for r in ts_data.get('cumulative_returns', [])],  # Normalized to 100
                'technical': tech_data
            }
            price_fig = create_price_chart(chart_data, selected_symbol)
            st.plotly_chart(price_fig, use_container_width=True)
        
        with col2:
            st.subheader("Returns Analysis")
            returns_fig = create_returns_chart(ts_data, selected_symbol)
            st.plotly_chart(returns_fig, use_container_width=True)
        
        # Drawdown chart
        st.subheader("Drawdown Analysis")
        drawdown_fig = create_drawdown_chart(ts_data, selected_symbol)
        st.plotly_chart(drawdown_fig, use_container_width=True)
        
        # Live data tiles
        st.subheader("Live Market Data")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            last_return = ts_data.get('returns_simple', [0])[-1] if ts_data.get('returns_simple') else 0
            st.metric(
                label="Last Return",
                value=f"{last_return*100:.2f}%",
                delta=f"{'📈' if last_return > 0 else '📉'}"
            )
        
        with col2:
            current_vol = ts_data.get('realized_volatility', 0)
            st.metric(
                label="Current Volatility",
                value=f"{current_vol*100:.1f}%"
            )
        
        with col3:
            autocorr = ts_data.get('autocorrelation_lag1', 0)
            st.metric(
                label="Autocorrelation",
                value=f"{autocorr:.3f}"
            )
        
        with col4:
            if tech_data.get('rsi'):
                current_rsi = tech_data['rsi'][-1] if tech_data['rsi'] else 50
                rsi_signal = "Overbought" if current_rsi > 70 else "Oversold" if current_rsi < 30 else "Neutral"
                st.metric(
                    label="RSI Signal",
                    value=rsi_signal,
                    delta=f"RSI: {current_rsi:.1f}"
                )
    else:
        st.error("Failed to load market data. Please check the analytics service.")

elif tab_selection == "Risk":
    st.header(f"⚠️ Risk Analysis - {selected_symbol}")
    
    # Fetch risk data
    with st.spinner("Loading risk analysis..."):
        risk_data = api_client.get_risk_metrics(selected_symbol, lookback_days, selected_source, benchmark_symbol)
        ts_data = api_client.get_time_series_metrics(selected_symbol, lookback_days, selected_source)
    
    if risk_data and ts_data:
        # Risk metrics table
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Risk Metrics")
            risk_table = create_risk_metrics_table(risk_data)
            st.dataframe(risk_table, use_container_width=True)
        
        with col2:
            st.subheader("Risk Assessment")
            
            # Risk level assessment
            sharpe = risk_data.get('sharpe_ratio', 0)
            max_dd = abs(risk_data.get('max_drawdown', 0))
            var_95 = abs(risk_data.get('var_95', 0))
            
            if sharpe > 1.0 and max_dd < 0.2:
                st.markdown('<div class="alert-success">✅ <strong>Low Risk Profile</strong><br>Good risk-adjusted returns with manageable drawdowns.</div>', unsafe_allow_html=True)
            elif sharpe > 0.5 or max_dd < 0.3:
                st.markdown('<div class="alert-warning">⚠️ <strong>Moderate Risk Profile</strong><br>Acceptable risk levels but monitor closely.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="alert-danger">🚨 <strong>High Risk Profile</strong><br>Elevated risk levels require careful consideration.</div>', unsafe_allow_html=True)
            
            # VaR interpretation
            st.markdown("### Value at Risk Interpretation")
            st.write(f"**95% VaR**: {var_95*100:.2f}%")
            st.write("This means there's a 5% chance of losing more than this amount in a single day.")
            
            # Beta interpretation
            beta = risk_data.get('beta')
            if beta:
                if beta > 1.2:
                    beta_desc = "High Beta - More volatile than market"
                elif beta > 0.8:
                    beta_desc = "Market Beta - Similar to market volatility"
                else:
                    beta_desc = "Low Beta - Less volatile than market"
                st.write(f"**Beta vs {benchmark_symbol}**: {beta:.2f} ({beta_desc})")
        
        # Drawdown analysis
        st.subheader("Drawdown Analysis")
        drawdown_fig = create_drawdown_chart(ts_data, selected_symbol)
        st.plotly_chart(drawdown_fig, use_container_width=True)
        
        # Portfolio correlation (if multiple symbols)
        st.subheader("Asset Correlation Analysis")
        
        # For demo, create correlation with benchmark
        correlation_symbols = [selected_symbol, benchmark_symbol]
        
        # Fetch benchmark data
        benchmark_data = api_client.get_time_series_metrics(benchmark_symbol, lookback_days, 'yfinance')
        
        if benchmark_data:
            # Calculate correlation (simplified)
            asset_returns = ts_data.get('returns_simple', [])
            benchmark_returns = benchmark_data.get('returns_simple', [])
            
            if len(asset_returns) == len(benchmark_returns) and len(asset_returns) > 10:
                correlation = np.corrcoef(asset_returns, benchmark_returns)[0, 1]
                correlation_matrix = [[1.0, correlation], [correlation, 1.0]]
                
                corr_fig = create_correlation_heatmap(correlation_matrix, correlation_symbols)
                st.plotly_chart(corr_fig, use_container_width=True)
            else:
                st.info("Insufficient data for correlation analysis")
        else:
            st.info("Benchmark data not available for correlation analysis")
    
    else:
        st.error("Failed to load risk data. Please check the analytics service.")

elif tab_selection == "Options":
    st.header(f"📋 Options Analysis")
    
    # Options symbol selection
    options_symbols = ['AAPL', 'SPY', 'MSFT', 'TSLA']  # Common options underlyings
    selected_underlying = st.selectbox(
        "Select Options Underlying",
        options_symbols,
        index=0
    )
    
    risk_free_rate = st.slider(
        "Risk-Free Rate (%)",
        min_value=0.0,
        max_value=10.0,
        value=5.0,
        step=0.1
    ) / 100
    
    # Fetch options data
    with st.spinner("Loading options data..."):
        options_data = api_client.get_options_analytics(selected_underlying, risk_free_rate)
    
    if options_data and options_data.get('greeks'):
        # Options summary
        st.subheader("Options Summary")
        display_options_summary(options_data, selected_underlying)
        
        st.markdown("---")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Implied Volatility Surface")
            iv_fig = create_options_iv_surface(options_data.get('iv_surface', []), selected_underlying)
            st.plotly_chart(iv_fig, use_container_width=True)
        
        with col2:
            st.subheader("Options Greeks")
            greek_selection = st.selectbox(
                "Select Greek",
                ['delta', 'gamma', 'theta', 'vega', 'rho'],
                index=0
            )
            
            greeks_fig = create_options_greeks_chart(
                options_data.get('greeks', []), 
                selected_underlying, 
                greek_selection
            )
            st.plotly_chart(greeks_fig, use_container_width=True)
        
        # Options chain table
        st.subheader("Options Chain with Greeks")
        
        # Expiry filter
        greeks_data = options_data.get('greeks', [])
        if greeks_data:
            expiries = sorted(list(set([str(g['expiry'])[:10] for g in greeks_data])))
            selected_expiry = st.selectbox("Filter by Expiry", ["All"] + expiries)
            
            expiry_filter = None if selected_expiry == "All" else selected_expiry
            
            options_table = create_options_chain_table(greeks_data, expiry_filter)
            st.dataframe(options_table, use_container_width=True)
        
        # Skew metrics
        skew_metrics = options_data.get('skew_metrics', {})
        if skew_metrics:
            st.subheader("Volatility Skew Metrics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                atm_iv = skew_metrics.get('atm_iv', 0)
                st.metric("ATM IV", f"{atm_iv*100:.1f}%" if atm_iv else "N/A")
            
            with col2:
                skew = skew_metrics.get('skew', 0)
                st.metric("Put-Call Skew", f"{skew*100:.1f}%" if skew else "N/A")
            
            with col3:
                term_slope = skew_metrics.get('term_structure_slope', 0)
                st.metric("Term Structure Slope", f"{term_slope*100:.1f}%" if term_slope else "N/A")
    
    else:
        st.error("Failed to load options data. Please check if options data is available for this underlying.")

elif tab_selection == "Signals":
    st.header("🚨 Trading Signals & Alerts")
    
    # Signal filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        signal_limit = st.selectbox("Number of Signals", [25, 50, 100, 200], index=1)
    
    with col2:
        signal_type_filter = st.selectbox(
            "Signal Type", 
            ["All", "crossover", "volatility_regime", "options_skew"]
        )
    
    with col3:
        if st.button("🔄 Refresh Signals"):
            st.rerun()
    
    # Fetch signals
    with st.spinner("Loading trading signals..."):
        signal_type = None if signal_type_filter == "All" else signal_type_filter
        signals = api_client.get_trading_signals(signal_limit, signal_type)
    
    if signals:
        # Signals summary
        st.subheader("Signals Summary")
        display_signal_summary(signals)
        
        st.markdown("---")
        
        # Recent signals
        st.subheader("Recent Trading Signals")
        
        # Convert to DataFrame for better display
        signals_df = pd.DataFrame(signals)
        
        if not signals_df.empty:
            # Format timestamp
            signals_df['timestamp'] = pd.to_datetime(signals_df['timestamp'])
            signals_df['Time'] = signals_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Format display columns
            display_signals = signals_df[['Time', 'symbol', 'signal_type', 'message', 'strength']].copy()
            display_signals.columns = ['Time', 'Symbol', 'Type', 'Message', 'Strength']
            display_signals['Strength'] = display_signals['Strength'].apply(lambda x: f"{x:.1%}")
            
            # Color code by signal type
            def highlight_signals(row):
                if row['Type'] == 'crossover':
                    return ['background-color: #e3f2fd'] * len(row)
                elif row['Type'] == 'volatility_regime':
                    return ['background-color: #fff3e0'] * len(row)
                else:
                    return [''] * len(row)
            
            styled_df = display_signals.style.apply(highlight_signals, axis=1)
            st.dataframe(styled_df, use_container_width=True)
            
            # Signal details
            if st.checkbox("Show Signal Details"):
                selected_signal_idx = st.selectbox(
                    "Select Signal for Details",
                    range(len(signals)),
                    format_func=lambda x: f"{signals[x]['symbol']} - {signals[x]['signal_type']} - {signals[x]['timestamp'][:19]}"
                )
                
                if selected_signal_idx is not None:
                    signal_detail = signals[selected_signal_idx]
                    
                    st.json(signal_detail)
        else:
            st.info("No signals found matching the selected criteria.")
    
    else:
        st.info("No trading signals available. The signal generation system may still be initializing.")
    
    # Signal configuration
    st.markdown("---")
    st.subheader("Signal Configuration")
    
    st.info("""
    **Current Signal Types:**
    - **Crossover Signals**: SMA 20/50 crossovers
    - **Volatility Regime**: High volatility periods (>75th percentile)
    - **Options Skew**: Unusual put-call skew patterns
    
    Signals are generated every minute and stored for 7 days.
    """)

elif tab_selection == "Data":
    st.header("💾 Data Management")
    
    # Data status
    st.subheader("Data Status")
    
    if health_status:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Database Status:**", "✅ Connected" if health_status.get('database_connected') else "❌ Disconnected")
            st.write("**Active Symbols:**", len(health_status.get('active_symbols', [])))
            st.write("**Last Updated:**", health_status.get('timestamp', 'Unknown'))
        
        with col2:
            data_freshness = health_status.get('data_freshness', {})
            st.write("**Data Freshness:**")
            for source, timestamp in data_freshness.items():
                st.write(f"- {source}: {timestamp}")
    
    st.markdown("---")
    
    # Data preview
    st.subheader("Data Preview")
    
    preview_symbol = st.selectbox(
        "Select Symbol for Preview",
        available_symbols,
        key="preview_symbol"
    )
    
    preview_source = st.selectbox(
        "Select Source for Preview",
        ['yfinance', 'binance'],
        key="preview_source"
    )
    
    if st.button("Load Data Preview"):
        with st.spinner("Loading data preview..."):
            # Get recent data
            ts_data = api_client.get_time_series_metrics(preview_symbol, 30, preview_source)
            
            if ts_data:
                # Create preview DataFrame
                timestamps = pd.to_datetime(ts_data.get('timestamps', []))
                returns = ts_data.get('returns_simple', [])
                cumulative = ts_data.get('cumulative_returns', [])
                
                if len(timestamps) == len(returns) == len(cumulative):
                    preview_df = pd.DataFrame({
                        'Date': timestamps,
                        'Daily Return': [f"{r*100:.2f}%" for r in returns],
                        'Cumulative Return': [f"{c*100:.2f}%" for c in cumulative]
                    })
                    
                    st.dataframe(preview_df.tail(20), use_container_width=True)
                else:
                    st.error("Data format mismatch")
            else:
                st.error("Failed to load data preview")
    
    st.markdown("---")
    
    # Data export
    st.subheader("Data Export")
    
    export_symbol = st.selectbox(
        "Select Symbol for Export",
        available_symbols,
        key="export_symbol"
    )
    
    export_days = st.slider(
        "Export Days",
        min_value=30,
        max_value=1000,
        value=252,
        key="export_days"
    )
    
    if st.button("Generate Export"):
        with st.spinner("Generating export data..."):
            # Get comprehensive data
            ts_data = api_client.get_time_series_metrics(export_symbol, export_days, selected_source)
            risk_data = api_client.get_risk_metrics(export_symbol, export_days, selected_source, benchmark_symbol)
            tech_data = api_client.get_technical_indicators(export_symbol, export_days, selected_source)
            
            if ts_data and risk_data and tech_data:
                # Create comprehensive export DataFrame
                timestamps = pd.to_datetime(ts_data.get('timestamps', []))
                
                export_data = {
                    'Date': timestamps,
                    'Returns_Simple': ts_data.get('returns_simple', []),
                    'Returns_Log': ts_data.get('returns_log', []),
                    'Cumulative_Returns': ts_data.get('cumulative_returns', []),
                    'Rolling_Mean': ts_data.get('rolling_mean', []),
                    'Rolling_Std': ts_data.get('rolling_std', []),
                    'SMA_20': tech_data.get('sma_20', []),
                    'SMA_50': tech_data.get('sma_50', []),
                    'RSI': tech_data.get('rsi', []),
                    'MACD': tech_data.get('macd', []),
                    'MACD_Signal': tech_data.get('macd_signal', [])
                }
                
                # Ensure all arrays have the same length
                min_length = min(len(v) for v in export_data.values() if isinstance(v, list))
                for key, value in export_data.items():
                    if isinstance(value, list) and len(value) > min_length:
                        export_data[key] = value[:min_length]
                
                export_df = pd.DataFrame(export_data)
                
                # Download button
                csv = export_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"{export_symbol}_data_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
                
                st.success(f"Export ready! {len(export_df)} rows of data for {export_symbol}")
            else:
                st.error("Failed to generate export data")
    
    # System information
    st.markdown("---")
    st.subheader("System Information")
    
    st.info("""
    **Data Sources:**
    - **yfinance**: Equity and options data from Yahoo Finance
    - **Binance**: Real-time cryptocurrency data via WebSocket
    
    **Storage:**
    - **DuckDB**: Fast analytical queries
    - **Parquet**: Partitioned data storage
    
    **Update Frequency:**
    - **Crypto**: Real-time (WebSocket)
    - **Equities**: Every 15 minutes during market hours
    - **Options**: Every 30 minutes during market hours
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>Quant Dashboard MVP | Built with Streamlit, FastAPI, and DuckDB</p>
        <p>Real-time financial analytics and risk management</p>
    </div>
    """,
    unsafe_allow_html=True
)

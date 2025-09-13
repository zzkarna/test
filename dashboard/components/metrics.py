"""Metrics display components."""

import streamlit as st
import pandas as pd
from typing import Dict, Any, Optional
import numpy as np


def display_key_metrics(risk_data: Dict[str, Any], ts_data: Dict[str, Any], 
                       symbol: str) -> None:
    """Display key metrics in a grid layout."""
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Sharpe Ratio",
            value=f"{risk_data.get('sharpe_ratio', 0):.2f}",
            help="Risk-adjusted return measure"
        )
        
        st.metric(
            label="Max Drawdown",
            value=f"{risk_data.get('max_drawdown', 0)*100:.1f}%",
            delta=f"{risk_data.get('current_drawdown', 0)*100:.1f}%",
            help="Maximum peak-to-trough decline"
        )
    
    with col2:
        st.metric(
            label="Sortino Ratio",
            value=f"{risk_data.get('sortino_ratio', 0):.2f}",
            help="Downside risk-adjusted return"
        )
        
        st.metric(
            label="VaR (95%)",
            value=f"{risk_data.get('var_95', 0)*100:.1f}%",
            help="Value at Risk at 95% confidence"
        )
    
    with col3:
        st.metric(
            label="Realized Volatility",
            value=f"{ts_data.get('realized_volatility', 0)*100:.1f}%",
            help="Annualized realized volatility"
        )
        
        st.metric(
            label="Beta",
            value=f"{risk_data.get('beta', 0):.2f}" if risk_data.get('beta') else "N/A",
            help="Sensitivity to market movements"
        )
    
    with col4:
        st.metric(
            label="EWMA Volatility",
            value=f"{ts_data.get('ewma_volatility', 0)*100:.1f}%",
            help="Exponentially weighted volatility"
        )
        
        st.metric(
            label="Information Ratio",
            value=f"{risk_data.get('information_ratio', 0):.2f}" if not np.isnan(risk_data.get('information_ratio', np.nan)) else "N/A",
            help="Excess return per unit of tracking error"
        )


def display_options_summary(options_data: Dict[str, Any], underlying: str) -> None:
    """Display options summary metrics."""
    
    col1, col2, col3, col4 = st.columns(4)
    
    spot_price = options_data.get('spot_price', 0)
    total_contracts = options_data.get('total_contracts', 0)
    skew_metrics = options_data.get('skew_metrics', {})
    
    with col1:
        st.metric(
            label="Spot Price",
            value=f"${spot_price:.2f}",
            help="Current underlying price"
        )
    
    with col2:
        st.metric(
            label="Total Contracts",
            value=f"{total_contracts:,}",
            help="Total options contracts available"
        )
    
    with col3:
        atm_iv = skew_metrics.get('atm_iv', 0)
        st.metric(
            label="ATM Implied Vol",
            value=f"{atm_iv*100:.1f}%" if atm_iv else "N/A",
            help="At-the-money implied volatility"
        )
    
    with col4:
        skew = skew_metrics.get('skew', 0)
        st.metric(
            label="Put-Call Skew",
            value=f"{skew*100:.1f}%" if skew else "N/A",
            help="Difference between put and call implied volatility"
        )


def display_signal_summary(signals: list) -> None:
    """Display trading signals summary."""
    
    if not signals:
        st.info("No recent trading signals")
        return
    
    # Count signals by type
    signal_counts = {}
    for signal in signals:
        signal_type = signal.get('signal_type', 'unknown')
        signal_counts[signal_type] = signal_counts.get(signal_type, 0) + 1
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Total Signals",
            value=len(signals),
            help="Total number of recent signals"
        )
    
    with col2:
        crossover_count = signal_counts.get('crossover', 0)
        st.metric(
            label="Crossover Signals",
            value=crossover_count,
            help="Moving average crossover signals"
        )
    
    with col3:
        vol_count = signal_counts.get('volatility_regime', 0)
        st.metric(
            label="Volatility Alerts",
            value=vol_count,
            help="Volatility regime change alerts"
        )


def create_risk_metrics_table(risk_data: Dict[str, Any]) -> pd.DataFrame:
    """Create risk metrics table."""
    
    metrics = [
        ("Sharpe Ratio", f"{risk_data.get('sharpe_ratio', 0):.3f}"),
        ("Sortino Ratio", f"{risk_data.get('sortino_ratio', 0):.3f}"),
        ("Calmar Ratio", f"{risk_data.get('calmar_ratio', 0):.3f}"),
        ("Information Ratio", f"{risk_data.get('information_ratio', 0):.3f}" if not np.isnan(risk_data.get('information_ratio', np.nan)) else "N/A"),
        ("Maximum Drawdown", f"{risk_data.get('max_drawdown', 0)*100:.2f}%"),
        ("Current Drawdown", f"{risk_data.get('current_drawdown', 0)*100:.2f}%"),
        ("VaR (95%)", f"{risk_data.get('var_95', 0)*100:.2f}%"),
        ("VaR (99%)", f"{risk_data.get('var_99', 0)*100:.2f}%"),
        ("Expected Shortfall (95%)", f"{risk_data.get('expected_shortfall_95', 0)*100:.2f}%"),
        ("Expected Shortfall (99%)", f"{risk_data.get('expected_shortfall_99', 0)*100:.2f}%"),
        ("Beta", f"{risk_data.get('beta', 0):.3f}" if risk_data.get('beta') else "N/A"),
        ("Alpha (Annualized)", f"{risk_data.get('alpha', 0)*100:.2f}%" if risk_data.get('alpha') else "N/A"),
    ]
    
    return pd.DataFrame(metrics, columns=["Metric", "Value"])


def create_options_chain_table(greeks_data: list, expiry_filter: Optional[str] = None) -> pd.DataFrame:
    """Create options chain table with Greeks."""
    
    if not greeks_data:
        return pd.DataFrame()
    
    df = pd.DataFrame(greeks_data)
    
    if expiry_filter:
        df = df[df['expiry'].astype(str).str.contains(expiry_filter)]
    
    # Format the data for display
    display_df = df.copy()
    
    # Format columns
    display_df['Strike'] = display_df['strike'].apply(lambda x: f"${x:.2f}")
    display_df['Type'] = display_df['option_type'].str.title()
    display_df['Market Price'] = display_df['market_price'].apply(lambda x: f"${x:.2f}")
    display_df['Theoretical'] = display_df['theoretical_price'].apply(lambda x: f"${x:.2f}")
    display_df['Delta'] = display_df['delta'].apply(lambda x: f"{x:.3f}")
    display_df['Gamma'] = display_df['gamma'].apply(lambda x: f"{x:.4f}")
    display_df['Theta'] = display_df['theta'].apply(lambda x: f"{x:.3f}")
    display_df['Vega'] = display_df['vega'].apply(lambda x: f"{x:.3f}")
    display_df['IV'] = display_df['implied_vol'].apply(lambda x: f"{x*100:.1f}%")
    
    # Select columns for display
    columns = ['Strike', 'Type', 'Market Price', 'Theoretical', 'Delta', 'Gamma', 'Theta', 'Vega', 'IV']
    
    return display_df[columns]


def format_large_number(num: float) -> str:
    """Format large numbers with appropriate suffixes."""
    if abs(num) >= 1e9:
        return f"{num/1e9:.1f}B"
    elif abs(num) >= 1e6:
        return f"{num/1e6:.1f}M"
    elif abs(num) >= 1e3:
        return f"{num/1e3:.1f}K"
    else:
        return f"{num:.2f}"


def display_portfolio_metrics(portfolio_data: Dict[str, Any]) -> None:
    """Display portfolio-level metrics."""
    
    if not portfolio_data:
        st.info("No portfolio data available")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Portfolio Return",
            value=f"{portfolio_data.get('portfolio_return', 0)*100:.2f}%",
            help="Annualized portfolio return"
        )
    
    with col2:
        st.metric(
            label="Portfolio Volatility",
            value=f"{portfolio_data.get('portfolio_volatility', 0)*100:.2f}%",
            help="Annualized portfolio volatility"
        )
    
    with col3:
        st.metric(
            label="Portfolio Sharpe",
            value=f"{portfolio_data.get('portfolio_sharpe', 0):.2f}",
            help="Portfolio Sharpe ratio"
        )
    
    with col4:
        symbols = portfolio_data.get('symbols', [])
        st.metric(
            label="Number of Assets",
            value=len(symbols),
            help="Number of assets in portfolio"
        )

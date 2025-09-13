"""Chart components for the dashboard."""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import streamlit as st


def create_price_chart(data: Dict[str, Any], symbol: str) -> go.Figure:
    """Create price chart with technical indicators."""
    if not data or 'timestamps' not in data:
        return go.Figure()
    
    timestamps = pd.to_datetime(data['timestamps'])
    
    # Get technical indicators
    tech_data = data.get('technical', {})
    
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(f'{symbol} Price & Technical Indicators', 'MACD', 'RSI'),
        row_heights=[0.6, 0.2, 0.2]
    )
    
    # Price and moving averages
    if 'close_prices' in data:
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=data['close_prices'],
                name='Close Price',
                line=dict(color='#2E86AB', width=2)
            ),
            row=1, col=1
        )
    
    if tech_data.get('sma_20'):
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=tech_data['sma_20'],
                name='SMA 20',
                line=dict(color='#A23B72', width=1)
            ),
            row=1, col=1
        )
    
    if tech_data.get('sma_50'):
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=tech_data['sma_50'],
                name='SMA 50',
                line=dict(color='#F18F01', width=1)
            ),
            row=1, col=1
        )
    
    # Bollinger Bands
    if all(k in tech_data for k in ['bb_upper', 'bb_middle', 'bb_lower']):
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=tech_data['bb_upper'],
                name='BB Upper',
                line=dict(color='rgba(128,128,128,0.3)', width=1),
                showlegend=False
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=tech_data['bb_lower'],
                name='BB Lower',
                line=dict(color='rgba(128,128,128,0.3)', width=1),
                fill='tonexty',
                fillcolor='rgba(128,128,128,0.1)',
                showlegend=False
            ),
            row=1, col=1
        )
    
    # MACD
    if tech_data.get('macd') and tech_data.get('macd_signal'):
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=tech_data['macd'],
                name='MACD',
                line=dict(color='#2E86AB', width=1)
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=tech_data['macd_signal'],
                name='MACD Signal',
                line=dict(color='#A23B72', width=1)
            ),
            row=2, col=1
        )
    
    # RSI
    if tech_data.get('rsi'):
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=tech_data['rsi'],
                name='RSI',
                line=dict(color='#F18F01', width=1)
            ),
            row=3, col=1
        )
        
        # RSI overbought/oversold lines
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    
    fig.update_layout(
        height=800,
        showlegend=True,
        title_text=f"{symbol} Technical Analysis",
        template="plotly_white"
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    
    return fig


def create_returns_chart(data: Dict[str, Any], symbol: str) -> go.Figure:
    """Create returns and volatility chart."""
    if not data or 'timestamps' not in data:
        return go.Figure()
    
    timestamps = pd.to_datetime(data['timestamps'])
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(f'{symbol} Cumulative Returns', 'Rolling Volatility'),
        row_heights=[0.6, 0.4]
    )
    
    # Cumulative returns
    if 'cumulative_returns' in data:
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=[r * 100 for r in data['cumulative_returns']],  # Convert to percentage
                name='Cumulative Returns (%)',
                line=dict(color='#2E86AB', width=2),
                fill='tonexty'
            ),
            row=1, col=1
        )
    
    # Rolling volatility
    if 'rolling_std' in data:
        annualized_vol = [v * np.sqrt(252) * 100 for v in data['rolling_std']]  # Annualized %
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=annualized_vol,
                name='Rolling Volatility (%)',
                line=dict(color='#A23B72', width=2)
            ),
            row=2, col=1
        )
    
    fig.update_layout(
        height=600,
        showlegend=True,
        title_text=f"{symbol} Returns Analysis",
        template="plotly_white"
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    
    return fig


def create_drawdown_chart(data: Dict[str, Any], symbol: str) -> go.Figure:
    """Create drawdown chart."""
    if not data or 'timestamps' not in data or 'cumulative_returns' not in data:
        return go.Figure()
    
    timestamps = pd.to_datetime(data['timestamps'])
    cumulative_returns = data['cumulative_returns']
    
    # Calculate drawdown
    cumulative_wealth = [1 + r for r in cumulative_returns]
    running_max = []
    current_max = 1
    
    for wealth in cumulative_wealth:
        if wealth > current_max:
            current_max = wealth
        running_max.append(current_max)
    
    drawdown = [(wealth - max_wealth) / max_wealth * 100 
                for wealth, max_wealth in zip(cumulative_wealth, running_max)]
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=drawdown,
            name='Drawdown (%)',
            line=dict(color='#C73E1D', width=2),
            fill='tonexty',
            fillcolor='rgba(199, 62, 29, 0.3)'
        )
    )
    
    fig.update_layout(
        title=f"{symbol} Drawdown Analysis",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        height=400,
        template="plotly_white"
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    
    return fig


def create_correlation_heatmap(correlation_data: List[List[float]], 
                             symbols: List[str]) -> go.Figure:
    """Create correlation heatmap."""
    if not correlation_data or not symbols:
        return go.Figure()
    
    fig = go.Figure(data=go.Heatmap(
        z=correlation_data,
        x=symbols,
        y=symbols,
        colorscale='RdBu',
        zmid=0,
        text=[[f"{val:.2f}" for val in row] for row in correlation_data],
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title="Asset Correlation Matrix",
        height=500,
        template="plotly_white"
    )
    
    return fig


def create_options_iv_surface(iv_data: List[Dict[str, Any]], underlying: str) -> go.Figure:
    """Create implied volatility surface."""
    if not iv_data:
        return go.Figure()
    
    df = pd.DataFrame(iv_data)
    
    # Separate calls and puts
    calls = df[df['option_type'] == 'call']
    puts = df[df['option_type'] == 'put']
    
    fig = go.Figure()
    
    if not calls.empty:
        fig.add_trace(
            go.Scatter(
                x=calls['moneyness'],
                y=calls['implied_vol'] * 100,
                mode='markers+lines',
                name='Calls',
                marker=dict(color='#2E86AB', size=6),
                line=dict(color='#2E86AB', width=2)
            )
        )
    
    if not puts.empty:
        fig.add_trace(
            go.Scatter(
                x=puts['moneyness'],
                y=puts['implied_vol'] * 100,
                mode='markers+lines',
                name='Puts',
                marker=dict(color='#A23B72', size=6),
                line=dict(color='#A23B72', width=2)
            )
        )
    
    fig.update_layout(
        title=f"{underlying} Implied Volatility Smile",
        xaxis_title="Moneyness (S/K)",
        yaxis_title="Implied Volatility (%)",
        height=500,
        template="plotly_white"
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    
    return fig


def create_options_greeks_chart(greeks_data: List[Dict[str, Any]], 
                              underlying: str, greek: str = 'delta') -> go.Figure:
    """Create options Greeks chart."""
    if not greeks_data:
        return go.Figure()
    
    df = pd.DataFrame(greeks_data)
    
    # Filter for nearest expiry
    nearest_expiry = df['expiry'].min()
    df_filtered = df[df['expiry'] == nearest_expiry]
    
    calls = df_filtered[df_filtered['option_type'] == 'call']
    puts = df_filtered[df_filtered['option_type'] == 'put']
    
    fig = go.Figure()
    
    if not calls.empty:
        fig.add_trace(
            go.Scatter(
                x=calls['strike'],
                y=calls[greek],
                mode='markers+lines',
                name=f'Calls {greek.title()}',
                marker=dict(color='#2E86AB', size=6),
                line=dict(color='#2E86AB', width=2)
            )
        )
    
    if not puts.empty:
        fig.add_trace(
            go.Scatter(
                x=puts['strike'],
                y=puts[greek],
                mode='markers+lines',
                name=f'Puts {greek.title()}',
                marker=dict(color='#A23B72', size=6),
                line=dict(color='#A23B72', width=2)
            )
        )
    
    # Add vertical line at current spot price
    if greeks_data:
        spot_price = greeks_data[0]['spot_price']
        fig.add_vline(x=spot_price, line_dash="dash", line_color="gray", 
                     annotation_text=f"Spot: ${spot_price:.2f}")
    
    fig.update_layout(
        title=f"{underlying} {greek.title()} by Strike",
        xaxis_title="Strike Price",
        yaxis_title=greek.title(),
        height=500,
        template="plotly_white"
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    
    return fig

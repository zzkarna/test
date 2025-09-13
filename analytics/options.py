"""Options analytics and Black-Scholes calculations."""

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq
from typing import Tuple, Optional, Dict, List
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class OptionsAnalyzer:
    """Options analytics and Greeks calculations."""
    
    @staticmethod
    def black_scholes_price(S: float, K: float, T: float, r: float, sigma: float, 
                          option_type: str = 'call') -> float:
        """Calculate Black-Scholes option price."""
        try:
            if T <= 0:
                # Option has expired
                if option_type.lower() == 'call':
                    return max(S - K, 0)
                else:
                    return max(K - S, 0)
            
            d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            
            if option_type.lower() == 'call':
                price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            else:  # put
                price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            
            return max(price, 0)  # Price cannot be negative
        except Exception as e:
            logger.error(f"Error calculating Black-Scholes price: {e}")
            return np.nan
    
    @staticmethod
    def calculate_delta(S: float, K: float, T: float, r: float, sigma: float, 
                       option_type: str = 'call') -> float:
        """Calculate option delta."""
        try:
            if T <= 0:
                if option_type.lower() == 'call':
                    return 1.0 if S > K else 0.0
                else:
                    return -1.0 if S < K else 0.0
            
            d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            
            if option_type.lower() == 'call':
                return norm.cdf(d1)
            else:  # put
                return norm.cdf(d1) - 1
        except Exception as e:
            logger.error(f"Error calculating delta: {e}")
            return np.nan
    
    @staticmethod
    def calculate_gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate option gamma."""
        try:
            if T <= 0:
                return 0.0
            
            d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            return norm.pdf(d1) / (S * sigma * np.sqrt(T))
        except Exception as e:
            logger.error(f"Error calculating gamma: {e}")
            return np.nan
    
    @staticmethod
    def calculate_theta(S: float, K: float, T: float, r: float, sigma: float, 
                       option_type: str = 'call') -> float:
        """Calculate option theta (time decay)."""
        try:
            if T <= 0:
                return 0.0
            
            d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            
            if option_type.lower() == 'call':
                theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - 
                        r * K * np.exp(-r * T) * norm.cdf(d2))
            else:  # put
                theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + 
                        r * K * np.exp(-r * T) * norm.cdf(-d2))
            
            return theta / 365  # Convert to daily theta
        except Exception as e:
            logger.error(f"Error calculating theta: {e}")
            return np.nan
    
    @staticmethod
    def calculate_vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate option vega."""
        try:
            if T <= 0:
                return 0.0
            
            d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            return S * norm.pdf(d1) * np.sqrt(T) / 100  # Divide by 100 for 1% vol change
        except Exception as e:
            logger.error(f"Error calculating vega: {e}")
            return np.nan
    
    @staticmethod
    def calculate_rho(S: float, K: float, T: float, r: float, sigma: float, 
                     option_type: str = 'call') -> float:
        """Calculate option rho."""
        try:
            if T <= 0:
                return 0.0
            
            d2 = (np.log(S / K) + (r - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            
            if option_type.lower() == 'call':
                return K * T * np.exp(-r * T) * norm.cdf(d2) / 100
            else:  # put
                return -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
        except Exception as e:
            logger.error(f"Error calculating rho: {e}")
            return np.nan
    
    @staticmethod
    def calculate_all_greeks(S: float, K: float, T: float, r: float, sigma: float, 
                           option_type: str = 'call') -> Dict[str, float]:
        """Calculate all Greeks at once."""
        return {
            'delta': OptionsAnalyzer.calculate_delta(S, K, T, r, sigma, option_type),
            'gamma': OptionsAnalyzer.calculate_gamma(S, K, T, r, sigma),
            'theta': OptionsAnalyzer.calculate_theta(S, K, T, r, sigma, option_type),
            'vega': OptionsAnalyzer.calculate_vega(S, K, T, r, sigma),
            'rho': OptionsAnalyzer.calculate_rho(S, K, T, r, sigma, option_type)
        }
    
    @staticmethod
    def implied_volatility(market_price: float, S: float, K: float, T: float, 
                         r: float, option_type: str = 'call') -> float:
        """Calculate implied volatility using Brent's method."""
        try:
            if T <= 0:
                return np.nan
            
            # Intrinsic value
            if option_type.lower() == 'call':
                intrinsic = max(S - K, 0)
            else:
                intrinsic = max(K - S, 0)
            
            if market_price <= intrinsic:
                return 0.01  # Minimum volatility
            
            def objective(sigma):
                return OptionsAnalyzer.black_scholes_price(S, K, T, r, sigma, option_type) - market_price
            
            # Use Brent's method to find implied volatility
            try:
                iv = brentq(objective, 0.001, 5.0, xtol=1e-6)
                return iv
            except ValueError:
                # If Brent's method fails, try a simple search
                for sigma in np.arange(0.01, 3.0, 0.01):
                    if abs(objective(sigma)) < 0.01:
                        return sigma
                return np.nan
        except Exception as e:
            logger.error(f"Error calculating implied volatility: {e}")
            return np.nan
    
    @staticmethod
    def calculate_moneyness(S: float, K: float) -> float:
        """Calculate moneyness (S/K)."""
        return S / K if K != 0 else np.nan
    
    @staticmethod
    def put_call_parity_check(call_price: float, put_price: float, S: float, 
                            K: float, T: float, r: float) -> float:
        """Check put-call parity and return deviation."""
        try:
            # Put-call parity: C - P = S - K * e^(-rT)
            theoretical_diff = S - K * np.exp(-r * T)
            actual_diff = call_price - put_price
            deviation = actual_diff - theoretical_diff
            return deviation
        except Exception as e:
            logger.error(f"Error checking put-call parity: {e}")
            return np.nan
    
    @staticmethod
    def calculate_iv_surface(options_data: pd.DataFrame, spot_price: float, 
                           risk_free_rate: float = 0.05) -> pd.DataFrame:
        """Calculate implied volatility surface."""
        try:
            iv_data = []
            
            for _, row in options_data.iterrows():
                # Calculate time to expiry in years
                expiry_date = pd.to_datetime(row['expiry'])
                current_date = pd.to_datetime(row['ts'])
                T = (expiry_date - current_date).days / 365.0
                
                if T > 0 and row['mid'] > 0:
                    iv = OptionsAnalyzer.implied_volatility(
                        row['mid'], spot_price, row['strike'], T, 
                        risk_free_rate, row['option_type']
                    )
                    
                    if not np.isnan(iv):
                        iv_data.append({
                            'strike': row['strike'],
                            'expiry': row['expiry'],
                            'option_type': row['option_type'],
                            'time_to_expiry': T,
                            'moneyness': OptionsAnalyzer.calculate_moneyness(spot_price, row['strike']),
                            'implied_vol': iv,
                            'mid_price': row['mid']
                        })
            
            return pd.DataFrame(iv_data)
        except Exception as e:
            logger.error(f"Error calculating IV surface: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def calculate_skew_metrics(iv_surface: pd.DataFrame) -> Dict[str, float]:
        """Calculate volatility skew metrics."""
        try:
            if iv_surface.empty:
                return {}
            
            # Focus on near-term options (closest expiry)
            nearest_expiry = iv_surface['expiry'].min()
            near_term = iv_surface[iv_surface['expiry'] == nearest_expiry]
            
            if len(near_term) < 3:
                return {}
            
            # Separate calls and puts
            calls = near_term[near_term['option_type'] == 'call']
            puts = near_term[near_term['option_type'] == 'put']
            
            metrics = {}
            
            # ATM volatility (closest to moneyness = 1.0)
            if not near_term.empty:
                atm_row = near_term.loc[(near_term['moneyness'] - 1.0).abs().idxmin()]
                metrics['atm_iv'] = atm_row['implied_vol']
            
            # OTM put volatility (moneyness < 0.95)
            otm_puts = puts[puts['moneyness'] < 0.95]
            if not otm_puts.empty:
                metrics['otm_put_iv'] = otm_puts['implied_vol'].mean()
            
            # OTM call volatility (moneyness > 1.05)
            otm_calls = calls[calls['moneyness'] > 1.05]
            if not otm_calls.empty:
                metrics['otm_call_iv'] = otm_calls['implied_vol'].mean()
            
            # Skew calculation
            if 'otm_put_iv' in metrics and 'otm_call_iv' in metrics:
                metrics['skew'] = metrics['otm_put_iv'] - metrics['otm_call_iv']
            
            # Term structure (if multiple expiries available)
            expiries = sorted(iv_surface['expiry'].unique())
            if len(expiries) > 1:
                term_structure = []
                for expiry in expiries[:3]:  # First 3 expiries
                    expiry_data = iv_surface[iv_surface['expiry'] == expiry]
                    if not expiry_data.empty:
                        avg_iv = expiry_data['implied_vol'].mean()
                        term_structure.append(avg_iv)
                
                if len(term_structure) >= 2:
                    metrics['term_structure_slope'] = term_structure[1] - term_structure[0]
            
            return metrics
        except Exception as e:
            logger.error(f"Error calculating skew metrics: {e}")
            return {}

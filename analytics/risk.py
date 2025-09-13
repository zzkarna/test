"""Risk analysis and calculations."""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class RiskAnalyzer:
    """Risk analysis for financial data."""
    
    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0, 
                             annualize: bool = True) -> float:
        """Calculate Sharpe ratio."""
        try:
            excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
            mean_excess = excess_returns.mean()
            std_excess = excess_returns.std()
            
            if std_excess == 0:
                return 0.0
            
            sharpe = mean_excess / std_excess
            
            if annualize:
                sharpe *= np.sqrt(252)
            
            return sharpe
        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {e}")
            return np.nan
    
    @staticmethod
    def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0,
                              annualize: bool = True) -> float:
        """Calculate Sortino ratio (downside deviation)."""
        try:
            excess_returns = returns - risk_free_rate / 252
            mean_excess = excess_returns.mean()
            
            # Calculate downside deviation
            downside_returns = excess_returns[excess_returns < 0]
            if len(downside_returns) == 0:
                return np.inf
            
            downside_std = downside_returns.std()
            
            if downside_std == 0:
                return 0.0
            
            sortino = mean_excess / downside_std
            
            if annualize:
                sortino *= np.sqrt(252)
            
            return sortino
        except Exception as e:
            logger.error(f"Error calculating Sortino ratio: {e}")
            return np.nan
    
    @staticmethod
    def calculate_calmar_ratio(returns: pd.Series, annualize: bool = True) -> float:
        """Calculate Calmar ratio (return/max drawdown)."""
        try:
            annual_return = returns.mean()
            if annualize:
                annual_return *= 252
            
            max_dd = RiskAnalyzer.calculate_max_drawdown(returns)[0]
            
            if max_dd == 0:
                return np.inf
            
            return annual_return / abs(max_dd)
        except Exception as e:
            logger.error(f"Error calculating Calmar ratio: {e}")
            return np.nan
    
    @staticmethod
    def calculate_information_ratio(returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate Information ratio."""
        try:
            # Align series and remove NaN
            aligned_data = pd.concat([returns, benchmark_returns], axis=1).dropna()
            if len(aligned_data) < 10:
                return np.nan
            
            asset_returns = aligned_data.iloc[:, 0]
            bench_returns = aligned_data.iloc[:, 1]
            
            excess_returns = asset_returns - bench_returns
            tracking_error = excess_returns.std()
            
            if tracking_error == 0:
                return 0.0
            
            return (excess_returns.mean() * 252) / (tracking_error * np.sqrt(252))
        except Exception as e:
            logger.error(f"Error calculating Information ratio: {e}")
            return np.nan
    
    @staticmethod
    def calculate_max_drawdown(returns: pd.Series) -> Tuple[float, pd.Series]:
        """Calculate maximum drawdown and drawdown series."""
        try:
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()
            
            return max_drawdown, drawdown
        except Exception as e:
            logger.error(f"Error calculating max drawdown: {e}")
            return np.nan, pd.Series()
    
    @staticmethod
    def calculate_var(returns: pd.Series, confidence_level: float = 0.05, 
                     method: str = 'historical') -> float:
        """Calculate Value at Risk (VaR)."""
        try:
            if method == 'historical':
                return np.percentile(returns, confidence_level * 100)
            elif method == 'parametric':
                mean = returns.mean()
                std = returns.std()
                return stats.norm.ppf(confidence_level, mean, std)
            else:
                raise ValueError("Method must be 'historical' or 'parametric'")
        except Exception as e:
            logger.error(f"Error calculating VaR: {e}")
            return np.nan
    
    @staticmethod
    def calculate_expected_shortfall(returns: pd.Series, confidence_level: float = 0.05) -> float:
        """Calculate Expected Shortfall (Conditional VaR)."""
        try:
            var = RiskAnalyzer.calculate_var(returns, confidence_level, 'historical')
            tail_returns = returns[returns <= var]
            
            if len(tail_returns) == 0:
                return var
            
            return tail_returns.mean()
        except Exception as e:
            logger.error(f"Error calculating Expected Shortfall: {e}")
            return np.nan
    
    @staticmethod
    def calculate_portfolio_var(returns_matrix: pd.DataFrame, weights: np.ndarray,
                              confidence_level: float = 0.05) -> float:
        """Calculate portfolio VaR using variance-covariance method."""
        try:
            # Calculate portfolio returns
            portfolio_returns = (returns_matrix * weights).sum(axis=1)
            return RiskAnalyzer.calculate_var(portfolio_returns, confidence_level)
        except Exception as e:
            logger.error(f"Error calculating portfolio VaR: {e}")
            return np.nan
    
    @staticmethod
    def calculate_component_var(returns_matrix: pd.DataFrame, weights: np.ndarray,
                              confidence_level: float = 0.05) -> np.ndarray:
        """Calculate component VaR for portfolio."""
        try:
            portfolio_returns = (returns_matrix * weights).sum(axis=1)
            portfolio_var = RiskAnalyzer.calculate_var(portfolio_returns, confidence_level)
            
            # Calculate marginal VaR for each asset
            marginal_vars = []
            epsilon = 0.01
            
            for i in range(len(weights)):
                # Perturb weight slightly
                perturbed_weights = weights.copy()
                perturbed_weights[i] += epsilon
                perturbed_weights = perturbed_weights / perturbed_weights.sum()  # Renormalize
                
                perturbed_returns = (returns_matrix * perturbed_weights).sum(axis=1)
                perturbed_var = RiskAnalyzer.calculate_var(perturbed_returns, confidence_level)
                
                marginal_var = (perturbed_var - portfolio_var) / epsilon
                marginal_vars.append(marginal_var)
            
            # Component VaR = weight * marginal VaR
            component_vars = weights * np.array(marginal_vars)
            
            return component_vars
        except Exception as e:
            logger.error(f"Error calculating component VaR: {e}")
            return np.array([np.nan] * len(weights))
    
    @staticmethod
    def calculate_kelly_criterion(returns: pd.Series, benchmark_returns: Optional[pd.Series] = None) -> float:
        """Calculate Kelly criterion for optimal position sizing."""
        try:
            if benchmark_returns is not None:
                # Multi-asset Kelly approximation
                excess_returns = returns - benchmark_returns
                mean_excess = excess_returns.mean()
                variance = excess_returns.var()
            else:
                # Single asset Kelly
                mean_excess = returns.mean()
                variance = returns.var()
            
            if variance == 0:
                return 0.0
            
            # Kelly fraction = mean / variance
            kelly_fraction = mean_excess / variance
            
            # Cap at reasonable levels (25% max)
            kelly_fraction = np.clip(kelly_fraction, -0.25, 0.25)
            
            return kelly_fraction
        except Exception as e:
            logger.error(f"Error calculating Kelly criterion: {e}")
            return np.nan
    
    @staticmethod
    def calculate_risk_parity_weights(returns_matrix: pd.DataFrame) -> np.ndarray:
        """Calculate risk parity portfolio weights."""
        try:
            # Calculate covariance matrix
            cov_matrix = returns_matrix.cov().values
            
            # Initialize equal weights
            n_assets = len(cov_matrix)
            weights = np.ones(n_assets) / n_assets
            
            # Iterative algorithm for risk parity
            for _ in range(100):  # Max iterations
                # Calculate risk contributions
                portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)
                marginal_contrib = (cov_matrix @ weights) / portfolio_vol
                risk_contrib = weights * marginal_contrib
                
                # Target risk contribution (equal for all assets)
                target_risk = portfolio_vol / n_assets
                
                # Update weights
                weights = weights * target_risk / risk_contrib
                weights = weights / weights.sum()  # Normalize
                
                # Check convergence
                if np.max(np.abs(risk_contrib - target_risk)) < 1e-6:
                    break
            
            return weights
        except Exception as e:
            logger.error(f"Error calculating risk parity weights: {e}")
            return np.ones(len(returns_matrix.columns)) / len(returns_matrix.columns)

"""Comprehensive portfolio performance analysis and position monitoring."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

from quantdesk.core.portfolio import Portfolio, Position
from quantdesk.core.metrics import sharpe_ratio, tearsheet
from quantdesk.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class PositionAnalysis:
    """Detailed analysis of a single position."""
    symbol: str
    quantity: int
    avg_cost: float
    market_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    day_pnl: float
    day_pnl_pct: float
    weight: float  # Position weight in portfolio
    sector: str = "Unknown"
    asset_class: str = "Equity"
    beta: float = 1.0
    volatility: float = 0.0
    correlation_to_market: float = 0.0


@dataclass
class PortfolioSnapshot:
    """Snapshot of portfolio at a point in time."""
    timestamp: datetime
    total_value: float
    cash: float
    equity_value: float
    positions_count: int
    long_value: float
    short_value: float
    net_exposure: float
    gross_exposure: float
    leverage: float


@dataclass
class RiskMetrics:
    """Portfolio risk metrics."""
    portfolio_beta: float
    portfolio_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    value_at_risk_95: float
    value_at_risk_99: float
    conditional_var_95: float
    correlation_to_spy: float
    tracking_error: float
    information_ratio: float


class PortfolioAnalyzer:
    """Comprehensive portfolio performance analysis and monitoring."""
    
    def __init__(self, portfolio: Portfolio, benchmark_symbol: str = "SPY") -> None:
        """Initialize portfolio analyzer.
        
        :param portfolio: Portfolio object to analyze.
        :param benchmark_symbol: Benchmark symbol for comparison.
        """
        self.portfolio = portfolio
        self.benchmark_symbol = benchmark_symbol
        
        # Historical data storage
        self.snapshots: List[PortfolioSnapshot] = []
        self.returns_history: List[float] = []
        self.benchmark_returns: List[float] = []
        
        # Market data cache
        self.market_prices: Dict[str, float] = {}
        self.sector_mapping: Dict[str, str] = {}
        self.beta_cache: Dict[str, float] = {}
        
        # Performance tracking
        self.inception_date: Optional[datetime] = None
        self.high_water_mark: float = 0.0
        self.current_drawdown: float = 0.0
        self.max_drawdown_ever: float = 0.0
        
    def update_market_data(self, prices: Dict[str, float], timestamp: Optional[datetime] = None) -> None:
        """Update market prices and calculate portfolio metrics.
        
        :param prices: Dictionary of symbol -> current price.
        :param timestamp: Timestamp for the update.
        """
        self.market_prices.update(prices)
        
        if timestamp is None:
            timestamp = datetime.now()
            
        # Create portfolio snapshot
        snapshot = self._create_snapshot(timestamp)
        self.snapshots.append(snapshot)
        
        # Calculate returns if we have previous snapshot
        if len(self.snapshots) > 1:
            prev_value = self.snapshots[-2].total_value
            current_return = (snapshot.total_value - prev_value) / prev_value
            self.returns_history.append(current_return)
            
        # Update high water mark and drawdown
        if snapshot.total_value > self.high_water_mark:
            self.high_water_mark = snapshot.total_value
            self.current_drawdown = 0.0
        else:
            self.current_drawdown = (self.high_water_mark - snapshot.total_value) / self.high_water_mark
            self.max_drawdown_ever = max(self.max_drawdown_ever, self.current_drawdown)
            
        # Set inception date on first update
        if self.inception_date is None:
            self.inception_date = timestamp
            
        log.debug(
            "portfolio.updated",
            timestamp=timestamp.isoformat(),
            total_value=snapshot.total_value,
            positions=snapshot.positions_count,
            drawdown=self.current_drawdown,
        )
    
    def _create_snapshot(self, timestamp: datetime) -> PortfolioSnapshot:
        """Create a portfolio snapshot at given timestamp."""
        total_value = self.portfolio.value(self.market_prices)
        equity_value = sum(
            pos.qty * self.market_prices.get(sym, 0.0) 
            for sym, pos in self.portfolio.positions.items()
        )
        
        # Calculate long/short exposures
        long_value = sum(
            pos.qty * self.market_prices.get(sym, 0.0)
            for sym, pos in self.portfolio.positions.items()
            if pos.qty > 0
        )
        short_value = sum(
            abs(pos.qty) * self.market_prices.get(sym, 0.0)
            for sym, pos in self.portfolio.positions.items()
            if pos.qty < 0
        )
        
        net_exposure = long_value - short_value
        gross_exposure = long_value + short_value
        leverage = gross_exposure / max(total_value, 1)
        
        return PortfolioSnapshot(
            timestamp=timestamp,
            total_value=total_value,
            cash=self.portfolio.cash,
            equity_value=equity_value,
            positions_count=len([p for p in self.portfolio.positions.values() if p.qty != 0]),
            long_value=long_value,
            short_value=short_value,
            net_exposure=net_exposure,
            gross_exposure=gross_exposure,
            leverage=leverage,
        )
    
    def analyze_positions(self) -> List[PositionAnalysis]:
        """Analyze all current positions in detail."""
        position_analyses = []
        total_value = self.portfolio.value(self.market_prices)
        
        for symbol, position in self.portfolio.positions.items():
            if position.qty == 0:
                continue
                
            market_price = self.market_prices.get(symbol, 0.0)
            market_value = position.qty * market_price
            unrealized_pnl = market_value - (position.qty * position.avg_price)
            unrealized_pnl_pct = unrealized_pnl / max(abs(position.qty * position.avg_price), 1)
            weight = market_value / max(total_value, 1)
            
            # Calculate day P&L (placeholder - would need previous day's price)
            day_pnl = 0.0  # Would calculate from previous close
            day_pnl_pct = 0.0
            
            analysis = PositionAnalysis(
                symbol=symbol,
                quantity=position.qty,
                avg_cost=position.avg_price,
                market_price=market_price,
                market_value=market_value,
                unrealized_pnl=unrealized_pnl,
                unrealized_pnl_pct=unrealized_pnl_pct,
                day_pnl=day_pnl,
                day_pnl_pct=day_pnl_pct,
                weight=weight,
                sector=self.sector_mapping.get(symbol, "Unknown"),
                beta=self.beta_cache.get(symbol, 1.0),
            )
            
            position_analyses.append(analysis)
            
        # Sort by absolute market value (largest positions first)
        position_analyses.sort(key=lambda x: abs(x.market_value), reverse=True)
        
        return position_analyses
    
    def calculate_risk_metrics(self, lookback_days: int = 252) -> RiskMetrics:
        """Calculate comprehensive risk metrics.
        
        :param lookback_days: Number of days to look back for calculations.
        :return: RiskMetrics object with calculated values.
        """
        if len(self.returns_history) < 30:
            log.warning("Insufficient return history for reliable risk metrics")
            
        returns_array = np.array(self.returns_history[-lookback_days:])
        
        # Basic risk metrics
        portfolio_vol = np.std(returns_array) * np.sqrt(252) if len(returns_array) > 1 else 0.0
        portfolio_sharpe = sharpe_ratio(pd.Series(returns_array)) if len(returns_array) > 1 else 0.0
        
        # Sortino ratio (using downside deviation)
        downside_returns = returns_array[returns_array < 0]
        downside_vol = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 1 else 0.0
        sortino = (np.mean(returns_array) * 252) / downside_vol if downside_vol > 0 else 0.0
        
        # Value at Risk
        var_95 = np.percentile(returns_array, 5) if len(returns_array) > 20 else 0.0
        var_99 = np.percentile(returns_array, 1) if len(returns_array) > 20 else 0.0
        
        # Conditional VaR (Expected Shortfall)
        cvar_95 = np.mean(returns_array[returns_array <= var_95]) if len(returns_array) > 20 else 0.0
        
        # Portfolio beta (placeholder - would need benchmark returns)
        portfolio_beta = 1.0
        
        return RiskMetrics(
            portfolio_beta=portfolio_beta,
            portfolio_volatility=portfolio_vol,
            sharpe_ratio=portfolio_sharpe,
            sortino_ratio=sortino,
            max_drawdown=self.max_drawdown_ever,
            value_at_risk_95=var_95,
            value_at_risk_99=var_99,
            conditional_var_95=cvar_95,
            correlation_to_spy=0.0,  # Placeholder
            tracking_error=0.0,      # Placeholder
            information_ratio=0.0,   # Placeholder
        )
    
    def get_sector_allocation(self) -> Dict[str, float]:
        """Calculate portfolio allocation by sector.
        
        :return: Dictionary of sector -> allocation percentage.
        """
        sector_values: Dict[str, float] = {}
        total_value = self.portfolio.value(self.market_prices)
        
        for symbol, position in self.portfolio.positions.items():
            if position.qty == 0:
                continue
                
            market_value = position.qty * self.market_prices.get(symbol, 0.0)
            sector = self.sector_mapping.get(symbol, "Unknown")
            sector_values[sector] = sector_values.get(sector, 0.0) + abs(market_value)
            
        # Convert to percentages
        return {
            sector: (value / max(total_value, 1)) * 100
            for sector, value in sector_values.items()
        }
    
    def get_top_positions(self, n: int = 10) -> List[PositionAnalysis]:
        """Get top N positions by market value.
        
        :param n: Number of top positions to return.
        :return: List of top position analyses.
        """
        all_positions = self.analyze_positions()
        return all_positions[:n]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary.
        
        :return: Dictionary with performance metrics.
        """
        if not self.snapshots:
            return {"error": "No portfolio snapshots available"}
            
        current_snapshot = self.snapshots[-1]
        risk_metrics = self.calculate_risk_metrics()
        
        # Calculate returns over different periods
        returns_1d = self._calculate_period_return(1)
        returns_1w = self._calculate_period_return(7)
        returns_1m = self._calculate_period_return(30)
        returns_3m = self._calculate_period_return(90)
        returns_ytd = self._calculate_ytd_return()
        returns_inception = self._calculate_inception_return()
        
        return {
            "current_value": current_snapshot.total_value,
            "cash": current_snapshot.cash,
            "positions_count": current_snapshot.positions_count,
            "net_exposure": current_snapshot.net_exposure,
            "gross_exposure": current_snapshot.gross_exposure,
            "leverage": current_snapshot.leverage,
            "returns": {
                "1_day": returns_1d,
                "1_week": returns_1w,
                "1_month": returns_1m,
                "3_months": returns_3m,
                "year_to_date": returns_ytd,
                "inception": returns_inception,
            },
            "risk_metrics": {
                "volatility": risk_metrics.portfolio_volatility,
                "sharpe_ratio": risk_metrics.sharpe_ratio,
                "sortino_ratio": risk_metrics.sortino_ratio,
                "max_drawdown": risk_metrics.max_drawdown,
                "current_drawdown": self.current_drawdown,
                "var_95": risk_metrics.value_at_risk_95,
                "var_99": risk_metrics.value_at_risk_99,
            },
            "sector_allocation": self.get_sector_allocation(),
            "top_positions": [
                {
                    "symbol": pos.symbol,
                    "weight": pos.weight,
                    "unrealized_pnl_pct": pos.unrealized_pnl_pct,
                    "market_value": pos.market_value,
                }
                for pos in self.get_top_positions(5)
            ],
        }
    
    def _calculate_period_return(self, days: int) -> float:
        """Calculate return over specified number of days."""
        if len(self.snapshots) < 2:
            return 0.0
            
        # Find snapshot from 'days' ago
        target_date = self.snapshots[-1].timestamp - timedelta(days=days)
        
        # Find closest snapshot to target date
        closest_snapshot = None
        min_diff = float('inf')
        
        for snapshot in self.snapshots:
            diff = abs((snapshot.timestamp - target_date).total_seconds())
            if diff < min_diff:
                min_diff = diff
                closest_snapshot = snapshot
                
        if closest_snapshot is None:
            return 0.0
            
        start_value = closest_snapshot.total_value
        end_value = self.snapshots[-1].total_value
        
        return (end_value - start_value) / max(start_value, 1)
    
    def _calculate_ytd_return(self) -> float:
        """Calculate year-to-date return."""
        if not self.snapshots:
            return 0.0
            
        current_year = self.snapshots[-1].timestamp.year
        
        # Find first snapshot of current year
        ytd_snapshot = None
        for snapshot in self.snapshots:
            if snapshot.timestamp.year == current_year:
                ytd_snapshot = snapshot
                break
                
        if ytd_snapshot is None:
            return 0.0
            
        start_value = ytd_snapshot.total_value
        end_value = self.snapshots[-1].total_value
        
        return (end_value - start_value) / max(start_value, 1)
    
    def _calculate_inception_return(self) -> float:
        """Calculate return since inception."""
        if len(self.snapshots) < 2:
            return 0.0
            
        start_value = self.snapshots[0].total_value
        end_value = self.snapshots[-1].total_value
        
        return (end_value - start_value) / max(start_value, 1)
    
    def generate_positions_report(self) -> pd.DataFrame:
        """Generate detailed positions report as DataFrame.
        
        :return: DataFrame with position details.
        """
        positions = self.analyze_positions()
        
        if not positions:
            return pd.DataFrame()
            
        data = []
        for pos in positions:
            data.append({
                "Symbol": pos.symbol,
                "Quantity": pos.quantity,
                "Avg Cost": pos.avg_cost,
                "Market Price": pos.market_price,
                "Market Value": pos.market_value,
                "Unrealized P&L": pos.unrealized_pnl,
                "Unrealized P&L %": pos.unrealized_pnl_pct * 100,
                "Weight %": pos.weight * 100,
                "Sector": pos.sector,
                "Beta": pos.beta,
            })
            
        df = pd.DataFrame(data)
        
        # Format numeric columns
        numeric_cols = ["Avg Cost", "Market Price", "Market Value", "Unrealized P&L"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].round(2)
                
        percentage_cols = ["Unrealized P&L %", "Weight %"]
        for col in percentage_cols:
            if col in df.columns:
                df[col] = df[col].round(2)
                
        return df
    
    def generate_performance_report(self) -> pd.DataFrame:
        """Generate performance history report as DataFrame.
        
        :return: DataFrame with performance history.
        """
        if not self.snapshots:
            return pd.DataFrame()
            
        data = []
        for i, snapshot in enumerate(self.snapshots):
            daily_return = 0.0
            if i > 0:
                prev_value = self.snapshots[i-1].total_value
                daily_return = (snapshot.total_value - prev_value) / max(prev_value, 1)
                
            data.append({
                "Date": snapshot.timestamp.date(),
                "Portfolio Value": snapshot.total_value,
                "Cash": snapshot.cash,
                "Equity Value": snapshot.equity_value,
                "Daily Return %": daily_return * 100,
                "Positions Count": snapshot.positions_count,
                "Net Exposure": snapshot.net_exposure,
                "Leverage": snapshot.leverage,
            })
            
        df = pd.DataFrame(data)
        
        # Format numeric columns
        numeric_cols = ["Portfolio Value", "Cash", "Equity Value", "Net Exposure"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].round(2)
                
        if "Daily Return %" in df.columns:
            df["Daily Return %"] = df["Daily Return %"].round(3)
            
        if "Leverage" in df.columns:
            df["Leverage"] = df["Leverage"].round(2)
                
        return df
    
    def set_sector_mapping(self, sector_map: Dict[str, str]) -> None:
        """Set sector mapping for symbols.
        
        :param sector_map: Dictionary mapping symbol -> sector.
        """
        self.sector_mapping.update(sector_map)
        
    def set_beta_values(self, beta_map: Dict[str, float]) -> None:
        """Set beta values for symbols.
        
        :param beta_map: Dictionary mapping symbol -> beta.
        """
        self.beta_cache.update(beta_map)
        
    def export_to_excel(self, filename: str) -> None:
        """Export portfolio analysis to Excel file.
        
        :param filename: Output Excel filename.
        """
        try:
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # Positions sheet
                positions_df = self.generate_positions_report()
                positions_df.to_excel(writer, sheet_name='Positions', index=False)
                
                # Performance sheet
                performance_df = self.generate_performance_report()
                performance_df.to_excel(writer, sheet_name='Performance', index=False)
                
                # Summary sheet
                summary = self.get_performance_summary()
                summary_data = []
                
                # Flatten summary for Excel
                for key, value in summary.items():
                    if isinstance(value, dict):
                        for subkey, subvalue in value.items():
                            summary_data.append({"Metric": f"{key}.{subkey}", "Value": subvalue})
                    else:
                        summary_data.append({"Metric": key, "Value": value})
                        
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
            log.info("portfolio.exported", filename=filename)
            
        except Exception as e:
            log.error("portfolio.export_failed", filename=filename, error=str(e))
            
    def get_correlation_matrix(self, lookback_days: int = 60) -> pd.DataFrame:
        """Calculate correlation matrix of positions.
        
        :param lookback_days: Number of days for correlation calculation.
        :return: Correlation matrix DataFrame.
        """
        # This would require historical price data for each position
        # Placeholder implementation
        symbols = [sym for sym, pos in self.portfolio.positions.items() if pos.qty != 0]
        
        if len(symbols) < 2:
            return pd.DataFrame()
            
        # Create placeholder correlation matrix
        correlation_data = np.eye(len(symbols))  # Identity matrix as placeholder
        
        return pd.DataFrame(
            correlation_data,
            index=symbols,
            columns=symbols
        ) 
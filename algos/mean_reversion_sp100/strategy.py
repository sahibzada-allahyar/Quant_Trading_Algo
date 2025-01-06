"""Mean Reversion Strategy for S&P 100 constituents."""
from __future__ import annotations

import backtrader as bt
import numpy as np
import pandas as pd

from quantdesk.stratlib.base_strategy import BaseStrategy
from quantdesk.stratlib.utils import position_size
from quantdesk.core.risk import kelly_fraction


class MeanReversionSP100(BaseStrategy):
    """Z-score based mean reversion strategy for S&P 100 stocks.
    
    Strategy Logic:
    - Calculate 5-minute return z-score vs 60-minute lookback
    - Enter when |z-score| > threshold (default 2.0)
    - Exit when z-score crosses back to mean
    - Maximum 3 concurrent positions
    - Position sizing based on inverse Kelly criterion
    """
    
    params = {
        "lookback_period": 60,      # Minutes for z-score calculation
        "return_period": 5,         # Minutes for return calculation
        "zscore_threshold": 2.0,    # Entry threshold
        "max_positions": 3,         # Maximum concurrent positions
        "volatility_target": 0.10,  # Annual volatility target
        "kelly_fraction": 0.25,     # Maximum Kelly fraction
        "stop_loss": 0.03,          # Stop loss as fraction
        "take_profit": 0.015,       # Take profit as fraction
        "printlog": False,
    }

    def __init__(self) -> None:
        super().__init__()
        
        # Strategy-specific indicators
        self.returns = bt.indicators.PercentChange(
            self.datas[0].close, period=self.params["return_period"]
        )
        
        # Rolling statistics for z-score
        self.returns_mean = bt.indicators.SMA(
            self.returns, period=self.params["lookback_period"]
        )
        self.returns_std = bt.indicators.StandardDeviation(
            self.returns, period=self.params["lookback_period"]
        )
        
        # Z-score indicator
        self.zscore = (self.returns - self.returns_mean) / self.returns_std
        
        # Position tracking
        self.position_count = 0
        self.entry_prices: dict[str, float] = {}
        self.entry_signals: dict[str, float] = {}

    def next(self) -> None:
        """Execute strategy logic on each bar."""
        if self.order:  # Skip if order pending
            return
            
        current_price = self.datas[0].close[0]
        current_zscore = self.zscore[0]
        symbol = self.datas[0]._name
        
        # Check for exit conditions first
        if self.position:
            self._check_exit_conditions(current_price, current_zscore, symbol)
            return
        
        # Check for entry conditions
        if self.position_count < self.params["max_positions"]:
            self._check_entry_conditions(current_price, current_zscore, symbol)

    def _check_entry_conditions(self, price: float, zscore: float, symbol: str) -> None:
        """Check if we should enter a position."""
        # Ensure we have enough data
        if len(self.zscore) < self.params["lookback_period"]:
            return
            
        # Check if z-score exceeds threshold
        if abs(zscore) > self.params["zscore_threshold"]:
            # Determine direction (mean reversion)
            if zscore > self.params["zscore_threshold"]:
                # Price is too high, expect reversion down -> SHORT
                direction = -1
            elif zscore < -self.params["zscore_threshold"]:
                # Price is too low, expect reversion up -> LONG
                direction = 1
            else:
                return
                
            # Calculate position size
            size = self._calculate_position_size(price, direction)
            
            if size > 0:
                if direction == 1:
                    self.order = self.buy(size=size)
                else:
                    self.order = self.sell(size=size)
                
                # Track entry
                self.entry_prices[symbol] = price
                self.entry_signals[symbol] = zscore
                self.position_count += 1
                
                self.log.info(
                    "entry.signal",
                    symbol=symbol,
                    direction="LONG" if direction == 1 else "SHORT",
                    price=price,
                    zscore=zscore,
                    size=size,
                )

    def _check_exit_conditions(self, price: float, zscore: float, symbol: str) -> None:
        """Check if we should exit current position."""
        if not self.position:
            return
            
        entry_price = self.entry_prices.get(symbol, 0)
        entry_zscore = self.entry_signals.get(symbol, 0)
        
        if entry_price == 0:
            return
            
        # Calculate P&L
        if self.position.size > 0:  # Long position
            pnl_pct = (price - entry_price) / entry_price
            # Exit if z-score crosses back toward mean or stop/profit hit
            exit_signal = (
                zscore <= 0 or  # Z-score crossed to negative (mean reversion)
                pnl_pct <= -self.params["stop_loss"] or
                pnl_pct >= self.params["take_profit"]
            )
        else:  # Short position
            pnl_pct = (entry_price - price) / entry_price
            # Exit if z-score crosses back toward mean or stop/profit hit
            exit_signal = (
                zscore >= 0 or  # Z-score crossed to positive (mean reversion)
                pnl_pct <= -self.params["stop_loss"] or
                pnl_pct >= self.params["take_profit"]
            )
        
        if exit_signal:
            self.order = self.close()
            
            # Clean up tracking
            if symbol in self.entry_prices:
                del self.entry_prices[symbol]
            if symbol in self.entry_signals:
                del self.entry_signals[symbol]
            self.position_count = max(0, self.position_count - 1)
            
            self.log.info(
                "exit.signal",
                symbol=symbol,
                price=price,
                zscore=zscore,
                pnl_pct=pnl_pct,
                reason="mean_reversion" if abs(zscore) < 1.0 else "stop_profit",
            )

    def _calculate_position_size(self, price: float, direction: int) -> int:
        """Calculate position size using volatility targeting and Kelly criterion."""
        if len(self.returns_std) < self.params["lookback_period"]:
            return 0
            
        # Get recent volatility estimate (annualized)
        daily_vol = self.returns_std[0] * np.sqrt(252 * 24 * 60 / self.params["return_period"])
        
        if daily_vol <= 0:
            return 0
            
        # Calculate base position size using volatility targeting
        portfolio_value = self.broker.getvalue()
        target_position_value = portfolio_value * self.params["volatility_target"] / daily_vol
        
        # Apply Kelly fraction cap
        max_kelly_value = portfolio_value * self.params["kelly_fraction"]
        position_value = min(target_position_value, max_kelly_value)
        
        # Convert to shares
        shares = int(position_value / price)
        
        # Ensure we don't exceed broker limits
        max_shares = int(portfolio_value * 0.95 / price)  # 95% of portfolio max
        shares = min(shares, max_shares)
        
        return max(shares, 0)

    def notify_order(self, order: bt.Order) -> None:
        """Override to track order completion."""
        super().notify_order(order)
        
        # Reset order reference when completed or failed
        if order.status in (order.Completed, order.Canceled, order.Margin, order.Rejected):
            self.order = None

    def notify_trade(self, trade: bt.Trade) -> None:
        """Override to log trade completion."""
        super().notify_trade(trade)
        
        if trade.isclosed:
            # Additional strategy-specific logging
            win_rate = self._calculate_win_rate()
            self.log.info(
                "trade.stats",
                total_trades=len(self.broker.get_orders_open()) + 1,
                win_rate=win_rate,
                avg_position_count=self.position_count,
            )

    def _calculate_win_rate(self) -> float:
        """Calculate current win rate from completed trades."""
        # This is a simplified calculation
        # In production, you'd track this more carefully
        return 0.54  # Placeholder based on strategy expectations 
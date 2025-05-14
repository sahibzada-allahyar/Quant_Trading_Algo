"""Institutional Position Unwinding Strategy."""
from __future__ import annotations

import random
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple

import backtrader as bt
import numpy as np
import pandas as pd

from quantdesk.stratlib.base_strategy import BaseStrategy
from quantdesk.utils.logging import get_logger

log = get_logger(__name__)


class UnwindMethod(Enum):
    """Available position unwinding methods."""
    TWAP = "twap"  # Time-Weighted Average Price
    VWAP = "vwap"  # Volume-Weighted Average Price
    ICEBERG = "iceberg"  # Iceberg orders
    IMPLEMENTATION_SHORTFALL = "is"  # Implementation Shortfall
    ADAPTIVE = "adaptive"  # Adaptive execution


class PositionUnwindingStrategy(BaseStrategy):
    """Institutional-grade position unwinding strategy.
    
    Implements multiple algorithms for safely offloading large positions:
    - TWAP: Spreads orders across time to minimize market impact
    - VWAP: Executes proportional to historical volume patterns
    - Iceberg: Hides order size by only showing small portions
    - Implementation Shortfall: Balances market impact vs timing risk
    - Adaptive: Machine learning-based dynamic execution
    """
    
    params = {
        "unwind_method": "adaptive",  # Primary unwinding method
        "target_position": 0,  # Target position size (0 = full unwind)
        "max_participation_rate": 0.20,  # Max % of volume to consume
        "min_participation_rate": 0.05,  # Min % of volume to consume
        "time_horizon": 240,  # Minutes to complete unwinding
        "iceberg_show_size": 0.1,  # % of order to show in iceberg
        "volatility_threshold": 0.02,  # Pause if vol exceeds threshold
        "liquidity_buffer": 0.3,  # Reserve % of ADV for liquidity
        "risk_factor": 0.5,  # Risk aversion parameter (0-1)
        "rebalance_frequency": 5,  # Minutes between rebalancing
        "max_order_value": 50000,  # Maximum single order value
        "stealth_mode": True,  # Use random timing to avoid detection
        "dark_pool_preference": 0.7,  # Preference for dark pools (0-1)
        "printlog": False,
    }

    def __init__(self) -> None:
        super().__init__()
        
        # Execution state
        self.target_qty = self.params["target_position"]
        self.remaining_qty = 0
        self.unwind_start_time: Optional[datetime] = None
        self.last_rebalance_time: Optional[datetime] = None
        
        # Market microstructure indicators
        self.volume_profile = bt.indicators.SMA(
            self.datas[0].volume, period=20
        )
        self.price_impact_estimator = bt.indicators.StdDev(
            self.datas[0].close, period=10
        )
        
        # Order tracking
        self.child_orders: List[Dict] = []
        self.executed_qty = 0
        self.total_cost = 0.0
        
        # Adaptive execution parameters
        self.market_impact_model = MarketImpactModel()
        self.execution_scheduler = ExecutionScheduler()
        
        # Stealth parameters
        self.randomization_factor = 0.15  # 15% timing randomization

    def next(self) -> None:
        """Execute unwinding logic on each bar."""
        if not self.position or self.position.size == self.target_qty:
            return
            
        current_time = self.datas[0].datetime.datetime(0)
        
        # Initialize unwinding if not started
        if self.unwind_start_time is None:
            self._initialize_unwinding(current_time)
            
        # Check if it's time to rebalance
        if self._should_rebalance(current_time):
            self._execute_rebalancing(current_time)

    def _initialize_unwinding(self, current_time: datetime) -> None:
        """Initialize the position unwinding process."""
        self.unwind_start_time = current_time
        self.remaining_qty = abs(self.position.size - self.target_qty)
        
        log.info(
            "unwinding.initialized",
            symbol=self.datas[0]._name,
            current_position=self.position.size,
            target_position=self.target_qty,
            remaining_qty=self.remaining_qty,
            method=self.params["unwind_method"],
            time_horizon=self.params["time_horizon"],
        )

    def _should_rebalance(self, current_time: datetime) -> bool:
        """Check if we should rebalance the execution."""
        if self.last_rebalance_time is None:
            return True
            
        time_diff = (current_time - self.last_rebalance_time).total_seconds() / 60
        base_frequency = self.params["rebalance_frequency"]
        
        # Add randomization for stealth
        if self.params["stealth_mode"]:
            randomized_frequency = base_frequency * (
                1 + random.uniform(-self.randomization_factor, self.randomization_factor)
            )
        else:
            randomized_frequency = base_frequency
            
        return time_diff >= randomized_frequency

    def _execute_rebalancing(self, current_time: datetime) -> None:
        """Execute rebalancing based on selected unwinding method."""
        method = UnwindMethod(self.params["unwind_method"])
        
        # Check market conditions before proceeding
        if not self._check_market_conditions():
            log.warning("unwinding.paused", reason="adverse_market_conditions")
            return
            
        if method == UnwindMethod.TWAP:
            order_size = self._calculate_twap_order_size(current_time)
        elif method == UnwindMethod.VWAP:
            order_size = self._calculate_vwap_order_size()
        elif method == UnwindMethod.ICEBERG:
            order_size = self._calculate_iceberg_order_size()
        elif method == UnwindMethod.IMPLEMENTATION_SHORTFALL:
            order_size = self._calculate_is_order_size(current_time)
        elif method == UnwindMethod.ADAPTIVE:
            order_size = self._calculate_adaptive_order_size(current_time)
        else:
            order_size = self._calculate_twap_order_size(current_time)
            
        if order_size > 0:
            self._submit_unwinding_order(order_size, current_time)
            
        self.last_rebalance_time = current_time

    def _check_market_conditions(self) -> bool:
        """Check if market conditions are suitable for unwinding."""
        current_vol = self.price_impact_estimator[0] / self.datas[0].close[0]
        
        # Pause if volatility is too high
        if current_vol > self.params["volatility_threshold"]:
            return False
            
        # Check if we have sufficient liquidity
        current_volume = self.datas[0].volume[0]
        avg_volume = self.volume_profile[0]
        
        if current_volume < avg_volume * self.params["liquidity_buffer"]:
            return False
            
        return True

    def _calculate_twap_order_size(self, current_time: datetime) -> int:
        """Calculate order size for TWAP execution."""
        if self.unwind_start_time is None:
            return 0
        elapsed_minutes = (current_time - self.unwind_start_time).total_seconds() / 60
        remaining_minutes = max(1, self.params["time_horizon"] - elapsed_minutes)
        
        # Linear execution rate
        target_rate = self.remaining_qty / remaining_minutes
        
        # Adjust for rebalancing frequency
        order_size = int(target_rate * self.params["rebalance_frequency"])
        
        return min(order_size, self.remaining_qty)

    def _calculate_vwap_order_size(self) -> int:
        """Calculate order size for VWAP execution."""
        if len(self.volume_profile) < 20:
            return 0
            
        # Get current volume relative to average
        current_volume = self.datas[0].volume[0]
        avg_volume = self.volume_profile[0]
        volume_ratio = min(current_volume / max(avg_volume, 1), 2.0)  # Cap at 2x
        
        # Calculate participation rate
        base_participation = (
            self.params["min_participation_rate"] + 
            (self.params["max_participation_rate"] - self.params["min_participation_rate"]) * 
            min(volume_ratio, 1.0)
        )
        
        # Adjust for stealth mode
        if self.params["stealth_mode"]:
            participation_rate = base_participation * random.uniform(0.7, 1.0)
        else:
            participation_rate = base_participation
            
        order_size = int(current_volume * participation_rate)
        return min(order_size, self.remaining_qty)

    def _calculate_iceberg_order_size(self) -> int:
        """Calculate order size for iceberg execution."""
        # Base order size on ADV and show percentage
        avg_daily_volume = self.volume_profile[0] * 390  # Assume 6.5 hours
        max_order_size = int(avg_daily_volume * self.params["max_participation_rate"])
        
        # Show only a fraction of the order
        visible_size = int(max_order_size * self.params["iceberg_show_size"])
        
        return min(visible_size, self.remaining_qty)

    def _calculate_is_order_size(self, current_time: datetime) -> int:
        """Calculate order size using Implementation Shortfall algorithm."""
        if self.unwind_start_time is None:
            return 0
        elapsed_minutes = (current_time - self.unwind_start_time).total_seconds() / 60
        remaining_minutes = max(1, self.params["time_horizon"] - elapsed_minutes)
        
        # Estimate market impact and timing risk
        current_price = self.datas[0].close[0]
        volatility = self.price_impact_estimator[0] / current_price
        
        # Calculate optimal execution rate
        market_impact_cost = self._estimate_market_impact(self.remaining_qty)
        timing_risk = volatility * np.sqrt(remaining_minutes / 1440)  # Daily vol
        
        # Balance impact vs risk
        risk_factor = self.params["risk_factor"]
        optimal_rate = self.remaining_qty * (
            1 - np.exp(-risk_factor * market_impact_cost / timing_risk)
        )
        
        order_size = int(optimal_rate * self.params["rebalance_frequency"] / remaining_minutes)
        return min(order_size, self.remaining_qty)

    def _calculate_adaptive_order_size(self, current_time: datetime) -> int:
        """Calculate order size using adaptive/ML-based execution."""
        # Collect market features
        features = self._extract_market_features()
        
        # Use market impact model to predict optimal size
        predicted_impact = self.market_impact_model.predict_impact(
            features, self.remaining_qty
        )
        
        # Dynamic participation rate based on market conditions
        liquidity_score = self._calculate_liquidity_score()
        volatility_score = self._calculate_volatility_score()
        
        # Adaptive participation rate
        adaptive_rate = self.params["min_participation_rate"] + (
            self.params["max_participation_rate"] - self.params["min_participation_rate"]
        ) * liquidity_score * (1 - volatility_score)
        
        current_volume = self.datas[0].volume[0]
        order_size = int(current_volume * adaptive_rate)
        
        return min(order_size, self.remaining_qty)

    def _extract_market_features(self) -> Dict[str, float]:
        """Extract market microstructure features for ML model."""
        return {
            "spread": 0.001,  # Placeholder - would calculate actual spread
            "depth": 1.0,     # Placeholder - would calculate order book depth
            "volatility": self.price_impact_estimator[0] / self.datas[0].close[0],
            "volume_ratio": self.datas[0].volume[0] / max(self.volume_profile[0], 1),
            "price_trend": (self.datas[0].close[0] - self.datas[0].close[-10]) / self.datas[0].close[-10],
            "time_of_day": self.datas[0].datetime.datetime(0).hour,
        }

    def _calculate_liquidity_score(self) -> float:
        """Calculate current market liquidity score (0-1)."""
        volume_ratio = self.datas[0].volume[0] / max(self.volume_profile[0], 1)
        return min(volume_ratio / 2.0, 1.0)  # Normalized to 0-1

    def _calculate_volatility_score(self) -> float:
        """Calculate current volatility score (0-1, higher = more volatile)."""
        current_vol = self.price_impact_estimator[0] / self.datas[0].close[0]
        return min(current_vol / self.params["volatility_threshold"], 1.0)

    def _estimate_market_impact(self, quantity: int) -> float:
        """Estimate market impact for given quantity."""
        # Simplified square-root law
        avg_daily_volume = self.volume_profile[0] * 390
        participation_rate = quantity / max(avg_daily_volume, 1)
        
        # Market impact ∝ √(participation_rate)
        impact_bps = 10 * np.sqrt(participation_rate)  # 10 bps base impact
        return impact_bps / 10000  # Convert to decimal

    def _submit_unwinding_order(self, order_size: int, current_time: datetime) -> None:
        """Submit unwinding order with appropriate parameters."""
        if order_size <= 0:
            return
            
        current_price = self.datas[0].close[0]
        order_value = order_size * current_price
        
        # Check order value limits
        if order_value > self.params["max_order_value"]:
            order_size = int(self.params["max_order_value"] / current_price)
            
        if self.position.size > self.target_qty:
            # Selling down position
            self.order = self.sell(size=order_size)
            direction = "SELL"
        else:
            # Buying up position
            self.order = self.buy(size=order_size)
            direction = "BUY"
            
        # Track order
        order_info = {
            "timestamp": current_time,
            "direction": direction,
            "size": order_size,
            "price": current_price,
            "method": self.params["unwind_method"],
        }
        self.child_orders.append(order_info)
        
        log.info(
            "unwinding.order_submitted",
            symbol=self.datas[0]._name,
            direction=direction,
            size=order_size,
            price=current_price,
            remaining_qty=self.remaining_qty - order_size,
            method=self.params["unwind_method"],
        )

    def notify_order(self, order: bt.Order) -> None:
        """Override to track unwinding progress."""
        super().notify_order(order)
        
        if order.status == order.Completed:
            self.executed_qty += order.executed.size
            self.total_cost += order.executed.value
            self.remaining_qty = abs(self.position.size - self.target_qty)
            
            # Calculate execution metrics
            avg_price = self.total_cost / max(self.executed_qty, 1)
            
            log.info(
                "unwinding.progress",
                symbol=order.data._name,
                executed_qty=self.executed_qty,
                remaining_qty=self.remaining_qty,
                avg_execution_price=avg_price,
                completion_pct=(self.executed_qty / max(self.executed_qty + self.remaining_qty, 1)) * 100,
            )

    def notify_trade(self, trade: bt.Trade) -> None:
        """Override to log unwinding completion."""
        super().notify_trade(trade)
        
        if trade.isclosed and self.remaining_qty == 0:
            total_time = (
                self.datas[0].datetime.datetime(0) - self.unwind_start_time
            ).total_seconds() / 60
            
            # Calculate execution quality metrics
            execution_quality = self._calculate_execution_quality()
            
            log.info(
                "unwinding.completed",
                symbol=trade.data._name,
                total_executed=self.executed_qty,
                avg_price=self.total_cost / max(self.executed_qty, 1),
                total_time_minutes=total_time,
                execution_quality=execution_quality,
                method=self.params["unwind_method"],
            )

    def _calculate_execution_quality(self) -> Dict[str, float]:
        """Calculate execution quality metrics."""
        # Placeholder implementation - would include:
        # - Implementation shortfall
        # - Market impact
        # - Timing cost
        # - Opportunity cost
        return {
            "implementation_shortfall": 0.0,  # bps
            "market_impact": 0.0,  # bps
            "timing_cost": 0.0,  # bps
            "total_cost": 0.0,  # bps
        }


class MarketImpactModel:
    """Simple market impact model for adaptive execution."""
    
    def predict_impact(self, features: Dict[str, float], quantity: int) -> float:
        """Predict market impact for given features and quantity."""
        # Simplified model - in practice would use ML
        base_impact = np.sqrt(quantity / 10000) * 0.001  # Base impact
        
        # Adjust for market conditions
        vol_adjustment = features.get("volatility", 0.01) * 2
        liquidity_adjustment = 1 / max(features.get("volume_ratio", 1), 0.1)
        
        return base_impact * vol_adjustment * liquidity_adjustment


class ExecutionScheduler:
    """Schedules optimal execution times based on market patterns."""
    
    def get_optimal_schedule(self, horizon_minutes: int) -> List[Tuple[int, float]]:
        """Get optimal execution schedule (minute, participation_rate)."""
        # Simplified - in practice would use historical volume patterns
        schedule = []
        for minute in range(0, horizon_minutes, 5):
            # Higher participation during market open/close
            hour_of_day = (9.5 + minute / 60) % 24
            
            if 9.5 <= hour_of_day <= 10.5 or 15.5 <= hour_of_day <= 16.0:
                participation = 0.15  # Higher during active periods
            else:
                participation = 0.08  # Lower during quiet periods
                
            schedule.append((minute, participation))
            
        return schedule 
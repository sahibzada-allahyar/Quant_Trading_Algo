"""Execution analytics for position unwinding strategies."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from quantdesk.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class ExecutionFill:
    """Represents a single execution fill."""
    timestamp: datetime
    symbol: str
    side: str  # BUY or SELL
    quantity: int
    price: float
    venue: str = "primary"
    order_type: str = "market"


@dataclass
class ExecutionBenchmark:
    """Benchmark prices for execution analysis."""
    arrival_price: float
    twap_price: float
    vwap_price: float
    close_price: float


class ExecutionAnalytics:
    """Analytics for measuring execution quality and market impact."""
    
    def __init__(self) -> None:
        self.fills: List[ExecutionFill] = []
        self.benchmarks: Dict[str, ExecutionBenchmark] = {}
        
    def add_fill(self, fill: ExecutionFill) -> None:
        """Add an execution fill to the analytics."""
        self.fills.append(fill)
        
    def set_benchmark(self, symbol: str, benchmark: ExecutionBenchmark) -> None:
        """Set benchmark prices for a symbol."""
        self.benchmarks[symbol] = benchmark
        
    def calculate_implementation_shortfall(self, symbol: str) -> Dict[str, float]:
        """Calculate implementation shortfall metrics.
        
        :param symbol: Symbol to analyze.
        :return: Implementation shortfall breakdown in basis points.
        """
        symbol_fills = [f for f in self.fills if f.symbol == symbol]
        if not symbol_fills or symbol not in self.benchmarks:
            return {}
            
        benchmark = self.benchmarks[symbol]
        arrival_price = benchmark.arrival_price
        
        # Calculate weighted average execution price
        total_value = sum(f.quantity * f.price for f in symbol_fills)
        total_quantity = sum(f.quantity for f in symbol_fills)
        avg_execution_price = total_value / max(total_quantity, 1)
        
        # Determine if we were buying or selling
        net_quantity = sum(f.quantity if f.side == "BUY" else -f.quantity for f in symbol_fills)
        is_buying = net_quantity > 0
        
        # Calculate implementation shortfall components
        market_impact = self._calculate_market_impact(symbol_fills, arrival_price, is_buying)
        timing_cost = self._calculate_timing_cost(symbol_fills, benchmark, is_buying)
        
        total_is = market_impact + timing_cost
        
        return {
            "total_implementation_shortfall": total_is,
            "market_impact": market_impact,
            "timing_cost": timing_cost,
            "arrival_price": arrival_price,
            "avg_execution_price": avg_execution_price,
        }
        
    def _calculate_market_impact(self, fills: List[ExecutionFill], arrival_price: float, is_buying: bool) -> float:
        """Calculate market impact in basis points."""
        total_value = sum(f.quantity * f.price for f in fills)
        total_quantity = sum(f.quantity for f in fills)
        avg_price = total_value / max(total_quantity, 1)
        
        if is_buying:
            impact = (avg_price - arrival_price) / arrival_price
        else:
            impact = (arrival_price - avg_price) / arrival_price
            
        return impact * 10000  # Convert to basis points
        
    def _calculate_timing_cost(self, fills: List[ExecutionFill], benchmark: ExecutionBenchmark, is_buying: bool) -> float:
        """Calculate timing cost in basis points."""
        arrival_price = benchmark.arrival_price
        close_price = benchmark.close_price
        
        if is_buying:
            timing_cost = (close_price - arrival_price) / arrival_price
        else:
            timing_cost = (arrival_price - close_price) / arrival_price
            
        return timing_cost * 10000  # Convert to basis points
        
    def calculate_participation_rate(self, symbol: str, market_volume: float) -> float:
        """Calculate participation rate as percentage of market volume.
        
        :param symbol: Symbol to analyze.
        :param market_volume: Total market volume during execution period.
        :return: Participation rate as percentage.
        """
        symbol_fills = [f for f in self.fills if f.symbol == symbol]
        total_executed = sum(f.quantity for f in symbol_fills)
        
        return (total_executed / max(market_volume, 1)) * 100
        
    def calculate_venue_breakdown(self, symbol: str) -> Dict[str, float]:
        """Calculate execution breakdown by venue.
        
        :param symbol: Symbol to analyze.
        :return: Dictionary of venue -> percentage executed.
        """
        symbol_fills = [f for f in self.fills if f.symbol == symbol]
        total_quantity = sum(f.quantity for f in symbol_fills)
        
        venue_quantities = {}
        for fill in symbol_fills:
            venue_quantities[fill.venue] = venue_quantities.get(fill.venue, 0) + fill.quantity
            
        return {
            venue: (qty / max(total_quantity, 1)) * 100
            for venue, qty in venue_quantities.items()
        }
        
    def calculate_execution_rate(self, symbol: str) -> Dict[str, float]:
        """Calculate execution rate statistics.
        
        :param symbol: Symbol to analyze.
        :return: Execution rate statistics.
        """
        symbol_fills = [f for f in self.fills if f.symbol == symbol]
        if len(symbol_fills) < 2:
            return {}
            
        # Sort by timestamp
        symbol_fills.sort(key=lambda x: x.timestamp)
        
        # Calculate time intervals
        intervals = []
        quantities = []
        
        for i in range(1, len(symbol_fills)):
            time_diff = (symbol_fills[i].timestamp - symbol_fills[i-1].timestamp).total_seconds() / 60
            intervals.append(time_diff)
            quantities.append(symbol_fills[i].quantity)
            
        # Calculate rates (shares per minute)
        rates = [q / max(t, 1) for q, t in zip(quantities, intervals)]
        
        return {
            "avg_execution_rate": np.mean(rates),
            "median_execution_rate": np.median(rates),
            "std_execution_rate": np.std(rates),
            "min_execution_rate": np.min(rates),
            "max_execution_rate": np.max(rates),
        }
        
    def generate_execution_report(self, symbol: str) -> Dict[str, any]:
        """Generate comprehensive execution report.
        
        :param symbol: Symbol to analyze.
        :return: Comprehensive execution report.
        """
        symbol_fills = [f for f in self.fills if f.symbol == symbol]
        if not symbol_fills:
            return {"error": "No fills found for symbol"}
            
        # Basic statistics
        total_quantity = sum(f.quantity for f in symbol_fills)
        total_value = sum(f.quantity * f.price for f in symbol_fills)
        avg_price = total_value / max(total_quantity, 1)
        
        # Time statistics
        symbol_fills.sort(key=lambda x: x.timestamp)
        start_time = symbol_fills[0].timestamp
        end_time = symbol_fills[-1].timestamp
        duration_minutes = (end_time - start_time).total_seconds() / 60
        
        report = {
            "symbol": symbol,
            "execution_summary": {
                "total_quantity": total_quantity,
                "total_value": total_value,
                "avg_execution_price": avg_price,
                "num_fills": len(symbol_fills),
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_minutes": duration_minutes,
            },
            "venue_breakdown": self.calculate_venue_breakdown(symbol),
            "execution_rates": self.calculate_execution_rate(symbol),
        }
        
        # Add implementation shortfall if benchmarks available
        if symbol in self.benchmarks:
            report["implementation_shortfall"] = self.calculate_implementation_shortfall(symbol)
            
        return report
        
    def export_to_dataframe(self) -> pd.DataFrame:
        """Export execution fills to pandas DataFrame for analysis."""
        if not self.fills:
            return pd.DataFrame()
            
        data = []
        for fill in self.fills:
            data.append({
                "timestamp": fill.timestamp,
                "symbol": fill.symbol,
                "side": fill.side,
                "quantity": fill.quantity,
                "price": fill.price,
                "value": fill.quantity * fill.price,
                "venue": fill.venue,
                "order_type": fill.order_type,
            })
            
        return pd.DataFrame(data)
        
    def calculate_slippage(self, symbol: str, reference_price: float) -> float:
        """Calculate slippage relative to reference price.
        
        :param symbol: Symbol to analyze.
        :param reference_price: Reference price (e.g., mid-price at order time).
        :return: Slippage in basis points.
        """
        symbol_fills = [f for f in self.fills if f.symbol == symbol]
        if not symbol_fills:
            return 0.0
            
        total_value = sum(f.quantity * f.price for f in symbol_fills)
        total_quantity = sum(f.quantity for f in symbol_fills)
        avg_price = total_value / max(total_quantity, 1)
        
        # Calculate slippage (positive means worse execution)
        net_quantity = sum(f.quantity if f.side == "BUY" else -f.quantity for f in symbol_fills)
        is_buying = net_quantity > 0
        
        if is_buying:
            slippage = (avg_price - reference_price) / reference_price
        else:
            slippage = (reference_price - avg_price) / reference_price
            
        return slippage * 10000  # Convert to basis points


class ExecutionBenchmarkCalculator:
    """Calculates execution benchmarks from market data."""
    
    @staticmethod
    def calculate_twap(prices: List[float], start_time: datetime, end_time: datetime) -> float:
        """Calculate Time-Weighted Average Price.
        
        :param prices: List of prices during execution period.
        :param start_time: Start of execution period.
        :param end_time: End of execution period.
        :return: TWAP price.
        """
        if not prices:
            return 0.0
        return np.mean(prices)
        
    @staticmethod
    def calculate_vwap(prices: List[float], volumes: List[float]) -> float:
        """Calculate Volume-Weighted Average Price.
        
        :param prices: List of prices.
        :param volumes: List of corresponding volumes.
        :return: VWAP price.
        """
        if not prices or not volumes or len(prices) != len(volumes):
            return 0.0
            
        total_value = sum(p * v for p, v in zip(prices, volumes))
        total_volume = sum(volumes)
        
        return total_value / max(total_volume, 1)
        
    @staticmethod
    def create_benchmark(
        arrival_price: float,
        prices: List[float],
        volumes: List[float],
        close_price: float,
        start_time: datetime,
        end_time: datetime,
    ) -> ExecutionBenchmark:
        """Create execution benchmark with all reference prices.
        
        :param arrival_price: Price when unwinding started.
        :param prices: Prices during execution period.
        :param volumes: Volumes during execution period.
        :param close_price: Closing price.
        :param start_time: Start of execution.
        :param end_time: End of execution.
        :return: ExecutionBenchmark object.
        """
        twap = ExecutionBenchmarkCalculator.calculate_twap(prices, start_time, end_time)
        vwap = ExecutionBenchmarkCalculator.calculate_vwap(prices, volumes)
        
        return ExecutionBenchmark(
            arrival_price=arrival_price,
            twap_price=twap,
            vwap_price=vwap,
            close_price=close_price,
        ) 
#!/usr/bin/env python
"""Example usage of the position unwinding strategy."""
from __future__ import annotations

import backtrader as bt
from datetime import datetime

from algos.position_unwinding.strategy import PositionUnwindingStrategy
from algos.position_unwinding.execution_analytics import ExecutionAnalytics
from quantdesk.utils.logging import get_logger

log = get_logger(__name__)


def create_sample_data():
    """Create sample data for demonstration."""
    # This would normally come from your data feeds
    # For demo purposes, we'll create a simple data feed
    data = bt.feeds.YahooFinanceData(
        dataname='AAPL',
        fromdate=datetime(2024, 1, 1),
        todate=datetime(2024, 12, 31),
        timeframe=bt.TimeFrame.Minutes,
        compression=1
    )
    return data


def run_twap_unwinding_example():
    """Example of using TWAP unwinding strategy."""
    log.info("example.twap_unwinding.starting")
    
    cerebro = bt.Cerebro()
    
    # Add data
    data = create_sample_data()
    cerebro.adddata(data)
    
    # Configure TWAP unwinding strategy
    cerebro.addstrategy(
        PositionUnwindingStrategy,
        unwind_method="twap",
        target_position=0,  # Full unwind
        time_horizon=240,   # 4 hours
        max_participation_rate=0.15,
        min_participation_rate=0.05,
        stealth_mode=True,
        printlog=True
    )
    
    # Set initial cash and commission
    cerebro.broker.setcash(100000.0)
    cerebro.broker.setcommission(commission=0.001)
    
    # Simulate existing position by manually setting
    # In practice, this would come from your portfolio
    log.info("Simulating large AAPL position that needs unwinding...")
    
    # Run backtest
    results = cerebro.run()
    
    log.info("example.twap_unwinding.completed")
    return results


def run_vwap_unwinding_example():
    """Example of using VWAP unwinding strategy."""
    log.info("example.vwap_unwinding.starting")
    
    cerebro = bt.Cerebro()
    
    # Add data
    data = create_sample_data()
    cerebro.adddata(data)
    
    # Configure VWAP unwinding strategy
    cerebro.addstrategy(
        PositionUnwindingStrategy,
        unwind_method="vwap",
        target_position=0,
        time_horizon=300,   # 5 hours
        max_participation_rate=0.20,
        min_participation_rate=0.08,
        liquidity_buffer=0.25,
        stealth_mode=True,
        printlog=True
    )
    
    cerebro.broker.setcash(100000.0)
    cerebro.broker.setcommission(commission=0.001)
    
    results = cerebro.run()
    
    log.info("example.vwap_unwinding.completed")
    return results


def run_adaptive_unwinding_example():
    """Example of using adaptive unwinding strategy."""
    log.info("example.adaptive_unwinding.starting")
    
    cerebro = bt.Cerebro()
    
    # Add data
    data = create_sample_data()
    cerebro.adddata(data)
    
    # Configure adaptive unwinding strategy
    cerebro.addstrategy(
        PositionUnwindingStrategy,
        unwind_method="adaptive",
        target_position=0,
        time_horizon=180,   # 3 hours
        max_participation_rate=0.25,
        min_participation_rate=0.03,
        volatility_threshold=0.015,
        risk_factor=0.6,
        stealth_mode=True,
        dark_pool_preference=0.8,
        printlog=True
    )
    
    cerebro.broker.setcash(100000.0)
    cerebro.broker.setcommission(commission=0.001)
    
    results = cerebro.run()
    
    log.info("example.adaptive_unwinding.completed")
    return results


def run_iceberg_unwinding_example():
    """Example of using iceberg order unwinding."""
    log.info("example.iceberg_unwinding.starting")
    
    cerebro = bt.Cerebro()
    
    # Add data
    data = create_sample_data()
    cerebro.adddata(data)
    
    # Configure iceberg unwinding strategy
    cerebro.addstrategy(
        PositionUnwindingStrategy,
        unwind_method="iceberg",
        target_position=0,
        time_horizon=360,   # 6 hours
        iceberg_show_size=0.08,  # Show only 8% of order
        max_participation_rate=0.18,
        stealth_mode=True,
        printlog=True
    )
    
    cerebro.broker.setcash(100000.0)
    cerebro.broker.setcommission(commission=0.001)
    
    results = cerebro.run()
    
    log.info("example.iceberg_unwinding.completed")
    return results


def run_conservative_unwinding_example():
    """Example of conservative unwinding profile."""
    log.info("example.conservative_unwinding.starting")
    
    cerebro = bt.Cerebro()
    
    # Add data
    data = create_sample_data()
    cerebro.adddata(data)
    
    # Configure conservative unwinding (based on config.yaml profile)
    cerebro.addstrategy(
        PositionUnwindingStrategy,
        unwind_method="adaptive",
        target_position=0,
        time_horizon=480,   # 8 hours
        max_participation_rate=0.10,
        min_participation_rate=0.03,
        volatility_threshold=0.015,
        risk_factor=0.8,
        stealth_mode=True,
        dark_pool_preference=0.9,
        printlog=True
    )
    
    cerebro.broker.setcash(100000.0)
    cerebro.broker.setcommission(commission=0.001)
    
    results = cerebro.run()
    
    log.info("example.conservative_unwinding.completed")
    return results


def analyze_execution_performance():
    """Example of execution performance analysis."""
    log.info("example.execution_analysis.starting")
    
    # Create analytics instance
    analytics = ExecutionAnalytics()
    
    # Simulate some execution fills (in practice, these would come from your strategy)
    from algos.position_unwinding.execution_analytics import ExecutionFill, ExecutionBenchmark
    
    # Sample fills for AAPL unwinding
    fills = [
        ExecutionFill(
            timestamp=datetime(2024, 1, 15, 9, 35),
            symbol="AAPL",
            side="SELL",
            quantity=500,
            price=185.25,
            venue="dark_pool"
        ),
        ExecutionFill(
            timestamp=datetime(2024, 1, 15, 9, 45),
            symbol="AAPL",
            side="SELL",
            quantity=750,
            price=185.10,
            venue="primary"
        ),
        ExecutionFill(
            timestamp=datetime(2024, 1, 15, 10, 15),
            symbol="AAPL",
            side="SELL",
            quantity=1000,
            price=184.95,
            venue="ecn"
        ),
    ]
    
    # Add fills to analytics
    for fill in fills:
        analytics.add_fill(fill)
    
    # Set benchmark prices
    benchmark = ExecutionBenchmark(
        arrival_price=185.50,  # Price when unwinding started
        twap_price=185.15,     # TWAP during execution period
        vwap_price=185.08,     # VWAP during execution period
        close_price=184.80     # Close price
    )
    analytics.set_benchmark("AAPL", benchmark)
    
    # Generate execution report
    report = analytics.generate_execution_report("AAPL")
    
    log.info("execution.report", report=report)
    
    # Calculate specific metrics
    is_metrics = analytics.calculate_implementation_shortfall("AAPL")
    log.info("implementation_shortfall", metrics=is_metrics)
    
    venue_breakdown = analytics.calculate_venue_breakdown("AAPL")
    log.info("venue_breakdown", breakdown=venue_breakdown)
    
    # Export to DataFrame for further analysis
    df = analytics.export_to_dataframe()
    log.info("execution_data_exported", shape=df.shape)
    
    log.info("example.execution_analysis.completed")


def main():
    """Run all unwinding strategy examples."""
    log.info("position_unwinding_examples.starting")
    
    # Run different unwinding method examples
    examples = [
        ("TWAP Unwinding", run_twap_unwinding_example),
        ("VWAP Unwinding", run_vwap_unwinding_example),
        ("Adaptive Unwinding", run_adaptive_unwinding_example),
        ("Iceberg Unwinding", run_iceberg_unwinding_example),
        ("Conservative Unwinding", run_conservative_unwinding_example),
    ]
    
    for name, example_func in examples:
        try:
            log.info(f"Running {name} example...")
            example_func()
            log.info(f"{name} example completed successfully")
        except Exception as e:
            log.error(f"Error in {name} example", error=str(e))
    
    # Run execution analysis example
    try:
        log.info("Running execution analysis example...")
        analyze_execution_performance()
        log.info("Execution analysis example completed successfully")
    except Exception as e:
        log.error("Error in execution analysis example", error=str(e))
    
    log.info("position_unwinding_examples.completed")


if __name__ == "__main__":
    main() 
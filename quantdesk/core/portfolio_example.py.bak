#!/usr/bin/env python
"""Example usage of the comprehensive portfolio analyzer."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict

from quantdesk.core.portfolio import Portfolio, Position
from quantdesk.core.portfolio_analyzer import PortfolioAnalyzer
from quantdesk.core.event_engine import FillEvent
from quantdesk.utils.logging import get_logger

log = get_logger(__name__)


def create_sample_portfolio() -> Portfolio:
    """Create a sample portfolio with some positions for demonstration."""
    portfolio = Portfolio(cash=50000.0)
    
    # Simulate some filled orders by adding positions directly
    portfolio.positions["AAPL"] = Position()
    portfolio.positions["AAPL"].qty = 100
    portfolio.positions["AAPL"].avg_price = 180.50
    
    portfolio.positions["MSFT"] = Position()
    portfolio.positions["MSFT"].qty = 150
    portfolio.positions["MSFT"].avg_price = 420.75
    
    portfolio.positions["GOOGL"] = Position()
    portfolio.positions["GOOGL"].qty = 50
    portfolio.positions["GOOGL"].avg_price = 2850.25
    
    portfolio.positions["TSLA"] = Position()
    portfolio.positions["TSLA"].qty = 75
    portfolio.positions["TSLA"].avg_price = 245.80
    
    portfolio.positions["NVDA"] = Position()
    portfolio.positions["NVDA"].qty = 120
    portfolio.positions["NVDA"].avg_price = 875.30
    
    portfolio.positions["SPY"] = Position()
    portfolio.positions["SPY"].qty = 200
    portfolio.positions["SPY"].avg_price = 485.60
    
    portfolio.positions["QQQ"] = Position()
    portfolio.positions["QQQ"].qty = 100
    portfolio.positions["QQQ"].avg_price = 395.45
    
    # Add a short position
    portfolio.positions["VIX"] = Position()
    portfolio.positions["VIX"].qty = -50
    portfolio.positions["VIX"].avg_price = 18.75
    
    log.info("sample.portfolio.created", positions=len(portfolio.positions))
    return portfolio


def get_sample_market_data() -> Dict[str, float]:
    """Get sample current market prices."""
    return {
        "AAPL": 185.25,    # Up from avg cost
        "MSFT": 425.80,    # Up from avg cost  
        "GOOGL": 2795.60,  # Down from avg cost
        "TSLA": 252.15,    # Up from avg cost
        "NVDA": 890.75,    # Up from avg cost
        "SPY": 488.90,     # Up from avg cost
        "QQQ": 392.30,     # Down from avg cost
        "VIX": 16.45,      # Down from avg cost (good for short)
    }


def get_sample_sector_mapping() -> Dict[str, str]:
    """Get sample sector mapping for positions."""
    return {
        "AAPL": "Technology",
        "MSFT": "Technology", 
        "GOOGL": "Technology",
        "TSLA": "Consumer Discretionary",
        "NVDA": "Technology",
        "SPY": "Broad Market ETF",
        "QQQ": "Technology ETF",
        "VIX": "Volatility Index",
    }


def get_sample_beta_values() -> Dict[str, float]:
    """Get sample beta values for positions."""
    return {
        "AAPL": 1.15,
        "MSFT": 0.95,
        "GOOGL": 1.25,
        "TSLA": 1.85,
        "NVDA": 1.65,
        "SPY": 1.00,
        "QQQ": 1.10,
        "VIX": -0.25,  # Negative correlation
    }


def demonstrate_basic_analysis():
    """Demonstrate basic portfolio analysis functionality."""
    log.info("demo.basic_analysis.starting")
    
    # Create sample portfolio
    portfolio = create_sample_portfolio()
    
    # Create analyzer
    analyzer = PortfolioAnalyzer(portfolio, benchmark_symbol="SPY")
    
    # Set sector and beta mappings
    analyzer.set_sector_mapping(get_sample_sector_mapping())
    analyzer.set_beta_values(get_sample_beta_values())
    
    # Update with current market data
    market_data = get_sample_market_data()
    analyzer.update_market_data(market_data)
    
    # Get performance summary
    summary = analyzer.get_performance_summary()
    log.info("performance.summary", summary=summary)
    
    # Analyze individual positions
    positions = analyzer.analyze_positions()
    log.info("positions.analyzed", count=len(positions))
    
    for pos in positions[:3]:  # Show top 3 positions
        log.info(
            "position.detail",
            symbol=pos.symbol,
            quantity=pos.quantity,
            market_value=pos.market_value,
            unrealized_pnl=pos.unrealized_pnl,
            weight_pct=pos.weight * 100,
            sector=pos.sector,
        )
    
    # Get sector allocation
    sector_allocation = analyzer.get_sector_allocation()
    log.info("sector.allocation", allocation=sector_allocation)
    
    # Calculate risk metrics
    risk_metrics = analyzer.calculate_risk_metrics()
    log.info("risk.metrics", 
             volatility=risk_metrics.portfolio_volatility,
             sharpe=risk_metrics.sharpe_ratio,
             max_dd=risk_metrics.max_drawdown)
    
    log.info("demo.basic_analysis.completed")
    return analyzer


def demonstrate_time_series_analysis():
    """Demonstrate time series analysis with multiple snapshots."""
    log.info("demo.time_series.starting")
    
    portfolio = create_sample_portfolio()
    analyzer = PortfolioAnalyzer(portfolio)
    analyzer.set_sector_mapping(get_sample_sector_mapping())
    
    # Simulate daily updates for a week
    base_prices = get_sample_market_data()
    
    for day in range(7):
        # Simulate price movements
        daily_prices = {}
        for symbol, base_price in base_prices.items():
            # Random price movement ±2%
            import random
            change = random.uniform(-0.02, 0.02)
            daily_prices[symbol] = base_price * (1 + change)
            
        # Create timestamp for each day
        timestamp = datetime.now() - timedelta(days=6-day)
        
        # Update analyzer
        analyzer.update_market_data(daily_prices, timestamp)
        
        log.info(f"day.{day+1}.updated", 
                 portfolio_value=analyzer.snapshots[-1].total_value,
                 timestamp=timestamp.date())
    
    # Generate performance report
    performance_df = analyzer.generate_performance_report()
    log.info("performance.report.generated", shape=performance_df.shape)
    
    # Show some performance metrics
    if len(analyzer.returns_history) > 0:
        total_return = analyzer._calculate_inception_return()
        log.info("week.performance", 
                 total_return_pct=total_return * 100,
                 daily_returns=len(analyzer.returns_history))
    
    log.info("demo.time_series.completed")
    return analyzer


def demonstrate_reporting():
    """Demonstrate comprehensive reporting capabilities."""
    log.info("demo.reporting.starting")
    
    # Use analyzer from time series demo
    analyzer = demonstrate_time_series_analysis()
    
    # Generate positions report
    positions_df = analyzer.generate_positions_report()
    log.info("positions.report.generated", shape=positions_df.shape)
    
    # Show positions report
    if not positions_df.empty:
        log.info("positions.sample")
        for _, row in positions_df.head(3).iterrows():
            log.info("position.row",
                     symbol=row['Symbol'],
                     quantity=row['Quantity'],
                     market_value=row['Market Value'],
                     unrealized_pnl_pct=row['Unrealized P&L %'])
    
    # Generate performance report
    performance_df = analyzer.generate_performance_report()
    log.info("performance.report.generated", shape=performance_df.shape)
    
    # Export to Excel (optional - requires openpyxl)
    try:
        filename = f"portfolio_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        analyzer.export_to_excel(filename)
        log.info("excel.export.completed", filename=filename)
    except Exception as e:
        log.warning("excel.export.failed", error=str(e))
    
    log.info("demo.reporting.completed")
    return analyzer


def demonstrate_risk_analysis():
    """Demonstrate risk analysis capabilities."""
    log.info("demo.risk_analysis.starting")
    
    portfolio = create_sample_portfolio()
    analyzer = PortfolioAnalyzer(portfolio)
    analyzer.set_sector_mapping(get_sample_sector_mapping())
    analyzer.set_beta_values(get_sample_beta_values())
    
    # Simulate more data for better risk metrics
    base_prices = get_sample_market_data()
    
    for day in range(30):  # 30 days of data
        daily_prices = {}
        for symbol, base_price in base_prices.items():
            # More realistic price simulation
            import random
            import math
            
            # Use different volatilities by asset
            if symbol in ["TSLA", "NVDA"]:
                vol = 0.035  # Higher vol for growth stocks
            elif symbol == "VIX":
                vol = 0.08   # Very high vol for VIX
            else:
                vol = 0.02   # Lower vol for others
                
            change = random.normalvariate(0, vol)
            daily_prices[symbol] = base_price * (1 + change)
            
        timestamp = datetime.now() - timedelta(days=29-day)
        analyzer.update_market_data(daily_prices, timestamp)
    
    # Calculate comprehensive risk metrics
    risk_metrics = analyzer.calculate_risk_metrics()
    
    log.info("risk.analysis.results",
             portfolio_vol=risk_metrics.portfolio_volatility,
             sharpe_ratio=risk_metrics.sharpe_ratio,
             sortino_ratio=risk_metrics.sortino_ratio,
             max_drawdown=risk_metrics.max_drawdown,
             var_95=risk_metrics.value_at_risk_95,
             var_99=risk_metrics.value_at_risk_99,
             cvar_95=risk_metrics.conditional_var_95)
    
    # Analyze portfolio composition
    current_snapshot = analyzer.snapshots[-1]
    log.info("portfolio.composition",
             leverage=current_snapshot.leverage,
             net_exposure=current_snapshot.net_exposure,
             gross_exposure=current_snapshot.gross_exposure,
             positions_count=current_snapshot.positions_count)
    
    # Get correlation matrix (placeholder)
    corr_matrix = analyzer.get_correlation_matrix()
    if not corr_matrix.empty:
        log.info("correlation.matrix.generated", shape=corr_matrix.shape)
    
    log.info("demo.risk_analysis.completed")
    return analyzer


def demonstrate_live_monitoring():
    """Demonstrate live portfolio monitoring simulation."""
    log.info("demo.live_monitoring.starting")
    
    portfolio = create_sample_portfolio()
    analyzer = PortfolioAnalyzer(portfolio)
    analyzer.set_sector_mapping(get_sample_sector_mapping())
    analyzer.set_beta_values(get_sample_beta_values())
    
    base_prices = get_sample_market_data()
    
    # Simulate intraday monitoring
    for minute in range(0, 60, 5):  # Every 5 minutes for 1 hour
        # Simulate small price movements
        current_prices = {}
        for symbol, base_price in base_prices.items():
            import random
            # Small intraday movements
            change = random.uniform(-0.005, 0.005)  # ±0.5%
            current_prices[symbol] = base_price * (1 + change)
        
        timestamp = datetime.now() + timedelta(minutes=minute)
        analyzer.update_market_data(current_prices, timestamp)
        
        # Get current performance
        summary = analyzer.get_performance_summary()
        
        log.info(f"minute.{minute}.update",
                 timestamp=timestamp.strftime("%H:%M"),
                 portfolio_value=summary['current_value'],
                 current_drawdown=analyzer.current_drawdown * 100,
                 positions_count=summary['positions_count'])
        
        # Check for any risk alerts (example)
        if analyzer.current_drawdown > 0.02:  # 2% drawdown alert
            log.warning("risk.alert.drawdown", 
                       drawdown_pct=analyzer.current_drawdown * 100)
    
    # Final snapshot
    final_summary = analyzer.get_performance_summary()
    log.info("session.summary",
             final_value=final_summary['current_value'],
             max_drawdown=analyzer.max_drawdown_ever * 100,
             total_snapshots=len(analyzer.snapshots))
    
    log.info("demo.live_monitoring.completed")
    return analyzer


def main():
    """Run all portfolio analyzer demonstrations."""
    log.info("portfolio.analyzer.demos.starting")
    
    demos = [
        ("Basic Analysis", demonstrate_basic_analysis),
        ("Time Series Analysis", demonstrate_time_series_analysis),
        ("Comprehensive Reporting", demonstrate_reporting),
        ("Risk Analysis", demonstrate_risk_analysis),
        ("Live Monitoring", demonstrate_live_monitoring),
    ]
    
    for name, demo_func in demos:
        try:
            log.info(f"Running {name} demo...")
            analyzer = demo_func()
            log.info(f"{name} demo completed successfully")
            
            # Show final stats
            if analyzer.snapshots:
                final_value = analyzer.snapshots[-1].total_value
                log.info(f"{name}.final_stats", portfolio_value=final_value)
                
        except Exception as e:
            log.error(f"Error in {name} demo", error=str(e))
    
    log.info("portfolio.analyzer.demos.completed")


if __name__ == "__main__":
    main() 
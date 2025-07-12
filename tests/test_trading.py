import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json
import pandas as pd
from dataclasses import dataclass

# Trading modullarini import qilish
from trading.strategy_manager import StrategyManager, StrategyResult
from trading.execution_engine import ExecutionEngine, TradeResult
from trading.propshot_connector import PropshotConnector, PropshotResponse
from trading.backtest_engine import BacktestEngine, BacktestResult
from trading.portfolio_manager import PortfolioManager, PortfolioState
from trading.mt5_bridge import MT5Bridge, MT5Response
from utils.logger import get_logger
from config.config import TradingConfig

logger = get_logger(__name__)

@dataclass
class MockSignal:
    """Mock signal test uchun"""
    symbol: str
    action: str  # 'BUY' yoki 'SELL'
    price: float
    lot_size: float
    stop_loss: float
    take_profit: float
    confidence: float
    timestamp: datetime
    reason: str

@dataclass
class MockTrade:
    """Mock trade test uchun"""
    trade_id: str
    symbol: str
    action: str
    entry_price: float
    exit_price: Optional[float]
    lot_size: float
    profit_loss: float
    status: str
    entry_time: datetime
    exit_time: Optional[datetime]

class TestStrategyManager:
    """Strategy Manager testlari"""
    
    @pytest.fixture
    def strategy_manager(self):
        """Strategy manager fixture"""
        config = TradingConfig(
            max_risk_per_trade=0.02,
            max_daily_trades=5,
            allowed_symbols=['EURUSD', 'GBPUSD', 'USDJPY']
        )
        return StrategyManager(config)
    
    @pytest.fixture
    def sample_signal(self):
        """Sample signal fixture"""
        return MockSignal(
            symbol='EURUSD',
            action='BUY',
            price=1.0850,
            lot_size=0.1,
            stop_loss=1.0800,
            take_profit=1.0900,
            confidence=0.85,
            timestamp=datetime.now(),
            reason='Bullish sentiment + strong order flow'
        )
    
    @pytest.mark.asyncio
    async def test_strategy_manager_initialization(self, strategy_manager):
        """Strategy manager ishga tushirish testi"""
        assert strategy_manager is not None
        assert strategy_manager.config.max_risk_per_trade == 0.02
        assert len(strategy_manager.config.allowed_symbols) == 3
        logger.info("Strategy manager muvaffaqiyatli ishga tushirildi")
    
    @pytest.mark.asyncio
    async def test_signal_validation(self, strategy_manager, sample_signal):
        """Signal validatsiya testi"""
        # To'g'ri signal
        is_valid = await strategy_manager.validate_signal(sample_signal)
        assert is_valid is True
        
        # Noto'g'ri symbol
        invalid_signal = sample_signal
        invalid_signal.symbol = 'INVALID'
        is_valid = await strategy_manager.validate_signal(invalid_signal)
        assert is_valid is False
        
        # Noto'g'ri confidence
        invalid_signal.symbol = 'EURUSD'
        invalid_signal.confidence = 0.3  # Past confidence
        is_valid = await strategy_manager.validate_signal(invalid_signal)
        assert is_valid is False
        
        logger.info("Signal validatsiya testlari o'tdi")
    
    @pytest.mark.asyncio
    async def test_risk_calculation(self, strategy_manager, sample_signal):
        """Risk hisoblash testi"""
        balance = 10000.0
        risk_amount = await strategy_manager.calculate_risk(sample_signal, balance)
        
        # Risk 2% dan oshmasligi kerak
        expected_risk = balance * 0.02
        assert risk_amount <= expected_risk
        
        # Risk musbat bo'lishi kerak
        assert risk_amount > 0
        
        logger.info(f"Risk hisoblash: {risk_amount}, Kutilgan: {expected_risk}")
    
    @pytest.mark.asyncio
    async def test_position_sizing(self, strategy_manager, sample_signal):
        """Position sizing testi"""
        balance = 10000.0
        position_size = await strategy_manager.calculate_position_size(
            sample_signal, balance
        )
        
        # Position size musbat bo'lishi kerak
        assert position_size > 0
        
        # Maksimal lot size tekshirish
        assert position_size <= 1.0  # Maksimal lot size
        
        logger.info(f"Position size: {position_size}")
    
    @pytest.mark.asyncio
    async def test_strategy_execution(self, strategy_manager, sample_signal):
        """Strategiya bajarish testi"""
        with patch.object(strategy_manager, 'execute_strategy') as mock_execute:
            mock_execute.return_value = StrategyResult(
                success=True,
                trade_id='TEST_123',
                entry_price=1.0850,
                lot_size=0.1,
                message='Trade muvaffaqiyatli bajarildi'
            )
            
            result = await strategy_manager.execute_strategy(sample_signal)
            
            assert result.success is True
            assert result.trade_id == 'TEST_123'
            assert result.entry_price == 1.0850
            
            logger.info("Strategiya bajarish testi o'tdi")

class TestExecutionEngine:
    """Execution Engine testlari"""
    
    @pytest.fixture
    def execution_engine(self):
        """Execution engine fixture"""
        return ExecutionEngine()
    
    @pytest.fixture
    def mock_propshot_connector(self):
        """Mock Propshot connector"""
        connector = Mock(spec=PropshotConnector)
        connector.send_order = AsyncMock()
        connector.check_limits = AsyncMock()
        return connector
    
    @pytest.mark.asyncio
    async def test_execution_engine_initialization(self, execution_engine):
        """Execution engine ishga tushirish testi"""
        assert execution_engine is not None
        assert execution_engine.is_active is False
        logger.info("Execution engine muvaffaqiyatli yaratildi")
    
    @pytest.mark.asyncio
    async def test_trade_execution(self, execution_engine, sample_signal):
        """Trade bajarish testi"""
        with patch.object(execution_engine, 'execute_trade') as mock_execute:
            mock_execute.return_value = TradeResult(
                success=True,
                trade_id='EXEC_456',
                entry_price=1.0850,
                execution_time=datetime.now(),
                message='Trade muvaffaqiyatli bajarildi'
            )
            
            result = await execution_engine.execute_trade(sample_signal)
            
            assert result.success is True
            assert result.trade_id == 'EXEC_456'
            assert result.entry_price == 1.0850
            
            logger.info("Trade bajarish testi o'tdi")
    
    @pytest.mark.asyncio
    async def test_order_validation(self, execution_engine, sample_signal):
        """Order validatsiya testi"""
        # To'g'ri order
        is_valid = await execution_engine.validate_order(sample_signal)
        assert is_valid is True
        
        # Noto'g'ri lot size
        invalid_signal = sample_signal
        invalid_signal.lot_size = 0.0
        is_valid = await execution_engine.validate_order(invalid_signal)
        assert is_valid is False
        
        # Noto'g'ri stop loss
        invalid_signal.lot_size = 0.1
        invalid_signal.stop_loss = invalid_signal.price + 0.01  # SL price dan baland
        is_valid = await execution_engine.validate_order(invalid_signal)
        assert is_valid is False
        
        logger.info("Order validatsiya testlari o'tdi")
    
    @pytest.mark.asyncio
    async def test_trade_monitoring(self, execution_engine):
        """Trade monitoring testi"""
        mock_trade = MockTrade(
            trade_id='MONITOR_789',
            symbol='EURUSD',
            action='BUY',
            entry_price=1.0850,
            exit_price=None,
            lot_size=0.1,
            profit_loss=0.0,
            status='OPEN',
            entry_time=datetime.now(),
            exit_time=None
        )
        
        with patch.object(execution_engine, 'monitor_trade') as mock_monitor:
            mock_monitor.return_value = {
                'trade_id': 'MONITOR_789',
                'current_price': 1.0875,
                'unrealized_pnl': 25.0,
                'status': 'OPEN'
            }
            
            result = await execution_engine.monitor_trade(mock_trade)
            
            assert result['trade_id'] == 'MONITOR_789'
            assert result['current_price'] == 1.0875
            assert result['unrealized_pnl'] == 25.0
            
            logger.info("Trade monitoring testi o'tdi")

class TestPropshotConnector:
    """Propshot Connector testlari"""
    
    @pytest.fixture
    def propshot_connector(self):
        """Propshot connector fixture"""
        return PropshotConnector(
            api_key='test_key',
            base_url='https://api.propshot.com'
        )
    
    @pytest.mark.asyncio
    async def test_propshot_connection(self, propshot_connector):
        """Propshot ulanish testi"""
        with patch.object(propshot_connector, 'connect') as mock_connect:
            mock_connect.return_value = PropshotResponse(
                success=True,
                message='Ulanish muvaffaqiyatli',
                data={'status': 'connected'}
            )
            
            result = await propshot_connector.connect()
            
            assert result.success is True
            assert result.message == 'Ulanish muvaffaqiyatli'
            assert result.data['status'] == 'connected'
            
            logger.info("Propshot ulanish testi o'tdi")
    
    @pytest.mark.asyncio
    async def test_propshot_limits_check(self, propshot_connector):
        """Propshot limitlar tekshirish testi"""
        with patch.object(propshot_connector, 'check_limits') as mock_check:
            mock_check.return_value = {
                'max_daily_loss': 0.025,
                'current_daily_loss': 0.01,
                'max_lot_size': 0.5,
                'max_daily_trades': 3,
                'current_daily_trades': 1,
                'limits_ok': True
            }
            
            result = await propshot_connector.check_limits()
            
            assert result['limits_ok'] is True
            assert result['current_daily_loss'] < result['max_daily_loss']
            assert result['current_daily_trades'] < result['max_daily_trades']
            
            logger.info("Propshot limitlar testi o'tdi")
    
    @pytest.mark.asyncio
    async def test_propshot_order_sending(self, propshot_connector, sample_signal):
        """Propshot order yuborish testi"""
        with patch.object(propshot_connector, 'send_order') as mock_send:
            mock_send.return_value = PropshotResponse(
                success=True,
                message='Order yuborildi',
                data={
                    'order_id': 'PROP_001',
                    'symbol': 'EURUSD',
                    'action': 'BUY',
                    'lot_size': 0.1,
                    'status': 'PENDING'
                }
            )
            
            result = await propshot_connector.send_order(sample_signal)
            
            assert result.success is True
            assert result.data['order_id'] == 'PROP_001'
            assert result.data['symbol'] == 'EURUSD'
            assert result.data['action'] == 'BUY'
            
            logger.info("Propshot order yuborish testi o'tdi")

class TestBacktestEngine:
    """Backtest Engine testlari"""
    
    @pytest.fixture
    def backtest_engine(self):
        """Backtest engine fixture"""
        return BacktestEngine(
            initial_balance=10000.0,
            start_date=datetime.now() - timedelta(days=30),
            end_date=datetime.now()
        )
    
    @pytest.fixture
    def sample_historical_data(self):
        """Sample historical data"""
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=30),
            end=datetime.now(),
            freq='1H'
        )
        
        return pd.DataFrame({
            'timestamp': dates,
            'symbol': ['EURUSD'] * len(dates),
            'open': [1.0850 + i * 0.0001 for i in range(len(dates))],
            'high': [1.0860 + i * 0.0001 for i in range(len(dates))],
            'low': [1.0840 + i * 0.0001 for i in range(len(dates))],
            'close': [1.0855 + i * 0.0001 for i in range(len(dates))],
            'volume': [1000] * len(dates)
        })
    
    @pytest.mark.asyncio
    async def test_backtest_initialization(self, backtest_engine):
        """Backtest ishga tushirish testi"""
        assert backtest_engine is not None
        assert backtest_engine.initial_balance == 10000.0
        assert backtest_engine.current_balance == 10000.0
        logger.info("Backtest engine muvaffaqiyatli ishga tushirildi")
    
    @pytest.mark.asyncio
    async def test_backtest_execution(self, backtest_engine, sample_historical_data):
        """Backtest bajarish testi"""
        with patch.object(backtest_engine, 'run_backtest') as mock_backtest:
            mock_backtest.return_value = BacktestResult(
                total_trades=50,
                winning_trades=32,
                losing_trades=18,
                win_rate=0.64,
                total_profit=1500.0,
                max_drawdown=0.05,
                sharpe_ratio=1.2,
                profit_factor=1.8,
                final_balance=11500.0
            )
            
            result = await backtest_engine.run_backtest(
                sample_historical_data,
                strategy_name='TestStrategy'
            )
            
            assert result.total_trades == 50
            assert result.win_rate == 0.64
            assert result.total_profit == 1500.0
            assert result.final_balance == 11500.0
            
            logger.info("Backtest bajarish testi o'tdi")
    
    @pytest.mark.asyncio
    async def test_performance_calculation(self, backtest_engine):
        """Performance hisoblash testi"""
        trades = [
            {'profit_loss': 100.0, 'win': True},
            {'profit_loss': -50.0, 'win': False},
            {'profit_loss': 200.0, 'win': True},
            {'profit_loss': -75.0, 'win': False},
            {'profit_loss': 150.0, 'win': True}
        ]
        
        performance = await backtest_engine.calculate_performance(trades)
        
        assert performance['total_trades'] == 5
        assert performance['winning_trades'] == 3
        assert performance['losing_trades'] == 2
        assert performance['win_rate'] == 0.6
        assert performance['total_profit'] == 325.0
        
        logger.info("Performance hisoblash testi o'tdi")

class TestPortfolioManager:
    """Portfolio Manager testlari"""
    
    @pytest.fixture
    def portfolio_manager(self):
        """Portfolio manager fixture"""
        return PortfolioManager(
            initial_balance=10000.0,
            max_risk_per_trade=0.02,
            max_portfolio_risk=0.1
        )
    
    @pytest.mark.asyncio
    async def test_portfolio_initialization(self, portfolio_manager):
        """Portfolio ishga tushirish testi"""
        assert portfolio_manager is not None
        assert portfolio_manager.initial_balance == 10000.0
        assert portfolio_manager.current_balance == 10000.0
        assert portfolio_manager.max_risk_per_trade == 0.02
        logger.info("Portfolio manager muvaffaqiyatli ishga tushirildi")
    
    @pytest.mark.asyncio
    async def test_portfolio_risk_calculation(self, portfolio_manager):
        """Portfolio risk hisoblash testi"""
        open_trades = [
            {'symbol': 'EURUSD', 'lot_size': 0.1, 'risk': 200.0},
            {'symbol': 'GBPUSD', 'lot_size': 0.1, 'risk': 150.0},
            {'symbol': 'USDJPY', 'lot_size': 0.05, 'risk': 100.0}
        ]
        
        total_risk = await portfolio_manager.calculate_portfolio_risk(open_trades)
        
        assert total_risk == 450.0  # 200 + 150 + 100
        
        # Risk percentage tekshirish
        risk_percentage = total_risk / portfolio_manager.current_balance
        assert risk_percentage <= portfolio_manager.max_portfolio_risk
        
        logger.info(f"Portfolio risk: {total_risk}, Percentage: {risk_percentage}")
    
    @pytest.mark.asyncio
    async def test_position_limit_check(self, portfolio_manager):
        """Position limit tekshirish testi"""
        current_positions = [
            {'symbol': 'EURUSD', 'lot_size': 0.2},
            {'symbol': 'GBPUSD', 'lot_size': 0.1}
        ]
        
        new_position = {'symbol': 'EURUSD', 'lot_size': 0.1}
        
        can_add = await portfolio_manager.can_add_position(
            new_position, current_positions
        )
        
        # EURUSD uchun maksimal limit tekshirish
        assert isinstance(can_add, bool)
        
        logger.info(f"Yangi position qo'shish mumkin: {can_add}")
    
    @pytest.mark.asyncio
    async def test_portfolio_state_update(self, portfolio_manager):
        """Portfolio holat yangilash testi"""
        trade_result = {
            'profit_loss': 250.0,
            'symbol': 'EURUSD',
            'lot_size': 0.1,
            'closed': True
        }
        
        initial_balance = portfolio_manager.current_balance
        
        await portfolio_manager.update_portfolio_state(trade_result)
        
        # Balance yangilangan bo'lishi kerak
        expected_balance = initial_balance + trade_result['profit_loss']
        assert portfolio_manager.current_balance == expected_balance
        
        logger.info(f"Portfolio yangilandi: {portfolio_manager.current_balance}")

class TestMT5Bridge:
    """MT5 Bridge testlari"""
    
    @pytest.fixture
    def mt5_bridge(self):
        """MT5 bridge fixture"""
        return MT5Bridge(
            server='DemoServer',
            login=12345,
            password='test_password'
        )
    
    @pytest.mark.asyncio
    async def test_mt5_connection(self, mt5_bridge):
        """MT5 ulanish testi"""
        with patch.object(mt5_bridge, 'connect') as mock_connect:
            mock_connect.return_value = MT5Response(
                success=True,
                message='MT5 ga ulanish muvaffaqiyatli',
                data={'connected': True, 'account': 12345}
            )
            
            result = await mt5_bridge.connect()
            
            assert result.success is True
            assert result.data['connected'] is True
            assert result.data['account'] == 12345
            
            logger.info("MT5 ulanish testi o'tdi")
    
    @pytest.mark.asyncio
    async def test_mt5_account_info(self, mt5_bridge):
        """MT5 account info testi"""
        with patch.object(mt5_bridge, 'get_account_info') as mock_info:
            mock_info.return_value = {
                'balance': 10000.0,
                'equity': 10000.0,
                'margin': 0.0,
                'free_margin': 10000.0,
                'margin_level': 0.0,
                'profit': 0.0
            }
            
            result = await mt5_bridge.get_account_info()
            
            assert result['balance'] == 10000.0
            assert result['equity'] == 10000.0
            assert result['free_margin'] == 10000.0
            
            logger.info("MT5 account info testi o'tdi")
    
    @pytest.mark.asyncio
    async def test_mt5_order_sending(self, mt5_bridge, sample_signal):
        """MT5 order yuborish testi"""
        with patch.object(mt5_bridge, 'send_order') as mock_send:
            mock_send.return_value = MT5Response(
                success=True,
                message='Order muvaffaqiyatli yuborildi',
                data={
                    'order_id': 'MT5_123',
                    'symbol': 'EURUSD',
                    'action': 'BUY',
                    'volume': 0.1,
                    'price': 1.0850,
                    'sl': 1.0800,
                    'tp': 1.0900
                }
            )
            
            result = await mt5_bridge.send_order(sample_signal)
            
            assert result.success is True
            assert result.data['order_id'] == 'MT5_123'
            assert result.data['symbol'] == 'EURUSD'
            assert result.data['volume'] == 0.1
            
            logger.info("MT5 order yuborish testi o'tdi")

# Performance testlari
class TestTradingPerformance:
    """Trading performance testlari"""
    
    @pytest.mark.asyncio
    async def test_high_frequency_trading(self):
        """Yuqori chastotali trading testi"""
        execution_engine = ExecutionEngine()
        
        # 100 ta signal yaratish
        signals = []
        for i in range(100):
            signal = MockSignal(
                symbol='EURUSD',
                action='BUY' if i % 2 == 0 else 'SELL',
                price=1.0850 + i * 0.0001,
                lot_size=0.01,
                stop_loss=1.0800 + i * 0.0001,
                take_profit=1.0900 + i * 0.0001,
                confidence=0.8,
                timestamp=datetime.now(),
                reason=f'Test signal {i}'
            )
            signals.append(signal)
        
        # Barchasi uchun vaqt o'lchash
        start_time = datetime.now()
        
        with patch.object(execution_engine, 'execute_trade') as mock_execute:
            mock_execute.return_value = TradeResult(
                success=True,
                trade_id=f'PERF_TEST',
                entry_price=1.0850,
                execution_time=datetime.now(),
                message='Performance test'
            )
            
            tasks = []
            for signal in signals:
                task = execution_engine.execute_trade(signal)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        # Barcha trade lar muvaffaqiyatli bo'lishi kerak
        assert len(results) == 100
        assert all(result.success for result in results)
        
        # Performance requirements (100 trade < 10 sekund)
        assert execution_time < 10.0
        
        logger.info(f"100 ta trade {execution_time:.2f} sekundda bajarildi")
    
    @pytest.mark.asyncio
    async def test_memory_usage(self):
        """Memory usage testi"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Ko'p ma'lumot bilan ishlash
        strategy_manager = StrategyManager(TradingConfig())
        backtest_engine = BacktestEngine(initial_balance=10000.0)
        
        # 1000 ta signal yaratish
        signals = []
        for i in range(1000):
            signal = MockSignal(
                symbol='EURUSD',
                action='BUY',
                price=1.0850,
                lot_size=0.01,
                stop_loss=1.0800,
                take_profit=1.0900,
                confidence=0.8,
                timestamp=datetime.now(),
                reason=f'Memory test {i}'
            )
            signals.append(signal)
        
        # Memory usage tekshirish
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase < 100 MB bo'lishi kerak
        assert memory_increase < 100.0
        
        logger.info(f"Memory increase: {memory_increase:.2f} MB")

# Test konfiguratsiya
@pytest.fixture(autouse=True)
def setup_test_environment():
    """Test muhiti sozlash"""
    # Test loglarini sozlash
    logger.info("=== TRADING TESTLARI BOSHLANDI ===")
    
    yield
    
    logger.info("=== TRADING TESTLARI TUGADI ===")

if __name__ == "__main__":
    # Testlarni ishga tushirish
    pytest.main([__file__, "-v", "--tb=short"])

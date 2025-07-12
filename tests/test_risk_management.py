"""
AI OrderFlow & Signal Bot - Risk Management Test Module
Risk management komponentlari uchun test suite
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import json
from datetime import datetime, timedelta

# Risk management modullarini import qilish
from risk_management.risk_calculator import RiskCalculator, RiskResult
from risk_management.position_sizer import PositionSizer, PositionResult
from risk_management.trade_monitor import TradeMonitor, TradeStatus
from utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class MockTradeData:
    """Test uchun mock savdo ma'lumotlari"""
    symbol: str
    price: float
    account_balance: float
    risk_percent: float
    stop_loss: float
    take_profit: float
    lot_size: float = 0.0
    
@dataclass
class MockAccountData:
    """Test uchun mock akkaunt ma'lumotlari"""
    balance: float
    equity: float
    margin: float
    free_margin: float
    daily_loss: float = 0.0
    total_trades: int = 0
    
class TestRiskCalculator:
    """Risk Calculator test sinfi"""
    
    @pytest.fixture
    def risk_calculator(self):
        """Risk calculator fixture"""
        return RiskCalculator()
    
    @pytest.fixture
    def sample_trade_data(self):
        """Sample trade data fixture"""
        return MockTradeData(
            symbol="EURUSD",
            price=1.0500,
            account_balance=10000.0,
            risk_percent=0.02,  # 2%
            stop_loss=1.0450,   # 50 pips
            take_profit=1.0600  # 100 pips
        )
    
    def test_calculate_risk_basic(self, risk_calculator, sample_trade_data):
        """Asosiy risk hisoblash testi"""
        # Test bajarish
        result = risk_calculator.calculate_risk(sample_trade_data)
        
        # Tekshirish
        assert result.success is True
        assert result.risk_amount == 200.0  # 2% of 10000
        assert result.pip_value > 0
        assert result.max_lot_size > 0
        assert result.error is None
        
        logger.info(f"Risk hisoblash muvaffaqiyatli: {result.risk_amount}")
    
    def test_calculate_risk_invalid_data(self, risk_calculator):
        """Noto'g'ri ma'lumotlar bilan test"""
        invalid_data = MockTradeData(
            symbol="INVALID",
            price=0.0,  # Noto'g'ri narx
            account_balance=-1000.0,  # Manfiy balans
            risk_percent=0.5,  # 50% - juda yuqori
            stop_loss=0.0,
            take_profit=0.0
        )
        
        result = risk_calculator.calculate_risk(invalid_data)
        
        # Xatolik bo'lishi kerak
        assert result.success is False
        assert result.error is not None
        assert "invalid" in result.error.lower()
        
        logger.info(f"Noto'g'ri ma'lumotlar xatoligi: {result.error}")
    
    def test_calculate_risk_zero_stop_loss(self, risk_calculator):
        """Stop loss bo'lmaganda test"""
        data = MockTradeData(
            symbol="GBPUSD",
            price=1.2500,
            account_balance=5000.0,
            risk_percent=0.01,
            stop_loss=0.0,  # Stop loss yo'q
            take_profit=1.2600
        )
        
        result = risk_calculator.calculate_risk(data)
        
        # Default stop loss qo'yilishi kerak
        assert result.success is True
        assert result.suggested_stop_loss > 0
        assert result.risk_amount > 0
        
        logger.info(f"Default stop loss qo'yildi: {result.suggested_stop_loss}")
    
    def test_calculate_risk_high_risk_percent(self, risk_calculator):
        """Yuqori risk foizi bilan test"""
        data = MockTradeData(
            symbol="USDJPY",
            price=150.00,
            account_balance=1000.0,
            risk_percent=0.10,  # 10% - juda yuqori
            stop_loss=149.00,
            take_profit=152.00
        )
        
        result = risk_calculator.calculate_risk(data)
        
        # Risk kamaytirilishi kerak
        assert result.success is True
        assert result.adjusted_risk_percent < 0.10
        assert result.warning is not None
        assert "yuqori risk" in result.warning.lower()
        
        logger.info(f"Risk kamaytirildi: {result.adjusted_risk_percent}")
    
    def test_propshot_risk_limits(self, risk_calculator):
        """Propshot risk limitlari testi"""
        data = MockTradeData(
            symbol="AUDUSD",
            price=0.6500,
            account_balance=2000.0,
            risk_percent=0.03,  # 3%
            stop_loss=0.6450,
            take_profit=0.6600
        )
        
        with patch('risk_management.risk_calculator.PROPSHOT_LIMITS', {
            'max_daily_loss': 0.025,
            'max_total_loss': 0.05,
            'max_lot_size': 0.5
        }):
            result = risk_calculator.calculate_risk(data)
            
            # Propshot limitlari tekshirilishi kerak
            assert result.success is True
            assert result.propshot_compliant is True
            assert result.max_lot_size <= 0.5
            
        logger.info(f"Propshot limitlari bajarildi: {result.propshot_compliant}")

class TestPositionSizer:
    """Position Sizer test sinfi"""
    
    @pytest.fixture
    def position_sizer(self):
        """Position sizer fixture"""
        return PositionSizer()
    
    @pytest.fixture
    def sample_account_data(self):
        """Sample account data fixture"""
        return MockAccountData(
            balance=15000.0,
            equity=14800.0,
            margin=2000.0,
            free_margin=12800.0
        )
    
    def test_calculate_position_size_kelly(self, position_sizer, sample_account_data):
        """Kelly formula bilan position size hisoblash"""
        trade_data = MockTradeData(
            symbol="EURUSD",
            price=1.0500,
            account_balance=sample_account_data.balance,
            risk_percent=0.02,
            stop_loss=1.0450,
            take_profit=1.0600
        )
        
        result = position_sizer.calculate_position_size(
            trade_data, 
            sample_account_data,
            method="kelly"
        )
        
        # Tekshirish
        assert result.success is True
        assert result.lot_size > 0
        assert result.method == "kelly"
        assert result.risk_reward_ratio > 0
        
        logger.info(f"Kelly position size: {result.lot_size}")
    
    def test_calculate_position_size_fixed_percent(self, position_sizer, sample_account_data):
        """Fixed percent bilan position size hisoblash"""
        trade_data = MockTradeData(
            symbol="GBPUSD",
            price=1.2500,
            account_balance=sample_account_data.balance,
            risk_percent=0.015,  # 1.5%
            stop_loss=1.2450,
            take_profit=1.2600
        )
        
        result = position_sizer.calculate_position_size(
            trade_data,
            sample_account_data,
            method="fixed_percent"
        )
        
        # Tekshirish
        assert result.success is True
        assert result.lot_size > 0
        assert result.method == "fixed_percent"
        assert result.risk_amount == trade_data.account_balance * trade_data.risk_percent
        
        logger.info(f"Fixed percent position size: {result.lot_size}")
    
    def test_calculate_position_size_insufficient_margin(self, position_sizer):
        """Margin yetmagan holatda test"""
        low_margin_account = MockAccountData(
            balance=1000.0,
            equity=900.0,
            margin=800.0,
            free_margin=100.0  # Kam margin
        )
        
        trade_data = MockTradeData(
            symbol="USDJPY",
            price=150.00,
            account_balance=low_margin_account.balance,
            risk_percent=0.05,  # 5%
            stop_loss=149.00,
            take_profit=152.00
        )
        
        result = position_sizer.calculate_position_size(
            trade_data,
            low_margin_account,
            method="fixed_percent"
        )
        
        # Margin yetmazligi haqida ogohlantirish
        assert result.success is False
        assert result.error is not None
        assert "margin" in result.error.lower()
        
        logger.info(f"Margin yetmadi: {result.error}")
    
    def test_calculate_position_size_max_position_limit(self, position_sizer, sample_account_data):
        """Maksimal position limiti testi"""
        large_trade_data = MockTradeData(
            symbol="EURUSD",
            price=1.0500,
            account_balance=sample_account_data.balance,
            risk_percent=0.10,  # 10% - katta risk
            stop_loss=1.0490,   # Kichik stop loss
            take_profit=1.0600
        )
        
        result = position_sizer.calculate_position_size(
            large_trade_data,
            sample_account_data,
            method="fixed_percent"
        )
        
        # Position limit qo'llanilishi kerak
        assert result.success is True
        assert result.lot_size <= 2.0  # Maksimal lot size
        assert result.position_limited is True
        
        logger.info(f"Position limite qo'llanildi: {result.lot_size}")

class TestTradeMonitor:
    """Trade Monitor test sinfi"""
    
    @pytest.fixture
    def trade_monitor(self):
        """Trade monitor fixture"""
        return TradeMonitor()
    
    @pytest.fixture
    def sample_active_trades(self):
        """Sample active trades fixture"""
        return [
            {
                'id': 1,
                'symbol': 'EURUSD',
                'lot_size': 0.5,
                'entry_price': 1.0500,
                'stop_loss': 1.0450,
                'take_profit': 1.0600,
                'current_price': 1.0520,
                'profit_loss': 100.0,
                'open_time': datetime.now() - timedelta(hours=2)
            },
            {
                'id': 2,
                'symbol': 'GBPUSD',
                'lot_size': 0.3,
                'entry_price': 1.2500,
                'stop_loss': 1.2450,
                'take_profit': 1.2600,
                'current_price': 1.2480,
                'profit_loss': -60.0,
                'open_time': datetime.now() - timedelta(hours=1)
            }
        ]
    
    def test_monitor_daily_limits(self, trade_monitor):
        """Kunlik limitlar monitoring testi"""
        account_data = MockAccountData(
            balance=10000.0,
            equity=9500.0,
            margin=1000.0,
            free_margin=8500.0,
            daily_loss=400.0,  # 4% kunlik yo'qotish
            total_trades=8
        )
        
        result = trade_monitor.check_daily_limits(account_data)
        
        # Kunlik limit tekshirilishi kerak
        assert result.success is True
        assert result.daily_loss_percent == 0.04
        assert result.within_limits is True  # Hali limit ichida
        
        logger.info(f"Kunlik yo'qotish: {result.daily_loss_percent:.2%}")
    
    def test_monitor_daily_limits_exceeded(self, trade_monitor):
        """Kunlik limitlar oshib ketganda test"""
        account_data = MockAccountData(
            balance=10000.0,
            equity=9200.0,
            margin=1000.0,
            free_margin=8200.0,
            daily_loss=800.0,  # 8% kunlik yo'qotish - limit dan oshgan
            total_trades=12
        )
        
        result = trade_monitor.check_daily_limits(account_data)
        
        # Limit oshib ketgani haqida ogohlantirish
        assert result.success is False
        assert result.within_limits is False
        assert result.limit_exceeded is True
        assert result.warning is not None
        
        logger.info(f"Kunlik limit oshib ketdi: {result.warning}")
    
    def test_monitor_active_trades(self, trade_monitor, sample_active_trades):
        """Faol savdolarni monitoring qilish"""
        result = trade_monitor.monitor_active_trades(sample_active_trades)
        
        # Faol savdolar tekshirilishi kerak
        assert result.success is True
        assert result.total_trades == 2
        assert result.profitable_trades == 1
        assert result.losing_trades == 1
        assert result.total_profit_loss == 40.0  # 100 - 60
        
        logger.info(f"Faol savdolar: {result.total_trades}, P/L: {result.total_profit_loss}")
    
    def test_monitor_risk_exposure(self, trade_monitor, sample_active_trades):
        """Risk exposure monitoring"""
        account_balance = 10000.0
        
        result = trade_monitor.calculate_risk_exposure(sample_active_trades, account_balance)
        
        # Risk exposure tekshirilishi kerak
        assert result.success is True
        assert result.total_exposure > 0
        assert result.exposure_percent > 0
        assert result.exposure_percent <= 1.0  # 100% dan kam bo'lishi kerak
        
        logger.info(f"Risk exposure: {result.exposure_percent:.2%}")
    
    def test_monitor_propshot_compliance(self, trade_monitor):
        """Propshot compliance monitoring"""
        account_data = MockAccountData(
            balance=5000.0,
            equity=4900.0,
            margin=500.0,
            free_margin=4400.0,
            daily_loss=125.0,  # 2.5% kunlik yo'qotish
            total_trades=2
        )
        
        trades = [
            {
                'id': 1,
                'symbol': 'EURUSD',
                'lot_size': 0.3,  # Propshot limit ichida
                'profit_loss': -50.0
            }
        ]
        
        result = trade_monitor.check_propshot_compliance(account_data, trades)
        
        # Propshot compliance tekshirilishi kerak
        assert result.success is True
        assert result.compliant is True
        assert result.daily_loss_limit_ok is True
        assert result.lot_size_limit_ok is True
        assert result.trade_count_limit_ok is True
        
        logger.info(f"Propshot compliance: {result.compliant}")
    
    def test_monitor_propshot_compliance_violation(self, trade_monitor):
        """Propshot compliance buzilganda test"""
        account_data = MockAccountData(
            balance=5000.0,
            equity=4700.0,
            margin=1000.0,
            free_margin=3700.0,
            daily_loss=300.0,  # 6% kunlik yo'qotish - limit dan oshgan
            total_trades=5  # Ko'p savdo
        )
        
        trades = [
            {
                'id': 1,
                'symbol': 'EURUSD',
                'lot_size': 0.8,  # Propshot limit dan oshgan
                'profit_loss': -150.0
            }
        ]
        
        result = trade_monitor.check_propshot_compliance(account_data, trades)
        
        # Compliance buzilgani haqida ogohlantirish
        assert result.success is False
        assert result.compliant is False
        assert result.violations is not None
        assert len(result.violations) > 0
        
        logger.info(f"Propshot compliance buzildi: {result.violations}")
    
    def test_monitor_correlation_risk(self, trade_monitor):
        """Korrelyatsiya risk monitoring"""
        correlated_trades = [
            {
                'id': 1,
                'symbol': 'EURUSD',
                'lot_size': 0.5,
                'direction': 'buy'
            },
            {
                'id': 2,
                'symbol': 'GBPUSD',
                'lot_size': 0.3,
                'direction': 'buy'
            },
            {
                'id': 3,
                'symbol': 'AUDUSD',
                'lot_size': 0.2,
                'direction': 'buy'
            }
        ]
        
        result = trade_monitor.check_correlation_risk(correlated_trades)
        
        # Korrelyatsiya riski tekshirilishi kerak
        assert result.success is True
        assert result.correlation_risk > 0
        assert result.high_correlation_pairs is not None
        
        logger.info(f"Korrelyatsiya riski: {result.correlation_risk:.2f}")

@pytest.mark.asyncio
class TestRiskManagementIntegration:
    """Risk management integratsiya testlari"""
    
    async def test_full_risk_assessment_pipeline(self):
        """To'liq risk baholash pipeline testi"""
        # Komponentlarni yaratish
        risk_calculator = RiskCalculator()
        position_sizer = PositionSizer()
        trade_monitor = TradeMonitor()
        
        # Test ma'lumotlari
        trade_data = MockTradeData(
            symbol="EURUSD",
            price=1.0500,
            account_balance=20000.0,
            risk_percent=0.02,
            stop_loss=1.0450,
            take_profit=1.0600
        )
        
        account_data = MockAccountData(
            balance=20000.0,
            equity=19800.0,
            margin=3000.0,
            free_margin=16800.0,
            daily_loss=200.0,
            total_trades=3
        )
        
        # 1. Risk hisoblash
        risk_result = risk_calculator.calculate_risk(trade_data)
        assert risk_result.success is True
        
        # 2. Position size hisoblash
        position_result = position_sizer.calculate_position_size(
            trade_data, account_data, method="kelly"
        )
        assert position_result.success is True
        
        # 3. Trade monitoring
        monitor_result = trade_monitor.check_daily_limits(account_data)
        assert monitor_result.success is True
        
        # 4. Propshot compliance
        compliance_result = trade_monitor.check_propshot_compliance(account_data, [])
        assert compliance_result.success is True
        
        logger.info("To'liq risk baholash pipeline muvaffaqiyatli o'tdi")
    
    async def test_risk_management_error_handling(self):
        """Risk management error handling testi"""
        risk_calculator = RiskCalculator()
        
        # Noto'g'ri ma'lumotlar bilan test
        with pytest.raises(Exception):
            invalid_data = None
            risk_calculator.calculate_risk(invalid_data)
        
        logger.info("Error handling test muvaffaqiyatli")
    
    async def test_risk_management_performance(self):
        """Risk management performance testi"""
        import time
        
        risk_calculator = RiskCalculator()
        
        # Performance test
        start_time = time.time()
        
        for i in range(100):
            trade_data = MockTradeData(
                symbol="EURUSD",
                price=1.0500 + i * 0.0001,
                account_balance=10000.0,
                risk_percent=0.02,
                stop_loss=1.0450,
                take_profit=1.0600
            )
            
            result = risk_calculator.calculate_risk(trade_data)
            assert result.success is True
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Performance tekshirilishi - 100 ta hisoblash 1 soniyadan kam bo'lishi kerak
        assert execution_time < 1.0
        
        logger.info(f"100 ta risk hisoblash vaqti: {execution_time:.3f} soniya")

# Fixtures va utility funktsiolar
@pytest.fixture
def mock_propshot_config():
    """Mock Propshot konfiguratsiyasi"""
    return {
        'max_daily_loss': 0.05,    # 5%
        'max_total_loss': 0.10,    # 10%
        'max_lot_size': 1.0,       # 1 lot
        'max_daily_trades': 10,    # 10 ta savdo
        'max_correlation_risk': 0.7 # 70%
    }

@pytest.fixture
def mock_market_data():
    """Mock market ma'lumotlari"""
    return {
        'EURUSD': {
            'bid': 1.0500,
            'ask': 1.0502,
            'spread': 0.0002,
            'volatility': 0.015
        },
        'GBPUSD': {
            'bid': 1.2500,
            'ask': 1.2503,
            'spread': 0.0003,
            'volatility': 0.020
        }
    }

def test_risk_management_config_validation():
    """Risk management konfiguratsiya validatsiya testi"""
    # Test konfiguratsiya
    config = {
        'max_risk_per_trade': 0.02,
        'max_daily_loss': 0.05,
        'max_correlation_risk': 0.7,
        'position_sizing_method': 'kelly'
    }
    
    # Validatsiya
    assert 0 < config['max_risk_per_trade'] <= 0.05
    assert 0 < config['max_daily_loss'] <= 0.10
    assert 0 < config['max_correlation_risk'] <= 1.0
    assert config['position_sizing_method'] in ['kelly', 'fixed_percent', 'volatility']
    
    logger.info("Risk management konfiguratsiya validatsiya muvaffaqiyatli")

if __name__ == "__main__":
    # Test ishga tushirish
    logger.info("Risk Management testlari boshlandi")
    pytest.main([__file__, "-v"])
    logger.info("Risk Management testlari tugadi")

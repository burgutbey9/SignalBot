"""
AI OrderFlow & Signal Bot - PyTest Konfiguratsiya
PyTest fixtures va mock objects yaratish
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path

# Mock data va responses
MOCK_CONFIG_DATA = {
    "api_limits": {
        "oneinch": {
            "rate_limit": 100,
            "timeout": 30,
            "max_retries": 3
        },
        "alchemy": {
            "rate_limit": 300,
            "timeout": 15,
            "max_retries": 5
        },
        "huggingface": {
            "rate_limit": 1000,
            "timeout": 60,
            "max_retries": 2
        }
    },
    "trading": {
        "max_risk_per_trade": 0.02,
        "max_daily_loss": 0.05,
        "position_size_method": "kelly"
    }
}

MOCK_API_KEYS = {
    "ONEINCH_API_KEY": "test_oneinch_key",
    "ALCHEMY_API_KEY": "test_alchemy_key",
    "HUGGINGFACE_API_KEY": "test_huggingface_key",
    "TELEGRAM_BOT_TOKEN": "test_telegram_token",
    "TELEGRAM_CHAT_ID": "test_chat_id"
}

MOCK_MARKET_DATA = {
    "symbol": "EURUSD",
    "price": 1.0850,
    "bid": 1.0848,
    "ask": 1.0852,
    "volume": 1000000,
    "change": 0.0015,
    "timestamp": "2024-01-15T10:30:00Z"
}

@pytest.fixture(scope="session")
def event_loop():
    """Async test uchun event loop yaratish"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def mock_config():
    """Mock konfiguratsiya objekti"""
    config = Mock()
    config.api_limits = MOCK_CONFIG_DATA["api_limits"]
    config.trading = MOCK_CONFIG_DATA["trading"]
    config.database_url = "sqlite:///:memory:"
    config.log_level = "DEBUG"
    config.timezone = "Asia/Tashkent"
    
    # Konfiguratsiya metodlari
    config.get_api_limit = Mock(return_value=100)
    config.get_timeout = Mock(return_value=30)
    config.get_max_retries = Mock(return_value=3)
    config.validate_config = Mock(return_value=True)
    
    return config

@pytest.fixture
def mock_api_keys():
    """Mock API kalitlari"""
    keys = Mock()
    keys.oneinch_api_key = MOCK_API_KEYS["ONEINCH_API_KEY"]
    keys.alchemy_api_key = MOCK_API_KEYS["ALCHEMY_API_KEY"]
    keys.huggingface_api_key = MOCK_API_KEYS["HUGGINGFACE_API_KEY"]
    keys.telegram_bot_token = MOCK_API_KEYS["TELEGRAM_BOT_TOKEN"]
    keys.telegram_chat_id = MOCK_API_KEYS["TELEGRAM_CHAT_ID"]
    
    # API kalitlarini tekshirish
    keys.validate_keys = Mock(return_value=True)
    keys.get_key = Mock(side_effect=lambda name: MOCK_API_KEYS.get(name))
    
    return keys

@pytest.fixture
def mock_database():
    """Mock database objekti"""
    db = AsyncMock()
    
    # Database metodlari
    db.connect = AsyncMock(return_value=True)
    db.disconnect = AsyncMock(return_value=True)
    db.execute = AsyncMock(return_value=True)
    db.fetch_one = AsyncMock(return_value={"id": 1, "name": "test"})
    db.fetch_all = AsyncMock(return_value=[{"id": 1, "name": "test"}])
    db.create_tables = AsyncMock(return_value=True)
    
    # Signal va trade ma'lumotlari
    db.save_signal = AsyncMock(return_value=1)
    db.save_trade = AsyncMock(return_value=1)
    db.get_signals = AsyncMock(return_value=[])
    db.get_trades = AsyncMock(return_value=[])
    
    return db

@pytest.fixture
def mock_api_client():
    """Mock API client obyekti"""
    client = AsyncMock()
    
    # API client metodlari
    client.make_request = AsyncMock(return_value={
        "success": True,
        "data": MOCK_MARKET_DATA,
        "error": None,
        "rate_limit_remaining": 99
    })
    
    client.get_market_data = AsyncMock(return_value=MOCK_MARKET_DATA)
    client.get_order_flow = AsyncMock(return_value={
        "buy_pressure": 0.65,
        "sell_pressure": 0.35,
        "large_orders": 5,
        "whale_activity": True
    })
    
    # Rate limiting
    client.rate_limiter = Mock()
    client.rate_limiter.wait = AsyncMock()
    client.rate_limiter.remaining = 99
    
    return client

@pytest.fixture
def mock_telegram():
    """Mock Telegram client"""
    telegram = AsyncMock()
    
    # Telegram metodlari
    telegram.send_message = AsyncMock(return_value=True)
    telegram.send_signal = AsyncMock(return_value=True)
    telegram.send_log = AsyncMock(return_value=True)
    telegram.send_error = AsyncMock(return_value=True)
    
    # Telegram response
    telegram.get_updates = AsyncMock(return_value=[])
    telegram.handle_callback = AsyncMock(return_value=True)
    
    return telegram

@pytest.fixture
def sample_order_flow_data():
    """Sample order flow ma'lumotlari"""
    return {
        "timestamp": datetime.now().isoformat(),
        "symbol": "EURUSD",
        "buy_orders": [
            {"size": 1000000, "price": 1.0850},
            {"size": 500000, "price": 1.0851}
        ],
        "sell_orders": [
            {"size": 800000, "price": 1.0849},
            {"size": 600000, "price": 1.0848}
        ],
        "net_flow": 100000,
        "buy_pressure": 0.62,
        "sell_pressure": 0.38,
        "large_orders_count": 3,
        "whale_activity": True
    }

@pytest.fixture
def sample_sentiment_data():
    """Sample sentiment ma'lumotlari"""
    return {
        "timestamp": datetime.now().isoformat(),
        "symbol": "EURUSD",
        "news_sentiment": 0.7,
        "social_sentiment": 0.6,
        "reddit_buzz": 0.8,
        "twitter_sentiment": 0.5,
        "overall_score": 0.65,
        "confidence": 0.85,
        "sources": ["newsapi", "reddit", "twitter"],
        "news_count": 15,
        "social_mentions": 250
    }

@pytest.fixture
def sample_signal_data():
    """Sample signal ma'lumotlari"""
    return {
        "timestamp": datetime.now().isoformat(),
        "symbol": "EURUSD",
        "action": "BUY",
        "price": 1.0850,
        "lot_size": 0.1,
        "stop_loss": 1.0820,
        "take_profit": 1.0900,
        "risk_percent": 2.0,
        "confidence": 85.0,
        "signal_strength": "STRONG",
        "strategy": "OrderFlow + Sentiment",
        "reason": "Strong buy pressure + positive sentiment"
    }

@pytest.fixture
def sample_trade_data():
    """Sample trade ma'lumotlari"""
    return {
        "trade_id": 1,
        "signal_id": 1,
        "symbol": "EURUSD",
        "action": "BUY",
        "entry_time": datetime.now().isoformat(),
        "entry_price": 1.0850,
        "lot_size": 0.1,
        "stop_loss": 1.0820,
        "take_profit": 1.0900,
        "status": "OPEN",
        "account_id": "test_account",
        "platform": "propshot"
    }

@pytest.fixture
def sample_historical_data():
    """Sample tarixiy ma'lumotlar"""
    dates = pd.date_range(start="2024-01-01", end="2024-01-31", freq="H")
    data = []
    
    for date in dates:
        data.append({
            "timestamp": date.isoformat(),
            "symbol": "EURUSD",
            "open": 1.0850 + (0.01 * (len(data) % 10) - 0.005),
            "high": 1.0860 + (0.01 * (len(data) % 10) - 0.005),
            "low": 1.0840 + (0.01 * (len(data) % 10) - 0.005),
            "close": 1.0855 + (0.01 * (len(data) % 10) - 0.005),
            "volume": 1000000 + (100000 * (len(data) % 5))
        })
    
    return data

@pytest.fixture
def mock_logger():
    """Mock logger obyekti"""
    logger = Mock()
    logger.info = Mock()
    logger.warning = Mock()
    logger.error = Mock()
    logger.debug = Mock()
    logger.critical = Mock()
    
    return logger

@pytest.fixture
def mock_rate_limiter():
    """Mock rate limiter"""
    limiter = AsyncMock()
    limiter.wait = AsyncMock()
    limiter.remaining = 99
    limiter.reset_time = datetime.now() + timedelta(minutes=1)
    
    return limiter

@pytest.fixture
def mock_fallback_manager():
    """Mock fallback manager"""
    manager = Mock()
    manager.get_next_client = Mock(return_value="backup_client")
    manager.mark_failed = Mock()
    manager.mark_success = Mock()
    manager.reset_failures = Mock()
    
    return manager

@pytest.fixture
def mock_risk_calculator():
    """Mock risk calculator"""
    calc = Mock()
    calc.calculate_position_size = Mock(return_value=0.1)
    calc.calculate_risk_percent = Mock(return_value=2.0)
    calc.calculate_stop_loss = Mock(return_value=1.0820)
    calc.calculate_take_profit = Mock(return_value=1.0900)
    calc.validate_risk = Mock(return_value=True)
    
    return calc

@pytest.fixture
def mock_propshot_connector():
    """Mock Propshot connector"""
    connector = AsyncMock()
    connector.connect = AsyncMock(return_value=True)
    connector.disconnect = AsyncMock(return_value=True)
    connector.send_order = AsyncMock(return_value={"success": True, "order_id": "12345"})
    connector.get_positions = AsyncMock(return_value=[])
    connector.get_account_info = AsyncMock(return_value={
        "balance": 10000,
        "equity": 10000,
        "free_margin": 9000,
        "margin_level": 1000
    })
    
    return connector

@pytest.fixture
def test_data_dir(tmp_path):
    """Test ma'lumotlar uchun vaqtinchalik papka"""
    data_dir = tmp_path / "test_data"
    data_dir.mkdir()
    
    # Test fayllar yaratish
    config_file = data_dir / "test_config.json"
    config_file.write_text(json.dumps(MOCK_CONFIG_DATA))
    
    keys_file = data_dir / "test_keys.json"
    keys_file.write_text(json.dumps(MOCK_API_KEYS))
    
    return data_dir

@pytest.fixture
def cleanup_test_files():
    """Test fayllarni tozalash"""
    test_files = []
    
    def add_file(filepath):
        test_files.append(filepath)
    
    yield add_file
    
    # Cleanup
    for filepath in test_files:
        if Path(filepath).exists():
            Path(filepath).unlink()

# Pytest hooks va konfiguratsiya
def pytest_configure(config):
    """PyTest konfiguratsiya"""
    config.addinivalue_line(
        "markers", "slow: slow running tests"
    )
    config.addinivalue_line(
        "markers", "integration: integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: unit tests"
    )

def pytest_collection_modifyitems(config, items):
    """Test itemlarni o'zgartirish"""
    for item in items:
        # Async testlar uchun marker
        if "async" in item.name:
            item.add_marker(pytest.mark.asyncio)
        
        # Integration testlar uchun marker
        if "integration" in item.name:
            item.add_marker(pytest.mark.integration)

@pytest.fixture(autouse=True)
def setup_test_environment():
    """Har test uchun muhit sozlash"""
    # Test muhitini sozlash
    with patch.dict("os.environ", MOCK_API_KEYS):
        with patch("config.config.ConfigManager") as mock_config:
            mock_config.return_value.load_config.return_value = MOCK_CONFIG_DATA
            yield

@pytest.fixture
def mock_async_context_manager():
    """Async context manager mock"""
    manager = AsyncMock()
    manager.__aenter__ = AsyncMock(return_value=manager)
    manager.__aexit__ = AsyncMock(return_value=None)
    
    return manager

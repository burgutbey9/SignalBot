"""
AI OrderFlow & Signal Bot - Config Testlari
===========================================

Konfiguratsiya modullarini sinash va tekshirish

Bu fayl config modulidagi barcha funksiyalar va klasslarni sinaydi:
- Konfiguratsiya yuklash
- API kalitlar tekshirish
- Fallback tizimi
- Validatsiya jarayonlari
"""

import pytest
import os
import json
import tempfile
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from typing import Dict, Any, Optional

# Test uchun import qilinadigan modullar
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config.config import ConfigManager, DatabaseConfig, APIConfig
from config.api_keys import APIKeyManager
from config.fallback_config import FallbackManager
from utils.logger import get_logger

logger = get_logger(__name__)


class TestConfigManager:
    """ConfigManager klassini sinash"""
    
    def setup_method(self):
        """Har test oldidan ishlaydi - test muhitini tayyorlash"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.temp_dir, 'test_settings.json')
        self.env_file = os.path.join(self.temp_dir, '.env')
        
        # Test uchun sample konfiguratsiya
        self.sample_config = {
            "database": {
                "url": "sqlite:///test.db",
                "max_connections": 5,
                "timeout": 30
            },
            "api_limits": {
                "oneinch": {"rate_limit": 100, "timeout": 30},
                "alchemy": {"rate_limit": 300, "timeout": 15}
            },
            "trading": {
                "max_risk_per_trade": 0.02,
                "max_daily_loss": 0.05
            }
        }
        
        # Test konfiguratsiya faylini yaratish
        with open(self.config_file, 'w') as f:
            json.dump(self.sample_config, f, indent=2)
    
    def teardown_method(self):
        """Har test oxirida ishlaydi - tozalash"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_config_loading(self):
        """Konfiguratsiya yuklash testini tekshirish"""
        # Test qilish
        with patch('config.config.os.path.exists', return_value=True):
            with patch('builtins.open', mock_open_json(self.sample_config)):
                config_manager = ConfigManager()
                config_manager.config_file = self.config_file
                config_manager.load_config()
                
                # Natijalarni tekshirish
                assert config_manager.database_config.url == "sqlite:///test.db"
                assert config_manager.database_config.max_connections == 5
                assert config_manager.api_config.timeout == 30
                
        logger.info("✅ Konfiguratsiya yuklash testi muvaffaqiyatli")
    
    def test_config_validation(self):
        """Konfiguratsiya validatsiya testini tekshirish"""
        config_manager = ConfigManager()
        
        # To'g'ri konfiguratsiya
        valid_config = {
            "database": {"url": "sqlite:///test.db"},
            "api_limits": {"oneinch": {"rate_limit": 100}}
        }
        
        # Noto'g'ri konfiguratsiya
        invalid_config = {
            "database": {},  # url yo'q
            "api_limits": {"oneinch": {"rate_limit": "invalid"}}  # noto'g'ri tip
        }
        
        # Validatsiya testlari
        assert config_manager.validate_config(valid_config) == True
        assert config_manager.validate_config(invalid_config) == False
        
        logger.info("✅ Konfiguratsiya validatsiya testi muvaffaqiyatli")
    
    def test_config_missing_file(self):
        """Konfiguratsiya fayli yo'q bo'lgan holatni tekshirish"""
        with patch('config.config.os.path.exists', return_value=False):
            config_manager = ConfigManager()
            
            # Fayl yo'q bo'lsa, default qiymatlar ishlatilishi kerak
            with pytest.raises(FileNotFoundError):
                config_manager.load_config()
        
        logger.info("✅ Konfiguratsiya fayli yo'q test muvaffaqiyatli")
    
    def test_config_invalid_json(self):
        """Noto'g'ri JSON formati testini tekshirish"""
        invalid_json = "{'invalid': json format}"
        
        with patch('config.config.os.path.exists', return_value=True):
            with patch('builtins.open', mock_open_text(invalid_json)):
                config_manager = ConfigManager()
                
                with pytest.raises(json.JSONDecodeError):
                    config_manager.load_config()
        
        logger.info("✅ Noto'g'ri JSON format testi muvaffaqiyatli")


class TestAPIKeyManager:
    """APIKeyManager klassini sinash"""
    
    def setup_method(self):
        """Test muhitini tayyorlash"""
        self.temp_dir = tempfile.mkdtemp()
        self.env_file = os.path.join(self.temp_dir, '.env')
        
        # Test uchun sample API keys
        self.sample_env = """
ONEINCH_API_KEY=test_oneinch_key
ALCHEMY_API_KEY=test_alchemy_key
GEMINI_API_KEY_1=test_gemini_key_1
GEMINI_API_KEY_2=test_gemini_key_2
TELEGRAM_BOT_TOKEN=test_telegram_token
"""
        
        with open(self.env_file, 'w') as f:
            f.write(self.sample_env)
    
    def teardown_method(self):
        """Test muhitini tozalash"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_api_keys_loading(self):
        """API kalitlarni yuklash testini tekshirish"""
        with patch('config.api_keys.load_dotenv') as mock_load:
            with patch.dict(os.environ, {
                'ONEINCH_API_KEY': 'test_oneinch_key',
                'ALCHEMY_API_KEY': 'test_alchemy_key',
                'GEMINI_API_KEY_1': 'test_gemini_key_1'
            }):
                key_manager = APIKeyManager()
                key_manager.load_api_keys()
                
                # API kalitlarni tekshirish
                assert key_manager.get_api_key('oneinch') == 'test_oneinch_key'
                assert key_manager.get_api_key('alchemy') == 'test_alchemy_key'
                assert key_manager.get_api_key('gemini', 1) == 'test_gemini_key_1'
                
        logger.info("✅ API kalitlar yuklash testi muvaffaqiyatli")
    
    def test_api_keys_validation(self):
        """API kalitlar validatsiya testini tekshirish"""
        key_manager = APIKeyManager()
        
        # To'g'ri kalitlar
        valid_keys = {
            'oneinch': 'valid_key_length_32_characters',
            'alchemy': 'valid_alchemy_key_format',
            'telegram': '1234567890:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefg'
        }
        
        # Noto'g'ri kalitlar
        invalid_keys = {
            'oneinch': 'short',  # juda qisqa
            'alchemy': '',       # bo'sh
            'telegram': 'invalid_format'  # noto'g'ri format
        }
        
        # Validatsiya testlari
        for key_name, key_value in valid_keys.items():
            assert key_manager.validate_api_key(key_name, key_value) == True
        
        for key_name, key_value in invalid_keys.items():
            assert key_manager.validate_api_key(key_name, key_value) == False
        
        logger.info("✅ API kalitlar validatsiya testi muvaffaqiyatli")
    
    def test_gemini_multiple_keys(self):
        """Gemini uchun bir nechta kalitlar testini tekshirish"""
        with patch.dict(os.environ, {
            'GEMINI_API_KEY_1': 'gemini_key_1',
            'GEMINI_API_KEY_2': 'gemini_key_2',
            'GEMINI_API_KEY_3': 'gemini_key_3',
            'GEMINI_API_KEY_4': 'gemini_key_4',
            'GEMINI_API_KEY_5': 'gemini_key_5'
        }):
            key_manager = APIKeyManager()
            key_manager.load_api_keys()
            
            # Barcha Gemini kalitlarni tekshirish
            gemini_keys = key_manager.get_gemini_keys()
            assert len(gemini_keys) == 5
            assert 'gemini_key_1' in gemini_keys
            assert 'gemini_key_5' in gemini_keys
            
        logger.info("✅ Gemini bir nechta kalitlar testi muvaffaqiyatli")


class TestFallbackManager:
    """FallbackManager klassini sinash"""
    
    def setup_method(self):
        """Test muhitini tayyorlash"""
        self.fallback_config = {
            "order_flow": ["oneinch", "thegraph", "alchemy"],
            "sentiment": ["huggingface", "gemini", "claude"],
            "news": ["newsapi", "reddit", "claude"]
        }
        
        self.fallback_manager = FallbackManager(self.fallback_config)
    
    def test_fallback_config(self):
        """Fallback konfiguratsiya testini tekshirish"""
        # Fallback tartibi tekshirish
        order_flow_fallback = self.fallback_manager.get_fallback_order('order_flow')
        assert order_flow_fallback == ["oneinch", "thegraph", "alchemy"]
        
        sentiment_fallback = self.fallback_manager.get_fallback_order('sentiment')
        assert sentiment_fallback == ["huggingface", "gemini", "claude"]
        
        # Noto'g'ri service nomi
        invalid_fallback = self.fallback_manager.get_fallback_order('invalid_service')
        assert invalid_fallback == []
        
        logger.info("✅ Fallback konfiguratsiya testi muvaffaqiyatli")
    
    def test_fallback_switching(self):
        """Fallback o'tish testini tekshirish"""
        # Birinchi service ishlamay qolgan holatni simulyatsiya qilish
        current_service = self.fallback_manager.get_current_service('order_flow')
        assert current_service == "oneinch"  # birinchi service
        
        # Keyingi service ga o'tish
        next_service = self.fallback_manager.switch_to_next('order_flow')
        assert next_service == "thegraph"  # ikkinchi service
        
        # Yana keyingi service ga o'tish
        next_service = self.fallback_manager.switch_to_next('order_flow')
        assert next_service == "alchemy"  # uchinchi service
        
        # Oxirgi service dan keyin None qaytarish
        next_service = self.fallback_manager.switch_to_next('order_flow')
        assert next_service is None
        
        logger.info("✅ Fallback o'tish testi muvaffaqiyatli")
    
    def test_fallback_reset(self):
        """Fallback qayta tiklash testini tekshirish"""
        # Fallback ni keyingi service ga o'tkazish
        self.fallback_manager.switch_to_next('sentiment')
        current = self.fallback_manager.get_current_service('sentiment')
        assert current == "gemini"  # ikkinchi service
        
        # Fallback ni qayta tiklash
        self.fallback_manager.reset_fallback('sentiment')
        current = self.fallback_manager.get_current_service('sentiment')
        assert current == "huggingface"  # birinchi service ga qaytdi
        
        logger.info("✅ Fallback qayta tiklash testi muvaffaqiyatli")


class TestDatabaseConfig:
    """DatabaseConfig klassini sinash"""
    
    def test_database_config_creation(self):
        """Database konfiguratsiya yaratish testini tekshirish"""
        db_config = DatabaseConfig(
            url="postgresql://user:pass@localhost/db",
            max_connections=20,
            timeout=60
        )
        
        # Konfiguratsiya qiymatlarini tekshirish
        assert db_config.url == "postgresql://user:pass@localhost/db"
        assert db_config.max_connections == 20
        assert db_config.timeout == 60
        
        logger.info("✅ Database konfiguratsiya yaratish testi muvaffaqiyatli")
    
    def test_database_config_defaults(self):
        """Database konfiguratsiya default qiymatlar testini tekshirish"""
        db_config = DatabaseConfig(url="sqlite:///test.db")
        
        # Default qiymatlarni tekshirish
        assert db_config.url == "sqlite:///test.db"
        assert db_config.max_connections == 10  # default qiymat
        assert db_config.timeout == 30  # default qiymat
        
        logger.info("✅ Database konfiguratsiya default qiymatlar testi muvaffaqiyatli")


class TestAPIConfig:
    """APIConfig klassini sinash"""
    
    def test_api_config_creation(self):
        """API konfiguratsiya yaratish testini tekshirish"""
        api_config = APIConfig(
            timeout=45,
            max_retries=5,
            rate_limit=200
        )
        
        # Konfiguratsiya qiymatlarini tekshirish
        assert api_config.timeout == 45
        assert api_config.max_retries == 5
        assert api_config.rate_limit == 200
        
        logger.info("✅ API konfiguratsiya yaratish testi muvaffaqiyatli")
    
    def test_api_config_defaults(self):
        """API konfiguratsiya default qiymatlar testini tekshirish"""
        api_config = APIConfig()
        
        # Default qiymatlarni tekshirish
        assert api_config.timeout == 30
        assert api_config.max_retries == 3
        assert api_config.rate_limit == 100
        
        logger.info("✅ API konfiguratsiya default qiymatlar testi muvaffaqiyatli")


# Test uchun yordamchi funksiyalar
def mock_open_json(data: Dict[str, Any]):
    """JSON ma'lumotlar uchun mock open funksiyasi"""
    import json
    from unittest.mock import mock_open
    return mock_open(read_data=json.dumps(data))


def mock_open_text(text: str):
    """Matn ma'lumotlar uchun mock open funksiyasi"""
    from unittest.mock import mock_open
    return mock_open(read_data=text)


# Test fixtures
@pytest.fixture
def sample_config():
    """Test uchun sample konfiguratsiya"""
    return {
        "database": {
            "url": "sqlite:///test.db",
            "max_connections": 10,
            "timeout": 30
        },
        "api_limits": {
            "oneinch": {"rate_limit": 100, "timeout": 30},
            "alchemy": {"rate_limit": 300, "timeout": 15}
        },
        "trading": {
            "max_risk_per_trade": 0.02,
            "max_daily_loss": 0.05
        }
    }


@pytest.fixture
def sample_env_vars():
    """Test uchun sample environment variables"""
    return {
        'ONEINCH_API_KEY': 'test_oneinch_key_12345678',
        'ALCHEMY_API_KEY': 'test_alchemy_key_12345678',
        'GEMINI_API_KEY_1': 'test_gemini_key_1_12345678',
        'GEMINI_API_KEY_2': 'test_gemini_key_2_12345678',
        'TELEGRAM_BOT_TOKEN': '1234567890:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefg'
    }


@pytest.fixture
def fallback_config():
    """Test uchun fallback konfiguratsiya"""
    return {
        "order_flow": ["oneinch", "thegraph", "alchemy"],
        "sentiment": ["huggingface", "gemini", "claude"],
        "news": ["newsapi", "reddit", "claude"]
    }


if __name__ == "__main__":
    # Testlarni ishga tushirish
    pytest.main([__file__, "-v"])
    logger.info("🎯 Barcha config testlari tugallandi")

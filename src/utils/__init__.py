"""
Utils moduli - Barcha yordamchi class va functionlarni import qilish
=================================================================

Bu modul utils package ning asosiy entry point hisoblanadi.
Barcha utility classlarni import qilish va package sifatida ishlashni ta'minlaydi.

Author: AI OrderFlow Signal Bot
Version: 1.0.0
"""

# Standart importlar
import sys
import os
from pathlib import Path

# Utils modulining versiya ma'lumotlari
__version__ = "1.0.0"
__author__ = "AI OrderFlow Signal Bot"
__description__ = "Utility functions and classes for OrderFlow Signal Bot"

# Asosiy utility classlarni import qilish
try:
    from .logger import get_logger, setup_logger
    from .error_handler import handle_api_error, handle_processing_error, ErrorHandler
    from .rate_limiter import RateLimiter, RateLimit
    from .retry_handler import retry_async, retry_sync, RetryHandler
    from .fallback_manager import FallbackManager, FallbackConfig
    
    # Logger ni o'rnatish
    logger = get_logger(__name__)
    logger.info("Utils moduli muvaffaqiyatli yuklandi")
    
except ImportError as e:
    # Agar biror utility import bo'lmasa, console ga xabar
    print(f"Utils import xatosi: {e}")
    print("Ba'zi utility classlar mavjud emas")

# Package da mavjud bo'lgan barcha classlar
__all__ = [
    # Logger utilities
    'get_logger',
    'setup_logger',
    
    # Error handling
    'handle_api_error',
    'handle_processing_error',
    'ErrorHandler',
    
    # Rate limiting
    'RateLimiter',
    'RateLimit',
    
    # Retry logic
    'retry_async',
    'retry_sync',
    'RetryHandler',
    
    # Fallback system
    'FallbackManager',
    'FallbackConfig',
    
    # Package info
    '__version__',
    '__author__',
    '__description__'
]

# Utils modulining asosiy konfiguratsiyasi
class UtilsConfig:
    """Utils moduli uchun asosiy konfiguratsiya"""
    
    # Log fayl yo'llari
    LOG_DIR = Path("logs")
    DEFAULT_LOG_FILE = LOG_DIR / "bot.log"
    
    # Rate limiting default qiymatlar
    DEFAULT_RATE_LIMIT = 100
    DEFAULT_RATE_PERIOD = 60  # seconds
    
    # Retry default qiymatlar
    DEFAULT_MAX_RETRIES = 3
    DEFAULT_RETRY_DELAY = 1  # seconds
    
    # Error handling
    MAX_ERROR_MESSAGE_LENGTH = 500
    
    @classmethod
    def ensure_log_dir(cls):
        """Log direktoriyasini yaratish"""
        cls.LOG_DIR.mkdir(exist_ok=True)
        return cls.LOG_DIR
    
    @classmethod
    def get_log_path(cls, module_name: str) -> Path:
        """Modul uchun log fayl yo'lini olish"""
        cls.ensure_log_dir()
        return cls.LOG_DIR / f"{module_name}.log"

# Utils modulini ishga tushirish
def initialize_utils():
    """Utils modulini ishga tushirish"""
    try:
        # Log direktoriyasini yaratish
        UtilsConfig.ensure_log_dir()
        
        # Asosiy logger ni o'rnatish
        main_logger = get_logger("utils")
        main_logger.info("Utils moduli ishga tushirildi")
        main_logger.info(f"Utils versiya: {__version__}")
        
        return True
        
    except Exception as e:
        print(f"Utils modulini ishga tushirishda xato: {e}")
        return False

# Modulni import qilganda avtomatik ishga tushirish
_initialized = initialize_utils()

# Debug ma'lumotlari
def get_utils_info():
    """Utils moduli haqida ma'lumot"""
    return {
        'version': __version__,
        'author': __author__,
        'description': __description__,
        'available_utilities': __all__,
        'initialized': _initialized,
        'log_directory': str(UtilsConfig.LOG_DIR),
        'config': {
            'default_rate_limit': UtilsConfig.DEFAULT_RATE_LIMIT,
            'default_rate_period': UtilsConfig.DEFAULT_RATE_PERIOD,
            'default_max_retries': UtilsConfig.DEFAULT_MAX_RETRIES,
            'default_retry_delay': UtilsConfig.DEFAULT_RETRY_DELAY
        }
    }

# Test funksiyasi
def test_utils():
    """Utils modulini test qilish"""
    print("🔧 Utils moduli test qilinmoqda...")
    
    try:
        # Logger test
        test_logger = get_logger("test")
        test_logger.info("Logger test muvaffaqiyatli")
        
        # Rate limiter test
        rate_limiter = RateLimiter(calls=5, period=60)
        print("✅ RateLimiter yaratildi")
        
        # Error handler test
        error_handler = ErrorHandler()
        print("✅ ErrorHandler yaratildi")
        
        # Fallback manager test
        fallback_manager = FallbackManager()
        print("✅ FallbackManager yaratildi")
        
        print("🎉 Barcha utils testlari muvaffaqiyatli!")
        return True
        
    except Exception as e:
        print(f"❌ Utils test xatosi: {e}")
        return False

# Modul import qilinganda chop etish
if _initialized:
    pass  # Sukut bilan ishlash
else:
    print("⚠️  Utils moduli to'liq ishga tushmadi")

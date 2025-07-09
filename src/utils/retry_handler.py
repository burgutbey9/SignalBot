"""
utils/retry_handler.py - Qayta urinish logika implementatsiyasi
Barcha API so'rovlar uchun qayta urinish mexanizmi
"""

import asyncio
import functools
import random
import time
from typing import Callable, Any, Optional, Union, Type
from dataclasses import dataclass
from utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class RetryConfig:
    """Qayta urinish konfiguratsiyasi"""
    max_retries: int = 3              # Maksimal urinish soni
    base_delay: float = 1.0           # Asosiy kechikish vaqti (sekund)
    max_delay: float = 60.0           # Maksimal kechikish vaqti
    exponential_base: float = 2.0     # Eksponensial ortish koeffitsienti
    jitter: bool = True               # Tasodifiy kechikish qo'shish
    backoff_factor: float = 1.0       # Backoff koeffitsienti
    retry_on_exceptions: tuple = (Exception,)  # Qaysi xatolarda qayta urinish
    
class RetryExhaustedError(Exception):
    """Qayta urinish imkoniyatlari tugaganda"""
    def __init__(self, attempts: int, last_exception: Exception):
        self.attempts = attempts
        self.last_exception = last_exception
        super().__init__(f"Qayta urinish tugadi {attempts} urinishdan keyin: {last_exception}")

class RetryHandler:
    """Qayta urinish boshqaruvchisi"""
    
    def __init__(self, config: RetryConfig = None):
        self.config = config or RetryConfig()
        logger.info(f"RetryHandler yaratildi: max_retries={self.config.max_retries}")
    
    def calculate_delay(self, attempt: int) -> float:
        """Kechikish vaqtini hisoblash"""
        try:
            # Eksponensial backoff
            delay = self.config.base_delay * (
                self.config.exponential_base ** (attempt - 1)
            ) * self.config.backoff_factor
            
            # Maksimal kechikish chegarasi
            delay = min(delay, self.config.max_delay)
            
            # Jitter qo'shish (tasodifiy kechikish)
            if self.config.jitter:
                delay = delay * (0.5 + random.random() * 0.5)
            
            logger.debug(f"Urinish {attempt} uchun kechikish: {delay:.2f} sekund")
            return delay
            
        except Exception as e:
            logger.error(f"Kechikish hisoblashda xato: {e}")
            return self.config.base_delay
    
    def should_retry(self, exception: Exception, attempt: int) -> bool:
        """Qayta urinish kerakmi tekshirish"""
        try:
            # Maksimal urinish soni tekshirish
            if attempt >= self.config.max_retries:
                logger.warning(f"Maksimal urinish soni {self.config.max_retries} ga yetdi")
                return False
            
            # Xato turini tekshirish
            if not isinstance(exception, self.config.retry_on_exceptions):
                logger.warning(f"Xato turi qayta urinish uchun mos emas: {type(exception)}")
                return False
            
            # Maxsus xato holatlarini tekshirish
            if hasattr(exception, 'response') and exception.response is not None:
                status_code = getattr(exception.response, 'status_code', None)
                if status_code:
                    # 4xx xatolarda qayta urinmaslik (client error)
                    if 400 <= status_code < 500 and status_code != 429:
                        logger.warning(f"Client xato {status_code}, qayta urinish yo'q")
                        return False
            
            logger.info(f"Urinish {attempt}/{self.config.max_retries} qayta uriniladi")
            return True
            
        except Exception as e:
            logger.error(f"Qayta urinish tekshirishda xato: {e}")
            return False

    async def execute_async(self, func: Callable, *args, **kwargs) -> Any:
        """Async funksiyani qayta urinish bilan bajarish"""
        last_exception = None
        
        for attempt in range(1, self.config.max_retries + 1):
            try:
                logger.debug(f"Urinish {attempt}/{self.config.max_retries}: {func.__name__}")
                
                # Funksiyani bajarish
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                # Muvaffaqiyatli natija
                if attempt > 1:
                    logger.info(f"Muvaffaqiyat {attempt}-urinishda: {func.__name__}")
                
                return result
                
            except Exception as e:
                last_exception = e
                logger.warning(f"Urinish {attempt} xato: {e}")
                
                # Qayta urinish kerakmi tekshirish
                if not self.should_retry(e, attempt):
                    logger.error(f"Qayta urinish to'xtatildi: {e}")
                    raise e
                
                # Oxirgi urinish emas bo'lsa, kechikish
                if attempt < self.config.max_retries:
                    delay = self.calculate_delay(attempt)
                    logger.info(f"Kechikish: {delay:.2f} sekund")
                    await asyncio.sleep(delay)
        
        # Barcha urinishlar muvaffaqiyatsiz
        logger.error(f"Barcha urinishlar muvaffaqiyatsiz: {func.__name__}")
        raise RetryExhaustedError(self.config.max_retries, last_exception)

    def execute_sync(self, func: Callable, *args, **kwargs) -> Any:
        """Sync funksiyani qayta urinish bilan bajarish"""
        last_exception = None
        
        for attempt in range(1, self.config.max_retries + 1):
            try:
                logger.debug(f"Urinish {attempt}/{self.config.max_retries}: {func.__name__}")
                
                # Funksiyani bajarish
                result = func(*args, **kwargs)
                
                # Muvaffaqiyatli natija
                if attempt > 1:
                    logger.info(f"Muvaffaqiyat {attempt}-urinishda: {func.__name__}")
                
                return result
                
            except Exception as e:
                last_exception = e
                logger.warning(f"Urinish {attempt} xato: {e}")
                
                # Qayta urinish kerakmi tekshirish
                if not self.should_retry(e, attempt):
                    logger.error(f"Qayta urinish to'xtatildi: {e}")
                    raise e
                
                # Oxirgi urinish emas bo'lsa, kechikish
                if attempt < self.config.max_retries:
                    delay = self.calculate_delay(attempt)
                    logger.info(f"Kechikish: {delay:.2f} sekund")
                    time.sleep(delay)
        
        # Barcha urinishlar muvaffaqiyatsiz
        logger.error(f"Barcha urinishlar muvaffaqiyatsiz: {func.__name__}")
        raise RetryExhaustedError(self.config.max_retries, last_exception)

# Decorator funksiyalari
def retry_async(max_retries: int = 3, delay: float = 1.0, 
               exponential_base: float = 2.0, jitter: bool = True,
               retry_on: tuple = (Exception,)):
    """Async funksiyalar uchun qayta urinish decorator"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            config = RetryConfig(
                max_retries=max_retries,
                base_delay=delay,
                exponential_base=exponential_base,
                jitter=jitter,
                retry_on_exceptions=retry_on
            )
            handler = RetryHandler(config)
            return await handler.execute_async(func, *args, **kwargs)
        return wrapper
    return decorator

def retry_sync(max_retries: int = 3, delay: float = 1.0,
               exponential_base: float = 2.0, jitter: bool = True,
               retry_on: tuple = (Exception,)):
    """Sync funksiyalar uchun qayta urinish decorator"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            config = RetryConfig(
                max_retries=max_retries,
                base_delay=delay,
                exponential_base=exponential_base,
                jitter=jitter,
                retry_on_exceptions=retry_on
            )
            handler = RetryHandler(config)
            return handler.execute_sync(func, *args, **kwargs)
        return wrapper
    return decorator

# Maxsus xato turlari uchun konfiguratsiyalar
API_RETRY_CONFIG = RetryConfig(
    max_retries=3,
    base_delay=1.0,
    max_delay=30.0,
    exponential_base=2.0,
    jitter=True,
    retry_on_exceptions=(
        ConnectionError,
        TimeoutError,
        # HTTP xatolari
        Exception,  # Umumiy xato
    )
)

DATABASE_RETRY_CONFIG = RetryConfig(
    max_retries=5,
    base_delay=0.5,
    max_delay=10.0,
    exponential_base=1.5,
    jitter=True,
    retry_on_exceptions=(
        ConnectionError,
        TimeoutError,
        Exception,
    )
)

TRADING_RETRY_CONFIG = RetryConfig(
    max_retries=2,
    base_delay=0.5,
    max_delay=5.0,
    exponential_base=1.5,
    jitter=False,  # Trading uchun jitter yo'q
    retry_on_exceptions=(
        ConnectionError,
        TimeoutError,
    )
)

# Foydalanish misollari
async def example_async_function():
    """Async funksiya misoli"""
    
    @retry_async(max_retries=3, delay=1.0)
    async def api_call():
        # API so'rovi
        logger.info("API so'rovi yuborilmoqda...")
        # Xato simulatsiyasi
        if random.random() < 0.7:
            raise ConnectionError("API ulanish xatosi")
        return {"status": "success"}
    
    try:
        result = await api_call()
        logger.info(f"Natija: {result}")
        return result
    except RetryExhaustedError as e:
        logger.error(f"API so'rovi muvaffaqiyatsiz: {e}")
        raise

def example_sync_function():
    """Sync funksiya misoli"""
    
    @retry_sync(max_retries=3, delay=1.0)
    def database_query():
        # Database so'rovi
        logger.info("Database so'rovi...")
        # Xato simulatsiyasi
        if random.random() < 0.5:
            raise ConnectionError("Database ulanish xatosi")
        return {"data": "success"}
    
    try:
        result = database_query()
        logger.info(f"Natija: {result}")
        return result
    except RetryExhaustedError as e:
        logger.error(f"Database so'rovi muvaffaqiyatsiz: {e}")
        raise

# Manual retry handler foydalanish
async def manual_retry_example():
    """Manual retry handler misoli"""
    
    config = RetryConfig(
        max_retries=5,
        base_delay=1.0,
        exponential_base=2.0,
        jitter=True
    )
    
    handler = RetryHandler(config)
    
    async def risky_operation():
        logger.info("Xavfli operatsiya bajarilmoqda...")
        if random.random() < 0.8:
            raise Exception("Operatsiya xatosi")
        return "Muvaffaqiyat!"
    
    try:
        result = await handler.execute_async(risky_operation)
        logger.info(f"Natija: {result}")
        return result
    except RetryExhaustedError as e:
        logger.error(f"Operatsiya muvaffaqiyatsiz: {e}")
        return None

# Utility funksiyalar
def create_api_retry_handler() -> RetryHandler:
    """API uchun retry handler yaratish"""
    return RetryHandler(API_RETRY_CONFIG)

def create_database_retry_handler() -> RetryHandler:
    """Database uchun retry handler yaratish"""
    return RetryHandler(DATABASE_RETRY_CONFIG)

def create_trading_retry_handler() -> RetryHandler:
    """Trading uchun retry handler yaratish"""
    return RetryHandler(TRADING_RETRY_CONFIG)

if __name__ == "__main__":
    # Test
    import asyncio
    
    async def test_retry_handler():
        """Retry handler testlari"""
        logger.info("=== Retry Handler Testlari ===")
        
        # Async test
        logger.info("1. Async funksiya testi...")
        try:
            await example_async_function()
        except Exception as e:
            logger.error(f"Async test xato: {e}")
        
        # Sync test
        logger.info("2. Sync funksiya testi...")
        try:
            example_sync_function()
        except Exception as e:
            logger.error(f"Sync test xato: {e}")
        
        # Manual test
        logger.info("3. Manual retry test...")
        try:
            await manual_retry_example()
        except Exception as e:
            logger.error(f"Manual test xato: {e}")
        
        logger.info("=== Testlar tugadi ===")
    
    # Testlarni ishga tushirish
    asyncio.run(test_retry_handler())

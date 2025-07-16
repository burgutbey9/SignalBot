"""
Error Handling Tizimi
====================

Bu modul barcha bot komponentlari uchun xatoliklarni boshqarish va qayta ishlash tizimini o'rnatadi.
API xatoliklari, ma'lumot qayta ishlash xatoliklari, va umumiy xatoliklarni boshqaradi.

Author: AI OrderFlow Signal Bot
Version: 1.0.0
"""

import sys
import traceback
import functools
import asyncio
import time
from typing import Dict, Any, Optional, Callable, Union, List, Type
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
from datetime import datetime, timedelta
import threading
from contextlib import contextmanager

# Logger import qilish
from .logger import get_logger

# Logger yaratish
logger = get_logger(__name__)

class ErrorType(Enum):
    """Xato turlari"""
    API_ERROR = "api_error"
    NETWORK_ERROR = "network_error"
    RATE_LIMIT_ERROR = "rate_limit_error"
    AUTHENTICATION_ERROR = "auth_error"
    VALIDATION_ERROR = "validation_error"
    PROCESSING_ERROR = "processing_error"
    DATABASE_ERROR = "database_error"
    TRADING_ERROR = "trading_error"
    CONFIGURATION_ERROR = "config_error"
    UNKNOWN_ERROR = "unknown_error"

class ErrorSeverity(Enum):
    """Xato darajasi"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ErrorInfo:
    """Xato ma'lumotlari"""
    error_type: ErrorType
    severity: ErrorSeverity
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    module: Optional[str] = None
    function: Optional[str] = None
    line_number: Optional[int] = None
    traceback: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    max_retries: int = 3
    retry_delay: float = 1.0
    resolved: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """ErrorInfo ni dictionary ga aylantirish"""
        return {
            'error_type': self.error_type.value,
            'severity': self.severity.value,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'module': self.module,
            'function': self.function,
            'line_number': self.line_number,
            'traceback': self.traceback,
            'context': self.context,
            'retry_count': self.retry_count,
            'max_retries': self.max_retries,
            'retry_delay': self.retry_delay,
            'resolved': self.resolved
        }
    
    def to_telegram_message(self) -> str:
        """Telegram uchun xato xabarini yaratish"""
        emoji_map = {
            ErrorSeverity.LOW: "🟡",
            ErrorSeverity.MEDIUM: "🟠", 
            ErrorSeverity.HIGH: "🔴",
            ErrorSeverity.CRITICAL: "🚨"
        }
        
        emoji = emoji_map.get(self.severity, "⚠️")
        
        message = f"""{emoji} XATO YUZAGA KELDI
════════════════════════
📊 Tur: {self.error_type.value}
📈 Daraja: {self.severity.value}
💬 Xabar: {self.message}
📂 Modul: {self.module or 'Noma\'lum'}
🔧 Funksiya: {self.function or 'Noma\'lum'}
📍 Qator: {self.line_number or 'Noma\'lum'}
🔄 Qayta urinish: {self.retry_count}/{self.max_retries}
⏰ Vaqt: {self.timestamp.strftime('%H:%M:%S')}
════════════════════════"""
        
        if self.context:
            message += f"\n📝 Context: {json.dumps(self.context, indent=2, ensure_ascii=False)}"
            
        return message

class ErrorStats:
    """Xato statistikalari"""
    
    def __init__(self):
        self.errors: List[ErrorInfo] = []
        self.error_counts: Dict[ErrorType, int] = {}
        self.last_errors: Dict[ErrorType, datetime] = {}
        self.lock = threading.Lock()
    
    def add_error(self, error_info: ErrorInfo):
        """Xato qo'shish"""
        with self.lock:
            self.errors.append(error_info)
            
            # Hisoblagichni yangilash
            if error_info.error_type not in self.error_counts:
                self.error_counts[error_info.error_type] = 0
            self.error_counts[error_info.error_type] += 1
            
            # Oxirgi xato vaqtini yangilash
            self.last_errors[error_info.error_type] = error_info.timestamp
            
            # Eski xatolarni o'chirish (24 soat)
            cutoff_time = datetime.now() - timedelta(hours=24)
            self.errors = [e for e in self.errors if e.timestamp > cutoff_time]
    
    def get_stats(self) -> Dict[str, Any]:
        """Statistikalarni olish"""
        with self.lock:
            now = datetime.now()
            last_hour = now - timedelta(hours=1)
            last_day = now - timedelta(hours=24)
            
            # Soatlik va kunlik xatolar
            hourly_errors = [e for e in self.errors if e.timestamp > last_hour]
            daily_errors = [e for e in self.errors if e.timestamp > last_day]
            
            return {
                'total_errors': len(self.errors),
                'hourly_errors': len(hourly_errors),
                'daily_errors': len(daily_errors),
                'error_counts': dict(self.error_counts),
                'last_errors': {
                    error_type.value: timestamp.isoformat()
                    for error_type, timestamp in self.last_errors.items()
                },
                'error_types_last_hour': {
                    error_type.value: len([e for e in hourly_errors if e.error_type == error_type])
                    for error_type in ErrorType
                }
            }

class ErrorHandler:
    """Asosiy xato boshqarish classi"""
    
    def __init__(self, telegram_client=None):
        self.telegram_client = telegram_client
        self.stats = ErrorStats()
        self.error_callbacks: Dict[ErrorType, List[Callable]] = {}
        self.global_callbacks: List[Callable] = []
        self.suppress_telegram = False
        self.max_telegram_errors_per_hour = 10
        self.telegram_error_count = 0
        self.telegram_reset_time = datetime.now()
        
        logger.info("ErrorHandler ishga tushirildi")
    
    def set_telegram_client(self, telegram_client):
        """Telegram client ni o'rnatish"""
        self.telegram_client = telegram_client
        logger.info("Telegram client ErrorHandler ga o'rnatildi")
    
    def add_callback(self, error_type: Optional[ErrorType], callback: Callable):
        """Xato uchun callback qo'shish"""
        if error_type is None:
            self.global_callbacks.append(callback)
        else:
            if error_type not in self.error_callbacks:
                self.error_callbacks[error_type] = []
            self.error_callbacks[error_type].append(callback)
    
    def _determine_severity(self, error_type: ErrorType, exception: Exception) -> ErrorSeverity:
        """Xato darajasini aniqlash"""
        # Critical xatolar
        if error_type in [ErrorType.TRADING_ERROR, ErrorType.DATABASE_ERROR]:
            return ErrorSeverity.CRITICAL
        
        # High xatolar
        if error_type in [ErrorType.AUTHENTICATION_ERROR, ErrorType.CONFIGURATION_ERROR]:
            return ErrorSeverity.HIGH
        
        # Medium xatolar
        if error_type in [ErrorType.API_ERROR, ErrorType.PROCESSING_ERROR]:
            return ErrorSeverity.MEDIUM
        
        # Low xatolar
        return ErrorSeverity.LOW
    
    def _extract_error_info(self, exception: Exception, error_type: ErrorType, context: Dict[str, Any] = None) -> ErrorInfo:
        """Exception dan ErrorInfo yaratish"""
        # Traceback ma'lumotlarini olish
        tb = traceback.extract_tb(exception.__traceback__)
        if tb:
            last_frame = tb[-1]
            module = last_frame.filename
            function = last_frame.name
            line_number = last_frame.lineno
        else:
            module = None
            function = None
            line_number = None
        
        # Severity aniqlash
        severity = self._determine_severity(error_type, exception)
        
        return ErrorInfo(
            error_type=error_type,
            severity=severity,
            message=str(exception),
            module=module,
            function=function,
            line_number=line_number,
            traceback=traceback.format_exc(),
            context=context or {}
        )
    
    async def _send_to_telegram(self, error_info: ErrorInfo):
        """Telegram ga xato yuborish"""
        if not self.telegram_client or self.suppress_telegram:
            return
        
        # Rate limiting
        now = datetime.now()
        if now - self.telegram_reset_time > timedelta(hours=1):
            self.telegram_error_count = 0
            self.telegram_reset_time = now
        
        if self.telegram_error_count >= self.max_telegram_errors_per_hour:
            return
        
        try:
            # Faqat HIGH va CRITICAL xatolarni yuborish
            if error_info.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
                message = error_info.to_telegram_message()
                await self.telegram_client.send_message(message)
                self.telegram_error_count += 1
                
        except Exception as e:
            logger.error(f"Telegram ga xato yuborishda xato: {e}")
    
    def _run_callbacks(self, error_info: ErrorInfo):
        """Callback functionlarni ishga tushirish"""
        # Global callbacklar
        for callback in self.global_callbacks:
            try:
                callback(error_info)
            except Exception as e:
                logger.error(f"Global callback xatosi: {e}")
        
        # Specific callbacklar
        if error_info.error_type in self.error_callbacks:
            for callback in self.error_callbacks[error_info.error_type]:
                try:
                    callback(error_info)
                except Exception as e:
                    logger.error(f"Specific callback xatosi: {e}")
    
    async def handle_error(self, 
                          exception: Exception, 
                          error_type: ErrorType, 
                          context: Dict[str, Any] = None,
                          suppress_telegram: bool = False) -> ErrorInfo:
        """Xatoni boshqarish"""
        # ErrorInfo yaratish
        error_info = self._extract_error_info(exception, error_type, context)
        
        # Log yozish
        log_message = f"[{error_type.value}] {error_info.message}"
        if error_info.severity == ErrorSeverity.CRITICAL:
            logger.critical(log_message, extra={'error_info': error_info.to_dict()})
        elif error_info.severity == ErrorSeverity.HIGH:
            logger.error(log_message, extra={'error_info': error_info.to_dict()})
        elif error_info.severity == ErrorSeverity.MEDIUM:
            logger.warning(log_message, extra={'error_info': error_info.to_dict()})
        else:
            logger.info(log_message, extra={'error_info': error_info.to_dict()})
        
        # Statistikaga qo'shish
        self.stats.add_error(error_info)
        
        # Telegram ga yuborish
        if not suppress_telegram:
            await self._send_to_telegram(error_info)
        
        # Callbacklarni ishga tushirish
        self._run_callbacks(error_info)
        
        return error_info
    
    def get_stats(self) -> Dict[str, Any]:
        """Xato statistikalarini olish"""
        return self.stats.get_stats()
    
    def clear_stats(self):
        """Statistikalarni tozalash"""
        self.stats = ErrorStats()
        logger.info("Xato statistikalari tozalandi")
    
    def suppress_telegram_errors(self, suppress: bool = True):
        """Telegram xatolarini to'xtatish"""
        self.suppress_telegram = suppress
        logger.info(f"Telegram xatolari {'to\'xtatildi' if suppress else 'yoqildi'}")

# Global error handler instance
_error_handler = ErrorHandler()

# Decorator functionlar
def handle_api_error(error_type: ErrorType = ErrorType.API_ERROR, 
                    context: Dict[str, Any] = None,
                    suppress_telegram: bool = False):
    """API xatolarini boshqarish decoratori"""
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    error_info = await _error_handler.handle_error(
                        e, error_type, context, suppress_telegram
                    )
                    raise ErrorHandledException(error_info) from e
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    # Sync function uchun async handle_error ni ishlatish
                    loop = asyncio.get_event_loop()
                    error_info = loop.run_until_complete(
                        _error_handler.handle_error(e, error_type, context, suppress_telegram)
                    )
                    raise ErrorHandledException(error_info) from e
            return sync_wrapper
    return decorator

def handle_processing_error(context: Dict[str, Any] = None):
    """Ma'lumot qayta ishlash xatolarini boshqarish"""
    return handle_api_error(ErrorType.PROCESSING_ERROR, context)

def handle_trading_error(context: Dict[str, Any] = None):
    """Trading xatolarini boshqarish"""
    return handle_api_error(ErrorType.TRADING_ERROR, context)

def handle_database_error(context: Dict[str, Any] = None):
    """Database xatolarini boshqarish"""
    return handle_api_error(ErrorType.DATABASE_ERROR, context)

# Custom Exception
class ErrorHandledException(Exception):
    """Boshqarilgan xato exception"""
    
    def __init__(self, error_info: ErrorInfo):
        self.error_info = error_info
        super().__init__(error_info.message)

# Context manager
@contextmanager
def error_context(error_type: ErrorType, context: Dict[str, Any] = None):
    """Xato context manager"""
    try:
        yield
    except Exception as e:
        loop = asyncio.get_event_loop()
        error_info = loop.run_until_complete(
            _error_handler.handle_error(e, error_type, context)
        )
        raise ErrorHandledException(error_info) from e

# Utility functions
def set_telegram_client(telegram_client):
    """Telegram client ni o'rnatish"""
    _error_handler.set_telegram_client(telegram_client)

def add_error_callback(error_type: Optional[ErrorType], callback: Callable):
    """Xato callback qo'shish"""
    _error_handler.add_callback(error_type, callback)

def get_error_stats():
    """Xato statistikalarini olish"""
    return _error_handler.get_stats()

def clear_error_stats():
    """Xato statistikalarini tozalash"""
    _error_handler.clear_stats()

def suppress_telegram_errors(suppress: bool = True):
    """Telegram xatolarini to'xtatish"""
    _error_handler.suppress_telegram_errors(suppress)

# Maxsus xato handler functionlar
async def handle_rate_limit_error(exception: Exception, 
                                 retry_after: Optional[int] = None,
                                 context: Dict[str, Any] = None) -> ErrorInfo:
    """Rate limit xatosini boshqarish"""
    if context is None:
        context = {}
    
    if retry_after:
        context['retry_after'] = retry_after
    
    return await _error_handler.handle_error(
        exception, ErrorType.RATE_LIMIT_ERROR, context
    )

async def handle_network_error(exception: Exception, 
                              url: Optional[str] = None,
                              method: Optional[str] = None,
                              context: Dict[str, Any] = None) -> ErrorInfo:
    """Network xatosini boshqarish"""
    if context is None:
        context = {}
    
    if url:
        context['url'] = url
    if method:
        context['method'] = method
    
    return await _error_handler.handle_error(
        exception, ErrorType.NETWORK_ERROR, context
    )

# Test funksiyasi
async def test_error_handler():
    """Error handler ni test qilish"""
    print("🔧 Error handler test qilinmoqda...")
    
    try:
        # Test xato yaratish
        test_exception = ValueError("Test xato")
        
        # Xatoni boshqarish
        error_info = await _error_handler.handle_error(
            test_exception, 
            ErrorType.API_ERROR,
            {'test': True, 'operation': 'test_function'}
        )
        
        print(f"✅ Xato muvaffaqiyatli boshqarildi: {error_info.error_type}")
        
        # Statistikalarni ko'rsatish
        stats = get_error_stats()
        print(f"📊 Jami xatolar: {stats['total_errors']}")
        
        # Decorator test
        @handle_api_error(ErrorType.PROCESSING_ERROR)
        async def test_function():
            raise RuntimeError("Decorator test xato")
        
        try:
            await test_function()
        except ErrorHandledException as e:
            print(f"✅ Decorator xato muvaffaqiyatli boshqarildi: {e.error_info.error_type}")
        
        print("🎉 Error handler test muvaffaqiyatli!")
        return True
        
    except Exception as e:
        print(f"❌ Error handler test xatosi: {e}")
        return False

# Modulni import qilganda
if __name__ == "__main__":
    asyncio.run(test_error_handler())

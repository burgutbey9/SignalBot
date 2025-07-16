"""
Logger Setup va Formatlar
========================

Bu modul barcha bot komponentlari uchun logging tizimini o'rnatadi.
Fayl va console logging, log rotation, va turli xil log darajalarini qo'llab-quvvatlaydi.

Author: AI OrderFlow Signal Bot
Version: 1.0.0
"""

import logging
import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from datetime import datetime
import json
import threading
from dataclasses import dataclass
import colorama
from colorama import Fore, Back, Style

# Colorama ni ishga tushirish
colorama.init(autoreset=True)

@dataclass
class LoggerConfig:
    """Logger konfiguratsiya parametrlari"""
    name: str
    level: int = logging.INFO
    log_file: Optional[str] = None
    max_bytes: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    console_output: bool = True
    file_output: bool = True
    use_colors: bool = True
    format_string: Optional[str] = None
    date_format: Optional[str] = None
    
class ColoredFormatter(logging.Formatter):
    """Rangli console output uchun formatter"""
    
    # Rang kodlari
    COLORS = {
        'DEBUG': Fore.CYAN,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Back.WHITE + Style.BRIGHT,
    }
    
    def format(self, record):
        """Log record ni rangli formatga aylantirish"""
        # Asosiy formatni qo'llash
        formatted = super().format(record)
        
        # Rang qo'shish
        if record.levelname in self.COLORS:
            color = self.COLORS[record.levelname]
            formatted = f"{color}{formatted}{Style.RESET_ALL}"
        
        return formatted

class TelegramLogHandler(logging.Handler):
    """Telegram ga log yuborish uchun handler"""
    
    def __init__(self, telegram_client=None):
        super().__init__()
        self.telegram_client = telegram_client
        self.buffer = []
        self.buffer_lock = threading.Lock()
        
    def emit(self, record):
        """Log record ni Telegram ga yuborish"""
        if not self.telegram_client:
            return
            
        try:
            # Faqat ERROR va CRITICAL loglarni Telegram ga yuborish
            if record.levelno >= logging.ERROR:
                message = self.format(record)
                
                with self.buffer_lock:
                    self.buffer.append({
                        'timestamp': datetime.now().isoformat(),
                        'level': record.levelname,
                        'message': message,
                        'module': record.name
                    })
                
                # Telegram ga yuborish (async)
                # Bu real implementatsiyada telegram_client orqali amalga oshiriladi
                
        except Exception:
            # Logging xatolarini yashirish
            pass

class JSONFormatter(logging.Formatter):
    """JSON formatida log yozish uchun formatter"""
    
    def format(self, record):
        """Log record ni JSON formatga aylantirish"""
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'thread': record.thread,
            'thread_name': record.threadName,
        }
        
        # Exception ma'lumotlarini qo'shish
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
            
        # Qo'shimcha ma'lumotlarni qo'shish
        if hasattr(record, 'extra_data'):
            log_entry.update(record.extra_data)
            
        return json.dumps(log_entry, ensure_ascii=False, indent=None)

class BotLogger:
    """Bot uchun asosiy logger class"""
    
    def __init__(self):
        self.loggers: Dict[str, logging.Logger] = {}
        self.config = self._load_config()
        self.telegram_handler = None
        self._setup_root_logger()
        
    def _load_config(self) -> Dict[str, Any]:
        """Logger konfiguratsiyasini yuklash"""
        default_config = {
            'log_dir': 'logs',
            'default_level': 'INFO',
            'max_file_size': 10 * 1024 * 1024,  # 10MB
            'backup_count': 5,
            'console_colors': True,
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'date_format': '%Y-%m-%d %H:%M:%S',
            'json_logs': False,
            'telegram_errors': True
        }
        
        try:
            # Config fayldan o'qishni qo'shish mumkin
            config_file = Path('config/logger_config.json')
            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f:
                    file_config = json.load(f)
                    default_config.update(file_config)
        except Exception:
            pass  # Default config ishlatish
            
        return default_config
    
    def _setup_root_logger(self):
        """Root logger ni o'rnatish"""
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        
        # Mavjud handlerlarni o'chirish
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
    
    def _ensure_log_dir(self, log_dir: str) -> Path:
        """Log direktoriyasini yaratish"""
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        return log_path
    
    def _get_level(self, level_str: str) -> int:
        """String level ni int ga aylantirish"""
        levels = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }
        return levels.get(level_str.upper(), logging.INFO)
    
    def create_logger(self, name: str, config: Optional[LoggerConfig] = None) -> logging.Logger:
        """Yangi logger yaratish"""
        # Agar logger allaqachon mavjud bo'lsa, uni qaytarish
        if name in self.loggers:
            return self.loggers[name]
        
        # Default konfiguratsiya
        if config is None:
            config = LoggerConfig(
                name=name,
                level=self._get_level(self.config['default_level']),
                log_file=f"{self.config['log_dir']}/{name}.log",
                max_bytes=self.config['max_file_size'],
                backup_count=self.config['backup_count'],
                format_string=self.config['format'],
                date_format=self.config['date_format']
            )
        
        # Logger yaratish
        logger = logging.getLogger(name)
        logger.setLevel(config.level)
        
        # Mavjud handlerlarni o'chirish
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Formatter yaratish
        if self.config['json_logs']:
            formatter = JSONFormatter()
        else:
            formatter = logging.Formatter(
                config.format_string or self.config['format'],
                config.date_format or self.config['date_format']
            )
        
        # Console handler qo'shish
        if config.console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(config.level)
            
            if config.use_colors and self.config['console_colors']:
                console_formatter = ColoredFormatter(
                    config.format_string or self.config['format'],
                    config.date_format or self.config['date_format']
                )
                console_handler.setFormatter(console_formatter)
            else:
                console_handler.setFormatter(formatter)
            
            logger.addHandler(console_handler)
        
        # File handler qo'shish
        if config.file_output and config.log_file:
            log_dir = self._ensure_log_dir(Path(config.log_file).parent)
            
            file_handler = RotatingFileHandler(
                config.log_file,
                maxBytes=config.max_bytes,
                backupCount=config.backup_count,
                encoding='utf-8'
            )
            file_handler.setLevel(config.level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        # Telegram handler qo'shish
        if self.config['telegram_errors'] and self.telegram_handler:
            logger.addHandler(self.telegram_handler)
        
        # Logger ni saqlash
        self.loggers[name] = logger
        
        # Yaratilganlik haqida log
        logger.info(f"Logger '{name}' yaratildi va ishga tushirildi")
        
        return logger
    
    def set_telegram_handler(self, telegram_client):
        """Telegram handler ni o'rnatish"""
        self.telegram_handler = TelegramLogHandler(telegram_client)
        self.telegram_handler.setLevel(logging.ERROR)
        
        # Mavjud loggerlar uchun telegram handler qo'shish
        for logger in self.loggers.values():
            if self.config['telegram_errors']:
                logger.addHandler(self.telegram_handler)
    
    def get_logger(self, name: str) -> logging.Logger:
        """Logger ni olish yoki yaratish"""
        if name not in self.loggers:
            return self.create_logger(name)
        return self.loggers[name]
    
    def set_level(self, name: str, level: str):
        """Logger darajasini o'rnatish"""
        if name in self.loggers:
            self.loggers[name].setLevel(self._get_level(level))
    
    def flush_logs(self):
        """Barcha loglarni flush qilish"""
        for logger in self.loggers.values():
            for handler in logger.handlers:
                handler.flush()
    
    def get_log_stats(self) -> Dict[str, Any]:
        """Log statistikalarini olish"""
        stats = {
            'total_loggers': len(self.loggers),
            'loggers': {},
            'config': self.config
        }
        
        for name, logger in self.loggers.items():
            stats['loggers'][name] = {
                'level': logging.getLevelName(logger.level),
                'handlers': len(logger.handlers),
                'effective_level': logging.getLevelName(logger.getEffectiveLevel())
            }
        
        return stats

# Global logger instance
_bot_logger = BotLogger()

# Asosiy funksiyalar
def get_logger(name: str) -> logging.Logger:
    """Logger ni olish - asosiy funksiya"""
    return _bot_logger.get_logger(name)

def setup_logger(name: str, 
                level: str = 'INFO',
                log_file: Optional[str] = None,
                console_output: bool = True,
                use_colors: bool = True) -> logging.Logger:
    """Logger ni sozlash - sodda interfeys"""
    config = LoggerConfig(
        name=name,
        level=_bot_logger._get_level(level),
        log_file=log_file,
        console_output=console_output,
        use_colors=use_colors
    )
    
    return _bot_logger.create_logger(name, config)

def set_telegram_handler(telegram_client):
    """Telegram handler ni o'rnatish"""
    _bot_logger.set_telegram_handler(telegram_client)

def set_log_level(name: str, level: str):
    """Logger darajasini o'rnatish"""
    _bot_logger.set_level(name, level)

def flush_all_logs():
    """Barcha loglarni flush qilish"""
    _bot_logger.flush_logs()

def get_log_statistics():
    """Log statistikalarini olish"""
    return _bot_logger.get_log_stats()

# Contextual logger
class LogContext:
    """Log context manager"""
    
    def __init__(self, logger: logging.Logger, extra_data: Dict[str, Any]):
        self.logger = logger
        self.extra_data = extra_data
        self.old_extra = getattr(logger, '_extra_data', {})
    
    def __enter__(self):
        self.logger._extra_data = {**self.old_extra, **self.extra_data}
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger._extra_data = self.old_extra

def with_context(logger: logging.Logger, **kwargs) -> LogContext:
    """Logger uchun context yaratish"""
    return LogContext(logger, kwargs)

# Test va debug funksiyalari
def test_logger_system():
    """Logger tizimini test qilish"""
    print("🔧 Logger tizimi test qilinmoqda...")
    
    try:
        # Test logger yaratish
        test_logger = get_logger("test_logger")
        
        # Turli xil log darajalarini test qilish
        test_logger.debug("Bu DEBUG xabar")
        test_logger.info("Bu INFO xabar")
        test_logger.warning("Bu WARNING xabar")
        test_logger.error("Bu ERROR xabar")
        test_logger.critical("Bu CRITICAL xabar")
        
        # Context bilan test
        with with_context(test_logger, user_id=12345, operation="test"):
            test_logger.info("Context bilan log")
        
        print("✅ Logger tizimi muvaffaqiyatli test qilindi")
        
        # Statistikalarni ko'rsatish
        stats = get_log_statistics()
        print(f"📊 Jami loggerlar: {stats['total_loggers']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Logger test xatosi: {e}")
        return False

# Modulni import qilganda
if __name__ == "__main__":
    test_logger_system()

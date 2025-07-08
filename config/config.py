"""
AI OrderFlow & Signal Bot - Asosiy konfiguratsiya moduli
Barcha tizim sozlamalari va konfiguratsiyalarni boshqaradi
"""

import os
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime, time
import pytz

# Logger import qilish (keyinroq yaratiladi)
# from utils.logger import get_logger
# logger = get_logger(__name__)

@dataclass
class DatabaseConfig:
    """Database konfiguratsiyasi"""
    url: str = "sqlite:///data/bot.db"
    max_connections: int = 10
    timeout: int = 30
    pool_size: int = 20
    echo: bool = False

@dataclass
class APIConfig:
    """API konfiguratsiyasi"""
    timeout: int = 30
    max_retries: int = 3
    rate_limit: int = 100
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff_factor: float = 2.0

@dataclass
class TradingConfig:
    """Trading konfiguratsiyasi"""
    max_risk_per_trade: float = 0.02
    max_daily_loss: float = 0.05
    max_daily_trades: int = 20
    position_size_method: str = "kelly"
    min_confidence: float = 0.75
    
    # Propshot EA sozlamalari
    propshot_max_daily_loss: float = 0.025
    propshot_max_total_loss: float = 0.05
    propshot_max_lot_size: float = 0.5
    propshot_max_daily_trades: int = 3
    
    # Stop Loss va Take Profit
    default_sl_pips: int = 20
    default_tp_pips: int = 40
    max_sl_pips: int = 50
    max_tp_pips: int = 100

@dataclass
class TelegramConfig:
    """Telegram bot konfiguratsiyasi"""
    bot_token: str = ""
    chat_id: str = ""
    channel_id: str = ""
    admin_ids: List[str] = field(default_factory=list)
    message_format: str = "markdown"
    auto_delete_minutes: int = 60

@dataclass
class WorkingHours:
    """Ish vaqti konfiguratsiyasi"""
    timezone: str = "Asia/Tashkent"
    start_time: time = time(7, 0)  # 07:00
    end_time: time = time(19, 30)  # 19:30
    trading_days: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])  # Monday to Friday
    break_start: time = time(12, 0)  # 12:00
    break_end: time = time(13, 0)    # 13:00

@dataclass
class FallbackConfig:
    """Fallback tizimi konfiguratsiyasi"""
    order_flow_priority: List[str] = field(default_factory=lambda: ["oneinch", "thegraph", "alchemy"])
    sentiment_priority: List[str] = field(default_factory=lambda: ["huggingface", "gemini", "claude"])
    news_priority: List[str] = field(default_factory=lambda: ["newsapi", "reddit", "claude"])
    
    # Fallback timeout (soniya)
    fallback_timeout: int = 30
    max_fallback_attempts: int = 3
    cooldown_period: int = 300  # 5 minut

class ConfigManager:
    """Konfiguratsiya boshqaruvchi sinf"""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.settings_file = self.config_dir / "settings.json"
        self.env_file = Path(".env")
        
        # Konfiguratsiya ob'ektlari
        self.database: DatabaseConfig = DatabaseConfig()
        self.api: APIConfig = APIConfig()
        self.trading: TradingConfig = TradingConfig()
        self.telegram: TelegramConfig = TelegramConfig()
        self.working_hours: WorkingHours = WorkingHours()
        self.fallback: FallbackConfig = FallbackConfig()
        
        # API kalitlar
        self.api_keys: Dict[str, str] = {}
        self.api_limits: Dict[str, Dict[str, Any]] = {}
        
        # Konfiguratsiyani yuklash
        self.load_config()
    
    def load_config(self) -> None:
        """Barcha konfiguratsiya fayllarini yuklash"""
        try:
            print("Konfiguratsiya yuklanyapti...")
            
            # .env faylini yuklash
            self._load_env_file()
            
            # JSON settings faylini yuklash
            self._load_json_settings()
            
            # Konfiguratsiyani validatsiya qilish
            self._validate_config()
            
            print("Konfiguratsiya muvaffaqiyatli yuklandi")
            
        except Exception as e:
            print(f"Konfiguratsiya yuklashda xato: {e}")
            raise ConfigurationError(f"Konfiguratsiya yuklashda xato: {e}")
    
    def _load_env_file(self) -> None:
        """Environment o'zgaruvchilarini yuklash"""
        try:
            if self.env_file.exists():
                with open(self.env_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            key, value = line.split('=', 1)
                            os.environ[key.strip()] = value.strip().strip('"\'')
            
            # API kalitlarini yuklash
            self.api_keys = {
                'oneinch': os.getenv('ONEINCH_API_KEY', ''),
                'alchemy': os.getenv('ALCHEMY_API_KEY', ''),
                'huggingface': os.getenv('HUGGINGFACE_API_KEY', ''),
                'gemini_keys': [
                    os.getenv('GEMINI_API_KEY_1', ''),
                    os.getenv('GEMINI_API_KEY_2', ''),
                    os.getenv('GEMINI_API_KEY_3', ''),
                    os.getenv('GEMINI_API_KEY_4', ''),
                    os.getenv('GEMINI_API_KEY_5', '')
                ],
                'claude': os.getenv('CLAUDE_API_KEY', ''),
                'news_api': os.getenv('NEWS_API_KEY', ''),
                'reddit_client_id': os.getenv('REDDIT_CLIENT_ID', ''),
                'reddit_client_secret': os.getenv('REDDIT_CLIENT_SECRET', ''),
                'telegram_bot_token': os.getenv('TELEGRAM_BOT_TOKEN', ''),
                'telegram_chat_id': os.getenv('TELEGRAM_CHAT_ID', ''),
                'propshot_api': os.getenv('PROPSHOT_API_KEY', ''),
                'mt5_server': os.getenv('MT5_SERVER', ''),
                'mt5_login': os.getenv('MT5_LOGIN', ''),
                'mt5_password': os.getenv('MT5_PASSWORD', '')
            }
            
            # Database URL
            if os.getenv('DATABASE_URL'):
                self.database.url = os.getenv('DATABASE_URL')
            
            # Telegram konfiguratsiyasi
            self.telegram.bot_token = self.api_keys['telegram_bot_token']
            self.telegram.chat_id = self.api_keys['telegram_chat_id']
            
        except Exception as e:
            raise ConfigurationError(f"Environment faylini yuklashda xato: {e}")
    
    def _load_json_settings(self) -> None:
        """JSON settings faylini yuklash"""
        try:
            if self.settings_file.exists():
                with open(self.settings_file, 'r', encoding='utf-8') as f:
                    settings = json.load(f)
                
                # API limitlarini yuklash
                if 'api_limits' in settings:
                    self.api_limits = settings['api_limits']
                
                # Fallback tartibini yuklash
                if 'fallback_order' in settings:
                    fallback_data = settings['fallback_order']
                    self.fallback.order_flow_priority = fallback_data.get('order_flow', self.fallback.order_flow_priority)
                    self.fallback.sentiment_priority = fallback_data.get('sentiment', self.fallback.sentiment_priority)
                    self.fallback.news_priority = fallback_data.get('news', self.fallback.news_priority)
                
                # Trading sozlamalarini yuklash
                if 'trading' in settings:
                    trading_data = settings['trading']
                    self.trading.max_risk_per_trade = trading_data.get('max_risk_per_trade', self.trading.max_risk_per_trade)
                    self.trading.max_daily_loss = trading_data.get('max_daily_loss', self.trading.max_daily_loss)
                    self.trading.position_size_method = trading_data.get('position_size_method', self.trading.position_size_method)
                    
                    # Propshot sozlamalari
                    if 'propshot_settings' in trading_data:
                        propshot = trading_data['propshot_settings']
                        self.trading.propshot_max_daily_loss = propshot.get('max_daily_loss', self.trading.propshot_max_daily_loss)
                        self.trading.propshot_max_total_loss = propshot.get('max_total_loss', self.trading.propshot_max_total_loss)
                        self.trading.propshot_max_lot_size = propshot.get('max_lot_size', self.trading.propshot_max_lot_size)
                        self.trading.propshot_max_daily_trades = propshot.get('max_daily_trades', self.trading.propshot_max_daily_trades)
                
                # Ish vaqti sozlamalari
                if 'working_hours' in settings:
                    hours_data = settings['working_hours']
                    self.working_hours.timezone = hours_data.get('timezone', self.working_hours.timezone)
                    
                    if 'start_time' in hours_data:
                        start_parts = hours_data['start_time'].split(':')
                        self.working_hours.start_time = time(int(start_parts[0]), int(start_parts[1]))
                    
                    if 'end_time' in hours_data:
                        end_parts = hours_data['end_time'].split(':')
                        self.working_hours.end_time = time(int(end_parts[0]), int(end_parts[1]))
                
            else:
                # Default settings.json yaratish
                self._create_default_settings()
                
        except Exception as e:
            raise ConfigurationError(f"JSON settings faylini yuklashda xato: {e}")
    
    def _create_default_settings(self) -> None:
        """Default settings.json faylini yaratish"""
        default_settings = {
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
                },
                "gemini": {
                    "rate_limit": 60,
                    "timeout": 45,
                    "max_retries": 3
                },
                "claude": {
                    "rate_limit": 20,
                    "timeout": 60,
                    "max_retries": 2
                }
            },
            "fallback_order": {
                "order_flow": ["oneinch", "thegraph", "alchemy"],
                "sentiment": ["huggingface", "gemini", "claude"],
                "news": ["newsapi", "reddit", "claude"]
            },
            "trading": {
                "max_risk_per_trade": 0.02,
                "max_daily_loss": 0.05,
                "position_size_method": "kelly",
                "propshot_settings": {
                    "max_daily_loss": 0.025,
                    "max_total_loss": 0.05,
                    "max_lot_size": 0.5,
                    "max_daily_trades": 3
                }
            },
            "working_hours": {
                "timezone": "Asia/Tashkent",
                "start_time": "07:00",
                "end_time": "19:30",
                "trading_days": [0, 1, 2, 3, 4],
                "break_start": "12:00",
                "break_end": "13:00"
            }
        }
        
        # Papka yaratish
        self.config_dir.mkdir(exist_ok=True)
        
        # Faylni yozish
        with open(self.settings_file, 'w', encoding='utf-8') as f:
            json.dump(default_settings, f, indent=2, ensure_ascii=False)
        
        print(f"Default settings.json fayli yaratildi: {self.settings_file}")
    
    def _validate_config(self) -> None:
        """Konfiguratsiyani validatsiya qilish"""
        errors = []
        
        # Majburiy API kalitlarini tekshirish
        required_keys = ['telegram_bot_token', 'telegram_chat_id']
        for key in required_keys:
            if not self.api_keys.get(key):
                errors.append(f"Majburiy API kaliti topilmadi: {key}")
        
        # Trading sozlamalarini tekshirish
        if self.trading.max_risk_per_trade <= 0 or self.trading.max_risk_per_trade > 0.1:
            errors.append("max_risk_per_trade 0 va 0.1 orasida bo'lishi kerak")
        
        if self.trading.max_daily_loss <= 0 or self.trading.max_daily_loss > 0.2:
            errors.append("max_daily_loss 0 va 0.2 orasida bo'lishi kerak")
        
        # Xatolar bo'lsa exception
        if errors:
            raise ConfigurationError(f"Konfiguratsiya validatsiyasida xatolar: {'; '.join(errors)}")
    
    def is_working_hours(self) -> bool:
        """Hozir ish vaqti ekanligini tekshirish"""
        try:
            tz = pytz.timezone(self.working_hours.timezone)
            now = datetime.now(tz)
            current_time = now.time()
            current_day = now.weekday()
            
            # Ish kunlarini tekshirish
            if current_day not in self.working_hours.trading_days:
                return False
            
            # Ish vaqtini tekshirish
            if not (self.working_hours.start_time <= current_time <= self.working_hours.end_time):
                return False
            
            # Tushlik tanaffusini tekshirish
            if self.working_hours.break_start <= current_time <= self.working_hours.break_end:
                return False
            
            return True
            
        except Exception as e:
            print(f"Ish vaqtini tekshirishda xato: {e}")
            return False
    
    def get_api_config(self, api_name: str) -> Dict[str, Any]:
        """API uchun konfiguratsiya olish"""
        return self.api_limits.get(api_name, {
            'rate_limit': self.api.rate_limit,
            'timeout': self.api.timeout,
            'max_retries': self.api.max_retries
        })
    
    def get_fallback_order(self, service_type: str) -> List[str]:
        """Fallback tartibini olish"""
        if service_type == 'order_flow':
            return self.fallback.order_flow_priority
        elif service_type == 'sentiment':
            return self.fallback.sentiment_priority
        elif service_type == 'news':
            return self.fallback.news_priority
        else:
            return []
    
    def save_config(self) -> None:
        """Konfiguratsiyani saqlash"""
        try:
            settings = {
                "api_limits": self.api_limits,
                "fallback_order": {
                    "order_flow": self.fallback.order_flow_priority,
                    "sentiment": self.fallback.sentiment_priority,
                    "news": self.fallback.news_priority
                },
                "trading": {
                    "max_risk_per_trade": self.trading.max_risk_per_trade,
                    "max_daily_loss": self.trading.max_daily_loss,
                    "position_size_method": self.trading.position_size_method,
                    "propshot_settings": {
                        "max_daily_loss": self.trading.propshot_max_daily_loss,
                        "max_total_loss": self.trading.propshot_max_total_loss,
                        "max_lot_size": self.trading.propshot_max_lot_size,
                        "max_daily_trades": self.trading.propshot_max_daily_trades
                    }
                },
                "working_hours": {
                    "timezone": self.working_hours.timezone,
                    "start_time": self.working_hours.start_time.strftime('%H:%M'),
                    "end_time": self.working_hours.end_time.strftime('%H:%M'),
                    "trading_days": self.working_hours.trading_days,
                    "break_start": self.working_hours.break_start.strftime('%H:%M'),
                    "break_end": self.working_hours.break_end.strftime('%H:%M')
                }
            }
            
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(settings, f, indent=2, ensure_ascii=False)
            
            print("Konfiguratsiya muvaffaqiyatli saqlandi")
            
        except Exception as e:
            print(f"Konfiguratsiyani saqlashda xato: {e}")
            raise ConfigurationError(f"Konfiguratsiyani saqlashda xato: {e}")


class ConfigurationError(Exception):
    """Konfiguratsiya xatosi"""
    pass


# Global konfiguratsiya instance
config_manager = ConfigManager()


def get_config() -> ConfigManager:
    """Global konfiguratsiya manager olish"""
    return config_manager


def reload_config() -> None:
    """Konfiguratsiyani qayta yuklash"""
    global config_manager
    config_manager.load_config()


if __name__ == "__main__":
    # Test uchun
    try:
        config = get_config()
        print("✅ Konfiguratsiya muvaffaqiyatli yuklandi")
        print(f"📊 Database URL: {config.database.url}")
        print(f"💬 Telegram Bot Token: {'✅ Mavjud' if config.telegram.bot_token else '❌ Yo\'q'}")
        print(f"⏰ Ish vaqti: {config.working_hours.start_time} - {config.working_hours.end_time}")
        print(f"🔄 Fallback tartibi: {config.fallback.order_flow_priority}")
        print(f"🕐 Hozir ish vaqti: {'✅ Ha' if config.is_working_hours() else '❌ Yo\'q'}")
        
    except Exception as e:
        print(f"❌ Konfiguratsiya xatosi: {e}")

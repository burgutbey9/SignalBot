# -*- coding: utf-8 -*-
"""
🤖 AI OrderFlow & Signal Bot - Asosiy Konfiguratsiya
═══════════════════════════════════════════════════════════

Bu fayl barcha bot sozlamalarini boshqaradi:
- API timeout va rate limitlar
- Database ulanish parametrlari
- Risk management sozlamalar
- Fallback tizimi konfiguratsiyasi
- Propshot 2x himoya qoidalari

Author: AI OrderFlow Bot
Version: 1.0
Language: O'zbekcha
"""

import os
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import json

# =============================================================================
# 🛡️ PROPSHOT RISK QOIDALARI (2x HIMOYA TIZIMI)
# =============================================================================

@dataclass
class PropshotRiskConfig:
    """Propshot challenge qoidalari va bizning 2x himoya tizimi"""
    
    # Propshot standart qoidalar
    PROPSHOT_MAX_DAILY_LOSS: float = 0.05  # 5%
    PROPSHOT_MAX_TOTAL_LOSS: float = 0.10  # 10%
    PROPSHOT_PROFIT_TARGET: float = 0.10   # 10%
    PROPSHOT_MIN_TRADING_DAYS: int = 10
    PROPSHOT_MAX_LOT_SIZE: float = 1.0
    
    # Bizning 2x himoya tizimi (Propshot qoidalarining yarmi)
    OUR_MAX_DAILY_LOSS: float = 0.025       # 2.5% (5% ning yarmi)
    OUR_MAX_TOTAL_LOSS: float = 0.05        # 5% (10% ning yarmi)
    OUR_PROFIT_TARGET: float = 0.05         # 5% (ehtiyotkorlik bilan)
    OUR_MAX_LOT_SIZE: float = 0.5           # 0.5 lot (1.0 ning yarmi)
    OUR_MAX_RISK_PER_TRADE: float = 0.005   # 0.5% (1% ning yarmi)
    OUR_MAX_DAILY_TRADES: int = 3           # 3 ta (5 ta emas)
    
    # Alert chegaralar
    DAILY_LOSS_WARNING: float = 0.015       # 1.5% da ogohlantirish
    TOTAL_LOSS_WARNING: float = 0.03        # 3% da ogohlantirish
    CONSECUTIVE_LOSSES_LIMIT: int = 3       # Ketma-ket 3 ta loss da to'xtatish

# =============================================================================
# 📊 API RATE LIMITS VA TIMEOUTS
# =============================================================================

@dataclass
class APILimits:
    """API rate limitlar va timeout sozlamalar"""
    
    # 1inch API (asosiy DEX data)
    ONEINCH_RATE_LIMIT: int = 100          # 100 req/min
    ONEINCH_TIMEOUT: int = 30              # 30 sekund
    ONEINCH_RETRY_COUNT: int = 3
    
    # Alchemy API (on-chain monitoring)
    ALCHEMY_RATE_LIMIT: int = 300          # 300 req/sec
    ALCHEMY_TIMEOUT: int = 15              # 15 sekund
    ALCHEMY_RETRY_COUNT: int = 3
    
    # HuggingFace AI (asosiy sentiment)
    HUGGINGFACE_RATE_LIMIT: int = 1000     # 1000 req/month
    HUGGINGFACE_TIMEOUT: int = 60          # 60 sekund
    HUGGINGFACE_RETRY_COUNT: int = 2
    
    # Gemini AI (5 ta key, fallback)
    GEMINI_RATE_LIMIT: int = 60            # 60 req/min per key
    GEMINI_TIMEOUT: int = 45               # 45 sekund
    GEMINI_RETRY_COUNT: int = 2
    GEMINI_KEYS_COUNT: int = 5
    
    # Claude AI (limitli fallback)
    CLAUDE_RATE_LIMIT: int = 20            # 20 req/min
    CLAUDE_TIMEOUT: int = 60               # 60 sekund
    CLAUDE_RETRY_COUNT: int = 1
    
    # CCXT (CEX market data)
    CCXT_TIMEOUT: int = 20                 # 20 sekund
    CCXT_RETRY_COUNT: int = 3
    
    # NewsAPI
    NEWS_API_TIMEOUT: int = 30             # 30 sekund
    NEWS_API_RETRY_COUNT: int = 2
    
    # Reddit API
    REDDIT_TIMEOUT: int = 25               # 25 sekund
    REDDIT_RETRY_COUNT: int = 2
    
    # Telegram Bot
    TELEGRAM_TIMEOUT: int = 30             # 30 sekund
    TELEGRAM_RETRY_COUNT: int = 5

# =============================================================================
# 🗄️ DATABASE KONFIGURATSIYA
# =============================================================================

@dataclass
class DatabaseConfig:
    """Database ulanish va sozlamalar"""
    
    # SQLite (development)
    SQLITE_PATH: str = "data/bot_database.db"
    SQLITE_TIMEOUT: int = 30
    
    # PostgreSQL (production)
    POSTGRES_HOST: str = os.getenv("DB_HOST", "localhost")
    POSTGRES_PORT: int = int(os.getenv("DB_PORT", "5432"))
    POSTGRES_NAME: str = os.getenv("DB_NAME", "orderflow_bot")
    POSTGRES_USER: str = os.getenv("DB_USER", "postgres")
    POSTGRES_PASSWORD: str = os.getenv("DB_PASSWORD", "")
    
    # Connection pool sozlamalar
    CONNECTION_POOL_SIZE: int = 5
    MAX_OVERFLOW: int = 10
    POOL_TIMEOUT: int = 30
    
    # Backup sozlamalar
    BACKUP_INTERVAL_HOURS: int = 6
    BACKUP_RETENTION_DAYS: int = 30
    BACKUP_PATH: str = "data/backups/"

# =============================================================================
# 📈 TRADING KONFIGURATSIYA
# =============================================================================

@dataclass
class TradingConfig:
    """Trading strategiya va signal sozlamalar"""
    
    # Signal parametrlari
    MIN_SIGNAL_CONFIDENCE: float = 0.75    # 75% minimum ishonch
    MAX_SIGNALS_PER_HOUR: int = 5          # Soatiga maksimal 5 ta signal
    SIGNAL_COOLDOWN_MINUTES: int = 10      # Signallar orasida 10 daqiqa
    
    # Order Flow tahlil
    WHALE_THRESHOLD_USD: float = 100000    # $100k+ whale deb hisoblanadi
    VOLUME_SPIKE_MULTIPLIER: float = 2.0   # O'rtacha volumdan 2x ko'p
    ORDER_FLOW_TIMEFRAMES: list = [5, 15, 30, 60]  # Daqiqalarda
    
    # Sentiment tahlil
    SENTIMENT_SOURCES: list = ["news", "reddit", "twitter", "telegram"]
    SENTIMENT_WEIGHT_NEWS: float = 0.4     # 40% vazn
    SENTIMENT_WEIGHT_SOCIAL: float = 0.6   # 60% vazn
    SENTIMENT_THRESHOLD: float = 0.6       # 60% ijobiy/salbiy
    
    # Technical indicators
    RSI_PERIOD: int = 14
    RSI_OVERSOLD: float = 30
    RSI_OVERBOUGHT: float = 70
    MA_FAST: int = 10
    MA_SLOW: int = 20
    BOLLINGER_PERIOD: int = 20
    BOLLINGER_STD: float = 2.0
    
    # Risk management
    MAX_POSITION_SIZE_PERCENT: float = 0.02  # 2% maksimal pozitsiya
    STOP_LOSS_PERCENT: float = 0.015        # 1.5% stop loss
    TAKE_PROFIT_PERCENT: float = 0.03       # 3% take profit
    TRAILING_STOP_PERCENT: float = 0.01     # 1% trailing stop

# =============================================================================
# 🔄 FALLBACK TIZIMI KONFIGURATSIYA
# =============================================================================

@dataclass
class FallbackConfig:
    """Fallback tizimi sozlamalar"""
    
    # AI Services fallback zanjiri
    AI_FALLBACK_CHAIN: list = [
        "huggingface",  # Asosiy
        "gemini",       # 5 ta key
        "claude",       # Limitli
        "local_nlp"     # Oxirgi
    ]
    
    # DEX Data fallback zanjiri
    DEX_FALLBACK_CHAIN: list = [
        "oneinch",      # Asosiy, verifikatsiyali
        "thegraph",     # Uniswap V3 fallback
        "alchemy"       # On-chain backup
    ]
    
    # Market Data fallback
    MARKET_FALLBACK_CHAIN: list = [
        "ccxt_gateio",  # Gate.io
        "ccxt_kucoin",  # KuCoin
        "ccxt_mexc",    # MEXC
        "alchemy"       # On-chain
    ]
    
    # News fallback
    NEWS_FALLBACK_CHAIN: list = [
        "newsapi",      # Asosiy, verifikatsiyali
        "reddit",       # Ijtimoiy sentiment
        "claude"        # News tahlil backup
    ]
    
    # Fallback aktivatsiya sozlamalar
    FALLBACK_DELAY_SECONDS: int = 5         # 5 sekund kutish
    MAX_FALLBACK_ATTEMPTS: int = 3          # 3 ta fallback
    FALLBACK_SUCCESS_RESET_HOURS: int = 1   # 1 soat muvaffaqiyatdan keyin reset

# =============================================================================
# 📱 TELEGRAM BOT KONFIGURATSIYA
# =============================================================================

@dataclass
class TelegramConfig:
    """Telegram bot sozlamalar"""
    
    # Bot sozlamalar
    BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")
    
    # Xabar formatlash
    MESSAGE_LANGUAGE: str = "uzbek"
    USE_MARKDOWN: bool = True
    MAX_MESSAGE_LENGTH: int = 4000
    
    # Signal xabar templatelari
    SIGNAL_TEMPLATE: str = """
📊 SIGNAL KELDI
════════════════
📈 Savdo: {action} {symbol}
💰 Narx: {price}
📊 Lot: {lot_size} lot
🛡️ Stop Loss: {stop_loss} ({sl_pips} pips)
🎯 Take Profit: {take_profit} ({tp_pips} pips)
⚡ Ishonch: {confidence}%
🔥 Risk: {risk}% (Propshot 2x himoya)
════════════════
📝 Sabab: {reason}
⏰ Vaqt: {time} (UZB)
💼 Akavunt: {account}
"""
    
    # Button sozlamalar
    ENABLE_INLINE_BUTTONS: bool = True
    AUTO_TRADE_BUTTON: str = "🟢 AVTO SAVDO"
    CANCEL_BUTTON: str = "🔴 BEKOR QILISH"
    
    # Notification sozlamalar
    ENABLE_DAILY_REPORTS: bool = True
    DAILY_REPORT_TIME: str = "07:00"  # UZB vaqti
    ENABLE_WEEKLY_REPORTS: bool = True
    WEEKLY_REPORT_DAY: str = "sunday"
    
    # Error notification
    ENABLE_ERROR_NOTIFICATIONS: bool = True
    ERROR_NOTIFICATION_COOLDOWN: int = 300  # 5 daqiqa

# =============================================================================
# 🖥️ METATRADER 5 KONFIGURATSIYA
# =============================================================================

@dataclass
class MT5Config:
    """MetaTrader 5 terminal sozlamalar"""
    
    # Terminal sozlamalar
    TERMINAL_PATH: str = r"C:\Program Files\MetaTrader 5\terminal64.exe"
    CONNECTION_TIMEOUT: int = 30
    LOGIN_TIMEOUT: int = 60
    
    # Demo akavunt
    DEMO_SERVER: str = "Demo-Server"
    DEMO_LOGIN: int = 0
    DEMO_PASSWORD: str = ""
    
    # Propshot akavunt
    PROPSHOT_SERVER: str = "Propshot-Server"
    PROPSHOT_LOGIN: int = 0
    PROPSHOT_PASSWORD: str = ""
    
    # Order parametrlari
    SLIPPAGE: int = 3               # 3 pips slippage
    MAGIC_NUMBER: int = 123456      # Bot magic number
    ORDER_TIMEOUT: int = 30         # 30 sekund order timeout
    
    # Monitoring
    POSITION_CHECK_INTERVAL: int = 5  # 5 sekund
    ENABLE_TRAILING_STOP: bool = True
    TRAILING_STOP_STEP: int = 10     # 10 pips step

# =============================================================================
# 📊 LOGGING KONFIGURATSIYA
# =============================================================================

@dataclass
class LoggingConfig:
    """Logging tizimi sozlamalar"""
    
    # Log fayllar
    LOG_DIR: str = "logs"
    APP_LOG_FILE: str = "app.log"
    ERROR_LOG_FILE: str = "error.log"
    API_LOG_FILE: str = "api.log"
    TRADE_LOG_FILE: str = "trade.log"
    
    # Log levellari
    CONSOLE_LOG_LEVEL: str = "INFO"
    FILE_LOG_LEVEL: str = "DEBUG"
    
    # Log rotation
    MAX_LOG_SIZE_MB: int = 10
    BACKUP_COUNT: int = 5
    
    # Log formatlar
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"
    
    # Sensitive data masking
    MASK_API_KEYS: bool = True
    MASK_PASSWORDS: bool = True
    MASK_ACCOUNT_DATA: bool = True

# =============================================================================
# 🎯 ASOSIY KONFIGURATSIYA KLASSI
# =============================================================================

class Config:
    """Asosiy konfiguratsiya klassi - barcha sozlamalarni birlashtiradi"""
    
    def __init__(self):
        """Konfiguratsiya initsializatsiyasi"""
        
        # Fayl yo'llari
        self.BASE_DIR = Path(__file__).parent.parent
        self.CONFIG_DIR = self.BASE_DIR / "config"
        self.DATA_DIR = self.BASE_DIR / "data"
        self.LOGS_DIR = self.BASE_DIR / "logs"
        
        # Papkalarni yaratish
        self._create_directories()
        
        # Konfiguratsiya obyektlari
        self.propshot_risk = PropshotRiskConfig()
        self.api_limits = APILimits()
        self.database = DatabaseConfig()
        self.trading = TradingConfig()
        self.fallback = FallbackConfig()
        self.telegram = TelegramConfig()
        self.mt5 = MT5Config()
        self.logging = LoggingConfig()
        
        # Muhit o'zgaruvchilar
        self.ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
        self.DEBUG = os.getenv("DEBUG", "False").lower() == "true"
        
        # Settings.json faylini yuklash
        self._load_settings_from_file()
        
        # Konfiguratsiya validatsiyasi
        self._validate_config()
    
    def _create_directories(self):
        """Kerakli papkalarni yaratish"""
        directories = [
            self.DATA_DIR,
            self.LOGS_DIR,
            self.DATA_DIR / "backups",
            self.DATA_DIR / "cache"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _load_settings_from_file(self):
        """Settings.json faylidan sozlamalarni yuklash"""
        settings_file = self.CONFIG_DIR / "settings.json"
        
        if settings_file.exists():
            try:
                with open(settings_file, 'r', encoding='utf-8') as f:
                    settings = json.load(f)
                
                # Sozlamalarni qo'llash
                self._apply_settings(settings)
                
            except Exception as e:
                print(f"⚠️ Settings.json yuklashda xatolik: {e}")
    
    def _apply_settings(self, settings: Dict[str, Any]):
        """JSON sozlamalarni qo'llash"""
        
        # Risk sozlamalar
        if "risk" in settings:
            risk_settings = settings["risk"]
            for key, value in risk_settings.items():
                if hasattr(self.propshot_risk, key.upper()):
                    setattr(self.propshot_risk, key.upper(), value)
        
        # Trading sozlamalar
        if "trading" in settings:
            trading_settings = settings["trading"]
            for key, value in trading_settings.items():
                if hasattr(self.trading, key.upper()):
                    setattr(self.trading, key.upper(), value)
        
        # Telegram sozlamalar
        if "telegram" in settings:
            telegram_settings = settings["telegram"]
            for key, value in telegram_settings.items():
                if hasattr(self.telegram, key.upper()):
                    setattr(self.telegram, key.upper(), value)
    
    def _validate_config(self):
        """Konfiguratsiya validatsiyasi"""
        
        # Telegram bot token tekshirish
        if not self.telegram.BOT_TOKEN:
            raise ValueError("❌ TELEGRAM_BOT_TOKEN muhit o'zgaruvchisi kerak!")
        
        # Telegram chat ID tekshirish
        if not self.telegram.CHAT_ID:
            raise ValueError("❌ TELEGRAM_CHAT_ID muhit o'zgaruvchisi kerak!")
        
        # Risk limitlar tekshirish
        if self.propshot_risk.OUR_MAX_DAILY_LOSS >= self.propshot_risk.PROPSHOT_MAX_DAILY_LOSS:
            raise ValueError("❌ Bizning kunlik risk Propshot limitdan kam bo'lishi kerak!")
        
        if self.propshot_risk.OUR_MAX_TOTAL_LOSS >= self.propshot_risk.PROPSHOT_MAX_TOTAL_LOSS:
            raise ValueError("❌ Bizning umumiy risk Propshot limitdan kam bo'lishi kerak!")
    
    def save_settings_to_file(self):
        """Joriy sozlamalarni settings.json ga saqlash"""
        settings_file = self.CONFIG_DIR / "settings.json"
        
        settings = {
            "risk": {
                "our_max_daily_loss": self.propshot_risk.OUR_MAX_DAILY_LOSS,
                "our_max_total_loss": self.propshot_risk.OUR_MAX_TOTAL_LOSS,
                "our_max_risk_per_trade": self.propshot_risk.OUR_MAX_RISK_PER_TRADE,
                "our_max_daily_trades": self.propshot_risk.OUR_MAX_DAILY_TRADES
            },
            "trading": {
                "min_signal_confidence": self.trading.MIN_SIGNAL_CONFIDENCE,
                "max_signals_per_hour": self.trading.MAX_SIGNALS_PER_HOUR,
                "whale_threshold_usd": self.trading.WHALE_THRESHOLD_USD,
                "sentiment_threshold": self.trading.SENTIMENT_THRESHOLD
            },
            "telegram": {
                "message_language": self.telegram.MESSAGE_LANGUAGE,
                "enable_daily_reports": self.telegram.ENABLE_DAILY_REPORTS,
                "daily_report_time": self.telegram.DAILY_REPORT_TIME,
                "enable_weekly_reports": self.telegram.ENABLE_WEEKLY_REPORTS
            },
            "last_updated": str(Path(__file__).stat().st_mtime)
        }
        
        try:
            with open(settings_file, 'w', encoding='utf-8') as f:
                json.dump(settings, f, ensure_ascii=False, indent=2)
            
            print(f"✅ Sozlamalar saqlandi: {settings_file}")
            
        except Exception as e:
            print(f"❌ Sozlamalar saqlanmadi: {e}")
    
    def get_database_url(self) -> str:
        """Database URL yaratish"""
        if self.ENVIRONMENT == "production":
            return (
                f"postgresql://{self.database.POSTGRES_USER}:"
                f"{self.database.POSTGRES_PASSWORD}@"
                f"{self.database.POSTGRES_HOST}:"
                f"{self.database.POSTGRES_PORT}/"
                f"{self.database.POSTGRES_NAME}"
            )
        else:
            return f"sqlite:///{self.database.SQLITE_PATH}"
    
    def is_production(self) -> bool:
        """Production muhitmi?"""
        return self.ENVIRONMENT == "production"
    
    def is_debug(self) -> bool:
        """Debug rejimmi?"""
        return self.DEBUG
    
    def get_log_level(self) -> int:
        """Log level olish"""
        if self.is_debug():
            return logging.DEBUG
        return logging.INFO
    
    def __str__(self) -> str:
        """Konfiguratsiya string ko'rinishi"""
        return f"""
🤖 AI OrderFlow Bot Konfiguratsiya
═══════════════════════════════════
🌍 Muhit: {self.ENVIRONMENT}
🔧 Debug: {self.DEBUG}
🛡️ Propshot 2x himoya: ✅
📊 Kunlik risk: {self.propshot_risk.OUR_MAX_DAILY_LOSS*100:.1f}%
💰 Har savdo risk: {self.propshot_risk.OUR_MAX_RISK_PER_TRADE*100:.1f}%
📈 Minimum ishonch: {self.trading.MIN_SIGNAL_CONFIDENCE*100:.0f}%
🗄️ Database: {self.get_database_url()}
═══════════════════════════════════
"""

# =============================================================================
# 🚀 GLOBAL KONFIGURATSIYA OBYEKTI
# =============================================================================

# Global konfiguratsiya obyekti - barcha modullarda ishlatiladi
try:
    config = Config()
    print("✅ Konfiguratsiya muvaffaqiyatli yuklandi!")
    print(config)
except Exception as e:
    print(f"❌ Konfiguratsiya yuklashda xatolik: {e}")
    raise

# =============================================================================
# 🧪 TEST FUNKSIYASI
# =============================================================================

def test_config():
    """Konfiguratsiya test funksiyasi"""
    print("🧪 Konfiguratsiya test qilinmoqda...")
    
    # Asosiy parametrlar
    assert config.propshot_risk.OUR_MAX_DAILY_LOSS < config.propshot_risk.PROPSHOT_MAX_DAILY_LOSS
    assert config.propshot_risk.OUR_MAX_TOTAL_LOSS < config.propshot_risk.PROPSHOT_MAX_TOTAL_LOSS
    assert config.trading.MIN_SIGNAL_CONFIDENCE > 0.5
    assert config.trading.MIN_SIGNAL_CONFIDENCE < 1.0
    
    # API limitlar
    assert config.api_limits.ONEINCH_RATE_LIMIT > 0
    assert config.api_limits.ONEINCH_TIMEOUT > 0
    
    # Database
    db_url = config.get_database_url()
    assert db_url.startswith("sqlite://") or db_url.startswith("postgresql://")
    
    print("✅ Barcha testlar muvaffaqiyatli!")

if __name__ == "__main__":
    test_config()
    
    # Sozlamalarni faylga saqlash
    config.save_settings_to_file()
    
    print("\n🎉 Config.py tayyor va test qilindi!")
    print("🚀 Keyingi qadam: api_keys.py faylini yaratish")

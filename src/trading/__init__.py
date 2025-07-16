"""
Trading moduli - Savdo va strategiya boshqaruvi

Bu modul quyidagi komponentlarni o'z ichiga oladi:
- Strategy Manager: Savdo strategiyalarini boshqarish
- Backtest Engine: Tarixiy ma'lumotlarda strategiya sinovdan o'tkazish  
- Portfolio Manager: Portfolio boshqaruvi
- Execution Engine: Savdo buyruqlarini bajarish
- Propshot Connector: Propshot platformasiga bog'lanish
- MT5 Bridge: MetaTrader 5 bilan integratsiya

Asosiy xususiyatlar:
- Avtomatik savdo bajarish
- Risk boshqaruvi
- Strategiya optimizatsiyasi
- Real-time monitoring
- Propshot integration
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

# Utils modullarini import qilish
try:
    from utils.logger import get_logger
except ImportError:
    import logging
    def get_logger(name: str):
        return logging.getLogger(name)

logger = get_logger(__name__)

# Trading holatlari
class TradingStatus(Enum):
    """Savdo holatlari"""
    IDLE = "kutish"           # Kutish holati
    ANALYZING = "tahlil"      # Tahlil qilish
    SIGNAL_RECEIVED = "signal_keldi"  # Signal keldi
    EXECUTING = "bajarilmoqda"        # Bajarilmoqda
    MONITORING = "kuzatuv"            # Kuzatuv
    CLOSED = "yopildi"               # Yopildi
    ERROR = "xatolik"                # Xatolik

# Trading turlari
class TradeType(Enum):
    """Savdo turlari"""
    BUY = "sotib_olish"      # Sotib olish
    SELL = "sotish"          # Sotish
    
# Order turlari
class OrderType(Enum):
    """Buyruq turlari"""
    MARKET = "market"        # Market order
    LIMIT = "limit"          # Limit order
    STOP = "stop"            # Stop order
    STOP_LIMIT = "stop_limit" # Stop limit order

@dataclass
class TradingConfig:
    """Savdo konfiguratsiyasi"""
    max_risk_per_trade: float = 0.02      # Har savdo uchun maksimal risk (2%)
    max_daily_loss: float = 0.05          # Kunlik maksimal yo'qotish (5%)
    max_positions: int = 5                # Maksimal ochiq pozitsiyalar soni
    min_confidence: float = 0.75          # Minimal signal ishonch darajasi
    auto_trading: bool = True             # Avtomatik savdo
    propshot_enabled: bool = True         # Propshot yoqilganmi
    mt5_enabled: bool = False             # MT5 yoqilganmi
    
    def __post_init__(self):
        """Konfiguratsiya validatsiyasi"""
        if self.max_risk_per_trade > 0.1:
            raise ValueError("Har savdo uchun risk 10% dan oshmasligi kerak")
        if self.max_daily_loss > 0.2:
            raise ValueError("Kunlik yo'qotish 20% dan oshmasligi kerak")
        if self.min_confidence < 0.5:
            raise ValueError("Minimal ishonch darajasi 50% dan past bo'lmasligi kerak")

@dataclass
class TradingMetrics:
    """Savdo metrikalari"""
    total_trades: int = 0              # Jami savdolar
    winning_trades: int = 0            # Yutgan savdolar
    losing_trades: int = 0             # Yutqazgan savdolar
    total_profit: float = 0.0          # Jami foyda
    max_drawdown: float = 0.0          # Maksimal pasayish
    win_rate: float = 0.0              # Yutish foizi
    profit_factor: float = 0.0         # Foyda koeffitsienti
    sharpe_ratio: float = 0.0          # Sharpe koeffitsienti
    
    def calculate_metrics(self):
        """Metrikallarni hisoblash"""
        if self.total_trades > 0:
            self.win_rate = self.winning_trades / self.total_trades
            if self.losing_trades > 0:
                avg_win = self.total_profit / self.winning_trades if self.winning_trades > 0 else 0
                avg_loss = abs(self.total_profit - (avg_win * self.winning_trades)) / self.losing_trades
                self.profit_factor = avg_win / avg_loss if avg_loss > 0 else 0

# Trading modulini eksport qilish
__all__ = [
    'TradingStatus',
    'TradeType', 
    'OrderType',
    'TradingConfig',
    'TradingMetrics',
    'logger'
]

# Modul yuklanganda log yozish
logger.info("Trading moduli yuklandi - Savdo tizimi tayyor")
logger.info("Mavjud komponentlar: Strategy Manager, Backtest Engine, Portfolio Manager")
logger.info("Execution Engine, Propshot Connector, MT5 Bridge")

"""
AI OrderFlow & Signal Bot - Database Models
Ma'lumotlar bazasi modellari va sxemalar
"""

import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Boolean, Text, 
    ForeignKey, UniqueConstraint, Index, JSON
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from utils.logger import get_logger

logger = get_logger(__name__)

# Base class barcha modellar uchun
Base = declarative_base()

class SignalType(Enum):
    """Signal turlari enum"""
    BUY = "BUY"
    SELL = "SELL"
    CLOSE = "CLOSE"
    HOLD = "HOLD"

class SignalStatus(Enum):
    """Signal holati enum"""
    PENDING = "PENDING"           # Kutilayotgan
    CONFIRMED = "CONFIRMED"       # Tasdiqlangan
    EXECUTED = "EXECUTED"         # Bajarilgan
    CANCELLED = "CANCELLED"       # Bekor qilingan
    EXPIRED = "EXPIRED"           # Muddati tugagan
    FAILED = "FAILED"             # Muvaffaqiyatsiz

class RiskLevel(Enum):
    """Risk darajasi enum"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    EXTREME = "EXTREME"

class DataSource(Enum):
    """Ma'lumot manbai enum"""
    ONEINCH = "1INCH"
    THEGRAPH = "THEGRAPH"
    ALCHEMY = "ALCHEMY"
    HUGGINGFACE = "HUGGINGFACE"
    GEMINI = "GEMINI"
    CLAUDE = "CLAUDE"
    NEWSAPI = "NEWSAPI"
    REDDIT = "REDDIT"
    CCXT = "CCXT"

# ====== ASOSIY MODELLAR ======

class TradingPair(Base):
    """Trading juftliklari modeli"""
    __tablename__ = "trading_pairs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), unique=True, nullable=False, index=True)
    base_asset = Column(String(10), nullable=False)
    quote_asset = Column(String(10), nullable=False)
    exchange = Column(String(20), nullable=False)
    is_active = Column(Boolean, default=True)
    min_lot_size = Column(Float, default=0.01)
    max_lot_size = Column(Float, default=100.0)
    pip_value = Column(Float, default=0.0001)
    spread_avg = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=datetime.now(timezone.utc), onupdate=datetime.now(timezone.utc))
    
    # Relationship
    signals = relationship("Signal", back_populates="pair")
    orders = relationship("Order", back_populates="pair")
    market_data = relationship("MarketData", back_populates="pair")
    
    def __repr__(self):
        return f"<TradingPair(symbol='{self.symbol}', active={self.is_active})>"

class Signal(Base):
    """AI signal modeli"""
    __tablename__ = "signals"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    pair_id = Column(Integer, ForeignKey('trading_pairs.id'), nullable=False)
    signal_type = Column(String(10), nullable=False)  # SignalType enum
    signal_status = Column(String(20), default="PENDING")  # SignalStatus enum
    
    # Signal ma'lumotlari
    entry_price = Column(Float, nullable=False)
    stop_loss = Column(Float, nullable=True)
    take_profit = Column(Float, nullable=True)
    lot_size = Column(Float, nullable=False)
    
    # AI confidence va risk
    confidence = Column(Float, default=0.0)  # 0-100
    risk_level = Column(String(10), default="MEDIUM")  # RiskLevel enum
    risk_percent = Column(Float, default=0.02)  # Foiz hisobida
    
    # Boshqa ma'lumotlar
    reason = Column(Text, nullable=True)  # Signal sababi
    data_sources = Column(JSON, nullable=True)  # Ishlatilgan manbalar
    telegram_message_id = Column(String(50), nullable=True)
    
    # Vaqt ma'lumotlari
    created_at = Column(DateTime, default=datetime.now(timezone.utc))
    expires_at = Column(DateTime, nullable=True)
    executed_at = Column(DateTime, nullable=True)
    
    # Relationship
    pair = relationship("TradingPair", back_populates="signals")
    orders = relationship("Order", back_populates="signal")
    
    # Index - tezkor qidiruv uchun
    __table_args__ = (
        Index('idx_signal_status_created', 'signal_status', 'created_at'),
        Index('idx_signal_pair_type', 'pair_id', 'signal_type'),
    )
    
    def __repr__(self):
        return f"<Signal(pair={self.pair.symbol}, type={self.signal_type}, status={self.signal_status})>"

class Order(Base):
    """Savdo buyruqlar modeli"""
    __tablename__ = "orders"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    signal_id = Column(Integer, ForeignKey('signals.id'), nullable=False)
    pair_id = Column(Integer, ForeignKey('trading_pairs.id'), nullable=False)
    
    # Buyruq ma'lumotlari
    order_type = Column(String(20), nullable=False)  # MARKET, LIMIT, STOP
    side = Column(String(10), nullable=False)  # BUY, SELL
    quantity = Column(Float, nullable=False)
    price = Column(Float, nullable=True)
    
    # Execution ma'lumotlari
    executed_quantity = Column(Float, default=0.0)
    executed_price = Column(Float, nullable=True)
    commission = Column(Float, default=0.0)
    pnl = Column(Float, nullable=True)
    
    # Status
    status = Column(String(20), default="PENDING")
    external_order_id = Column(String(100), nullable=True)  # Propshot/MT5 ID
    
    # Vaqt ma'lumotlari
    created_at = Column(DateTime, default=datetime.now(timezone.utc))
    executed_at = Column(DateTime, nullable=True)
    cancelled_at = Column(DateTime, nullable=True)
    
    # Relationship
    signal = relationship("Signal", back_populates="orders")
    pair = relationship("TradingPair", back_populates="orders")
    
    def __repr__(self):
        return f"<Order(pair={self.pair.symbol}, side={self.side}, status={self.status})>"

class MarketData(Base):
    """Market ma'lumotlari modeli"""
    __tablename__ = "market_data"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    pair_id = Column(Integer, ForeignKey('trading_pairs.id'), nullable=False)
    
    # OHLCV ma'lumotlari
    open_price = Column(Float, nullable=False)
    high_price = Column(Float, nullable=False)
    low_price = Column(Float, nullable=False)
    close_price = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    
    # Qo'shimcha ma'lumotlar
    bid_price = Column(Float, nullable=True)
    ask_price = Column(Float, nullable=True)
    spread = Column(Float, nullable=True)
    
    # Vaqt ma'lumotlari
    timestamp = Column(DateTime, nullable=False)
    timeframe = Column(String(10), default="1m")  # 1m, 5m, 15m, 1h, 4h, 1d
    
    # Relationship
    pair = relationship("TradingPair", back_populates="market_data")
    
    # Index - tezkor qidiruv uchun
    __table_args__ = (
        Index('idx_market_data_pair_time', 'pair_id', 'timestamp'),
        Index('idx_market_data_timeframe', 'timeframe', 'timestamp'),
    )
    
    def __repr__(self):
        return f"<MarketData(pair={self.pair.symbol}, price={self.close_price}, time={self.timestamp})>"

class SentimentData(Base):
    """Sentiment tahlil ma'lumotlari"""
    __tablename__ = "sentiment_data"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    pair_id = Column(Integer, ForeignKey('trading_pairs.id'), nullable=True)
    
    # Sentiment ma'lumotlari
    source = Column(String(20), nullable=False)  # DataSource enum
    sentiment_score = Column(Float, nullable=False)  # -1 to 1
    confidence = Column(Float, nullable=False)  # 0 to 1
    
    # Matn ma'lumotlari
    text_content = Column(Text, nullable=True)
    keywords = Column(JSON, nullable=True)
    
    # Qo'shimcha ma'lumotlar
    language = Column(String(10), default="en")
    source_url = Column(String(500), nullable=True)
    author = Column(String(100), nullable=True)
    
    # Vaqt ma'lumotlari
    created_at = Column(DateTime, default=datetime.now(timezone.utc))
    published_at = Column(DateTime, nullable=True)
    
    # Relationship
    pair = relationship("TradingPair", back_populates="sentiment_data") if hasattr(TradingPair, 'sentiment_data') else None
    
    def __repr__(self):
        return f"<SentimentData(source={self.source}, score={self.sentiment_score}, confidence={self.confidence})>"

class OrderFlowData(Base):
    """Order Flow ma'lumotlari"""
    __tablename__ = "order_flow_data"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    pair_id = Column(Integer, ForeignKey('trading_pairs.id'), nullable=False)
    
    # Order Flow ma'lumotlari
    buy_volume = Column(Float, nullable=False)
    sell_volume = Column(Float, nullable=False)
    net_volume = Column(Float, nullable=False)
    
    # Qo'shimcha ma'lumotlar
    large_orders_count = Column(Integer, default=0)
    small_orders_count = Column(Integer, default=0)
    order_imbalance = Column(Float, nullable=True)
    
    # Manbalar
    source = Column(String(20), nullable=False)  # DataSource enum
    raw_data = Column(JSON, nullable=True)
    
    # Vaqt ma'lumotlari
    timestamp = Column(DateTime, nullable=False)
    timeframe = Column(String(10), default="1m")
    
    # Relationship
    pair = relationship("TradingPair")
    
    def __repr__(self):
        return f"<OrderFlowData(pair={self.pair.symbol}, net_volume={self.net_volume})>"

class BotSettings(Base):
    """Bot sozlamalari modeli"""
    __tablename__ = "bot_settings"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    key = Column(String(50), unique=True, nullable=False)
    value = Column(Text, nullable=False)
    value_type = Column(String(20), default="string")  # string, int, float, bool, json
    description = Column(Text, nullable=True)
    
    # Vaqt ma'lumotlari
    created_at = Column(DateTime, default=datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=datetime.now(timezone.utc), onupdate=datetime.now(timezone.utc))
    
    def __repr__(self):
        return f"<BotSettings(key='{self.key}', value='{self.value}')>"

class APIUsage(Base):
    """API foydalanish statistikasi"""
    __tablename__ = "api_usage"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    api_name = Column(String(50), nullable=False)
    endpoint = Column(String(200), nullable=True)
    
    # Statistika ma'lumotlari
    requests_count = Column(Integer, default=0)
    successful_requests = Column(Integer, default=0)
    failed_requests = Column(Integer, default=0)
    
    # Vaqt ma'lumotlari
    date = Column(DateTime, nullable=False)
    response_time_avg = Column(Float, nullable=True)
    
    # Qo'shimcha ma'lumotlar
    error_messages = Column(JSON, nullable=True)
    rate_limit_hits = Column(Integer, default=0)
    
    # Index
    __table_args__ = (
        Index('idx_api_usage_date', 'api_name', 'date'),
    )
    
    def __repr__(self):
        return f"<APIUsage(api={self.api_name}, requests={self.requests_count}, date={self.date})>"

class BacktestResult(Base):
    """Backtest natijalar modeli"""
    __tablename__ = "backtest_results"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    strategy_name = Column(String(100), nullable=False)
    pair_id = Column(Integer, ForeignKey('trading_pairs.id'), nullable=False)
    
    # Backtest parametrlari
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    initial_capital = Column(Float, nullable=False)
    
    # Natijalar
    final_capital = Column(Float, nullable=False)
    total_return = Column(Float, nullable=False)
    win_rate = Column(Float, nullable=False)
    profit_factor = Column(Float, nullable=False)
    max_drawdown = Column(Float, nullable=False)
    sharpe_ratio = Column(Float, nullable=True)
    
    # Savdo statistikasi
    total_trades = Column(Integer, nullable=False)
    winning_trades = Column(Integer, nullable=False)
    losing_trades = Column(Integer, nullable=False)
    avg_trade_return = Column(Float, nullable=False)
    
    # Qo'shimcha ma'lumotlar
    strategy_config = Column(JSON, nullable=True)
    detailed_results = Column(JSON, nullable=True)
    
    # Vaqt ma'lumotlari
    created_at = Column(DateTime, default=datetime.now(timezone.utc))
    
    # Relationship
    pair = relationship("TradingPair")
    
    def __repr__(self):
        return f"<BacktestResult(strategy={self.strategy_name}, return={self.total_return}%)>"

class RiskMetrics(Base):
    """Risk ko'rsatkichlari modeli"""
    __tablename__ = "risk_metrics"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Risk ko'rsatkichlari
    daily_var = Column(Float, nullable=True)  # Value at Risk
    daily_loss = Column(Float, default=0.0)
    max_drawdown = Column(Float, default=0.0)
    current_exposure = Column(Float, default=0.0)
    
    # Propshot qoidalari
    max_daily_loss_limit = Column(Float, default=0.025)  # 2.5%
    max_total_loss_limit = Column(Float, default=0.05)   # 5%
    daily_loss_remaining = Column(Float, default=0.025)
    total_loss_remaining = Column(Float, default=0.05)
    
    # Savdo limitlari
    max_lot_size = Column(Float, default=0.5)
    max_daily_trades = Column(Integer, default=3)
    trades_today = Column(Integer, default=0)
    
    # Vaqt ma'lumotlari
    date = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, default=datetime.now(timezone.utc), onupdate=datetime.now(timezone.utc))
    
    def __repr__(self):
        return f"<RiskMetrics(date={self.date}, daily_loss={self.daily_loss})>"

# ====== YORDAMCHI MODELLAR ======

class SystemLog(Base):
    """Tizim loglari modeli"""
    __tablename__ = "system_logs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    level = Column(String(10), nullable=False)  # INFO, WARNING, ERROR
    module = Column(String(50), nullable=False)
    message = Column(Text, nullable=False)
    
    # Qo'shimcha ma'lumotlar
    extra_data = Column(JSON, nullable=True)
    stack_trace = Column(Text, nullable=True)
    
    # Vaqt ma'lumotlari
    timestamp = Column(DateTime, default=datetime.now(timezone.utc))
    
    # Index
    __table_args__ = (
        Index('idx_logs_level_time', 'level', 'timestamp'),
        Index('idx_logs_module_time', 'module', 'timestamp'),
    )
    
    def __repr__(self):
        return f"<SystemLog(level={self.level}, module={self.module})>"

class TelegramMessage(Base):
    """Telegram xabar modeli"""
    __tablename__ = "telegram_messages"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    message_id = Column(String(50), nullable=False)
    chat_id = Column(String(50), nullable=False)
    
    # Xabar ma'lumotlari
    message_type = Column(String(20), nullable=False)  # SIGNAL, LOG, ALERT
    content = Column(Text, nullable=False)
    
    # Signal bilan bog'lanish
    signal_id = Column(Integer, ForeignKey('signals.id'), nullable=True)
    
    # Status
    sent = Column(Boolean, default=False)
    delivered = Column(Boolean, default=False)
    read = Column(Boolean, default=False)
    
    # Vaqt ma'lumotlari
    created_at = Column(DateTime, default=datetime.now(timezone.utc))
    sent_at = Column(DateTime, nullable=True)
    
    # Relationship
    signal = relationship("Signal") if hasattr(Signal, 'telegram_messages') else None
    
    def __repr__(self):
        return f"<TelegramMessage(type={self.message_type}, sent={self.sent})>"

# ====== DATACLASS MODELLARI ======

@dataclass
class SignalData:
    """Signal ma'lumotlari dataclass"""
    pair_symbol: str
    signal_type: str
    entry_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    lot_size: float = 0.01
    confidence: float = 0.0
    risk_percent: float = 0.02
    reason: Optional[str] = None
    expires_in_minutes: int = 60

@dataclass
class MarketDataPoint:
    """Market ma'lumot nuqtasi"""
    symbol: str
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: float
    timestamp: datetime
    timeframe: str = "1m"

@dataclass
class RiskCalculation:
    """Risk hisoblash natijasi"""
    max_lot_size: float
    risk_percent: float
    stop_loss_pips: float
    take_profit_pips: float
    risk_reward_ratio: float
    can_trade: bool
    reason: Optional[str] = None

# ====== UTILITY FUNKSIYALAR ======

def get_signal_by_id(session: AsyncSession, signal_id: int) -> Optional[Signal]:
    """Signal ID orqali signal olish"""
    try:
        return session.query(Signal).filter(Signal.id == signal_id).first()
    except Exception as e:
        logger.error(f"Signal olishda xato: {e}")
        return None

def get_active_signals(session: AsyncSession, limit: int = 10) -> List[Signal]:
    """Faol signallar ro'yxatini olish"""
    try:
        return session.query(Signal).filter(
            Signal.signal_status.in_(['PENDING', 'CONFIRMED'])
        ).order_by(Signal.created_at.desc()).limit(limit).all()
    except Exception as e:
        logger.error(f"Faol signallar olishda xato: {e}")
        return []

def get_trading_pairs(session: AsyncSession, active_only: bool = True) -> List[TradingPair]:
    """Trading juftliklari ro'yxatini olish"""
    try:
        query = session.query(TradingPair)
        if active_only:
            query = query.filter(TradingPair.is_active == True)
        return query.all()
    except Exception as e:
        logger.error(f"Trading juftliklari olishda xato: {e}")
        return []

def get_daily_risk_metrics(session: AsyncSession, date: datetime) -> Optional[RiskMetrics]:
    """Kunlik risk ko'rsatkichlarini olish"""
    try:
        return session.query(RiskMetrics).filter(
            RiskMetrics.date >= date.replace(hour=0, minute=0, second=0),
            RiskMetrics.date < date.replace(hour=23, minute=59, second=59)
        ).first()
    except Exception as e:
        logger.error(f"Risk ko'rsatkichlarini olishda xato: {e}")
        return None

def create_signal(session: AsyncSession, signal_data: SignalData) -> Optional[Signal]:
    """Yangi signal yaratish"""
    try:
        # Trading pair topish
        pair = session.query(TradingPair).filter(
            TradingPair.symbol == signal_data.pair_symbol
        ).first()
        
        if not pair:
            logger.error(f"Trading pair topilmadi: {signal_data.pair_symbol}")
            return None
        
        # Signal yaratish
        signal = Signal(
            pair_id=pair.id,
            signal_type=signal_data.signal_type,
            entry_price=signal_data.entry_price,
            stop_loss=signal_data.stop_loss,
            take_profit=signal_data.take_profit,
            lot_size=signal_data.lot_size,
            confidence=signal_data.confidence,
            risk_percent=signal_data.risk_percent,
            reason=signal_data.reason,
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=signal_data.expires_in_minutes)
        )
        
        session.add(signal)
        session.commit()
        session.refresh(signal)
        
        logger.info(f"Yangi signal yaratildi: {signal.id}")
        return signal
        
    except Exception as e:
        logger.error(f"Signal yaratishda xato: {e}")
        session.rollback()
        return None

def update_signal_status(session: AsyncSession, signal_id: int, status: str) -> bool:
    """Signal holatini yangilash"""
    try:
        signal = session.query(Signal).filter(Signal.id == signal_id).first()
        if not signal:
            logger.error(f"Signal topilmadi: {signal_id}")
            return False
        
        signal.signal_status = status
        if status == "EXECUTED":
            signal.executed_at = datetime.now(timezone.utc)
        
        session.commit()
        logger.info(f"Signal holati yangilandi: {signal_id} -> {status}")
        return True
        
    except Exception as e:
        logger.error(f"Signal holatini yangilashda xato: {e}")
        session.rollback()
        return False

async def cleanup_old_data(session: AsyncSession, days_to_keep: int = 30) -> None:
    """Eski ma'lumotlarni tozalash"""
    try:
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_to_keep)
        
        # Eski market datalarni o'chirish
        old_market_data = session.query(MarketData).filter(
            MarketData.timestamp < cutoff_date
        ).delete()
        
        # Eski loglarni o'chirish
        old_logs = session.query(SystemLog).filter(
            SystemLog.timestamp < cutoff_date
        ).delete()
        
        # Eski sentiment datalarni o'chirish
        old_sentiment = session.query(SentimentData).filter(
            SentimentData.created_at < cutoff_date
        ).delete()
        
        session.commit()
        logger.info(f"Eski ma'lumotlar tozalandi: {old_market_data + old_logs + old_sentiment} ta yozuv")
        
    except Exception as e:
        logger.error(f"Ma'lumotlar tozalashda xato: {e}")
        session.rollback()

logger.info("Database models yuklandi - barcha modellar tayyor")

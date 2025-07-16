"""
Database modul - AI OrderFlow & Signal Bot uchun
====================================================

Bu modul database bilan ishlash uchun barcha kerakli komponentlarni ta'minlaydi:
- Database manager
- Data modellari
- Migration tizimi
- Connection pool
- Async database operatsiyalari

Moduldan foydalanish:
    from database import DatabaseManager, SignalModel, TradeModel
    
    # Database manager yaratish
    db_manager = DatabaseManager()
    
    # Signal saqlash
    signal = SignalModel(symbol="EURUSD", action="BUY", ...)
    await db_manager.save_signal(signal)
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

# Utils import
from utils.logger import get_logger
from utils.error_handler import handle_database_error

# Database komponentlari import
from .db_manager import DatabaseManager
from .models import (
    Base,
    SignalModel,
    TradeModel,
    OrderFlowModel,
    SentimentModel,
    MarketDataModel,
    UserSettingsModel,
    LogModel,
    BacktestModel,
    PerformanceModel
)

# Logger sozlash
logger = get_logger(__name__)

# Database connection pool
_engine: Optional[AsyncEngine] = None
_session_factory: Optional[sessionmaker] = None
_db_manager: Optional[DatabaseManager] = None

@dataclass
class DatabaseConfig:
    """Database konfiguratsiyasi"""
    url: str
    echo: bool = False
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 3600
    
    def __post_init__(self):
        """Konfiguratsiya validatsiyasi"""
        if not self.url:
            raise ValueError("Database URL talab qilinadi")
        
        if self.pool_size < 1:
            raise ValueError("Pool size 1 dan kichik bo'lishi mumkin emas")

class DatabaseInitializer:
    """Database initialization class"""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self._initialized = False
        
    async def initialize(self) -> None:
        """Database ni ishga tushirish"""
        try:
            global _engine, _session_factory, _db_manager
            
            if self._initialized:
                logger.info("Database allaqachon ishga tushirilgan")
                return
            
            # SQLAlchemy engine yaratish
            _engine = create_async_engine(
                self.config.url,
                echo=self.config.echo,
                pool_size=self.config.pool_size,
                max_overflow=self.config.max_overflow,
                pool_timeout=self.config.pool_timeout,
                pool_recycle=self.config.pool_recycle,
                future=True
            )
            
            # Session factory yaratish
            _session_factory = sessionmaker(
                bind=_engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            # Database manager yaratish
            _db_manager = DatabaseManager(_engine, _session_factory)
            
            # Jadvalar yaratish
            await self._create_tables()
            
            # Migration ishga tushirish
            await self._run_migrations()
            
            self._initialized = True
            logger.info("Database muvaffaqiyatli ishga tushirildi")
            
        except Exception as e:
            logger.error(f"Database ishga tushirishda xato: {e}")
            raise
    
    async def _create_tables(self) -> None:
        """Jadvalar yaratish"""
        try:
            if not _engine:
                raise RuntimeError("Engine ishga tushirilmagan")
                
            async with _engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
                
            logger.info("Database jadvalar yaratildi")
            
        except Exception as e:
            logger.error(f"Jadvalar yaratishda xato: {e}")
            raise
    
    async def _run_migrations(self) -> None:
        """Migration ishga tushirish"""
        try:
            # Migration logikasi bu yerda bo'ladi
            # Hozircha basic version
            logger.info("Migration tekshirildi")
            
        except Exception as e:
            logger.error(f"Migration xatosi: {e}")
            raise
    
    async def close(self) -> None:
        """Database connection yopish"""
        try:
            global _engine, _session_factory, _db_manager
            
            if _engine:
                await _engine.dispose()
                logger.info("Database connection yopildi")
            
            _engine = None
            _session_factory = None
            _db_manager = None
            self._initialized = False
            
        except Exception as e:
            logger.error(f"Database yopishda xato: {e}")
            raise

# Global functions
async def init_database(database_url: str, **kwargs) -> DatabaseManager:
    """Database ni ishga tushirish - global function"""
    try:
        config = DatabaseConfig(url=database_url, **kwargs)
        initializer = DatabaseInitializer(config)
        await initializer.initialize()
        
        return get_database_manager()
        
    except Exception as e:
        logger.error(f"Database init da xato: {e}")
        raise

def get_database_manager() -> DatabaseManager:
    """Database manager olish"""
    if not _db_manager:
        raise RuntimeError("Database ishga tushirilmagan. Avval init_database() chaqiring")
    return _db_manager

def get_engine() -> AsyncEngine:
    """Database engine olish"""
    if not _engine:
        raise RuntimeError("Database engine ishga tushirilmagan")
    return _engine

def get_session_factory() -> sessionmaker:
    """Session factory olish"""
    if not _session_factory:
        raise RuntimeError("Session factory ishga tushirilmagan")
    return _session_factory

async def get_async_session() -> AsyncSession:
    """Async session olish"""
    session_factory = get_session_factory()
    return session_factory()

@handle_database_error
async def health_check() -> Dict[str, Any]:
    """Database sog'ligi tekshirish"""
    try:
        if not _engine:
            return {
                "status": "unhealthy",
                "error": "Database engine ishga tushirilmagan"
            }
        
        # Simple query ishga tushirish
        async with _engine.begin() as conn:
            result = await conn.execute("SELECT 1")
            await result.fetchone()
        
        return {
            "status": "healthy",
            "engine": str(_engine.url),
            "pool_size": _engine.pool.size(),
            "checked_in": _engine.pool.checkedin(),
            "checked_out": _engine.pool.checkedout()
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

# Cleanup function
async def cleanup_database():
    """Database tozalash"""
    try:
        if _db_manager:
            await _db_manager.cleanup()
            
        if _engine:
            await _engine.dispose()
            
        logger.info("Database tozalandi")
        
    except Exception as e:
        logger.error(f"Database tozalashda xato: {e}")

# Context manager
class DatabaseContext:
    """Database context manager"""
    
    def __init__(self, database_url: str, **kwargs):
        self.database_url = database_url
        self.kwargs = kwargs
        self.db_manager: Optional[DatabaseManager] = None
    
    async def __aenter__(self) -> DatabaseManager:
        """Context manager kirish"""
        self.db_manager = await init_database(self.database_url, **self.kwargs)
        return self.db_manager
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager chiqish"""
        await cleanup_database()

# Modul eksportlari
__all__ = [
    # Database manager
    'DatabaseManager',
    
    # Models
    'Base',
    'SignalModel',
    'TradeModel',
    'OrderFlowModel',
    'SentimentModel',
    'MarketDataModel',
    'UserSettingsModel',
    'LogModel',
    'BacktestModel',
    'PerformanceModel',
    
    # Config
    'DatabaseConfig',
    'DatabaseInitializer',
    
    # Functions
    'init_database',
    'get_database_manager',
    'get_engine',
    'get_session_factory',
    'get_async_session',
    'health_check',
    'cleanup_database',
    
    # Context manager
    'DatabaseContext'
]

# Modul ishga tushganda log
logger.info("Database modul yuklandi")

# Modul versiyasi
__version__ = "1.0.0"
__author__ = "AI OrderFlow Bot"
__description__ = "Database management modul - async SQLAlchemy bilan"

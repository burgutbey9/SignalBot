"""
Database Manager - Ma'lumotlar bazasi boshqaruv tizimi
SQLAlchemy async ORM va migratsiya tizimi bilan
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import select, update, delete, and_, or_, func, text
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.exc import SQLAlchemyError
import sqlite3
import aiosqlite

from utils.logger import get_logger
from utils.error_handler import handle_database_error
from utils.retry_handler import retry_async
from config.config import ConfigManager
from database.models import (
    Base, Signal, Trade, OrderFlow, Sentiment, MarketData, 
    News, Portfolio, RiskMetrics, BacktestResult, APIStatus
)

logger = get_logger(__name__)

@dataclass
class DatabaseStats:
    """Database statistika ma'lumotlari"""
    total_signals: int = 0
    total_trades: int = 0
    successful_trades: int = 0
    failed_trades: int = 0
    total_profit: float = 0.0
    win_rate: float = 0.0
    avg_profit: float = 0.0
    last_update: str = ""

@dataclass
class QueryResult:
    """So'rov natijasi"""
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    rows_affected: int = 0

class DatabaseManager:
    """Database boshqaruv tizimi"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        self.engine = None
        self.session_factory = None
        self.is_connected = False
        logger.info("DatabaseManager yaratildi")
    
    async def initialize(self) -> bool:
        """Database ulanish va jadvallar yaratish"""
        try:
            # Database URL olish
            db_url = self.config.get_database_url()
            logger.info(f"Database ga ulanish: {db_url}")
            
            # Async engine yaratish
            self.engine = create_async_engine(
                db_url,
                echo=False,  # SQL query log
                pool_size=20,
                max_overflow=30,
                pool_timeout=30,
                pool_recycle=3600
            )
            
            # Session factory yaratish
            self.session_factory = async_sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            # Jadvallar yaratish
            await self.create_tables()
            
            # Ulanish tekshirish
            await self.test_connection()
            
            self.is_connected = True
            logger.info("Database muvaffaqiyatli ishga tushdi")
            return True
            
        except Exception as e:
            logger.error(f"Database ishga tushirishda xato: {e}")
            self.is_connected = False
            return False
    
    async def create_tables(self) -> None:
        """Jadvallar yaratish"""
        try:
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            logger.info("Jadvallar muvaffaqiyatli yaratildi")
        except Exception as e:
            logger.error(f"Jadval yaratishda xato: {e}")
            raise
    
    async def test_connection(self) -> bool:
        """Database ulanish tekshirish"""
        try:
            async with self.get_session() as session:
                result = await session.execute(text("SELECT 1"))
                await session.commit()
                return True
        except Exception as e:
            logger.error(f"Database ulanish tekshirishda xato: {e}")
            return False
    
    @asynccontextmanager
    async def get_session(self):
        """Session context manager"""
        session = self.session_factory()
        try:
            yield session
        except Exception as e:
            await session.rollback()
            logger.error(f"Session xatosi: {e}")
            raise
        finally:
            await session.close()
    
    @retry_async(max_retries=3, delay=1)
    async def save_signal(self, signal_data: Dict) -> QueryResult:
        """Signal ma'lumotlarini saqlash"""
        try:
            async with self.get_session() as session:
                signal = Signal(
                    symbol=signal_data['symbol'],
                    action=signal_data['action'],
                    price=signal_data['price'],
                    confidence=signal_data['confidence'],
                    stop_loss=signal_data.get('stop_loss'),
                    take_profit=signal_data.get('take_profit'),
                    lot_size=signal_data.get('lot_size', 0.1),
                    risk_percent=signal_data.get('risk_percent', 2.0),
                    reason=signal_data.get('reason', ''),
                    status='pending',
                    metadata=json.dumps(signal_data.get('metadata', {}))
                )
                
                session.add(signal)
                await session.commit()
                
                logger.info(f"Signal saqlandi: {signal.id}")
                return QueryResult(
                    success=True,
                    data={'signal_id': signal.id},
                    rows_affected=1
                )
                
        except Exception as e:
            logger.error(f"Signal saqlashda xato: {e}")
            return QueryResult(success=False, error=str(e))
    
    @retry_async(max_retries=3, delay=1)
    async def save_trade(self, trade_data: Dict) -> QueryResult:
        """Trade ma'lumotlarini saqlash"""
        try:
            async with self.get_session() as session:
                trade = Trade(
                    signal_id=trade_data.get('signal_id'),
                    symbol=trade_data['symbol'],
                    action=trade_data['action'],
                    entry_price=trade_data['entry_price'],
                    lot_size=trade_data['lot_size'],
                    stop_loss=trade_data.get('stop_loss'),
                    take_profit=trade_data.get('take_profit'),
                    status='open',
                    platform=trade_data.get('platform', 'mt5'),
                    ticket=trade_data.get('ticket'),
                    metadata=json.dumps(trade_data.get('metadata', {}))
                )
                
                session.add(trade)
                await session.commit()
                
                logger.info(f"Trade saqlandi: {trade.id}")
                return QueryResult(
                    success=True,
                    data={'trade_id': trade.id},
                    rows_affected=1
                )
                
        except Exception as e:
            logger.error(f"Trade saqlashda xato: {e}")
            return QueryResult(success=False, error=str(e))
    
    @retry_async(max_retries=3, delay=1)
    async def update_trade(self, trade_id: int, update_data: Dict) -> QueryResult:
        """Trade ma'lumotlarini yangilash"""
        try:
            async with self.get_session() as session:
                stmt = update(Trade).where(Trade.id == trade_id).values(**update_data)
                result = await session.execute(stmt)
                await session.commit()
                
                logger.info(f"Trade yangilandi: {trade_id}")
                return QueryResult(
                    success=True,
                    rows_affected=result.rowcount
                )
                
        except Exception as e:
            logger.error(f"Trade yangilashda xato: {e}")
            return QueryResult(success=False, error=str(e))
    
    @retry_async(max_retries=3, delay=1)
    async def save_order_flow(self, flow_data: Dict) -> QueryResult:
        """Order flow ma'lumotlarini saqlash"""
        try:
            async with self.get_session() as session:
                order_flow = OrderFlow(
                    symbol=flow_data['symbol'],
                    buy_volume=flow_data.get('buy_volume', 0),
                    sell_volume=flow_data.get('sell_volume', 0),
                    net_flow=flow_data.get('net_flow', 0),
                    price=flow_data['price'],
                    source=flow_data.get('source', 'oneinch'),
                    metadata=json.dumps(flow_data.get('metadata', {}))
                )
                
                session.add(order_flow)
                await session.commit()
                
                return QueryResult(
                    success=True,
                    data={'flow_id': order_flow.id},
                    rows_affected=1
                )
                
        except Exception as e:
            logger.error(f"Order flow saqlashda xato: {e}")
            return QueryResult(success=False, error=str(e))
    
    @retry_async(max_retries=3, delay=1)
    async def save_sentiment(self, sentiment_data: Dict) -> QueryResult:
        """Sentiment ma'lumotlarini saqlash"""
        try:
            async with self.get_session() as session:
                sentiment = Sentiment(
                    symbol=sentiment_data['symbol'],
                    score=sentiment_data['score'],
                    confidence=sentiment_data['confidence'],
                    sentiment_type=sentiment_data.get('type', 'general'),
                    source=sentiment_data.get('source', 'huggingface'),
                    text_analyzed=sentiment_data.get('text', ''),
                    metadata=json.dumps(sentiment_data.get('metadata', {}))
                )
                
                session.add(sentiment)
                await session.commit()
                
                return QueryResult(
                    success=True,
                    data={'sentiment_id': sentiment.id},
                    rows_affected=1
                )
                
        except Exception as e:
            logger.error(f"Sentiment saqlashda xato: {e}")
            return QueryResult(success=False, error=str(e))
    
    @retry_async(max_retries=3, delay=1)
    async def save_market_data(self, market_data: Dict) -> QueryResult:
        """Market data saqlash"""
        try:
            async with self.get_session() as session:
                market = MarketData(
                    symbol=market_data['symbol'],
                    open_price=market_data['open'],
                    high_price=market_data['high'],
                    low_price=market_data['low'],
                    close_price=market_data['close'],
                    volume=market_data.get('volume', 0),
                    timeframe=market_data.get('timeframe', '1m'),
                    source=market_data.get('source', 'ccxt'),
                    metadata=json.dumps(market_data.get('metadata', {}))
                )
                
                session.add(market)
                await session.commit()
                
                return QueryResult(
                    success=True,
                    data={'market_id': market.id},
                    rows_affected=1
                )
                
        except Exception as e:
            logger.error(f"Market data saqlashda xato: {e}")
            return QueryResult(success=False, error=str(e))
    
    @retry_async(max_retries=3, delay=1)
    async def save_news(self, news_data: Dict) -> QueryResult:
        """Yangilik ma'lumotlarini saqlash"""
        try:
            async with self.get_session() as session:
                news = News(
                    title=news_data['title'],
                    content=news_data.get('content', ''),
                    source=news_data.get('source', 'newsapi'),
                    url=news_data.get('url', ''),
                    sentiment_score=news_data.get('sentiment_score', 0),
                    relevance_score=news_data.get('relevance_score', 0),
                    symbols=json.dumps(news_data.get('symbols', [])),
                    metadata=json.dumps(news_data.get('metadata', {}))
                )
                
                session.add(news)
                await session.commit()
                
                return QueryResult(
                    success=True,
                    data={'news_id': news.id},
                    rows_affected=1
                )
                
        except Exception as e:
            logger.error(f"News saqlashda xato: {e}")
            return QueryResult(success=False, error=str(e))
    
    @retry_async(max_retries=3, delay=1)
    async def get_signals(self, 
                         limit: int = 100,
                         status: Optional[str] = None,
                         symbol: Optional[str] = None,
                         from_date: Optional[datetime] = None) -> QueryResult:
        """Signallar ro'yxatini olish"""
        try:
            async with self.get_session() as session:
                query = select(Signal)
                
                # Filtrlar qo'llash
                if status:
                    query = query.where(Signal.status == status)
                if symbol:
                    query = query.where(Signal.symbol == symbol)
                if from_date:
                    query = query.where(Signal.created_at >= from_date)
                
                query = query.order_by(Signal.created_at.desc()).limit(limit)
                
                result = await session.execute(query)
                signals = result.scalars().all()
                
                return QueryResult(
                    success=True,
                    data=[{
                        'id': s.id,
                        'symbol': s.symbol,
                        'action': s.action,
                        'price': s.price,
                        'confidence': s.confidence,
                        'status': s.status,
                        'created_at': s.created_at.isoformat(),
                        'metadata': json.loads(s.metadata or '{}')
                    } for s in signals]
                )
                
        except Exception as e:
            logger.error(f"Signallar olishda xato: {e}")
            return QueryResult(success=False, error=str(e))
    
    @retry_async(max_retries=3, delay=1)
    async def get_trades(self,
                        limit: int = 100,
                        status: Optional[str] = None,
                        symbol: Optional[str] = None) -> QueryResult:
        """Trade'lar ro'yxatini olish"""
        try:
            async with self.get_session() as session:
                query = select(Trade)
                
                if status:
                    query = query.where(Trade.status == status)
                if symbol:
                    query = query.where(Trade.symbol == symbol)
                
                query = query.order_by(Trade.created_at.desc()).limit(limit)
                
                result = await session.execute(query)
                trades = result.scalars().all()
                
                return QueryResult(
                    success=True,
                    data=[{
                        'id': t.id,
                        'symbol': t.symbol,
                        'action': t.action,
                        'entry_price': t.entry_price,
                        'exit_price': t.exit_price,
                        'lot_size': t.lot_size,
                        'profit': t.profit,
                        'status': t.status,
                        'created_at': t.created_at.isoformat(),
                        'closed_at': t.closed_at.isoformat() if t.closed_at else None
                    } for t in trades]
                )
                
        except Exception as e:
            logger.error(f"Trade'lar olishda xato: {e}")
            return QueryResult(success=False, error=str(e))
    
    @retry_async(max_retries=3, delay=1)
    async def get_portfolio_stats(self) -> QueryResult:
        """Portfolio statistikasi"""
        try:
            async with self.get_session() as session:
                # Umumiy trade'lar
                total_trades = await session.execute(
                    select(func.count(Trade.id))
                )
                total_trades = total_trades.scalar()
                
                # Yopilgan trade'lar
                closed_trades = await session.execute(
                    select(func.count(Trade.id)).where(Trade.status == 'closed')
                )
                closed_trades = closed_trades.scalar()
                
                # Foydali trade'lar
                profitable_trades = await session.execute(
                    select(func.count(Trade.id)).where(
                        and_(Trade.status == 'closed', Trade.profit > 0)
                    )
                )
                profitable_trades = profitable_trades.scalar()
                
                # Umumiy foyda
                total_profit = await session.execute(
                    select(func.sum(Trade.profit)).where(Trade.status == 'closed')
                )
                total_profit = total_profit.scalar() or 0
                
                # Win rate hisoblash
                win_rate = (profitable_trades / closed_trades * 100) if closed_trades > 0 else 0
                
                # O'rtacha foyda
                avg_profit = total_profit / closed_trades if closed_trades > 0 else 0
                
                return QueryResult(
                    success=True,
                    data={
                        'total_trades': total_trades,
                        'closed_trades': closed_trades,
                        'profitable_trades': profitable_trades,
                        'total_profit': total_profit,
                        'win_rate': win_rate,
                        'avg_profit': avg_profit
                    }
                )
                
        except Exception as e:
            logger.error(f"Portfolio statistika olishda xato: {e}")
            return QueryResult(success=False, error=str(e))
    
    @retry_async(max_retries=3, delay=1)
    async def get_recent_order_flow(self, 
                                   symbol: str, 
                                   limit: int = 100) -> QueryResult:
        """So'nggi order flow ma'lumotlari"""
        try:
            async with self.get_session() as session:
                query = select(OrderFlow).where(
                    OrderFlow.symbol == symbol
                ).order_by(OrderFlow.created_at.desc()).limit(limit)
                
                result = await session.execute(query)
                flows = result.scalars().all()
                
                return QueryResult(
                    success=True,
                    data=[{
                        'id': f.id,
                        'buy_volume': f.buy_volume,
                        'sell_volume': f.sell_volume,
                        'net_flow': f.net_flow,
                        'price': f.price,
                        'created_at': f.created_at.isoformat(),
                        'metadata': json.loads(f.metadata or '{}')
                    } for f in flows]
                )
                
        except Exception as e:
            logger.error(f"Order flow olishda xato: {e}")
            return QueryResult(success=False, error=str(e))
    
    @retry_async(max_retries=3, delay=1)
    async def get_sentiment_analysis(self,
                                   symbol: str,
                                   hours: int = 24) -> QueryResult:
        """Sentiment tahlil ma'lumotlari"""
        try:
            async with self.get_session() as session:
                from_time = datetime.utcnow() - timedelta(hours=hours)
                
                query = select(Sentiment).where(
                    and_(
                        Sentiment.symbol == symbol,
                        Sentiment.created_at >= from_time
                    )
                ).order_by(Sentiment.created_at.desc())
                
                result = await session.execute(query)
                sentiments = result.scalars().all()
                
                # O'rtacha sentiment hisoblash
                avg_sentiment = sum(s.score for s in sentiments) / len(sentiments) if sentiments else 0
                
                return QueryResult(
                    success=True,
                    data={
                        'avg_sentiment': avg_sentiment,
                        'total_records': len(sentiments),
                        'sentiments': [{
                            'score': s.score,
                            'confidence': s.confidence,
                            'source': s.source,
                            'created_at': s.created_at.isoformat()
                        } for s in sentiments[:10]]  # Oxirgi 10 ta
                    }
                )
                
        except Exception as e:
            logger.error(f"Sentiment tahlil olishda xato: {e}")
            return QueryResult(success=False, error=str(e))
    
    @retry_async(max_retries=3, delay=1)
    async def update_api_status(self, 
                              api_name: str,
                              status: str,
                              response_time: float = 0) -> QueryResult:
        """API holat yangilash"""
        try:
            async with self.get_session() as session:
                # Upsert operation
                stmt = insert(APIStatus).values(
                    api_name=api_name,
                    status=status,
                    response_time=response_time,
                    last_check=datetime.utcnow()
                )
                stmt = stmt.on_conflict_do_update(
                    index_elements=['api_name'],
                    set_={
                        'status': stmt.excluded.status,
                        'response_time': stmt.excluded.response_time,
                        'last_check': stmt.excluded.last_check
                    }
                )
                
                await session.execute(stmt)
                await session.commit()
                
                return QueryResult(success=True, rows_affected=1)
                
        except Exception as e:
            logger.error(f"API status yangilashda xato: {e}")
            return QueryResult(success=False, error=str(e))
    
    @retry_async(max_retries=3, delay=1)
    async def cleanup_old_data(self, days: int = 30) -> QueryResult:
        """Eski ma'lumotlarni tozalash"""
        try:
            async with self.get_session() as session:
                cutoff_date = datetime.utcnow() - timedelta(days=days)
                
                # Eski order flow ma'lumotlari
                flow_result = await session.execute(
                    delete(OrderFlow).where(OrderFlow.created_at < cutoff_date)
                )
                
                # Eski sentiment ma'lumotlari
                sentiment_result = await session.execute(
                    delete(Sentiment).where(Sentiment.created_at < cutoff_date)
                )
                
                # Eski market data
                market_result = await session.execute(
                    delete(MarketData).where(MarketData.created_at < cutoff_date)
                )
                
                await session.commit()
                
                total_deleted = (flow_result.rowcount + 
                               sentiment_result.rowcount + 
                               market_result.rowcount)
                
                logger.info(f"Eski ma'lumotlar tozalandi: {total_deleted} ta record")
                
                return QueryResult(
                    success=True,
                    data={'deleted_records': total_deleted},
                    rows_affected=total_deleted
                )
                
        except Exception as e:
            logger.error(f"Ma'lumotlar tozalashda xato: {e}")
            return QueryResult(success=False, error=str(e))
    
    async def get_database_stats(self) -> DatabaseStats:
        """Database statistikasi"""
        try:
            async with self.get_session() as session:
                # Signallar soni
                total_signals = await session.execute(select(func.count(Signal.id)))
                total_signals = total_signals.scalar()
                
                # Trade'lar soni
                total_trades = await session.execute(select(func.count(Trade.id)))
                total_trades = total_trades.scalar()
                
                # Muvaffaqiyatli trade'lar
                successful_trades = await session.execute(
                    select(func.count(Trade.id)).where(
                        and_(Trade.status == 'closed', Trade.profit > 0)
                    )
                )
                successful_trades = successful_trades.scalar()
                
                # Muvaffaqiyatsiz trade'lar
                failed_trades = await session.execute(
                    select(func.count(Trade.id)).where(
                        and_(Trade.status == 'closed', Trade.profit <= 0)
                    )
                )
                failed_trades = failed_trades.scalar()
                
                # Umumiy foyda
                total_profit = await session.execute(
                    select(func.sum(Trade.profit)).where(Trade.status == 'closed')
                )
                total_profit = total_profit.scalar() or 0
                
                # Win rate
                closed_trades = successful_trades + failed_trades
                win_rate = (successful_trades / closed_trades * 100) if closed_trades > 0 else 0
                
                # O'rtacha foyda
                avg_profit = total_profit / closed_trades if closed_trades > 0 else 0
                
                return DatabaseStats(
                    total_signals=total_signals,
                    total_trades=total_trades,
                    successful_trades=successful_trades,
                    failed_trades=failed_trades,
                    total_profit=total_profit,
                    win_rate=win_rate,
                    avg_profit=avg_profit,
                    last_update=datetime.utcnow().isoformat()
                )
                
        except Exception as e:
            logger.error(f"Database statistika olishda xato: {e}")
            return DatabaseStats()
    
    async def close(self) -> None:
        """Database ulanishni yopish"""
        try:
            if self.engine:
                await self.engine.dispose()
                self.is_connected = False
                logger.info("Database ulanish yopildi")
        except Exception as e:
            logger.error(f"Database yopishda xato: {e}")
    
    async def __aenter__(self):
        """Async context manager kirish"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager chiqish"""
        await self.close()

# Singleton instance
_db_manager = None

async def get_db_manager(config_manager: ConfigManager = None) -> DatabaseManager:
    """Database manager singleton"""
    global _db_manager
    if _db_manager is None:
        if config_manager is None:
            from config.config import ConfigManager
            config_manager = ConfigManager()
        _db_manager = DatabaseManager(config_manager)
        await _db_manager.initialize()
    return _db_manager

# Qisqa yo'llar (shortcuts)
async def save_signal(signal_data: Dict) -> QueryResult:
    """Signal saqlash qisqa yo'li"""
    db = await get_db_manager()
    return await db.save_signal(signal_data)

async def save_trade(trade_data: Dict) -> QueryResult:
    """Trade saqlash qisqa yo'li"""
    db = await get_db_manager()
    return await db.save_trade(trade_data)

async def get_portfolio_stats() -> QueryResult:
    """Portfolio statistika qisqa yo'li"""
    db = await get_db_manager()
    return await db.get_portfolio_stats()

# Test funksiyasi
async def test_database():
    """Database test funksiyasi"""
    try:
        from config.config import ConfigManager
        config = ConfigManager()
        
        async with DatabaseManager(config) as db:
            # Test signal saqlash
            test_signal = {
                'symbol': 'EURUSD',
                'action': 'buy',
                'price': 1.0950,
                'confidence': 85.5,
                'stop_loss': 1.0900,
                'take_profit': 1.1000,
                'reason': 'Strong bullish sentiment'
            }
            
            result = await db.save_signal(test_signal)
            print(f"Signal saqlash: {result.success}")
            
            # Test trade saqlash
            test_trade = {
                'symbol': 'EURUSD',
                'action': 'buy',
                'entry_price': 1.0950,
                'lot_size': 0.1,
                'stop_loss': 1.0900,
                'take_profit': 1.1000
            }
            
            result = await db.save_trade(test_trade)
            print(f"Trade saqlash: {result.success}")
            
            # Portfolio statistika
            stats = await db.get_portfolio_stats()
            print(f"Portfolio stats: {stats.data}")
            
            # Database statistika
            db_stats = await db.get_database_stats()
            print(f"Database stats: {db_stats}")
            
    except Exception as e:
        print(f"Test xatosi: {e}")

if __name__ == "__main__":
    asyncio.run(test_database())

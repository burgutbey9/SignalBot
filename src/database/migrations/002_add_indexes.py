"""
Database migratsiya - Index va optimizatsiya qo'shish
Performance va qidiruv tezligini oshirish
"""

import asyncio
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncEngine
from utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class IndexResult:
    """Index yaratish natijasi"""
    success: bool
    index_name: str
    table_name: str
    columns: List[str]
    execution_time: float
    error: Optional[str] = None

@dataclass
class MigrationResult:
    """Migratsiya natijasi"""
    success: bool
    message: str
    indexes_created: List[IndexResult]
    total_execution_time: float
    error: Optional[str] = None

class AddIndexes:
    """Database index va optimizatsiyalarni qo'shish"""
    
    def __init__(self, engine: AsyncEngine):
        self.engine = engine
        self.indexes_created = []
        self.total_execution_time = 0.0
        logger.info("AddIndexes migratsiya tayyor")
    
    async def up(self) -> MigrationResult:
        """Migratsiyani yuqoriga ko'tarish - indexlarni yaratish"""
        try:
            start_time = datetime.now()
            logger.info("Index yaratish boshlandi")
            
            # Performance indexlari
            await self.create_performance_indexes()
            
            # Qidiruv indexlari
            await self.create_search_indexes()
            
            # Foreign key indexlari
            await self.create_foreign_key_indexes()
            
            # Compound indexlari
            await self.create_compound_indexes()
            
            # Partial indexlari
            await self.create_partial_indexes()
            
            # Unique indexlari
            await self.create_unique_indexes()
            
            end_time = datetime.now()
            self.total_execution_time = (end_time - start_time).total_seconds()
            
            logger.info(f"Jami {len(self.indexes_created)} ta index yaratildi")
            logger.info(f"Jami vaqt: {self.total_execution_time:.2f} soniya")
            
            return MigrationResult(
                success=True,
                message="Indexlar muvaffaqiyatli yaratildi",
                indexes_created=self.indexes_created,
                total_execution_time=self.total_execution_time
            )
            
        except Exception as e:
            logger.error(f"Index yaratishda xato: {e}")
            return MigrationResult(
                success=False,
                message="Index yaratishda xato",
                indexes_created=self.indexes_created,
                total_execution_time=self.total_execution_time,
                error=str(e)
            )
    
    async def down(self) -> MigrationResult:
        """Migratsiyani pastga tushirish - indexlarni o'chirish"""
        try:
            start_time = datetime.now()
            logger.info("Index o'chirish boshlandi")
            
            # Barcha indexlarni o'chirish
            indexes_to_drop = [
                # Performance indexes
                'idx_signals_created_at', 'idx_signals_status', 'idx_signals_symbol',
                'idx_trades_created_at', 'idx_trades_status', 'idx_trades_symbol',
                'idx_portfolio_updated_at', 'idx_api_logs_created_at',
                'idx_backtest_start_date', 'idx_strategies_last_used',
                
                # Search indexes
                'idx_users_username', 'idx_users_telegram_id', 'idx_users_email',
                'idx_signals_confidence', 'idx_trades_profit_loss',
                'idx_settings_category_key', 'idx_api_logs_api_name',
                'idx_backtest_symbol', 'idx_strategies_name',
                
                # Foreign key indexes
                'idx_signals_user_id', 'idx_trades_signal_id', 'idx_trades_user_id',
                'idx_portfolio_user_id', 'idx_api_logs_user_id',
                'idx_backtest_strategy_id', 'idx_backtest_user_id',
                'idx_strategies_user_id',
                
                # Compound indexes
                'idx_signals_user_symbol_status', 'idx_trades_user_symbol_status',
                'idx_api_logs_name_status', 'idx_backtest_strategy_date',
                'idx_strategies_user_status',
                
                # Partial indexes
                'idx_signals_active', 'idx_trades_open', 'idx_users_active',
                'idx_strategies_active', 'idx_api_logs_errors',
                
                # Unique indexes
                'idx_users_username_unique', 'idx_users_telegram_unique',
                'idx_settings_category_key_unique'
            ]
            
            async with self.engine.begin() as conn:
                for index_name in indexes_to_drop:
                    try:
                        await conn.execute(f"DROP INDEX IF EXISTS {index_name}")
                        logger.info(f"Index o'chirildi: {index_name}")
                    except Exception as e:
                        logger.warning(f"Index o'chirishda xato {index_name}: {e}")
            
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            return MigrationResult(
                success=True,
                message="Indexlar muvaffaqiyatli o'chirildi",
                indexes_created=[],
                total_execution_time=execution_time
            )
            
        except Exception as e:
            logger.error(f"Index o'chirishda xato: {e}")
            return MigrationResult(
                success=False,
                message="Index o'chirishda xato",
                indexes_created=[],
                total_execution_time=0.0,
                error=str(e)
            )
    
    async def create_performance_indexes(self) -> None:
        """Performance uchun asosiy indexlar yaratish"""
        logger.info("Performance indexlari yaratilmoqda...")
        
        # Vaqt asosidagi indexlar - eng ko'p ishlatiladigan
        performance_indexes = [
            # Signals jadvali - vaqt bo'yicha tezkor qidiruv
            ("idx_signals_created_at", "signals", ["created_at DESC"]),
            ("idx_signals_status", "signals", ["status"]),
            ("idx_signals_symbol", "signals", ["symbol"]),
            
            # Trades jadvali - treyding tarixini tezkor olish
            ("idx_trades_created_at", "trades", ["created_at DESC"]),
            ("idx_trades_status", "trades", ["status"]),
            ("idx_trades_symbol", "trades", ["symbol"]),
            ("idx_trades_opened_at", "trades", ["opened_at DESC"]),
            ("idx_trades_closed_at", "trades", ["closed_at DESC"]),
            
            # Portfolio jadvali - real-time yangilanish
            ("idx_portfolio_updated_at", "portfolio", ["last_updated DESC"]),
            
            # API logs - monitoring uchun
            ("idx_api_logs_created_at", "api_logs", ["created_at DESC"]),
            
            # Backtest - tarix bo'yicha qidiruv
            ("idx_backtest_start_date", "backtest_results", ["start_date DESC"]),
            ("idx_backtest_end_date", "backtest_results", ["end_date DESC"]),
            
            # Strategies - oxirgi foydalanish
            ("idx_strategies_last_used", "strategies", ["last_used DESC"]),
            ("idx_strategies_updated_at", "strategies", ["updated_at DESC"]),
        ]
        
        for index_name, table_name, columns in performance_indexes:
            await self._create_index(index_name, table_name, columns)
    
    async def create_search_indexes(self) -> None:
        """Qidiruv uchun indexlar yaratish"""
        logger.info("Qidiruv indexlari yaratilmoqda...")
        
        search_indexes = [
            # Users jadvali - foydalanuvchi qidiruvi
            ("idx_users_username", "users", ["username"]),
            ("idx_users_telegram_id", "users", ["telegram_id"]),
            ("idx_users_email", "users", ["email"]),
            ("idx_users_telegram_username", "users", ["telegram_username"]),
            
            # Signals jadvali - signal qidiruvi
            ("idx_signals_confidence", "signals", ["confidence DESC"]),
            ("idx_signals_price", "signals", ["price"]),
            ("idx_signals_lot_size", "signals", ["lot_size"]),
            
            # Trades jadvali - treyding qidiruvi
            ("idx_trades_profit_loss", "trades", ["profit_loss DESC"]),
            ("idx_trades_entry_price", "trades", ["entry_price"]),
            ("idx_trades_exit_price", "trades", ["exit_price"]),
            ("idx_trades_duration", "trades", ["duration_minutes"]),
            
            # Settings jadvali - sozlamalar qidiruvi
            ("idx_settings_category_key", "settings", ["category", "key"]),
            ("idx_settings_category", "settings", ["category"]),
            
            # API logs - API monitoring
            ("idx_api_logs_api_name", "api_logs", ["api_name"]),
            ("idx_api_logs_status_code", "api_logs", ["status_code"]),
            ("idx_api_logs_response_time", "api_logs", ["response_time"]),
            
            # Backtest - natijalar qidiruvi
            ("idx_backtest_symbol", "backtest_results", ["symbol"]),
            ("idx_backtest_win_rate", "backtest_results", ["win_rate DESC"]),
            ("idx_backtest_profit_loss", "backtest_results", ["total_profit_loss DESC"]),
            
            # Strategies - strategiya qidiruvi
            ("idx_strategies_name", "strategies", ["name"]),
            ("idx_strategies_type", "strategies", ["type"]),
            ("idx_strategies_win_rate", "strategies", ["win_rate DESC"]),
        ]
        
        for index_name, table_name, columns in search_indexes:
            await self._create_index(index_name, table_name, columns)
    
    async def create_foreign_key_indexes(self) -> None:
        """Foreign key bog'lanishlar uchun indexlar"""
        logger.info("Foreign key indexlari yaratilmoqda...")
        
        fk_indexes = [
            # Signals jadvali foreign keys
            ("idx_signals_user_id", "signals", ["user_id"]),
            
            # Trades jadvali foreign keys
            ("idx_trades_signal_id", "trades", ["signal_id"]),
            ("idx_trades_user_id", "trades", ["user_id"]),
            
            # Portfolio jadvali foreign keys
            ("idx_portfolio_user_id", "portfolio", ["user_id"]),
            
            # API logs foreign keys
            ("idx_api_logs_user_id", "api_logs", ["user_id"]),
            
            # Backtest foreign keys
            ("idx_backtest_strategy_id", "backtest_results", ["strategy_id"]),
            ("idx_backtest_user_id", "backtest_results", ["user_id"]),
            
            # Strategies foreign keys
            ("idx_strategies_user_id", "strategies", ["user_id"]),
        ]
        
        for index_name, table_name, columns in fk_indexes:
            await self._create_index(index_name, table_name, columns)
    
    async def create_compound_indexes(self) -> None:
        """Compound (murakkab) indexlar yaratish"""
        logger.info("Compound indexlari yaratilmoqda...")
        
        compound_indexes = [
            # Foydalanuvchi + Symbol + Status kombinatsiyasi
            ("idx_signals_user_symbol_status", "signals", ["user_id", "symbol", "status"]),
            ("idx_trades_user_symbol_status", "trades", ["user_id", "symbol", "status"]),
            
            # Vaqt + Status kombinatsiyasi
            ("idx_signals_status_created", "signals", ["status", "created_at DESC"]),
            ("idx_trades_status_opened", "trades", ["status", "opened_at DESC"]),
            
            # API name + Status kombinatsiyasi
            ("idx_api_logs_name_status", "api_logs", ["api_name", "success", "created_at DESC"]),
            
            # Backtest Strategy + Date kombinatsiyasi
            ("idx_backtest_strategy_date", "backtest_results", ["strategy_id", "start_date DESC"]),
            
            # User + Status kombinatsiyasi
            ("idx_strategies_user_status", "strategies", ["user_id", "status"]),
            
            # Symbol + Timeframe kombinatsiyasi
            ("idx_backtest_symbol_timeframe", "backtest_results", ["symbol", "timeframe"]),
            
            # Profit/Loss + Date kombinatsiyasi
            ("idx_trades_profit_date", "trades", ["profit_loss DESC", "created_at DESC"]),
            
            # Confidence + Symbol kombinatsiyasi
            ("idx_signals_confidence_symbol", "signals", ["confidence DESC", "symbol"]),
        ]
        
        for index_name, table_name, columns in compound_indexes:
            await self._create_index(index_name, table_name, columns)
    
    async def create_partial_indexes(self) -> None:
        """Partial (qisman) indexlar yaratish - faqat ma'lum shartlarga mos qatorlar"""
        logger.info("Partial indexlari yaratilmoqda...")
        
        partial_indexes = [
            # Faqat aktiv signallar
            ("idx_signals_active", "signals", ["created_at DESC"], "status IN ('pending', 'sent')"),
            
            # Faqat ochiq treydinglari
            ("idx_trades_open", "trades", ["opened_at DESC"], "status = 'open'"),
            
            # Faqat aktiv foydalanuvchilar
            ("idx_users_active", "users", ["last_activity DESC"], "is_active = 1"),
            
            # Faqat aktiv strategiyalar
            ("idx_strategies_active", "strategies", ["last_used DESC"], "status = 'active'"),
            
            # Faqat xatolar
            ("idx_api_logs_errors", "api_logs", ["created_at DESC"], "success = 0"),
            
            # Faqat foydali treydinglari
            ("idx_trades_profitable", "trades", ["profit_loss DESC"], "profit_loss > 0"),
            
            # Faqat yuqori confidence signallar
            ("idx_signals_high_confidence", "signals", ["created_at DESC"], "confidence >= 0.7"),
            
            # Faqat premium foydalanuvchilar
            ("idx_users_premium", "users", ["created_at DESC"], "is_premium = 1"),
        ]
        
        for index_name, table_name, columns, condition in partial_indexes:
            await self._create_partial_index(index_name, table_name, columns, condition)
    
    async def create_unique_indexes(self) -> None:
        """Unique indexlar yaratish - dublikatlarni oldini olish"""
        logger.info("Unique indexlari yaratilmoqda...")
        
        unique_indexes = [
            # Foydalanuvchi username unique
            ("idx_users_username_unique", "users", ["username"]),
            
            # Telegram ID unique
            ("idx_users_telegram_unique", "users", ["telegram_id"]),
            
            # Settings category + key unique
            ("idx_settings_category_key_unique", "settings", ["category", "key"]),
            
            # Portfolio user_id unique
            ("idx_portfolio_user_unique", "portfolio", ["user_id"]),
            
            # Trade ID unique (broker tomonidan)
            ("idx_trades_trade_id_unique", "trades", ["trade_id"]),
        ]
        
        for index_name, table_name, columns in unique_indexes:
            await self._create_unique_index(index_name, table_name, columns)
    
    async def _create_index(self, index_name: str, table_name: str, columns: List[str]) -> None:
        """Oddiy index yaratish"""
        try:
            start_time = datetime.now()
            columns_str = ", ".join(columns)
            
            sql = f"CREATE INDEX IF NOT EXISTS {index_name} ON {table_name} ({columns_str})"
            
            async with self.engine.begin() as conn:
                await conn.execute(sql)
            
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            result = IndexResult(
                success=True,
                index_name=index_name,
                table_name=table_name,
                columns=columns,
                execution_time=execution_time
            )
            
            self.indexes_created.append(result)
            logger.info(f"Index yaratildi: {index_name} ({execution_time:.3f}s)")
            
        except Exception as e:
            logger.error(f"Index yaratishda xato {index_name}: {e}")
            result = IndexResult(
                success=False,
                index_name=index_name,
                table_name=table_name,
                columns=columns,
                execution_time=0.0,
                error=str(e)
            )
            self.indexes_created.append(result)
    
    async def _create_partial_index(self, index_name: str, table_name: str, 
                                   columns: List[str], condition: str) -> None:
        """Partial index yaratish"""
        try:
            start_time = datetime.now()
            columns_str = ", ".join(columns)
            
            sql = f"""
            CREATE INDEX IF NOT EXISTS {index_name} 
            ON {table_name} ({columns_str}) 
            WHERE {condition}
            """
            
            async with self.engine.begin() as conn:
                await conn.execute(sql)
            
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            result = IndexResult(
                success=True,
                index_name=index_name,
                table_name=table_name,
                columns=columns,
                execution_time=execution_time
            )
            
            self.indexes_created.append(result)
            logger.info(f"Partial index yaratildi: {index_name} ({execution_time:.3f}s)")
            
        except Exception as e:
            logger.error(f"Partial index yaratishda xato {index_name}: {e}")
            result = IndexResult(
                success=False,
                index_name=index_name,
                table_name=table_name,
                columns=columns,
                execution_time=0.0,
                error=str(e)
            )
            self.indexes_created.append(result)
    
    async def _create_unique_index(self, index_name: str, table_name: str, 
                                  columns: List[str]) -> None:
        """Unique index yaratish"""
        try:
            start_time = datetime.now()
            columns_str = ", ".join(columns)
            
            sql = f"CREATE UNIQUE INDEX IF NOT EXISTS {index_name} ON {table_name} ({columns_str})"
            
            async with self.engine.begin() as conn:
                await conn.execute(sql)
            
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            result = IndexResult(
                success=True,
                index_name=index_name,
                table_name=table_name,
                columns=columns,
                execution_time=execution_time
            )
            
            self.indexes_created.append(result)
            logger.info(f"Unique index yaratildi: {index_name} ({execution_time:.3f}s)")
            
        except Exception as e:
            logger.error(f"Unique index yaratishda xato {index_name}: {e}")
            result = IndexResult(
                success=False,
                index_name=index_name,
                table_name=table_name,
                columns=columns,
                execution_time=0.0,
                error=str(e)
            )
            self.indexes_created.append(result)
    
    async def analyze_table_statistics(self) -> Dict[str, Dict]:
        """Jadval statistikalarini tahlil qilish"""
        try:
            stats = {}
            tables = ['users', 'signals', 'trades', 'portfolio', 'settings', 
                     'api_logs', 'backtest_results', 'strategies']
            
            async with self.engine.begin() as conn:
                for table in tables:
                    # Qatorlar soni
                    result = await conn.execute(f"SELECT COUNT(*) FROM {table}")
                    row_count = result.scalar()
                    
                    # Jadval hajmi (SQLite uchun taxminiy)
                    stats[table] = {
                        'row_count': row_count,
                        'estimated_size_mb': row_count * 0.001  # taxminiy hajm
                    }
            
            logger.info(f"Jadval statistikalari: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"Statistika tahlilida xato: {e}")
            return {}
    
    async def optimize_database(self) -> None:
        """Database optimizatsiya qilish"""
        try:
            logger.info("Database optimizatsiya boshlandi")
            
            async with self.engine.begin() as conn:
                # SQLite optimizatsiya buyruqlari
                await conn.execute("PRAGMA optimize")
                await conn.execute("VACUUM")
                await conn.execute("ANALYZE")
            
            logger.info("Database optimizatsiya tugadi")
            
        except Exception as e:
            logger.error(f"Database optimizatsiyada xato: {e}")

# Migratsiya versiyasi va meta ma'lumotlar
MIGRATION_VERSION = "002"
MIGRATION_NAME = "add_indexes"
MIGRATION_DESCRIPTION = "Database indexlari va optimizatsiya qo'shish"
MIGRATION_AUTHOR = "AI OrderFlow Bot"
MIGRATION_DATE = "2024-01-02"
MIGRATION_DEPENDENCIES = ["001_initial_schema"]

# Migratsiya funksiyalari
async def up(engine: AsyncEngine) -> MigrationResult:
    """Migratsiyani bajarish"""
    migration = AddIndexes(engine)
    result = await migration.up()
    
    # Statistika va optimizatsiya
    await migration.analyze_table_statistics()
    await migration.optimize_database()
    
    return result

async def down(engine: AsyncEngine) -> MigrationResult:
    """Migratsiyani bekor qilish"""
    migration = AddIndexes(engine)
    return await migration.down()

# Test funksiyasi
async def test_migration():
    """Migratsiya testini o'tkazish"""
    from sqlalchemy.ext.asyncio import create_async_engine
    
    # Test database yaratish
    engine = create_async_engine("sqlite+aiosqlite:///test_indexes.db")
    
    try:
        # Avval initial schema yaratish kerak
        from database.migrations.001_initial_schema import up as initial_up
        await initial_up(engine)
        
        # Index migratsiyasini bajarish
        result = await up(engine)
        print(f"Index migration test: {result.success}")
        print(f"Indexes created: {len(result.indexes_created)}")
        print(f"Execution time: {result.total_execution_time:.2f}s")
        
        # Rollback test
        rollback_result = await down(engine)
        print(f"Rollback test: {rollback_result.success}")
        
    finally:
        await engine.dispose()

if __name__ == "__main__":
    # Test ishga tushirish
    asyncio.run(test_migration())

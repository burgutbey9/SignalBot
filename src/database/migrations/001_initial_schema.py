"""
Database migratsiya - Boshlang'ich schema yaratish
Barcha asosiy jadvallarni yaratadi
"""

import asyncio
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Boolean, 
    Text, JSON, ForeignKey, Index, UniqueConstraint
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.asyncio import AsyncEngine
from utils.logger import get_logger

logger = get_logger(__name__)

Base = declarative_base()

@dataclass
class MigrationResult:
    """Migratsiya natijasi"""
    success: bool
    message: str
    tables_created: List[str]
    error: Optional[str] = None

class InitialSchema:
    """Boshlang'ich database schema yaratish"""
    
    def __init__(self, engine: AsyncEngine):
        self.engine = engine
        self.tables_created = []
        logger.info("InitialSchema migratsiya tayyor")
    
    async def up(self) -> MigrationResult:
        """Migratsiyani yuqoriga ko'tarish - jadvallarni yaratish"""
        try:
            logger.info("Boshlang'ich schema yaratish boshlandi")
            
            # Barcha jadvallarni yaratish
            await self.create_users_table()
            await self.create_signals_table()
            await self.create_trades_table()
            await self.create_portfolio_table()
            await self.create_settings_table()
            await self.create_api_logs_table()
            await self.create_backtest_table()
            await self.create_strategies_table()
            
            logger.info(f"Jami {len(self.tables_created)} ta jadval yaratildi")
            return MigrationResult(
                success=True,
                message="Boshlang'ich schema muvaffaqiyatli yaratildi",
                tables_created=self.tables_created
            )
            
        except Exception as e:
            logger.error(f"Schema yaratishda xato: {e}")
            return MigrationResult(
                success=False,
                message="Schema yaratishda xato",
                tables_created=self.tables_created,
                error=str(e)
            )
    
    async def down(self) -> MigrationResult:
        """Migratsiyani pastga tushirish - jadvallarni o'chirish"""
        try:
            logger.info("Schema o'chirish boshlandi")
            
            # Teskari tartibda jadvallarni o'chirish
            tables_to_drop = [
                'strategies', 'backtest_results', 'api_logs', 
                'settings', 'portfolio', 'trades', 'signals', 'users'
            ]
            
            async with self.engine.begin() as conn:
                for table_name in tables_to_drop:
                    await conn.execute(f"DROP TABLE IF EXISTS {table_name}")
                    logger.info(f"Jadval o'chirildi: {table_name}")
            
            return MigrationResult(
                success=True,
                message="Schema muvaffaqiyatli o'chirildi",
                tables_created=[]
            )
            
        except Exception as e:
            logger.error(f"Schema o'chirishda xato: {e}")
            return MigrationResult(
                success=False,
                message="Schema o'chirishda xato",
                tables_created=[],
                error=str(e)
            )
    
    async def create_users_table(self) -> None:
        """Foydalanuvchilar jadvali yaratish"""
        try:
            sql = """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username VARCHAR(100) UNIQUE NOT NULL,
                telegram_id BIGINT UNIQUE,
                telegram_username VARCHAR(100),
                full_name VARCHAR(200),
                phone VARCHAR(20),
                email VARCHAR(100),
                is_active BOOLEAN DEFAULT TRUE,
                is_premium BOOLEAN DEFAULT FALSE,
                trading_mode VARCHAR(20) DEFAULT 'manual',  -- manual, auto
                risk_level VARCHAR(20) DEFAULT 'medium',    -- low, medium, high
                max_daily_risk FLOAT DEFAULT 0.02,          -- 2% kunlik risk
                max_position_size FLOAT DEFAULT 0.01,       -- 1% pozitsiya hajmi
                notification_settings JSON DEFAULT '{}',    -- bildirishnoma sozlamalari
                api_settings JSON DEFAULT '{}',             -- API sozlamalari
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                last_login DATETIME,
                last_activity DATETIME
            )
            """
            
            async with self.engine.begin() as conn:
                await conn.execute(sql)
                logger.info("Users jadvali yaratildi")
                self.tables_created.append("users")
                
        except Exception as e:
            logger.error(f"Users jadval yaratishda xato: {e}")
            raise
    
    async def create_signals_table(self) -> None:
        """Signallar jadvali yaratish"""
        try:
            sql = """
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                symbol VARCHAR(20) NOT NULL,               -- EUR/USD, BTC/USD
                action VARCHAR(10) NOT NULL,               -- BUY, SELL
                signal_type VARCHAR(20) DEFAULT 'ai',      -- ai, manual, copy
                price FLOAT NOT NULL,                      -- kirish narxi
                stop_loss FLOAT,                           -- stop loss
                take_profit FLOAT,                         -- take profit
                lot_size FLOAT DEFAULT 0.01,               -- lot hajmi
                confidence FLOAT DEFAULT 0.0,              -- ishonch darajasi 0-100
                risk_percent FLOAT DEFAULT 0.02,           -- risk foizi
                reason TEXT,                               -- signal sababi
                source VARCHAR(50),                        -- signal manbai
                market_conditions JSON DEFAULT '{}',       -- bozor sharoitlari
                technical_analysis JSON DEFAULT '{}',      -- texnik tahlil
                sentiment_score FLOAT DEFAULT 0.0,         -- sentiment -1 to 1
                order_flow_data JSON DEFAULT '{}',         -- order flow ma'lumotlari
                status VARCHAR(20) DEFAULT 'pending',      -- pending, sent, executed, expired
                sent_at DATETIME,                          -- yuborilgan vaqt
                executed_at DATETIME,                      -- bajarilgan vaqt
                expires_at DATETIME,                       -- tugash vaqti
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
            """
            
            async with self.engine.begin() as conn:
                await conn.execute(sql)
                logger.info("Signals jadvali yaratildi")
                self.tables_created.append("signals")
                
        except Exception as e:
            logger.error(f"Signals jadval yaratishda xato: {e}")
            raise
    
    async def create_trades_table(self) -> None:
        """Treyding jadvali yaratish"""
        try:
            sql = """
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_id INTEGER,
                user_id INTEGER,
                trade_id VARCHAR(100),                     -- broker trade ID
                symbol VARCHAR(20) NOT NULL,
                action VARCHAR(10) NOT NULL,               -- BUY, SELL
                status VARCHAR(20) DEFAULT 'pending',      -- pending, open, closed, cancelled
                entry_price FLOAT,                         -- kirish narxi
                exit_price FLOAT,                          -- chiqish narxi
                current_price FLOAT,                       -- joriy narx
                lot_size FLOAT NOT NULL,
                stop_loss FLOAT,
                take_profit FLOAT,
                commission FLOAT DEFAULT 0.0,              -- komissiya
                swap FLOAT DEFAULT 0.0,                    -- swap
                profit_loss FLOAT DEFAULT 0.0,            -- foyda/zarar
                profit_loss_pips FLOAT DEFAULT 0.0,       -- pips hisobida
                profit_loss_percent FLOAT DEFAULT 0.0,    -- foiz hisobida
                max_profit FLOAT DEFAULT 0.0,             -- maksimal foyda
                max_loss FLOAT DEFAULT 0.0,               -- maksimal zarar
                duration_minutes INTEGER DEFAULT 0,        -- davomiyligi (daqiqa)
                trade_reason TEXT,                         -- treyding sababi
                exit_reason VARCHAR(50),                   -- chiqish sababi
                broker_name VARCHAR(50) DEFAULT 'propshot', -- broker nomi
                account_number VARCHAR(50),                -- akavunt raqami
                magic_number INTEGER,                      -- EA magic number
                opened_at DATETIME,                        -- ochilgan vaqt
                closed_at DATETIME,                        -- yopilgan vaqt
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                
                FOREIGN KEY (signal_id) REFERENCES signals(id),
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
            """
            
            async with self.engine.begin() as conn:
                await conn.execute(sql)
                logger.info("Trades jadvali yaratildi")
                self.tables_created.append("trades")
                
        except Exception as e:
            logger.error(f"Trades jadval yaratishda xato: {e}")
            raise
    
    async def create_portfolio_table(self) -> None:
        """Portfolio jadvali yaratish"""
        try:
            sql = """
            CREATE TABLE IF NOT EXISTS portfolio (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER UNIQUE,
                account_number VARCHAR(50),
                broker_name VARCHAR(50) DEFAULT 'propshot',
                balance FLOAT DEFAULT 0.0,                 -- balans
                equity FLOAT DEFAULT 0.0,                  -- equity
                margin FLOAT DEFAULT 0.0,                  -- margin
                free_margin FLOAT DEFAULT 0.0,             -- erkin margin
                margin_level FLOAT DEFAULT 0.0,            -- margin darajasi
                daily_profit_loss FLOAT DEFAULT 0.0,       -- kunlik P&L
                weekly_profit_loss FLOAT DEFAULT 0.0,      -- haftalik P&L
                monthly_profit_loss FLOAT DEFAULT 0.0,     -- oylik P&L
                total_profit_loss FLOAT DEFAULT 0.0,       -- jami P&L
                daily_trades_count INTEGER DEFAULT 0,      -- kunlik treyding soni
                weekly_trades_count INTEGER DEFAULT 0,     -- haftalik treyding soni
                monthly_trades_count INTEGER DEFAULT 0,    -- oylik treyding soni
                total_trades_count INTEGER DEFAULT 0,      -- jami treyding soni
                win_rate FLOAT DEFAULT 0.0,               -- g'alaba darajasi
                risk_used_today FLOAT DEFAULT 0.0,        -- bugungi ishlatilgan risk
                max_risk_per_day FLOAT DEFAULT 0.02,      -- maksimal kunlik risk
                max_daily_loss FLOAT DEFAULT 0.05,        -- maksimal kunlik zarar
                drawdown_current FLOAT DEFAULT 0.0,       -- joriy drawdown
                drawdown_max FLOAT DEFAULT 0.0,           -- maksimal drawdown
                last_trade_at DATETIME,                   -- oxirgi treyding vaqti
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
            """
            
            async with self.engine.begin() as conn:
                await conn.execute(sql)
                logger.info("Portfolio jadvali yaratildi")
                self.tables_created.append("portfolio")
                
        except Exception as e:
            logger.error(f"Portfolio jadval yaratishda xato: {e}")
            raise
    
    async def create_settings_table(self) -> None:
        """Sozlamalar jadvali yaratish"""
        try:
            sql = """
            CREATE TABLE IF NOT EXISTS settings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category VARCHAR(50) NOT NULL,             -- trading, risk, notification
                key VARCHAR(100) NOT NULL,                 -- setting kaliti
                value TEXT,                                -- setting qiymati
                value_type VARCHAR(20) DEFAULT 'string',   -- string, int, float, bool, json
                description TEXT,                          -- tavsif
                is_user_editable BOOLEAN DEFAULT TRUE,     -- foydalanuvchi o'zgartira oladimi
                is_system BOOLEAN DEFAULT FALSE,           -- tizim sozlamasi
                min_value FLOAT,                           -- minimal qiymat
                max_value FLOAT,                           -- maksimal qiymat
                default_value TEXT,                        -- default qiymat
                validation_rules JSON DEFAULT '{}',        -- validatsiya qoidalari
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                
                UNIQUE(category, key)
            )
            """
            
            async with self.engine.begin() as conn:
                await conn.execute(sql)
                logger.info("Settings jadvali yaratildi")
                self.tables_created.append("settings")
                
        except Exception as e:
            logger.error(f"Settings jadval yaratishda xato: {e}")
            raise
    
    async def create_api_logs_table(self) -> None:
        """API loglar jadvali yaratish"""
        try:
            sql = """
            CREATE TABLE IF NOT EXISTS api_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                api_name VARCHAR(50) NOT NULL,             -- oneinch, alchemy, gemini
                endpoint VARCHAR(200),                     -- API endpoint
                method VARCHAR(10) DEFAULT 'GET',          -- HTTP method
                request_data TEXT,                         -- so'rov ma'lumotlari
                response_data TEXT,                        -- javob ma'lumotlari
                status_code INTEGER,                       -- HTTP status code
                response_time FLOAT,                       -- javob vaqti (ms)
                success BOOLEAN DEFAULT FALSE,             -- muvaffaqiyatli bo'ldimi
                error_message TEXT,                        -- xato xabari
                rate_limit_remaining INTEGER,              -- qolgan rate limit
                fallback_used BOOLEAN DEFAULT FALSE,       -- fallback ishlatildimi
                fallback_reason TEXT,                      -- fallback sababi
                user_id INTEGER,                           -- foydalanuvchi ID
                ip_address VARCHAR(45),                    -- IP manzil
                user_agent TEXT,                           -- user agent
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
            """
            
            async with self.engine.begin() as conn:
                await conn.execute(sql)
                logger.info("API logs jadvali yaratildi")
                self.tables_created.append("api_logs")
                
        except Exception as e:
            logger.error(f"API logs jadval yaratishda xato: {e}")
            raise
    
    async def create_backtest_table(self) -> None:
        """Backtest natijalar jadvali yaratish"""
        try:
            sql = """
            CREATE TABLE IF NOT EXISTS backtest_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_id INTEGER,
                user_id INTEGER,
                name VARCHAR(100) NOT NULL,                -- test nomi
                symbol VARCHAR(20),                        -- test qilingan juftlik
                timeframe VARCHAR(10),                     -- vaqt oralig'i
                start_date DATETIME NOT NULL,              -- boshlanish sanasi
                end_date DATETIME NOT NULL,                -- tugash sanasi
                initial_balance FLOAT DEFAULT 10000.0,     -- boshlang'ich balans
                final_balance FLOAT DEFAULT 0.0,           -- yakuniy balans
                total_profit_loss FLOAT DEFAULT 0.0,       -- jami foyda/zarar
                total_profit_loss_percent FLOAT DEFAULT 0.0, -- foiz hisobida
                max_drawdown FLOAT DEFAULT 0.0,            -- maksimal drawdown
                max_drawdown_percent FLOAT DEFAULT 0.0,    -- foiz hisobida
                sharpe_ratio FLOAT DEFAULT 0.0,            -- Sharpe koeffitsienti
                sortino_ratio FLOAT DEFAULT 0.0,           -- Sortino koeffitsienti
                calmar_ratio FLOAT DEFAULT 0.0,            -- Calmar koeffitsienti
                total_trades INTEGER DEFAULT 0,            -- jami treyding soni
                winning_trades INTEGER DEFAULT 0,          -- g'olib treydinglari
                losing_trades INTEGER DEFAULT 0,           -- mag'lub treydinglari
                win_rate FLOAT DEFAULT 0.0,               -- g'alaba darajasi
                average_win FLOAT DEFAULT 0.0,            -- o'rtacha g'alaba
                average_loss FLOAT DEFAULT 0.0,           -- o'rtacha zarar
                largest_win FLOAT DEFAULT 0.0,            -- eng katta g'alaba
                largest_loss FLOAT DEFAULT 0.0,           -- eng katta zarar
                profit_factor FLOAT DEFAULT 0.0,          -- foyda faktori
                recovery_factor FLOAT DEFAULT 0.0,        -- tiklanish faktori
                max_consecutive_wins INTEGER DEFAULT 0,    -- ketma-ket g'alabalar
                max_consecutive_losses INTEGER DEFAULT 0,  -- ketma-ket zararlari
                average_trade_duration INTEGER DEFAULT 0,  -- o'rtacha treyding davomiyligi
                strategy_parameters JSON DEFAULT '{}',     -- strategiya parametrlari
                performance_metrics JSON DEFAULT '{}',     -- performance ko'rsatkichlari
                equity_curve JSON DEFAULT '{}',           -- equity egri chizig'i
                trade_history JSON DEFAULT '{}',          -- treyding tarixi
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                
                FOREIGN KEY (strategy_id) REFERENCES strategies(id),
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
            """
            
            async with self.engine.begin() as conn:
                await conn.execute(sql)
                logger.info("Backtest results jadvali yaratildi")
                self.tables_created.append("backtest_results")
                
        except Exception as e:
            logger.error(f"Backtest results jadval yaratishda xato: {e}")
            raise
    
    async def create_strategies_table(self) -> None:
        """Strategiyalar jadvali yaratish"""
        try:
            sql = """
            CREATE TABLE IF NOT EXISTS strategies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                name VARCHAR(100) NOT NULL,                -- strategiya nomi
                description TEXT,                          -- tavsif
                type VARCHAR(50) DEFAULT 'ai_signal',      -- ai_signal, technical, fundamental
                category VARCHAR(50) DEFAULT 'scalping',   -- scalping, swing, day_trading
                status VARCHAR(20) DEFAULT 'active',       -- active, inactive, testing
                version VARCHAR(10) DEFAULT '1.0',         -- strategiya versiyasi
                
                -- AI parametrlari
                ai_model VARCHAR(50),                      -- ishlatilgan AI model
                sentiment_weight FLOAT DEFAULT 0.3,       -- sentiment og'irligi
                technical_weight FLOAT DEFAULT 0.4,       -- texnik tahlil og'irligi
                orderflow_weight FLOAT DEFAULT 0.3,       -- order flow og'irligi
                confidence_threshold FLOAT DEFAULT 0.7,    -- minimal ishonch darajasi
                
                -- Risk parametrlari
                max_risk_per_trade FLOAT DEFAULT 0.02,    -- maksimal treyding riski
                max_daily_risk FLOAT DEFAULT 0.05,        -- maksimal kunlik risk
                max_positions INTEGER DEFAULT 3,          -- maksimal pozitsiyalar
                position_size_method VARCHAR(20) DEFAULT 'fixed', -- fixed, kelly, percent
                
                -- Vaqt parametrlari
                trading_hours JSON DEFAULT '{}',          -- treyding vaqtlari
                allowed_days JSON DEFAULT '[]',           -- ruxsat etilgan kunlar
                session_filters JSON DEFAULT '{}',        -- sessiya filtrlari
                
                -- Texnik parametrlari
                symbols JSON DEFAULT '[]',                 -- treyding juftliklari
                timeframes JSON DEFAULT '[]',              -- vaqt oralig'lari
                indicators JSON DEFAULT '{}',              -- indikatorlar
                entry_rules JSON DEFAULT '{}',            -- kirish qoidalari
                exit_rules JSON DEFAULT '{}',             -- chiqish qoidalari
                
                -- Performance ko'rsatkichlari
                total_trades INTEGER DEFAULT 0,
                winning_trades INTEGER DEFAULT 0,
                losing_trades INTEGER DEFAULT 0,
                win_rate FLOAT DEFAULT 0.0,
                profit_factor FLOAT DEFAULT 0.0,
                sharpe_ratio FLOAT DEFAULT 0.0,
                max_drawdown FLOAT DEFAULT 0.0,
                total_profit_loss FLOAT DEFAULT 0.0,
                
                -- Meta ma'lumotlar
                last_optimized DATETIME,                  -- oxirgi optimizatsiya
                last_backtest DATETIME,                   -- oxirgi backtest
                last_used DATETIME,                       -- oxirgi foydalanish
                use_count INTEGER DEFAULT 0,              -- foydalanish soni
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
            """
            
            async with self.engine.begin() as conn:
                await conn.execute(sql)
                logger.info("Strategies jadvali yaratildi")
                self.tables_created.append("strategies")
                
        except Exception as e:
            logger.error(f"Strategies jadval yaratishda xato: {e}")
            raise

# Migratsiya versiyasi va meta ma'lumotlar
MIGRATION_VERSION = "001"
MIGRATION_NAME = "initial_schema"
MIGRATION_DESCRIPTION = "Boshlang'ich database schema - barcha asosiy jadvallar"
MIGRATION_AUTHOR = "AI OrderFlow Bot"
MIGRATION_DATE = "2024-01-01"

# Migratsiya funksiyalari
async def up(engine: AsyncEngine) -> MigrationResult:
    """Migratsiyani bajarish"""
    migration = InitialSchema(engine)
    return await migration.up()

async def down(engine: AsyncEngine) -> MigrationResult:
    """Migratsiyani bekor qilish"""
    migration = InitialSchema(engine)
    return await migration.down()

# Test funksiyasi
async def test_migration():
    """Migratsiya testini o'tkazish"""
    from sqlalchemy.ext.asyncio import create_async_engine
    
    # Test database yaratish
    engine = create_async_engine("sqlite+aiosqlite:///test_migration.db")
    
    try:
        # Migratsiyani bajarish
        result = await up(engine)
        print(f"Migration test: {result.success}")
        print(f"Tables created: {result.tables_created}")
        
        # Rollback test
        rollback_result = await down(engine)
        print(f"Rollback test: {rollback_result.success}")
        
    finally:
        await engine.dispose()

if __name__ == "__main__":
    # Test ishga tushirish
    asyncio.run(test_migration())

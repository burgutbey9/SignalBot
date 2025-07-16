"""
Database Migration: Add Performance Indexes
Index qo'shish - Performance optimizatsiya uchun
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import text
from typing import List, Tuple
import logging

# Migration identifikatori
revision = '005'
down_revision = '004'
branch_labels = None
depends_on = None

# Logger setup
logger = logging.getLogger(__name__)

def get_indexes_to_create() -> List[Tuple[str, str, List[str], bool]]:
    """
    Yaratilishi kerak bo'lgan indexlar ro'yxati
    Format: (index_name, table_name, columns, unique)
    """
    indexes = [
        # Signals jadvali indexlari
        ('idx_signals_timestamp', 'signals', ['timestamp'], False),
        ('idx_signals_symbol', 'signals', ['symbol'], False),
        ('idx_signals_status', 'signals', ['status'], False),
        ('idx_signals_symbol_timestamp', 'signals', ['symbol', 'timestamp'], False),
        ('idx_signals_status_timestamp', 'signals', ['status', 'timestamp'], False),
        
        # Trades jadvali indexlari
        ('idx_trades_timestamp', 'trades', ['entry_time'], False),
        ('idx_trades_symbol', 'trades', ['symbol'], False),
        ('idx_trades_status', 'trades', ['status'], False),
        ('idx_trades_account', 'trades', ['account_id'], False),
        ('idx_trades_signal_id', 'trades', ['signal_id'], False),
        ('idx_trades_exit_time', 'trades', ['exit_time'], False),
        
        # Logs jadvali indexlari
        ('idx_logs_timestamp', 'logs', ['timestamp'], False),
        ('idx_logs_level', 'logs', ['level'], False),
        ('idx_logs_module', 'logs', ['module'], False),
        ('idx_logs_level_timestamp', 'logs', ['level', 'timestamp'], False),
        ('idx_logs_module_timestamp', 'logs', ['module', 'timestamp'], False),
        
        # Users jadvali indexlari
        ('idx_users_created_at', 'users', ['created_at'], False),
        ('idx_users_status', 'users', ['status'], False),
        ('idx_users_telegram_id', 'users', ['telegram_id'], True),
        
        # Config jadvali indexlari
        ('idx_config_key', 'config', ['key'], True),
        ('idx_config_updated_at', 'config', ['updated_at'], False),
        
        # API Keys jadvali indexlari
        ('idx_api_keys_service', 'api_keys', ['service_name'], False),
        ('idx_api_keys_status', 'api_keys', ['status'], False),
        ('idx_api_keys_created_at', 'api_keys', ['created_at'], False),
        
        # Settings jadvali indexlari
        ('idx_settings_category', 'settings', ['category'], False),
        ('idx_settings_key', 'settings', ['setting_key'], False),
        ('idx_settings_category_key', 'settings', ['category', 'setting_key'], True),
        
        # Murakkab indexlar - Performance uchun
        ('idx_signals_active', 'signals', ['status', 'symbol', 'timestamp'], False),
        ('idx_trades_performance', 'trades', ['status', 'entry_time', 'profit_loss'], False),
        ('idx_logs_error_search', 'logs', ['level', 'module', 'timestamp'], False),
    ]
    
    return indexes

def get_text_search_indexes() -> List[Tuple[str, str, List[str]]]:
    """
    Text search indexlari
    PostgreSQL uchun full-text search indexes
    """
    text_indexes = [
        ('idx_logs_message_search', 'logs', ['message']),
        ('idx_logs_error_search', 'logs', ['error_details']),
        ('idx_signals_notes_search', 'signals', ['notes']),
        ('idx_trades_notes_search', 'trades', ['notes']),
    ]
    
    return text_indexes

def create_standard_indexes():
    """
    Standart indexlarni yaratish
    """
    logger.info("Standart indexlar yaratilmoqda...")
    
    indexes = get_indexes_to_create()
    
    for index_name, table_name, columns, unique in indexes:
        try:
            # Index mavjudligini tekshirish
            if not index_exists(index_name):
                logger.info(f"Index yaratilmoqda: {index_name} on {table_name}")
                
                # Index yaratish
                op.create_index(
                    index_name,
                    table_name,
                    columns,
                    unique=unique
                )
                
                logger.info(f"Index yaratildi: {index_name}")
            else:
                logger.info(f"Index mavjud: {index_name}")
                
        except Exception as e:
            logger.error(f"Index yaratishda xato {index_name}: {str(e)}")
            # Critical bo'lmagan indexlar uchun davom etish
            continue

def create_composite_indexes():
    """
    Murakkab (composite) indexlar yaratish
    Query performance uchun
    """
    logger.info("Murakkab indexlar yaratilmoqda...")
    
    # Eng ko'p ishlatiladigan query patternlari uchun
    composite_indexes = [
        # Signals bo'yicha qidiruv
        {
            'name': 'idx_signals_search_optimized',
            'table': 'signals',
            'columns': ['symbol', 'status', 'timestamp', 'confidence'],
            'unique': False
        },
        
        # Trades performance tahlil
        {
            'name': 'idx_trades_analytics',
            'table': 'trades',
            'columns': ['symbol', 'entry_time', 'status', 'profit_loss'],
            'unique': False
        },
        
        # Log tahlil va debugging
        {
            'name': 'idx_logs_debug',
            'table': 'logs',
            'columns': ['timestamp', 'level', 'module'],
            'unique': False
        },
        
        # User activity tracking
        {
            'name': 'idx_user_activity',
            'table': 'users',
            'columns': ['status', 'created_at', 'telegram_id'],
            'unique': False
        },
    ]
    
    for idx_config in composite_indexes:
        try:
            if not index_exists(idx_config['name']):
                logger.info(f"Murakkab index yaratilmoqda: {idx_config['name']}")
                
                op.create_index(
                    idx_config['name'],
                    idx_config['table'],
                    idx_config['columns'],
                    unique=idx_config['unique']
                )
                
                logger.info(f"Murakkab index yaratildi: {idx_config['name']}")
            else:
                logger.info(f"Murakkab index mavjud: {idx_config['name']}")
                
        except Exception as e:
            logger.error(f"Murakkab index yaratishda xato {idx_config['name']}: {str(e)}")
            continue

def create_partial_indexes():
    """
    Partial indexlar yaratish
    Faqat ma'lum shartlarga mos qatorlar uchun
    """
    logger.info("Partial indexlar yaratilmoqda...")
    
    # Partial indexlar faqat active/valid ma'lumotlar uchun
    partial_indexes = [
        # Faqat active signallar uchun
        {
            'name': 'idx_signals_active_only',
            'table': 'signals',
            'columns': ['timestamp', 'symbol'],
            'where': "status = 'active'"
        },
        
        # Faqat open trades uchun
        {
            'name': 'idx_trades_open_only',
            'table': 'trades',
            'columns': ['entry_time', 'symbol'],
            'where': "status = 'open'"
        },
        
        # Faqat error loglar uchun
        {
            'name': 'idx_logs_errors_only',
            'table': 'logs',
            'columns': ['timestamp', 'module'],
            'where': "level = 'ERROR'"
        },
        
        # Faqat profitable trades uchun
        {
            'name': 'idx_trades_profitable',
            'table': 'trades',
            'columns': ['exit_time', 'profit_loss'],
            'where': "profit_loss > 0"
        },
    ]
    
    for idx_config in partial_indexes:
        try:
            if not index_exists(idx_config['name']):
                logger.info(f"Partial index yaratilmoqda: {idx_config['name']}")
                
                # Raw SQL ishlatish - partial index uchun
                sql = f"""
                CREATE INDEX {idx_config['name']} 
                ON {idx_config['table']} ({', '.join(idx_config['columns'])})
                WHERE {idx_config['where']}
                """
                
                op.execute(text(sql))
                logger.info(f"Partial index yaratildi: {idx_config['name']}")
            else:
                logger.info(f"Partial index mavjud: {idx_config['name']}")
                
        except Exception as e:
            logger.error(f"Partial index yaratishda xato {idx_config['name']}: {str(e)}")
            continue

def create_expression_indexes():
    """
    Expression indexlar yaratish
    Hisoblangan qiymatlar uchun
    """
    logger.info("Expression indexlar yaratilmoqda...")
    
    # Expression indexlar - hisoblangan qiymatlar uchun
    expression_indexes = [
        # Profit percentage calculation
        {
            'name': 'idx_trades_profit_percentage',
            'table': 'trades',
            'expression': '(profit_loss / (entry_price * lot_size)) * 100'
        },
        
        # Signal confidence rounded
        {
            'name': 'idx_signals_confidence_rounded',
            'table': 'signals',
            'expression': 'ROUND(confidence, 0)'
        },
        
        # Log timestamp date only
        {
            'name': 'idx_logs_date_only',
            'table': 'logs',
            'expression': 'DATE(timestamp)'
        },
        
        # Trade duration in hours
        {
            'name': 'idx_trades_duration_hours',
            'table': 'trades',
            'expression': 'EXTRACT(HOUR FROM (exit_time - entry_time))'
        },
    ]
    
    for idx_config in expression_indexes:
        try:
            if not index_exists(idx_config['name']):
                logger.info(f"Expression index yaratilmoqda: {idx_config['name']}")
                
                # Raw SQL ishlatish - expression index uchun
                sql = f"""
                CREATE INDEX {idx_config['name']} 
                ON {idx_config['table']} ({idx_config['expression']})
                """
                
                op.execute(text(sql))
                logger.info(f"Expression index yaratildi: {idx_config['name']}")
            else:
                logger.info(f"Expression index mavjud: {idx_config['name']}")
                
        except Exception as e:
            logger.error(f"Expression index yaratishda xato {idx_config['name']}: {str(e)}")
            continue

def index_exists(index_name: str) -> bool:
    """
    Index mavjudligini tekshirish
    """
    try:
        # Database engine ga qarab turli xil usullar
        connection = op.get_bind()
        
        # PostgreSQL uchun
        if 'postgresql' in str(connection.engine.url):
            result = connection.execute(text(f"""
                SELECT EXISTS (
                    SELECT 1 
                    FROM pg_indexes 
                    WHERE indexname = '{index_name}'
                )
            """))
            return result.scalar()
        
        # SQLite uchun
        elif 'sqlite' in str(connection.engine.url):
            result = connection.execute(text(f"""
                SELECT name 
                FROM sqlite_master 
                WHERE type='index' AND name='{index_name}'
            """))
            return result.fetchone() is not None
        
        # MySQL uchun
        elif 'mysql' in str(connection.engine.url):
            result = connection.execute(text(f"""
                SELECT INDEX_NAME 
                FROM INFORMATION_SCHEMA.STATISTICS 
                WHERE INDEX_NAME = '{index_name}'
            """))
            return result.fetchone() is not None
        
        return False
        
    except Exception as e:
        logger.error(f"Index tekshirishda xato {index_name}: {str(e)}")
        return False

def upgrade():
    """
    Indexlarni qo'shish - Migration upgrade
    """
    logger.info("=== INDEX MIGRATION BOSHLANDI ===")
    
    try:
        # 1. Standart indexlar
        create_standard_indexes()
        
        # 2. Murakkab indexlar
        create_composite_indexes()
        
        # 3. Partial indexlar (faqat PostgreSQL uchun)
        connection = op.get_bind()
        if 'postgresql' in str(connection.engine.url):
            create_partial_indexes()
            create_expression_indexes()
        else:
            logger.info("Partial va Expression indexlar faqat PostgreSQL uchun")
        
        logger.info("=== INDEX MIGRATION TUGALLANDI ===")
        
    except Exception as e:
        logger.error(f"Index migration xatosi: {str(e)}")
        raise

def downgrade():
    """
    Indexlarni o'chirish - Migration downgrade
    """
    logger.info("=== INDEX MIGRATION BEKOR QILISH ===")
    
    try:
        # Barcha yaratilgan indexlarni ro'yxati
        all_indexes = []
        
        # Standart indexlar
        indexes = get_indexes_to_create()
        all_indexes.extend([idx[0] for idx in indexes])
        
        # Murakkab indexlar
        composite_names = [
            'idx_signals_search_optimized',
            'idx_trades_analytics',
            'idx_logs_debug',
            'idx_user_activity'
        ]
        all_indexes.extend(composite_names)
        
        # Partial indexlar
        partial_names = [
            'idx_signals_active_only',
            'idx_trades_open_only',
            'idx_logs_errors_only',
            'idx_trades_profitable'
        ]
        all_indexes.extend(partial_names)
        
        # Expression indexlar
        expression_names = [
            'idx_trades_profit_percentage',
            'idx_signals_confidence_rounded',
            'idx_logs_date_only',
            'idx_trades_duration_hours'
        ]
        all_indexes.extend(expression_names)
        
        # Indexlarni o'chirish
        for index_name in all_indexes:
            try:
                if index_exists(index_name):
                    logger.info(f"Index o'chirilmoqda: {index_name}")
                    op.drop_index(index_name)
                    logger.info(f"Index o'chirildi: {index_name}")
                else:
                    logger.info(f"Index mavjud emas: {index_name}")
                    
            except Exception as e:
                logger.error(f"Index o'chirishda xato {index_name}: {str(e)}")
                continue
        
        logger.info("=== INDEX MIGRATION BEKOR QILINDI ===")
        
    except Exception as e:
        logger.error(f"Index downgrade xatosi: {str(e)}")
        raise

if __name__ == "__main__":
    # Test uchun
    print("Index Migration Script")
    print("Bu fayl Alembic orqali ishlatilishi kerak")
    print("Masalan: alembic upgrade 005")

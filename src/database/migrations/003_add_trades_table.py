"""
Savdo jadvali yaratish migratsiyasi
Revision ID: 003
Revises: 002
Create Date: 2024-01-15 10:00:00.000000
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import sqlite
from datetime import datetime
from typing import Optional

# revision identifiers
revision = '003'
down_revision = '002'
branch_labels: Optional[str] = None
depends_on: Optional[str] = None


def upgrade() -> None:
    """
    Savdo jadvali va bog'liq jadvallarni yaratish
    """
    # Savdo jadvali yaratish
    op.create_table(
        'trades',
        sa.Column('id', sa.Integer(), nullable=False, primary_key=True),
        sa.Column('trade_id', sa.String(50), nullable=False, unique=True, comment='Noyob savdo ID'),
        sa.Column('signal_id', sa.Integer(), nullable=False, comment='Signal ID ga havola'),
        sa.Column('symbol', sa.String(20), nullable=False, comment='Savdo simvoli (EURUSD, GBPUSD)'),
        sa.Column('action', sa.String(10), nullable=False, comment='BUY yoki SELL'),
        sa.Column('lot_size', sa.Float(), nullable=False, comment='Lot hajmi'),
        
        # Vaqt ma'lumotlari
        sa.Column('entry_time', sa.DateTime(), nullable=False, comment='Kirishda vaqti'),
        sa.Column('exit_time', sa.DateTime(), nullable=True, comment='Chiqishda vaqti'),
        sa.Column('duration_minutes', sa.Integer(), nullable=True, comment='Savdo davomiyligi (daqiqa)'),
        
        # Narx ma'lumotlari
        sa.Column('entry_price', sa.Float(), nullable=False, comment='Kirish narxi'),
        sa.Column('exit_price', sa.Float(), nullable=True, comment='Chiqish narxi'),
        sa.Column('stop_loss', sa.Float(), nullable=True, comment='Stop Loss narxi'),
        sa.Column('take_profit', sa.Float(), nullable=True, comment='Take Profit narxi'),
        
        # Natija ma'lumotlari
        sa.Column('profit_loss', sa.Float(), nullable=True, comment='Foyda/Zarar USD'),
        sa.Column('profit_loss_pips', sa.Float(), nullable=True, comment='Foyda/Zarar pips'),
        sa.Column('profit_loss_percent', sa.Float(), nullable=True, comment='Foyda/Zarar foizi'),
        
        # Holat ma'lumotlari
        sa.Column('status', sa.String(20), nullable=False, default='PENDING', comment='Savdo holati'),
        sa.Column('exit_reason', sa.String(50), nullable=True, comment='Chiqish sababi'),
        sa.Column('is_winner', sa.Boolean(), nullable=True, comment='Yutuqchi savdo'),
        
        # Hisob ma'lumotlari
        sa.Column('account_id', sa.String(50), nullable=False, comment='Hisob ID'),
        sa.Column('account_type', sa.String(20), nullable=False, default='DEMO', comment='Hisob turi'),
        sa.Column('broker', sa.String(30), nullable=False, default='PROPSHOT', comment='Broker nomi'),
        
        # Risk ma'lumotlari
        sa.Column('risk_percent', sa.Float(), nullable=True, comment='Risk foizi'),
        sa.Column('risk_amount', sa.Float(), nullable=True, comment='Risk miqdori USD'),
        sa.Column('account_balance_before', sa.Float(), nullable=True, comment='Savdo oldidan balans'),
        sa.Column('account_balance_after', sa.Float(), nullable=True, comment='Savdo keyingi balans'),
        
        # Qo'shimcha ma'lumotlar
        sa.Column('commission', sa.Float(), nullable=True, default=0.0, comment='Komissiya USD'),
        sa.Column('swap', sa.Float(), nullable=True, default=0.0, comment='Swap USD'),
        sa.Column('slippage', sa.Float(), nullable=True, default=0.0, comment='Slippage pips'),
        
        # Strategiya ma'lumotlari
        sa.Column('strategy_name', sa.String(50), nullable=True, comment='Strategiya nomi'),
        sa.Column('confidence_score', sa.Float(), nullable=True, comment='Ishonch darajasi'),
        sa.Column('ai_prediction', sa.String(100), nullable=True, comment='AI bashorat'),
        
        # Metadata
        sa.Column('notes', sa.Text(), nullable=True, comment='Qo'shimcha izohlar'),
        sa.Column('tags', sa.String(200), nullable=True, comment='Teglar (vergul bilan)'),
        sa.Column('created_at', sa.DateTime(), nullable=False, default=datetime.utcnow, comment='Yaratilgan vaqti'),
        sa.Column('updated_at', sa.DateTime(), nullable=False, default=datetime.utcnow, comment='Yangilangan vaqti'),
        
        # Foreign key bog'lanish
        sa.ForeignKeyConstraint(['signal_id'], ['signals.id'], ondelete='CASCADE'),
        
        # Unique constraints
        sa.UniqueConstraint('trade_id', name='uq_trades_trade_id'),
        
        comment='Savdo jadvali - barcha savdo operatsiyalari'
    )
    
    # Savdo statistikasi jadvali yaratish
    op.create_table(
        'trade_statistics',
        sa.Column('id', sa.Integer(), nullable=False, primary_key=True),
        sa.Column('date', sa.Date(), nullable=False, comment='Sana'),
        sa.Column('symbol', sa.String(20), nullable=True, comment='Simvol (null = barcha)'),
        sa.Column('account_id', sa.String(50), nullable=False, comment='Hisob ID'),
        
        # Kunlik statistika
        sa.Column('total_trades', sa.Integer(), nullable=False, default=0, comment='Jami savdolar'),
        sa.Column('winning_trades', sa.Integer(), nullable=False, default=0, comment='Yutuqchi savdolar'),
        sa.Column('losing_trades', sa.Integer(), nullable=False, default=0, comment='Yutqazuvchi savdolar'),
        sa.Column('win_rate', sa.Float(), nullable=True, comment='Yutuqchilik darajasi'),
        
        # Moliyaviy statistika
        sa.Column('total_profit_loss', sa.Float(), nullable=False, default=0.0, comment='Jami foyda/zarar'),
        sa.Column('gross_profit', sa.Float(), nullable=False, default=0.0, comment='Yalpi foyda'),
        sa.Column('gross_loss', sa.Float(), nullable=False, default=0.0, comment='Yalpi zarar'),
        sa.Column('profit_factor', sa.Float(), nullable=True, comment='Foyda omili'),
        
        # Risk statistikasi
        sa.Column('max_drawdown', sa.Float(), nullable=True, comment='Maksimal drawdown'),
        sa.Column('max_consecutive_wins', sa.Integer(), nullable=True, comment='Maksimal ketma-ket yutish'),
        sa.Column('max_consecutive_losses', sa.Integer(), nullable=True, comment='Maksimal ketma-ket yutqazish'),
        sa.Column('average_trade_duration', sa.Float(), nullable=True, comment='O'rtacha savdo davomiyligi'),
        
        # Metadata
        sa.Column('created_at', sa.DateTime(), nullable=False, default=datetime.utcnow),
        sa.Column('updated_at', sa.DateTime(), nullable=False, default=datetime.utcnow),
        
        # Unique constraint
        sa.UniqueConstraint('date', 'symbol', 'account_id', name='uq_trade_stats_date_symbol_account'),
        
        comment='Savdo statistikasi - kunlik hisobotlar'
    )
    
    # Savdo holati enum yaratish (SQLite uchun check constraint)
    op.create_check_constraint(
        'ck_trades_status',
        'trades',
        sa.text("status IN ('PENDING', 'OPEN', 'CLOSED', 'CANCELLED', 'EXPIRED', 'FILLED', 'PARTIAL')")
    )
    
    # Savdo harakati enum yaratish
    op.create_check_constraint(
        'ck_trades_action',
        'trades',
        sa.text("action IN ('BUY', 'SELL')")
    )
    
    # Chiqish sababi enum yaratish
    op.create_check_constraint(
        'ck_trades_exit_reason',
        'trades',
        sa.text("exit_reason IN ('TAKE_PROFIT', 'STOP_LOSS', 'MANUAL', 'TIMEOUT', 'REVERSE_SIGNAL', 'RISK_MANAGEMENT')")
    )
    
    # Hisob turi enum yaratish
    op.create_check_constraint(
        'ck_trades_account_type',
        'trades',
        sa.text("account_type IN ('DEMO', 'LIVE', 'CONTEST', 'PROP')")
    )
    
    # Indexlar yaratish
    # Vaqt bo'yicha index
    op.create_index('ix_trades_entry_time', 'trades', ['entry_time'])
    op.create_index('ix_trades_exit_time', 'trades', ['exit_time'])
    
    # Simvol bo'yicha index
    op.create_index('ix_trades_symbol', 'trades', ['symbol'])
    
    # Holat bo'yicha index
    op.create_index('ix_trades_status', 'trades', ['status'])
    
    # Hisob bo'yicha index
    op.create_index('ix_trades_account_id', 'trades', ['account_id'])
    
    # Signal ID bo'yicha index
    op.create_index('ix_trades_signal_id', 'trades', ['signal_id'])
    
    # Murakkab indexlar
    op.create_index('ix_trades_symbol_status', 'trades', ['symbol', 'status'])
    op.create_index('ix_trades_account_symbol', 'trades', ['account_id', 'symbol'])
    op.create_index('ix_trades_entry_time_symbol', 'trades', ['entry_time', 'symbol'])
    
    # Statistika jadvali uchun indexlar
    op.create_index('ix_trade_statistics_date', 'trade_statistics', ['date'])
    op.create_index('ix_trade_statistics_symbol', 'trade_statistics', ['symbol'])
    op.create_index('ix_trade_statistics_account_id', 'trade_statistics', ['account_id'])
    
    print("✅ Savdo jadvali va bog'liq jadvallar muvaffaqiyatli yaratildi")


def downgrade() -> None:
    """
    Savdo jadvalini o'chirish
    """
    # Indexlarni o'chirish
    op.drop_index('ix_trade_statistics_account_id', table_name='trade_statistics')
    op.drop_index('ix_trade_statistics_symbol', table_name='trade_statistics')
    op.drop_index('ix_trade_statistics_date', table_name='trade_statistics')
    
    op.drop_index('ix_trades_entry_time_symbol', table_name='trades')
    op.drop_index('ix_trades_account_symbol', table_name='trades')
    op.drop_index('ix_trades_symbol_status', table_name='trades')
    op.drop_index('ix_trades_signal_id', table_name='trades')
    op.drop_index('ix_trades_account_id', table_name='trades')
    op.drop_index('ix_trades_status', table_name='trades')
    op.drop_index('ix_trades_symbol', table_name='trades')
    op.drop_index('ix_trades_exit_time', table_name='trades')
    op.drop_index('ix_trades_entry_time', table_name='trades')
    
    # Check constraintlarni o'chirish
    op.drop_constraint('ck_trades_account_type', 'trades', type_='check')
    op.drop_constraint('ck_trades_exit_reason', 'trades', type_='check')
    op.drop_constraint('ck_trades_action', 'trades', type_='check')
    op.drop_constraint('ck_trades_status', 'trades', type_='check')
    
    # Jadvallarni o'chirish
    op.drop_table('trade_statistics')
    op.drop_table('trades')
    
    print("❌ Savdo jadvali va bog'liq jadvallar o'chirildi")


def get_trade_stats_query(account_id: str, symbol: str = None, start_date: str = None, end_date: str = None) -> str:
    """
    Savdo statistikasini olish uchun SQL so'rov
    
    Args:
        account_id: Hisob ID
        symbol: Simvol (ixtiyoriy)
        start_date: Boshlanish sanasi (ixtiyoriy)
        end_date: Tugash sanasi (ixtiyoriy)
        
    Returns:
        SQL so'rov matni
    """
    base_query = """
    SELECT 
        t.symbol,
        COUNT(*) as total_trades,
        SUM(CASE WHEN t.is_winner = 1 THEN 1 ELSE 0 END) as winning_trades,
        SUM(CASE WHEN t.is_winner = 0 THEN 1 ELSE 0 END) as losing_trades,
        ROUND(AVG(CASE WHEN t.is_winner = 1 THEN 1.0 ELSE 0.0 END) * 100, 2) as win_rate,
        ROUND(SUM(t.profit_loss), 2) as total_profit_loss,
        ROUND(SUM(CASE WHEN t.profit_loss > 0 THEN t.profit_loss ELSE 0 END), 2) as gross_profit,
        ROUND(ABS(SUM(CASE WHEN t.profit_loss < 0 THEN t.profit_loss ELSE 0 END)), 2) as gross_loss,
        ROUND(AVG(t.profit_loss), 2) as average_profit_loss,
        ROUND(MAX(t.profit_loss), 2) as best_trade,
        ROUND(MIN(t.profit_loss), 2) as worst_trade,
        ROUND(AVG(t.duration_minutes), 2) as avg_duration_minutes
    FROM trades t
    WHERE t.account_id = :account_id
    AND t.status = 'CLOSED'
    """
    
    conditions = []
    if symbol:
        conditions.append("AND t.symbol = :symbol")
    if start_date:
        conditions.append("AND t.entry_time >= :start_date")
    if end_date:
        conditions.append("AND t.entry_time <= :end_date")
    
    if conditions:
        base_query += " " + " ".join(conditions)
    
    base_query += " GROUP BY t.symbol ORDER BY total_trades DESC"
    
    return base_query


def get_daily_performance_query(account_id: str, days: int = 30) -> str:
    """
    Kunlik performance so'rovi
    
    Args:
        account_id: Hisob ID
        days: Necha kunlik ma'lumot (default: 30)
        
    Returns:
        SQL so'rov matni
    """
    return f"""
    SELECT 
        DATE(t.entry_time) as trade_date,
        COUNT(*) as daily_trades,
        SUM(CASE WHEN t.is_winner = 1 THEN 1 ELSE 0 END) as daily_wins,
        ROUND(SUM(t.profit_loss), 2) as daily_profit_loss,
        ROUND(AVG(t.confidence_score), 2) as avg_confidence,
        GROUP_CONCAT(DISTINCT t.symbol) as symbols_traded
    FROM trades t
    WHERE t.account_id = :account_id
    AND t.status = 'CLOSED'
    AND t.entry_time >= date('now', '-{days} days')
    GROUP BY DATE(t.entry_time)
    ORDER BY trade_date DESC
    """

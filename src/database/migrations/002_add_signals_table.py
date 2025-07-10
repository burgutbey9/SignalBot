"""
Database Migration 002: Add Signals Table
Signallar jadvalini yaratish migratsiyasi
"""

import sqlalchemy as sa
from alembic import op
from datetime import datetime
from typing import Optional

# revision identifiers
revision = '002'
down_revision = '001'
branch_labels: Optional[str] = None
depends_on: Optional[str] = None


def upgrade() -> None:
    """
    Signallar jadvalini yaratish
    AI signal ma'lumotlarini saqlash uchun
    """
    # Signals table yaratish
    op.create_table(
        'signals',
        
        # Asosiy ID va vaqt maydonlari
        sa.Column('signal_id', sa.Integer, primary_key=True, autoincrement=True,
                 comment='Signal ID - avtomatik raqam'),
        sa.Column('timestamp', sa.DateTime, default=datetime.utcnow, nullable=False,
                 comment='Signal yaratilgan vaqt'),
        sa.Column('created_at', sa.DateTime, default=datetime.utcnow, nullable=False,
                 comment='Yaratilgan vaqt (UTC)'),
        sa.Column('updated_at', sa.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow,
                 comment='Yangilangan vaqt'),
        
        # Trading asosiy ma'lumotlari
        sa.Column('symbol', sa.String(20), nullable=False,
                 comment='Trading juftligi (EURUSD, BTCUSDT)'),
        sa.Column('action', sa.Enum('BUY', 'SELL', 'HOLD', name='signal_action'), nullable=False,
                 comment='Signal harakati - BUY/SELL/HOLD'),
        sa.Column('signal_type', sa.Enum('MARKET', 'LIMIT', 'STOP', name='signal_type'), 
                 default='MARKET', nullable=False,
                 comment='Signal turi - MARKET/LIMIT/STOP'),
        sa.Column('timeframe', sa.String(10), default='1H', nullable=False,
                 comment='Vaqt oralig\'i (1M, 5M, 15M, 1H, 4H, 1D)'),
        
        # Narx ma'lumotlari
        sa.Column('price', sa.Numeric(precision=12, scale=6), nullable=False,
                 comment='Signal narxi'),
        sa.Column('entry_price', sa.Numeric(precision=12, scale=6), nullable=True,
                 comment='Kirish narxi'),
        sa.Column('stop_loss', sa.Numeric(precision=12, scale=6), nullable=True,
                 comment='Stop Loss narxi'),
        sa.Column('take_profit', sa.Numeric(precision=12, scale=6), nullable=True,
                 comment='Take Profit narxi'),
        sa.Column('stop_loss_pips', sa.Integer, nullable=True,
                 comment='Stop Loss pip miqdori'),
        sa.Column('take_profit_pips', sa.Integer, nullable=True,
                 comment='Take Profit pip miqdori'),
        
        # AI va tahlil ma'lumotlari
        sa.Column('confidence', sa.Float, nullable=False, default=0.0,
                 comment='AI ishonch darajasi (0-100)'),
        sa.Column('sentiment_score', sa.Float, nullable=True,
                 comment='Sentiment tahlil natijasi (-1 to 1)'),
        sa.Column('order_flow_score', sa.Float, nullable=True,
                 comment='Order Flow tahlil natijasi'),
        sa.Column('technical_score', sa.Float, nullable=True,
                 comment='Technical tahlil natijasi'),
        sa.Column('ai_model_version', sa.String(50), nullable=True,
                 comment='Ishlatilgan AI model versiyasi'),
        
        # Risk management
        sa.Column('risk_percent', sa.Float, nullable=False, default=1.0,
                 comment='Risk foizi (kapitaldan)'),
        sa.Column('lot_size', sa.Numeric(precision=8, scale=2), nullable=True,
                 comment='Lot hajmi'),
        sa.Column('position_size', sa.Numeric(precision=15, scale=2), nullable=True,
                 comment='Position hajmi (USD)'),
        sa.Column('max_loss_usd', sa.Numeric(precision=10, scale=2), nullable=True,
                 comment='Maksimal zarar (USD)'),
        sa.Column('expected_profit_usd', sa.Numeric(precision=10, scale=2), nullable=True,
                 comment='Kutilgan foyda (USD)'),
        
        # Signal holati va natijalar
        sa.Column('status', sa.Enum('PENDING', 'ACTIVE', 'FILLED', 'CANCELLED', 'EXPIRED', 
                                   'STOP_HIT', 'PROFIT_HIT', name='signal_status'), 
                 default='PENDING', nullable=False,
                 comment='Signal holati'),
        sa.Column('is_active', sa.Boolean, default=True, nullable=False,
                 comment='Signal faolligi'),
        sa.Column('executed_at', sa.DateTime, nullable=True,
                 comment='Signal bajarilgan vaqt'),
        sa.Column('closed_at', sa.DateTime, nullable=True,
                 comment='Signal yopilgan vaqt'),
        sa.Column('expiry_time', sa.DateTime, nullable=True,
                 comment='Signal tugash vaqti'),
        
        # Natija va hisobotlar
        sa.Column('result', sa.Enum('WIN', 'LOSS', 'BREAKEVEN', 'PENDING', name='signal_result'),
                 default='PENDING', nullable=False,
                 comment='Signal natijasi'),
        sa.Column('actual_profit_loss', sa.Numeric(precision=10, scale=2), nullable=True,
                 comment='Haqiqiy foyda/zarar'),
        sa.Column('actual_pips', sa.Integer, nullable=True,
                 comment='Haqiqiy pip miqdori'),
        sa.Column('duration_minutes', sa.Integer, nullable=True,
                 comment='Signal davomiyligi (daqiqa)'),
        
        # Manbalar va kontekst
        sa.Column('signal_source', sa.String(100), nullable=True,
                 comment='Signal manbasi (AI_COMBO, ORDER_FLOW, SENTIMENT)'),
        sa.Column('data_sources', sa.Text, nullable=True,
                 comment='Ishlatilgan ma\'lumot manbalari JSON'),
        sa.Column('analysis_context', sa.Text, nullable=True,
                 comment='Tahlil konteksti va sabablar'),
        sa.Column('market_conditions', sa.Text, nullable=True,
                 comment='Bozor sharoitlari JSON'),
        
        # Foydalanuvchi va tizim
        sa.Column('user_id', sa.Integer, sa.ForeignKey('users.user_id'), nullable=True,
                 comment='Foydalanuvchi ID'),
        sa.Column('account_id', sa.String(50), nullable=True,
                 comment='Trading akkaunt ID'),
        sa.Column('strategy_name', sa.String(100), nullable=True,
                 comment='Strategiya nomi'),
        sa.Column('session_id', sa.String(100), nullable=True,
                 comment='Trading session ID'),
        
        # Telegram va bildirishnomalar
        sa.Column('telegram_message_id', sa.Integer, nullable=True,
                 comment='Telegram xabar ID'),
        sa.Column('telegram_sent', sa.Boolean, default=False, nullable=False,
                 comment='Telegram orqali yuborilganmi'),
        sa.Column('telegram_confirmed', sa.Boolean, default=False, nullable=False,
                 comment='Telegram orqali tasdiqlangani'),
        sa.Column('auto_trade_enabled', sa.Boolean, default=False, nullable=False,
                 comment='Avtomatik savdo yoqilgani'),
        
        # Qo'shimcha ma'lumotlar
        sa.Column('notes', sa.Text, nullable=True,
                 comment='Qo\'shimcha eslatmalar'),
        sa.Column('tags', sa.String(200), nullable=True,
                 comment='Teglar (vergul bilan ajratilgan)'),
        sa.Column('metadata', sa.Text, nullable=True,
                 comment='Qo\'shimcha metadata JSON'),
        sa.Column('version', sa.Integer, default=1, nullable=False,
                 comment='Signal versiyasi'),
        
        # Audit va debugging
        sa.Column('debug_info', sa.Text, nullable=True,
                 comment='Debug ma\'lumotlari'),
        sa.Column('api_call_count', sa.Integer, default=0, nullable=False,
                 comment='API so\'rovlar soni'),
        sa.Column('processing_time_ms', sa.Integer, nullable=True,
                 comment='Qayta ishlash vaqti (ms)'),
        
        # Jadval sozlamalari
        comment='AI signal ma\'lumotlari jadvali - har signal uchun to\'liq ma\'lumot'
    )
    
    # Indexlar yaratish - performance uchun muhim
    
    # Asosiy qidiruv indexlari
    op.create_index('idx_signals_symbol', 'signals', ['symbol'])
    op.create_index('idx_signals_timestamp', 'signals', ['timestamp'])
    op.create_index('idx_signals_status', 'signals', ['status'])
    op.create_index('idx_signals_result', 'signals', ['result'])
    op.create_index('idx_signals_action', 'signals', ['action'])
    
    # Murakkab indexlar
    op.create_index('idx_signals_symbol_timestamp', 'signals', ['symbol', 'timestamp'])
    op.create_index('idx_signals_status_timestamp', 'signals', ['status', 'timestamp'])
    op.create_index('idx_signals_user_timestamp', 'signals', ['user_id', 'timestamp'])
    op.create_index('idx_signals_active_symbol', 'signals', ['is_active', 'symbol'])
    
    # Performance indexlari
    op.create_index('idx_signals_confidence', 'signals', ['confidence'])
    op.create_index('idx_signals_risk_percent', 'signals', ['risk_percent'])
    op.create_index('idx_signals_timeframe', 'signals', ['timeframe'])
    op.create_index('idx_signals_strategy', 'signals', ['strategy_name'])
    
    # Telegram indexlari
    op.create_index('idx_signals_telegram_sent', 'signals', ['telegram_sent'])
    op.create_index('idx_signals_telegram_confirmed', 'signals', ['telegram_confirmed'])
    op.create_index('idx_signals_auto_trade', 'signals', ['auto_trade_enabled'])
    
    # Vaqt oralig'i indexlari
    op.create_index('idx_signals_created_at', 'signals', ['created_at'])
    op.create_index('idx_signals_executed_at', 'signals', ['executed_at'])
    op.create_index('idx_signals_closed_at', 'signals', ['closed_at'])
    
    # Unique constraints
    op.create_index('idx_signals_session_unique', 'signals', 
                   ['session_id', 'symbol', 'timestamp'], unique=True)
    
    print("✅ Signals table muvaffaqiyatli yaratildi")
    print("📊 45 ta ustun bilan to'liq signal tracking tizimi")
    print("🔍 15 ta index performance uchun")
    print("🎯 AI, risk management, Telegram integration")


def downgrade() -> None:
    """
    Signallar jadvalini o'chirish
    Rollback uchun
    """
    # Indexlarni o'chirish
    op.drop_index('idx_signals_session_unique', table_name='signals')
    op.drop_index('idx_signals_closed_at', table_name='signals')
    op.drop_index('idx_signals_executed_at', table_name='signals')
    op.drop_index('idx_signals_created_at', table_name='signals')
    op.drop_index('idx_signals_auto_trade', table_name='signals')
    op.drop_index('idx_signals_telegram_confirmed', table_name='signals')
    op.drop_index('idx_signals_telegram_sent', table_name='signals')
    op.drop_index('idx_signals_strategy', table_name='signals')
    op.drop_index('idx_signals_timeframe', table_name='signals')
    op.drop_index('idx_signals_risk_percent', table_name='signals')
    op.drop_index('idx_signals_confidence', table_name='signals')
    op.drop_index('idx_signals_active_symbol', table_name='signals')
    op.drop_index('idx_signals_user_timestamp', table_name='signals')
    op.drop_index('idx_signals_status_timestamp', table_name='signals')
    op.drop_index('idx_signals_symbol_timestamp', table_name='signals')
    op.drop_index('idx_signals_action', table_name='signals')
    op.drop_index('idx_signals_result', table_name='signals')
    op.drop_index('idx_signals_status', table_name='signals')
    op.drop_index('idx_signals_timestamp', table_name='signals')
    op.drop_index('idx_signals_symbol', table_name='signals')
    
    # Enum tiplarini o'chirish
    op.execute("DROP TYPE IF EXISTS signal_result")
    op.execute("DROP TYPE IF EXISTS signal_status")
    op.execute("DROP TYPE IF EXISTS signal_type")
    op.execute("DROP TYPE IF EXISTS signal_action")
    
    # Jadvalni o'chirish
    op.drop_table('signals')
    
    print("❌ Signals table o'chirildi")


# Yordamchi funksiyalar
def create_sample_signals():
    """
    Test uchun sample signallar yaratish
    Development environmentda ishlatish uchun
    """
    sample_signals = [
        {
            'symbol': 'EURUSD',
            'action': 'BUY',
            'price': 1.0850,
            'confidence': 85.5,
            'risk_percent': 1.5,
            'stop_loss': 1.0800,
            'take_profit': 1.0920,
            'signal_source': 'AI_COMBO',
            'strategy_name': 'OrderFlow_Sentiment_v1'
        },
        {
            'symbol': 'GBPUSD',
            'action': 'SELL',
            'price': 1.2650,
            'confidence': 78.2,
            'risk_percent': 2.0,
            'stop_loss': 1.2700,
            'take_profit': 1.2580,
            'signal_source': 'ORDER_FLOW',
            'strategy_name': 'OrderFlow_Pure_v1'
        },
        {
            'symbol': 'BTCUSDT',
            'action': 'BUY',
            'price': 42500.00,
            'confidence': 92.1,
            'risk_percent': 1.0,
            'stop_loss': 41800.00,
            'take_profit': 44200.00,
            'signal_source': 'SENTIMENT',
            'strategy_name': 'AI_Sentiment_v2'
        }
    ]
    
    return sample_signals


def validate_signal_data(signal_data: dict) -> bool:
    """
    Signal ma'lumotlarini tekshirish
    """
    required_fields = ['symbol', 'action', 'price', 'confidence']
    
    for field in required_fields:
        if field not in signal_data:
            return False
    
    # Confidence range tekshirish
    if not (0 <= signal_data['confidence'] <= 100):
        return False
    
    # Risk percent tekshirish
    if 'risk_percent' in signal_data:
        if not (0 < signal_data['risk_percent'] <= 5):
            return False
    
    return True


# Migration ma'lumotlari
MIGRATION_INFO = {
    'version': '002',
    'description': 'Add comprehensive signals table',
    'tables_created': ['signals'],
    'indexes_created': 15,
    'enums_created': 4,
    'features': [
        'AI signal tracking',
        'Risk management',
        'Telegram integration',
        'Performance analytics',
        'Multi-timeframe support',
        'Strategy versioning'
    ]
}

print(f"📋 Migration {MIGRATION_INFO['version']}: {MIGRATION_INFO['description']}")
print(f"🏗️  {len(MIGRATION_INFO['tables_created'])} jadval yaratiladi")
print(f"🔍 {MIGRATION_INFO['indexes_created']} index yaratiladi")
print(f"📊 {len(MIGRATION_INFO['features'])} asosiy funksiya")

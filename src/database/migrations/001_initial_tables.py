"""
Birinchi migration - Asosiy jadvallar yaratish
Created: 2024-01-01
Description: Users, config, api_keys, settings jadvallarini yaratish
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import sqlite
from datetime import datetime
import json


# Revision identifiers
revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    """
    Asosiy jadvallarni yaratish
    """
    
    # 1. Users jadvali - Foydalanuvchilar
    op.create_table(
        'users',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('telegram_id', sa.BigInteger(), nullable=False, unique=True),
        sa.Column('username', sa.String(100), nullable=True),
        sa.Column('full_name', sa.String(200), nullable=True),
        sa.Column('phone', sa.String(20), nullable=True),
        sa.Column('email', sa.String(100), nullable=True),
        sa.Column('language', sa.String(10), default='uz', nullable=False),
        sa.Column('timezone', sa.String(50), default='Asia/Tashkent', nullable=False),
        sa.Column('is_active', sa.Boolean(), default=True, nullable=False),
        sa.Column('is_admin', sa.Boolean(), default=False, nullable=False),
        sa.Column('subscription_type', sa.String(20), default='free', nullable=False),
        sa.Column('subscription_expires', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), default=datetime.utcnow, nullable=False),
        sa.Column('updated_at', sa.DateTime(), default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False),
        sa.Column('last_login', sa.DateTime(), nullable=True),
        sa.Column('login_count', sa.Integer(), default=0, nullable=False),
        # Foydalanuvchi sozlamalari JSON formatda
        sa.Column('user_settings', sa.Text(), nullable=True),
        sa.Column('notes', sa.Text(), nullable=True),
    )
    
    # 2. Config jadvali - Asosiy konfiguratsiya
    op.create_table(
        'config',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('config_key', sa.String(100), nullable=False, unique=True),
        sa.Column('config_value', sa.Text(), nullable=True),
        sa.Column('config_type', sa.String(20), default='string', nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('is_active', sa.Boolean(), default=True, nullable=False),
        sa.Column('category', sa.String(50), default='general', nullable=False),
        sa.Column('created_at', sa.DateTime(), default=datetime.utcnow, nullable=False),
        sa.Column('updated_at', sa.DateTime(), default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False),
        sa.Column('created_by', sa.Integer(), nullable=True),
        sa.Column('updated_by', sa.Integer(), nullable=True),
    )
    
    # 3. API Keys jadvali - API kalitlar
    op.create_table(
        'api_keys',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('service_name', sa.String(50), nullable=False),
        sa.Column('api_key', sa.Text(), nullable=False),
        sa.Column('api_secret', sa.Text(), nullable=True),
        sa.Column('endpoint_url', sa.String(500), nullable=True),
        sa.Column('is_active', sa.Boolean(), default=True, nullable=False),
        sa.Column('rate_limit', sa.Integer(), default=100, nullable=False),
        sa.Column('timeout', sa.Integer(), default=30, nullable=False),
        sa.Column('max_retries', sa.Integer(), default=3, nullable=False),
        sa.Column('priority', sa.Integer(), default=1, nullable=False),
        sa.Column('last_used', sa.DateTime(), nullable=True),
        sa.Column('usage_count', sa.Integer(), default=0, nullable=False),
        sa.Column('error_count', sa.Integer(), default=0, nullable=False),
        sa.Column('created_at', sa.DateTime(), default=datetime.utcnow, nullable=False),
        sa.Column('updated_at', sa.DateTime(), default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False),
        sa.Column('notes', sa.Text(), nullable=True),
    )
    
    # 4. Settings jadvali - Tizim sozlamalari
    op.create_table(
        'settings',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('user_id', sa.Integer(), nullable=True),
        sa.Column('setting_key', sa.String(100), nullable=False),
        sa.Column('setting_value', sa.Text(), nullable=True),
        sa.Column('setting_type', sa.String(20), default='string', nullable=False),
        sa.Column('is_global', sa.Boolean(), default=False, nullable=False),
        sa.Column('category', sa.String(50), default='general', nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), default=datetime.utcnow, nullable=False),
        sa.Column('updated_at', sa.DateTime(), default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False),
    )
    
    # 5. System Status jadvali - Tizim holati
    op.create_table(
        'system_status',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('service_name', sa.String(50), nullable=False),
        sa.Column('status', sa.String(20), default='active', nullable=False),
        sa.Column('last_check', sa.DateTime(), default=datetime.utcnow, nullable=False),
        sa.Column('response_time', sa.Float(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('uptime_percentage', sa.Float(), default=100.0, nullable=False),
        sa.Column('total_requests', sa.Integer(), default=0, nullable=False),
        sa.Column('failed_requests', sa.Integer(), default=0, nullable=False),
        sa.Column('created_at', sa.DateTime(), default=datetime.utcnow, nullable=False),
        sa.Column('updated_at', sa.DateTime(), default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False),
    )
    
    # Indexlar yaratish
    create_indexes()
    
    # Boshlang'ich ma'lumotlar qo'shish
    insert_initial_data()


def downgrade():
    """
    Jadvallarni o'chirish
    """
    op.drop_table('system_status')
    op.drop_table('settings')
    op.drop_table('api_keys')
    op.drop_table('config')
    op.drop_table('users')


def create_indexes():
    """
    Kerakli indexlar yaratish
    """
    # Users jadvali uchun indexlar
    op.create_index('idx_users_telegram_id', 'users', ['telegram_id'])
    op.create_index('idx_users_username', 'users', ['username'])
    op.create_index('idx_users_active', 'users', ['is_active'])
    op.create_index('idx_users_created_at', 'users', ['created_at'])
    
    # Config jadvali uchun indexlar
    op.create_index('idx_config_key', 'config', ['config_key'])
    op.create_index('idx_config_category', 'config', ['category'])
    op.create_index('idx_config_active', 'config', ['is_active'])
    
    # API Keys jadvali uchun indexlar
    op.create_index('idx_api_keys_service', 'api_keys', ['service_name'])
    op.create_index('idx_api_keys_active', 'api_keys', ['is_active'])
    op.create_index('idx_api_keys_priority', 'api_keys', ['priority'])
    
    # Settings jadvali uchun indexlar
    op.create_index('idx_settings_user_id', 'settings', ['user_id'])
    op.create_index('idx_settings_key', 'settings', ['setting_key'])
    op.create_index('idx_settings_global', 'settings', ['is_global'])
    op.create_index('idx_settings_category', 'settings', ['category'])
    
    # System Status jadvali uchun indexlar
    op.create_index('idx_system_status_service', 'system_status', ['service_name'])
    op.create_index('idx_system_status_status', 'system_status', ['status'])
    op.create_index('idx_system_status_last_check', 'system_status', ['last_check'])


def insert_initial_data():
    """
    Boshlang'ich ma'lumotlar qo'shish
    """
    # Boshlang'ich konfiguratsiya
    config_data = [
        # Trading sozlamalar
        ('max_risk_per_trade', '0.02', 'float', 'Har bir savdo uchun maksimal risk', 'trading'),
        ('max_daily_loss', '0.05', 'float', 'Kunlik maksimal zarar', 'trading'),
        ('position_size_method', 'kelly', 'string', 'Position size hisoblash usuli', 'trading'),
        ('auto_trading_enabled', 'false', 'boolean', 'Avtomatik savdo yoqilganmi', 'trading'),
        
        # Bot sozlamalar
        ('bot_name', 'AI OrderFlow Signal Bot', 'string', 'Bot nomi', 'bot'),
        ('bot_version', '1.0.0', 'string', 'Bot versiyasi', 'bot'),
        ('working_hours_start', '07:00', 'string', 'Ish vaqti boshlanishi', 'bot'),
        ('working_hours_end', '19:30', 'string', 'Ish vaqti tugashi', 'bot'),
        ('timezone', 'Asia/Tashkent', 'string', 'Vaqt zonasi', 'bot'),
        
        # API sozlamalar
        ('api_timeout', '30', 'integer', 'API timeout (soniyalar)', 'api'),
        ('max_retries', '3', 'integer', 'Maksimal qayta urinish', 'api'),
        ('rate_limit_default', '100', 'integer', 'Standart rate limit', 'api'),
        
        # Fallback sozlamalar
        ('fallback_enabled', 'true', 'boolean', 'Fallback tizimi yoqilganmi', 'fallback'),
        ('fallback_delay', '5', 'integer', 'Fallback kutish vaqti (soniyalar)', 'fallback'),
        
        # Logging sozlamalar
        ('log_level', 'INFO', 'string', 'Log darajasi', 'logging'),
        ('log_rotation', 'true', 'boolean', 'Log fayllarni aylantirishmi', 'logging'),
        ('log_max_size', '10485760', 'integer', 'Log faylning maksimal hajmi (bytes)', 'logging'),
        
        # Telegram sozlamalar
        ('telegram_enabled', 'true', 'boolean', 'Telegram bot yoqilganmi', 'telegram'),
        ('telegram_language', 'uz', 'string', 'Telegram bot tili', 'telegram'),
        ('signal_format', 'detailed', 'string', 'Signal format turi', 'telegram'),
    ]
    
    # Konfiguratsiya ma'lumotlarini qo'shish
    config_table = sa.table('config',
        sa.column('config_key', sa.String),
        sa.column('config_value', sa.Text),
        sa.column('config_type', sa.String),
        sa.column('description', sa.Text),
        sa.column('category', sa.String),
        sa.column('is_active', sa.Boolean),
        sa.column('created_at', sa.DateTime),
    )
    
    for key, value, type_, desc, category in config_data:
        op.execute(
            config_table.insert().values(
                config_key=key,
                config_value=value,
                config_type=type_,
                description=desc,
                category=category,
                is_active=True,
                created_at=datetime.utcnow()
            )
        )
    
    # System Status boshlang'ich ma'lumotlar
    system_services = [
        ('oneinch_api', 'active'),
        ('alchemy_api', 'active'),
        ('telegram_bot', 'active'),
        ('huggingface_api', 'active'),
        ('gemini_api', 'active'),
        ('claude_api', 'active'),
        ('news_api', 'active'),
        ('reddit_api', 'active'),
        ('database', 'active'),
        ('main_bot', 'active'),
    ]
    
    status_table = sa.table('system_status',
        sa.column('service_name', sa.String),
        sa.column('status', sa.String),
        sa.column('last_check', sa.DateTime),
        sa.column('uptime_percentage', sa.Float),
        sa.column('created_at', sa.DateTime),
    )
    
    for service, status in system_services:
        op.execute(
            status_table.insert().values(
                service_name=service,
                status=status,
                last_check=datetime.utcnow(),
                uptime_percentage=100.0,
                created_at=datetime.utcnow()
            )
        )
    
    # Global settings
    settings_data = [
        ('default_language', 'uz', 'string', True, 'general', 'Standart til'),
        ('default_timezone', 'Asia/Tashkent', 'string', True, 'general', 'Standart vaqt zonasi'),
        ('maintenance_mode', 'false', 'boolean', True, 'system', 'Texnik xizmat rejimi'),
        ('max_users', '1000', 'integer', True, 'system', 'Maksimal foydalanuvchilar soni'),
        ('backup_enabled', 'true', 'boolean', True, 'system', 'Backup yoqilganmi'),
        ('backup_interval', '24', 'integer', True, 'system', 'Backup interval (soat)'),
    ]
    
    settings_table = sa.table('settings',
        sa.column('setting_key', sa.String),
        sa.column('setting_value', sa.Text),
        sa.column('setting_type', sa.String),
        sa.column('is_global', sa.Boolean),
        sa.column('category', sa.String),
        sa.column('description', sa.Text),
        sa.column('created_at', sa.DateTime),
    )
    
    for key, value, type_, is_global, category, desc in settings_data:
        op.execute(
            settings_table.insert().values(
                setting_key=key,
                setting_value=value,
                setting_type=type_,
                is_global=is_global,
                category=category,
                description=desc,
                created_at=datetime.utcnow()
            )
        )


# Qo'shimcha utility funksiyalar
def get_table_info():
    """
    Yaratilgan jadvallar haqida ma'lumot
    """
    return {
        'users': 'Foydalanuvchilar jadvali',
        'config': 'Tizim konfiguratsiyasi',
        'api_keys': 'API kalitlar',
        'settings': 'Sozlamalar',
        'system_status': 'Tizim holati'
    }


def validate_migration():
    """
    Migration to'g'riligini tekshirish
    """
    # Bu funksiya migration amalga oshirilgandan keyin
    # jadvallarning mavjudligini tekshirish uchun ishlatiladi
    pass


# Migration ma'lumotlari
MIGRATION_INFO = {
    'revision': '001',
    'description': 'Asosiy jadvallar yaratish',
    'tables_created': ['users', 'config', 'api_keys', 'settings', 'system_status'],
    'indexes_created': 15,
    'initial_data_inserted': True,
    'author': 'AI OrderFlow Bot',
    'date': '2024-01-01'
}

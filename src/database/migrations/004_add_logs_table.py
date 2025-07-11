"""
Database Migration 004: Log jadvali qo'shish
Barcha tizim loglarini saqlash uchun logs jadvali yaratish
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import sqlite
import enum
from datetime import datetime


class LogLevel(enum.Enum):
    """Log darajasi enum - O'zbekcha nomi bilan"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


# Revision identifiers
revision = '004'
down_revision = '003'
branch_labels = None
depends_on = None


def upgrade():
    """
    Logs jadvali va bog'liq strukturalarni yaratish
    """
    # Logs asosiy jadvali yaratish
    op.create_table(
        'logs',
        sa.Column('log_id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('timestamp', sa.DateTime(), nullable=False, default=datetime.utcnow),
        sa.Column('level', sa.Enum(LogLevel), nullable=False),
        sa.Column('module', sa.String(100), nullable=False),
        sa.Column('message', sa.Text(), nullable=False),
        sa.Column('error_details', sa.Text(), nullable=True),
        sa.Column('stack_trace', sa.Text(), nullable=True),
        sa.Column('user_id', sa.Integer(), nullable=True),
        sa.Column('session_id', sa.String(50), nullable=True),
        sa.Column('request_id', sa.String(50), nullable=True),
        sa.Column('api_endpoint', sa.String(200), nullable=True),
        sa.Column('response_time', sa.Float(), nullable=True),
        sa.Column('ip_address', sa.String(45), nullable=True),
        sa.Column('user_agent', sa.String(500), nullable=True),
        sa.Column('additional_data', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), default=datetime.utcnow),
        sa.Column('updated_at', sa.DateTime(), default=datetime.utcnow, onupdate=datetime.utcnow),
    )
    
    # API log jadvali - API so'rovlarini batafsil kuzatish
    op.create_table(
        'api_logs',
        sa.Column('api_log_id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('timestamp', sa.DateTime(), nullable=False, default=datetime.utcnow),
        sa.Column('api_name', sa.String(50), nullable=False),
        sa.Column('endpoint', sa.String(200), nullable=False),
        sa.Column('method', sa.String(10), nullable=False),
        sa.Column('status_code', sa.Integer(), nullable=True),
        sa.Column('response_time', sa.Float(), nullable=True),
        sa.Column('request_size', sa.Integer(), nullable=True),
        sa.Column('response_size', sa.Integer(), nullable=True),
        sa.Column('rate_limit_remaining', sa.Integer(), nullable=True),
        sa.Column('rate_limit_reset', sa.DateTime(), nullable=True),
        sa.Column('retry_count', sa.Integer(), default=0),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('fallback_used', sa.Boolean(), default=False),
        sa.Column('fallback_api', sa.String(50), nullable=True),
        sa.Column('request_headers', sa.JSON(), nullable=True),
        sa.Column('response_headers', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), default=datetime.utcnow),
    )
    
    # Error summary jadvali - xatolar statistikasi
    op.create_table(
        'error_summary',
        sa.Column('error_id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('error_type', sa.String(100), nullable=False),
        sa.Column('error_message', sa.Text(), nullable=False),
        sa.Column('module', sa.String(100), nullable=False),
        sa.Column('first_occurrence', sa.DateTime(), nullable=False),
        sa.Column('last_occurrence', sa.DateTime(), nullable=False),
        sa.Column('occurrence_count', sa.Integer(), default=1),
        sa.Column('resolved', sa.Boolean(), default=False),
        sa.Column('resolution_notes', sa.Text(), nullable=True),
        sa.Column('severity', sa.Enum(LogLevel), nullable=False),
        sa.Column('created_at', sa.DateTime(), default=datetime.utcnow),
        sa.Column('updated_at', sa.DateTime(), default=datetime.utcnow, onupdate=datetime.utcnow),
    )
    
    # Performance metrics jadvali - tizim ishlash ko'rsatkichlari
    op.create_table(
        'performance_logs',
        sa.Column('perf_id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('timestamp', sa.DateTime(), nullable=False, default=datetime.utcnow),
        sa.Column('module', sa.String(100), nullable=False),
        sa.Column('operation', sa.String(100), nullable=False),
        sa.Column('duration', sa.Float(), nullable=False),
        sa.Column('cpu_usage', sa.Float(), nullable=True),
        sa.Column('memory_usage', sa.Float(), nullable=True),
        sa.Column('success', sa.Boolean(), default=True),
        sa.Column('records_processed', sa.Integer(), nullable=True),
        sa.Column('data_size', sa.Integer(), nullable=True),
        sa.Column('additional_metrics', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), default=datetime.utcnow),
    )
    
    # Asosiy logs jadvali uchun indexlar
    op.create_index('idx_logs_timestamp', 'logs', ['timestamp'])
    op.create_index('idx_logs_level', 'logs', ['level'])
    op.create_index('idx_logs_module', 'logs', ['module'])
    op.create_index('idx_logs_user_id', 'logs', ['user_id'])
    op.create_index('idx_logs_session_id', 'logs', ['session_id'])
    op.create_index('idx_logs_level_timestamp', 'logs', ['level', 'timestamp'])
    op.create_index('idx_logs_module_timestamp', 'logs', ['module', 'timestamp'])
    
    # Full-text search index message ustunida
    op.create_index('idx_logs_message_fts', 'logs', ['message'])
    
    # API logs jadvali uchun indexlar
    op.create_index('idx_api_logs_timestamp', 'api_logs', ['timestamp'])
    op.create_index('idx_api_logs_api_name', 'api_logs', ['api_name'])
    op.create_index('idx_api_logs_endpoint', 'api_logs', ['endpoint'])
    op.create_index('idx_api_logs_status_code', 'api_logs', ['status_code'])
    op.create_index('idx_api_logs_response_time', 'api_logs', ['response_time'])
    op.create_index('idx_api_logs_api_timestamp', 'api_logs', ['api_name', 'timestamp'])
    
    # Error summary jadvali uchun indexlar
    op.create_index('idx_error_summary_type', 'error_summary', ['error_type'])
    op.create_index('idx_error_summary_module', 'error_summary', ['module'])
    op.create_index('idx_error_summary_resolved', 'error_summary', ['resolved'])
    op.create_index('idx_error_summary_severity', 'error_summary', ['severity'])
    op.create_index('idx_error_summary_last_occurrence', 'error_summary', ['last_occurrence'])
    
    # Performance logs jadvali uchun indexlar
    op.create_index('idx_performance_timestamp', 'performance_logs', ['timestamp'])
    op.create_index('idx_performance_module', 'performance_logs', ['module'])
    op.create_index('idx_performance_operation', 'performance_logs', ['operation'])
    op.create_index('idx_performance_duration', 'performance_logs', ['duration'])
    op.create_index('idx_performance_success', 'performance_logs', ['success'])
    op.create_index('idx_performance_module_operation', 'performance_logs', ['module', 'operation'])
    
    # Composite indexlar - murakkab so'rovlar uchun
    op.create_index('idx_logs_level_module_timestamp', 'logs', ['level', 'module', 'timestamp'])
    op.create_index('idx_api_logs_api_status_timestamp', 'api_logs', ['api_name', 'status_code', 'timestamp'])
    
    # Partitioning uchun check constraints
    op.create_check_constraint(
        'check_logs_timestamp_valid',
        'logs',
        'timestamp >= \'2024-01-01\''
    )
    
    op.create_check_constraint(
        'check_response_time_positive',
        'api_logs',
        'response_time >= 0'
    )
    
    op.create_check_constraint(
        'check_retry_count_positive',
        'api_logs',
        'retry_count >= 0'
    )
    
    op.create_check_constraint(
        'check_performance_duration_positive',
        'performance_logs',
        'duration >= 0'
    )
    
    print("✅ Logs jadvallari muvaffaqiyatli yaratildi")
    print("📊 Indexlar va constraintlar qo'shildi")
    print("🔍 Full-text search qo'llab-quvvatlanadi")


def downgrade():
    """
    Logs jadvallarini o'chirish
    """
    # Indexlarni o'chirish
    op.drop_index('idx_logs_level_module_timestamp', table_name='logs')
    op.drop_index('idx_api_logs_api_status_timestamp', table_name='api_logs')
    op.drop_index('idx_performance_module_operation', table_name='performance_logs')
    op.drop_index('idx_performance_success', table_name='performance_logs')
    op.drop_index('idx_performance_duration', table_name='performance_logs')
    op.drop_index('idx_performance_operation', table_name='performance_logs')
    op.drop_index('idx_performance_module', table_name='performance_logs')
    op.drop_index('idx_performance_timestamp', table_name='performance_logs')
    op.drop_index('idx_error_summary_last_occurrence', table_name='error_summary')
    op.drop_index('idx_error_summary_severity', table_name='error_summary')
    op.drop_index('idx_error_summary_resolved', table_name='error_summary')
    op.drop_index('idx_error_summary_module', table_name='error_summary')
    op.drop_index('idx_error_summary_type', table_name='error_summary')
    op.drop_index('idx_api_logs_api_timestamp', table_name='api_logs')
    op.drop_index('idx_api_logs_response_time', table_name='api_logs')
    op.drop_index('idx_api_logs_status_code', table_name='api_logs')
    op.drop_index('idx_api_logs_endpoint', table_name='api_logs')
    op.drop_index('idx_api_logs_api_name', table_name='api_logs')
    op.drop_index('idx_api_logs_timestamp', table_name='api_logs')
    op.drop_index('idx_logs_message_fts', table_name='logs')
    op.drop_index('idx_logs_module_timestamp', table_name='logs')
    op.drop_index('idx_logs_level_timestamp', table_name='logs')
    op.drop_index('idx_logs_session_id', table_name='logs')
    op.drop_index('idx_logs_user_id', table_name='logs')
    op.drop_index('idx_logs_module', table_name='logs')
    op.drop_index('idx_logs_level', table_name='logs')
    op.drop_index('idx_logs_timestamp', table_name='logs')
    
    # Jadvallarni o'chirish
    op.drop_table('performance_logs')
    op.drop_table('error_summary')
    op.drop_table('api_logs')
    op.drop_table('logs')
    
    print("🗑️ Logs jadvallari o'chirildi")


# Migratsiya metadata
migration_info = {
    'revision': revision,
    'down_revision': down_revision,
    'description': 'Log jadvali va bog\'liq strukturalarni yaratish',
    'tables_created': ['logs', 'api_logs', 'error_summary', 'performance_logs'],
    'indexes_created': 20,
    'constraints_added': 4,
    'features': [
        'Full-text search qo\'llab-quvvatlash',
        'Performance monitoring',
        'Error tracking va summary',
        'API call logging',
        'Composite indexlar',
        'Data validation constraints'
    ]
}

"""
Database Migrations Module
========================

Database migratsiya tizimi - versiya boshqaruvi va schema yangilanishi

Modules:
- migration_manager: Migratsiya boshqaruvi va versiya nazorati
- 001_initial_schema: Boshlang'ich database schema
- 002_add_indexes: Performance optimizatsiya va indexlar
- seed_data: Boshlang'ich ma'lumotlar va default sozlamalar

Author: AI OrderFlow Signal Bot
Created: 2025
"""

from .migration_manager import MigrationManager
from .seed_data import SeedData

__all__ = [
    'MigrationManager',
    'SeedData'
]

# Migratsiya ketma-ketligi
MIGRATION_ORDER = [
    '001_initial_schema',
    '002_add_indexes'
]

# Versiya ma'lumotlari
DATABASE_VERSION = '1.0.0'
MIGRATION_VERSION = '002'

# Default sozlamalar
DEFAULT_MIGRATION_TABLE = 'schema_migrations'
DEFAULT_BACKUP_DIR = 'data/backups'
DEFAULT_MIGRATION_DIR = 'database/migrations'

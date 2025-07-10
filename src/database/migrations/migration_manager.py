"""
Migration Manager - Database Migratsiya Boshqaruvi
=================================================

Database schema versiya nazorati va migratsiya boshqaruvi

Funksiyalar:
- Migratsiya ishga tushirish va bekor qilish
- Versiya nazorati va holat kuzatuvi
- Schema yangilanishi va backup
- Rollback va recovery

Author: AI OrderFlow Signal Bot
"""

import asyncio
import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import sqlite3
import aiosqlite
from utils.logger import get_logger
from utils.error_handler import handle_database_error
from config.config import ConfigManager

logger = get_logger(__name__)

@dataclass
class MigrationInfo:
    """Migratsiya ma'lumotlari"""
    version: str
    name: str
    description: str
    applied_at: Optional[datetime] = None
    status: str = 'pending'  # pending, applied, failed, rollback
    sql_up: str = ''
    sql_down: str = ''
    
@dataclass
class MigrationResult:
    """Migratsiya natijasi"""
    success: bool
    version: str
    message: str
    error: Optional[str] = None
    execution_time: float = 0.0

class MigrationManager:
    """Database migratsiya boshqaruvi"""
    
    def __init__(self, db_path: str = "data/bot.db"):
        self.db_path = db_path
        self.config = ConfigManager()
        self.migration_table = 'schema_migrations'
        self.backup_dir = Path('data/backups')
        self.migration_dir = Path('database/migrations')
        
        # Backup directory yaratish
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Available migratsiyalar
        self.available_migrations = {
            '001': {
                'name': 'initial_schema',
                'description': 'Boshlang\'ich database schema yaratish',
                'module': 'database.migrations.001_initial_schema'
            },
            '002': {
                'name': 'add_indexes',
                'description': 'Performance indexlar va optimizatsiya',
                'module': 'database.migrations.002_add_indexes'
            }
        }
        
        logger.info("MigrationManager ishga tushirildi")
    
    async def initialize(self) -> None:
        """Migratsiya tizimini ishga tushirish"""
        try:
            await self._create_migration_table()
            logger.info("Migratsiya tizimi tayyor")
        except Exception as e:
            logger.error(f"Migratsiya tizimi ishga tushirishda xato: {e}")
            raise
    
    async def _create_migration_table(self) -> None:
        """Migratsiya jadvalini yaratish"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(f'''
                    CREATE TABLE IF NOT EXISTS {self.migration_table} (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        version TEXT UNIQUE NOT NULL,
                        name TEXT NOT NULL,
                        description TEXT,
                        applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        status TEXT DEFAULT 'applied',
                        execution_time REAL DEFAULT 0.0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                await db.commit()
                logger.info(f"Migratsiya jadvali ({self.migration_table}) tayyor")
        except Exception as e:
            logger.error(f"Migratsiya jadvali yaratishda xato: {e}")
            raise
    
    async def run_migrations(self) -> List[MigrationResult]:
        """Barcha kutilayotgan migratsiyalarni ishga tushirish"""
        try:
            results = []
            pending_migrations = await self.get_pending_migrations()
            
            if not pending_migrations:
                logger.info("Barcha migratsiyalar qo'llanilgan")
                return results
            
            logger.info(f"{len(pending_migrations)} ta migratsiya ishga tushirilmoqda")
            
            for version in pending_migrations:
                # Backup yaratish
                backup_path = await self._create_backup(version)
                
                try:
                    result = await self._apply_migration(version)
                    results.append(result)
                    
                    if result.success:
                        logger.info(f"Migratsiya {version} muvaffaqiyatli qo'llanildi")
                    else:
                        logger.error(f"Migratsiya {version} xato: {result.error}")
                        break  # Xato bo'lsa to'xtatamiz
                        
                except Exception as e:
                    logger.error(f"Migratsiya {version} bajarilmadi: {e}")
                    # Backup dan qaytarish
                    await self._restore_backup(backup_path)
                    results.append(MigrationResult(
                        success=False,
                        version=version,
                        message=f"Migratsiya bajarilmadi: {e}",
                        error=str(e)
                    ))
                    break
            
            return results
            
        except Exception as e:
            logger.error(f"Migratsiya ishga tushirishda xato: {e}")
            raise
    
    async def _apply_migration(self, version: str) -> MigrationResult:
        """Bitta migratsiyani qo'llash"""
        start_time = datetime.now()
        
        try:
            if version not in self.available_migrations:
                raise ValueError(f"Migratsiya {version} topilmadi")
            
            migration_info = self.available_migrations[version]
            
            # Migratsiya modulini import qilish
            module_path = migration_info['module']
            module = __import__(module_path, fromlist=[''])
            
            # Migratsiya klassini olish
            class_name = ''.join(word.capitalize() for word in migration_info['name'].split('_'))
            migration_class = getattr(module, class_name)
            
            # Migratsiya obyektini yaratish
            migration = migration_class(self.db_path)
            
            # Migratsiyani bajarish
            await migration.up()
            
            # Migratsiya yozuvini qo'shish
            execution_time = (datetime.now() - start_time).total_seconds()
            await self._record_migration(version, migration_info, execution_time)
            
            return MigrationResult(
                success=True,
                version=version,
                message=f"Migratsiya {version} muvaffaqiyatli qo'llanildi",
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Migratsiya {version} bajarishda xato: {e}")
            
            return MigrationResult(
                success=False,
                version=version,
                message=f"Migratsiya {version} bajarilmadi",
                error=str(e),
                execution_time=execution_time
            )
    
    async def rollback_migration(self, version: str) -> MigrationResult:
        """Migratsiyani bekor qilish"""
        try:
            if version not in self.available_migrations:
                raise ValueError(f"Migratsiya {version} topilmadi")
            
            # Migratsiya qo'llanilganligini tekshirish
            applied = await self._is_migration_applied(version)
            if not applied:
                return MigrationResult(
                    success=False,
                    version=version,
                    message=f"Migratsiya {version} qo'llanilmagan",
                    error="Rollback mumkin emas"
                )
            
            # Backup yaratish
            backup_path = await self._create_backup(f"rollback_{version}")
            
            migration_info = self.available_migrations[version]
            
            # Migratsiya modulini import qilish
            module_path = migration_info['module']
            module = __import__(module_path, fromlist=[''])
            
            # Migratsiya klassini olish
            class_name = ''.join(word.capitalize() for word in migration_info['name'].split('_'))
            migration_class = getattr(module, class_name)
            
            # Migratsiya obyektini yaratish
            migration = migration_class(self.db_path)
            
            # Rollback bajarish
            await migration.down()
            
            # Migratsiya yozuvini o'chirish
            await self._remove_migration_record(version)
            
            logger.info(f"Migratsiya {version} bekor qilindi")
            
            return MigrationResult(
                success=True,
                version=version,
                message=f"Migratsiya {version} bekor qilindi"
            )
            
        except Exception as e:
            logger.error(f"Migratsiya {version} bekor qilishda xato: {e}")
            return MigrationResult(
                success=False,
                version=version,
                message=f"Migratsiya {version} bekor qilinmadi",
                error=str(e)
            )
    
    async def get_pending_migrations(self) -> List[str]:
        """Kutilayotgan migratsiyalarni olish"""
        try:
            applied_migrations = await self._get_applied_migrations()
            pending = []
            
            for version in sorted(self.available_migrations.keys()):
                if version not in applied_migrations:
                    pending.append(version)
            
            return pending
            
        except Exception as e:
            logger.error(f"Kutilayotgan migratsiyalarni olishda xato: {e}")
            return []
    
    async def get_migration_status(self) -> Dict[str, Any]:
        """Migratsiya holatini olish"""
        try:
            applied_migrations = await self._get_applied_migrations()
            pending_migrations = await self.get_pending_migrations()
            
            status = {
                'current_version': await self.get_current_version(),
                'total_migrations': len(self.available_migrations),
                'applied_migrations': len(applied_migrations),
                'pending_migrations': len(pending_migrations),
                'applied_list': applied_migrations,
                'pending_list': pending_migrations,
                'database_path': self.db_path,
                'last_migration': None
            }
            
            if applied_migrations:
                last_migration = await self._get_last_migration()
                status['last_migration'] = last_migration
            
            return status
            
        except Exception as e:
            logger.error(f"Migratsiya holatini olishda xato: {e}")
            return {'error': str(e)}
    
    async def get_current_version(self) -> str:
        """Joriy database versiyasini olish"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute(
                    f"SELECT version FROM {self.migration_table} ORDER BY applied_at DESC LIMIT 1"
                )
                result = await cursor.fetchone()
                
                if result:
                    return result[0]
                else:
                    return '000'  # Hech qanday migratsiya qo'llanilmagan
                    
        except Exception as e:
            logger.error(f"Joriy versiyani olishda xato: {e}")
            return '000'
    
    async def _get_applied_migrations(self) -> List[str]:
        """Qo'llanilgan migratsiyalar ro'yxati"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute(
                    f"SELECT version FROM {self.migration_table} WHERE status = 'applied' ORDER BY version"
                )
                results = await cursor.fetchall()
                
                return [row[0] for row in results]
                
        except Exception as e:
            logger.error(f"Qo'llanilgan migratsiyalarni olishda xato: {e}")
            return []
    
    async def _is_migration_applied(self, version: str) -> bool:
        """Migratsiya qo'llanilganligini tekshirish"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute(
                    f"SELECT 1 FROM {self.migration_table} WHERE version = ? AND status = 'applied'",
                    (version,)
                )
                result = await cursor.fetchone()
                return result is not None
                
        except Exception as e:
            logger.error(f"Migratsiya holatini tekshirishda xato: {e}")
            return False
    
    async def _record_migration(self, version: str, migration_info: Dict, execution_time: float) -> None:
        """Migratsiya yozuvini saqlash"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(f'''
                    INSERT INTO {self.migration_table} 
                    (version, name, description, execution_time, status)
                    VALUES (?, ?, ?, ?, 'applied')
                ''', (
                    version,
                    migration_info['name'],
                    migration_info['description'],
                    execution_time
                ))
                await db.commit()
                
        except Exception as e:
            logger.error(f"Migratsiya yozuvini saqlashda xato: {e}")
            raise
    
    async def _remove_migration_record(self, version: str) -> None:
        """Migratsiya yozuvini o'chirish"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    f"DELETE FROM {self.migration_table} WHERE version = ?",
                    (version,)
                )
                await db.commit()
                
        except Exception as e:
            logger.error(f"Migratsiya yozuvini o'chirishda xato: {e}")
            raise
    
    async def _get_last_migration(self) -> Optional[Dict]:
        """Oxirgi migratsiya ma'lumotini olish"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute(f'''
                    SELECT version, name, description, applied_at, execution_time
                    FROM {self.migration_table}
                    ORDER BY applied_at DESC
                    LIMIT 1
                ''')
                result = await cursor.fetchone()
                
                if result:
                    return {
                        'version': result[0],
                        'name': result[1],
                        'description': result[2],
                        'applied_at': result[3],
                        'execution_time': result[4]
                    }
                
                return None
                
        except Exception as e:
            logger.error(f"Oxirgi migratsiya ma'lumotini olishda xato: {e}")
            return None
    
    async def _create_backup(self, version: str) -> str:
        """Database backup yaratish"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_filename = f"backup_{version}_{timestamp}.db"
            backup_path = self.backup_dir / backup_filename
            
            # SQLite backup yaratish
            source_db = sqlite3.connect(self.db_path)
            backup_db = sqlite3.connect(str(backup_path))
            
            source_db.backup(backup_db)
            
            source_db.close()
            backup_db.close()
            
            logger.info(f"Backup yaratildi: {backup_path}")
            return str(backup_path)
            
        except Exception as e:
            logger.error(f"Backup yaratishda xato: {e}")
            raise
    
    async def _restore_backup(self, backup_path: str) -> None:
        """Backup dan qaytarish"""
        try:
            if not os.path.exists(backup_path):
                raise FileNotFoundError(f"Backup fayl topilmadi: {backup_path}")
            
            # Joriy database ni backup fayl bilan almashtirish
            backup_db = sqlite3.connect(backup_path)
            current_db = sqlite3.connect(self.db_path)
            
            backup_db.backup(current_db)
            
            backup_db.close()
            current_db.close()
            
            logger.info(f"Database qaytarildi: {backup_path}")
            
        except Exception as e:
            logger.error(f"Backup dan qaytarishda xato: {e}")
            raise
    
    async def create_migration(self, name: str, description: str) -> str:
        """Yangi migratsiya yaratish"""
        try:
            # Keyingi versiya raqamini hisoblash
            max_version = max(self.available_migrations.keys()) if self.available_migrations else '000'
            next_version = f"{int(max_version) + 1:03d}"
            
            # Migratsiya fayl nomi
            filename = f"{next_version}_{name}.py"
            filepath = self.migration_dir / filename
            
            # Migratsiya template
            template = f'''"""
Migratsiya {next_version}: {description}
{'='*50}

{description}

Author: AI OrderFlow Signal Bot
Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

import aiosqlite
from utils.logger import get_logger

logger = get_logger(__name__)

class {name.title().replace('_', '')}:
    """Migratsiya {next_version}: {description}"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
    
    async def up(self) -> None:
        """Migratsiyani qo'llash"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # TODO: Migratsiya kodini bu yerga yozing
                pass
                
        except Exception as e:
            logger.error(f"Migratsiya {next_version} up() xato: {{e}}")
            raise
    
    async def down(self) -> None:
        """Migratsiyani bekor qilish"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # TODO: Rollback kodini bu yerga yozing
                pass
                
        except Exception as e:
            logger.error(f"Migratsiya {next_version} down() xato: {{e}}")
            raise
'''
            
            # Faylni yaratish
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(template)
            
            logger.info(f"Yangi migratsiya yaratildi: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Yangi migratsiya yaratishda xato: {e}")
            raise
    
    async def check_database_integrity(self) -> bool:
        """Database yaxlitligini tekshirish"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute("PRAGMA integrity_check")
                result = await cursor.fetchone()
                
                if result and result[0] == 'ok':
                    logger.info("Database yaxlitlik tekshiruvi: OK")
                    return True
                else:
                    logger.error(f"Database yaxlitlik xatosi: {result}")
                    return False
                    
        except Exception as e:
            logger.error(f"Database yaxlitligini tekshirishda xato: {e}")
            return False
    
    async def optimize_database(self) -> None:
        """Database optimizatsiyasi"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # VACUUM operatsiyasi
                await db.execute("VACUUM")
                
                # ANALYZE operatsiyasi
                await db.execute("ANALYZE")
                
                await db.commit()
                
            logger.info("Database optimizatsiyasi tugallandi")
            
        except Exception as e:
            logger.error(f"Database optimizatsiyasida xato: {e}")
            raise

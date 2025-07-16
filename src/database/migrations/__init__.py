# AI yo'riqnomasi: Database migrations modulini yaratish
"""
Database migrations moduli - Ma'lumotlar bazasi migratsiyalarini boshqarish

Bu modul database migratsiyalarini tartib bilan bajarish va
migratsiya holatini kuzatish uchun ishlatiladi.

Modul tarkibi:
- Migratsiya fayllarini import qilish
- Migratsiya tartibini belgilash
- Migratsiya holatini kuzatish
- Alembic bilan integratsiya

Migratsiya fayllar:
1. 001_initial_tables.py - Asosiy jadvallar
2. 002_add_signals_table.py - Signal jadvali
3. 003_add_trades_table.py - Savdo jadvali
4. 004_add_logs_table.py - Log jadvali
5. 005_add_indexes.py - Index qo'shish
"""

from typing import List, Dict, Optional
import os
from pathlib import Path

# Migrations modulini versiyasi
__version__ = "1.0.0"

# Migration fayllarining ro'yxati (tartib bo'yicha)
MIGRATION_FILES = [
    "001_initial_tables.py",
    "002_add_signals_table.py", 
    "003_add_trades_table.py",
    "004_add_logs_table.py",
    "005_add_indexes.py"
]

# Migration fayllarining tavsifi
MIGRATION_DESCRIPTIONS = {
    "001_initial_tables.py": "Asosiy jadvallar - users, config, api_keys, settings",
    "002_add_signals_table.py": "Signal jadvali - trading signallari",
    "003_add_trades_table.py": "Savdo jadvali - bajarilgan savdolar",
    "004_add_logs_table.py": "Log jadvali - tizim loglari",
    "005_add_indexes.py": "Indexlar - performans optimizatsiyasi"
}

# Migration fayllari import qilish
def get_migration_modules():
    """
    Barcha migration modullarini import qilish
    
    Returns:
        Dict: Migration modullar ro'yxati
    """
    modules = {}
    current_dir = Path(__file__).parent
    
    for migration_file in MIGRATION_FILES:
        if migration_file.endswith('.py'):
            module_name = migration_file[:-3]  # .py ni olib tashlash
            try:
                # Migration modulini import qilish
                modules[module_name] = {
                    'file': migration_file,
                    'description': MIGRATION_DESCRIPTIONS.get(migration_file, "Ma'lumot yo'q"),
                    'status': 'available'
                }
            except ImportError as e:
                modules[module_name] = {
                    'file': migration_file,
                    'description': MIGRATION_DESCRIPTIONS.get(migration_file, "Ma'lumot yo'q"),
                    'status': 'error',
                    'error': str(e)
                }
    
    return modules

# Migration holatini tekshirish
def check_migration_status() -> Dict:
    """
    Migration holatini tekshirish
    
    Returns:
        Dict: Migration holati haqida ma'lumot
    """
    modules = get_migration_modules()
    
    status = {
        'total_migrations': len(MIGRATION_FILES),
        'available_migrations': len([m for m in modules.values() if m['status'] == 'available']),
        'error_migrations': len([m for m in modules.values() if m['status'] == 'error']),
        'migrations': modules
    }
    
    return status

# Migration ro'yxatini olish
def get_migration_list() -> List[str]:
    """
    Migration fayllar ro'yxatini olish
    
    Returns:
        List[str]: Migration fayllar ro'yxati
    """
    return MIGRATION_FILES.copy()

# Migration tavsifini olish
def get_migration_description(migration_file: str) -> Optional[str]:
    """
    Migration fayl tavsifini olish
    
    Args:
        migration_file: Migration fayl nomi
        
    Returns:
        Optional[str]: Migration tavsifi
    """
    return MIGRATION_DESCRIPTIONS.get(migration_file)

# Migration fayl mavjudligini tekshirish
def check_migration_exists(migration_file: str) -> bool:
    """
    Migration fayl mavjudligini tekshirish
    
    Args:
        migration_file: Migration fayl nomi
        
    Returns:
        bool: Fayl mavjudligi
    """
    current_dir = Path(__file__).parent
    migration_path = current_dir / migration_file
    return migration_path.exists()

# Migration modulini eksport qilish
__all__ = [
    'MIGRATION_FILES',
    'MIGRATION_DESCRIPTIONS',
    'get_migration_modules',
    'check_migration_status',
    'get_migration_list',
    'get_migration_description',
    'check_migration_exists'
]

# Modul yuklanganda migratsiya holatini tekshirish
if __name__ == "__main__":
    print(f"Database Migrations Module v{__version__}")
    print("=" * 50)
    
    status = check_migration_status()
    print(f"Jami migratsiyalar: {status['total_migrations']}")
    print(f"Mavjud migratsiyalar: {status['available_migrations']}")
    print(f"Xatolik bor: {status['error_migrations']}")
    print()
    
    print("Migration fayllar ro'yxati:")
    for i, migration_file in enumerate(MIGRATION_FILES, 1):
        description = get_migration_description(migration_file)
        exists = check_migration_exists(migration_file)
        status_icon = "✅" if exists else "❌"
        print(f"{i}. {status_icon} {migration_file} - {description}")

pass

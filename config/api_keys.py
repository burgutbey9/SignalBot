"""
AI OrderFlow & Signal Bot - API kalitlar boshqaruvi
Barcha API kalitlarni xavfsiz saqlash va boshqarish
"""

import os
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime, timedelta
import hashlib
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# Logger import qilish (keyinroq yaratiladi)
# from utils.logger import get_logger
# logger = get_logger(__name__)

@dataclass
class APIKeyInfo:
    """API kaliti ma'lumotlari"""
    service: str
    key: str
    encrypted: bool = True
    active: bool = True
    usage_count: int = 0
    last_used: Optional[datetime] = None
    rate_limit_reset: Optional[datetime] = None
    daily_limit: int = 1000
    daily_usage: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None

@dataclass
class APIServiceConfig:
    """API servis konfiguratsiyasi"""
    name: str
    base_url: str
    auth_type: str = "bearer"  # bearer, api_key, oauth
    headers: Dict[str, str] = field(default_factory=dict)
    required_scopes: List[str] = field(default_factory=list)
    rate_limit: int = 100
    timeout: int = 30
    max_retries: int = 3
    health_check_endpoint: str = ""

class APIKeyManager:
    """API kalitlar boshqaruvchi sinf"""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.keys_file = self.config_dir / "api_keys.json"
        self.encrypted_keys_file = self.config_dir / "encrypted_keys.json"
        self.env_file = Path(".env")
        
        # Shifrlash kaliti
        self.encryption_key = None
        self.cipher_suite = None
        
        # API kalitlar
        self.api_keys: Dict[str, APIKeyInfo] = {}
        self.service_configs: Dict[str, APIServiceConfig] = {}
        
        # Inicializatsiya
        self._init_encryption()
        self._load_service_configs()
        self._load_api_keys()
    
    def _init_encryption(self) -> None:
        """Shifrlash tizimini ishga tushirish"""
        try:
            # Master parol olish yoki yaratish
            master_password = os.getenv('MASTER_PASSWORD')
            if not master_password:
                master_password = self._generate_master_password()
                print(f"⚠️  Master parol yaratildi. .env fayliga qo'shing: MASTER_PASSWORD={master_password}")
            
            # Shifrlash kalitini yaratish
            password = master_password.encode()
            salt = b'ai_orderflow_salt_2024'  # Production da random bo'lishi kerak
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(password))
            self.encryption_key = key
            self.cipher_suite = Fernet(key)
            
            print("🔐 Shifrlash tizimi ishga tushirildi")
            
        except Exception as e:
            print(f"❌ Shifrlash tizimini ishga tushirishda xato: {e}")
            # Fallback - shifrlashsiz ishlash
            self.encryption_key = None
            self.cipher_suite = None
    
    def _generate_master_password(self) -> str:
        """Master parol yaratish"""
        import secrets
        import string
        
        alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
        password = ''.join(secrets.choice(alphabet) for _ in range(32))
        return password
    
    def _load_service_configs(self) -> None:
        """API servislar konfiguratsiyasini yuklash"""
        try:
            # Predefined service configs
            self.service_configs = {
                "oneinch": APIServiceConfig(
                    name="1inch",
                    base_url="https://api.1inch.io/v5.0",
                    auth_type="bearer",
                    headers={"Authorization": "Bearer {key}"},
                    rate_limit=100,
                    timeout=30,
                    health_check_endpoint="/1/healthcheck"
                ),
                "alchemy": APIServiceConfig(
                    name="Alchemy",
                    base_url="https://eth-mainnet.g.alchemy.com/v2",
                    auth_type="api_key",
                    headers={"Content-Type": "application/json"},
                    rate_limit=300,
                    timeout=15,
                    health_check_endpoint="/eth_blockNumber"
                ),
                "huggingface": APIServiceConfig(
                    name="HuggingFace",
                    base_url="https://api-inference.huggingface.co",
                    auth_type="bearer",
                    headers={"Authorization": "Bearer {key}"},
                    rate_limit=1000,
                    timeout=60,
                    health_check_endpoint="/models"
                ),
                "gemini": APIServiceConfig(
                    name="Gemini AI",
                    base_url="https://generativelanguage.googleapis.com/v1beta",
                    auth_type="api_key",
                    headers={"Content-Type": "application/json"},
                    rate_limit=60,
                    timeout=45,
                    health_check_endpoint="/models"
                ),
                "claude": APIServiceConfig(
                    name="Claude AI",
                    base_url="https://api.anthropic.com/v1",
                    auth_type="api_key",
                    headers={
                        "Content-Type": "application/json",
                        "x-api-key": "{key}",
                        "anthropic-version": "2023-06-01"
                    },
                    rate_limit=20,
                    timeout=60,
                    health_check_endpoint="/messages"
                ),
                "newsapi": APIServiceConfig(
                    name="News API",
                    base_url="https://newsapi.org/v2",
                    auth_type="api_key",
                    headers={"X-API-Key": "{key}"},
                    rate_limit=1000,
                    timeout=30,
                    health_check_endpoint="/sources"
                ),
                "reddit": APIServiceConfig(
                    name="Reddit API",
                    base_url="https://www.reddit.com/api/v1",
                    auth_type="oauth",
                    headers={"User-Agent": "AI OrderFlow Bot 1.0"},
                    rate_limit=60,
                    timeout=30,
                    health_check_endpoint="/me"
                ),
                "telegram": APIServiceConfig(
                    name="Telegram Bot",
                    base_url="https://api.telegram.org/bot{key}",
                    auth_type="token",
                    headers={"Content-Type": "application/json"},
                    rate_limit=30,
                    timeout=30,
                    health_check_endpoint="/getMe"
                ),
                "thegraph": APIServiceConfig(
                    name="The Graph",
                    base_url="https://gateway.thegraph.com/api",
                    auth_type="api_key",
                    headers={"Authorization": "Bearer {key}"},
                    rate_limit=100,
                    timeout=30,
                    health_check_endpoint="/subgraphs/health"
                ),
                "propshot": APIServiceConfig(
                    name="Propshot EA",
                    base_url="https://api.propshot.com/v1",
                    auth_type="api_key",
                    headers={
                        "Authorization": "Bearer {key}",
                        "Content-Type": "application/json"
                    },
                    rate_limit=100,
                    timeout=30,
                    health_check_endpoint="/account/info"
                )
            }
            
            print("📡 API servislar konfiguratsiyasi yuklandi")
            
        except Exception as e:
            print(f"❌ Servis konfiguratsiyasini yuklashda xato: {e}")
            raise APIKeyError(f"Servis konfiguratsiyasini yuklashda xato: {e}")
    
    def _load_api_keys(self) -> None:
        """API kalitlarni yuklash"""
        try:
            # Birinchi .env fayldan yuklash
            self._load_from_env()
            
            # Keyin shifrlangan fayldan yuklash
            self._load_from_encrypted_file()
            
            # Oxirida oddiy JSON fayldan yuklash (fallback)
            self._load_from_json_file()
            
            print(f"🔑 {len(self.api_keys)} ta API kaliti yuklandi")
            
        except Exception as e:
            print(f"❌ API kalitlarni yuklashda xato: {e}")
            raise APIKeyError(f"API kalitlarni yuklashda xato: {e}")
    
    def _load_from_env(self) -> None:
        """Environment faylidan API kalitlarni yuklash"""
        try:
            if not self.env_file.exists():
                return
            
            with open(self.env_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip().strip('"\'')
                        
                        # API kalitlarni tanib olish
                        if key.endswith('_API_KEY') or key.endswith('_TOKEN'):
                            service_name = self._extract_service_name(key)
                            if service_name:
                                self.api_keys[f"{service_name}_primary"] = APIKeyInfo(
                                    service=service_name,
                                    key=value,
                                    encrypted=False,
                                    active=True
                                )
                        
                        # Gemini kalitlari (5 ta)
                        elif key.startswith('GEMINI_API_KEY_'):
                            key_num = key.split('_')[-1]
                            self.api_keys[f"gemini_key_{key_num}"] = APIKeyInfo(
                                service="gemini",
                                key=value,
                                encrypted=False,
                                active=True
                            )
                        
                        # Telegram ma'lumotlari
                        elif key in ['TELEGRAM_BOT_TOKEN', 'TELEGRAM_CHAT_ID']:
                            service_name = key.lower().replace('_', '')
                            self.api_keys[service_name] = APIKeyInfo(
                                service="telegram",
                                key=value,
                                encrypted=False,
                                active=True
                            )
                        
                        # Reddit ma'lumotlari
                        elif key.startswith('REDDIT_'):
                            service_name = key.lower()
                            self.api_keys[service_name] = APIKeyInfo(
                                service="reddit",
                                key=value,
                                encrypted=False,
                                active=True
                            )
                        
                        # MT5 ma'lumotlari
                        elif key.startswith('MT5_'):
                            service_name = key.lower()
                            self.api_keys[service_name] = APIKeyInfo(
                                service="mt5",
                                key=value,
                                encrypted=False,
                                active=True
                            )
            
            print("📄 .env faylidan API kalitlar yuklandi")
            
        except Exception as e:
            print(f"❌ .env faylidan yuklashda xato: {e}")
    
    def _load_from_encrypted_file(self) -> None:
        """Shifrlangan fayldan API kalitlarni yuklash"""
        try:
            if not self.encrypted_keys_file.exists() or not self.cipher_suite:
                return
            
            with open(self.encrypted_keys_file, 'r', encoding='utf-8') as f:
                encrypted_data = json.load(f)
            
            for key_id, encrypted_info in encrypted_data.items():
                try:
                    # Kalitni shifrlash
                    decrypted_key = self.cipher_suite.decrypt(
                        encrypted_info['key'].encode()
                    ).decode()
                    
                    # APIKeyInfo obyektini yaratish
                    self.api_keys[key_id] = APIKeyInfo(
                        service=encrypted_info['service'],
                        key=decrypted_key,
                        encrypted=True,
                        active=encrypted_info.get('active', True),
                        usage_count=encrypted_info.get('usage_count', 0),
                        daily_limit=encrypted_info.get('daily_limit', 1000),
                        daily_usage=encrypted_info.get('daily_usage', 0)
                    )
                    
                except Exception as e:
                    print(f"❌ Kalitni shifrdan chiqarishda xato {key_id}: {e}")
                    continue
            
            print("🔐 Shifrlangan fayldan API kalitlar yuklandi")
            
        except Exception as e:
            print(f"❌ Shifrlangan fayldan yuklashda xato: {e}")
    
    def _load_from_json_file(self) -> None:
        """JSON fayldan API kalitlarni yuklash (fallback)"""
        try:
            if not self.keys_file.exists():
                return
            
            with open(self.keys_file, 'r', encoding='utf-8') as f:
                keys_data = json.load(f)
            
            for key_id, key_info in keys_data.items():
                if key_id not in self.api_keys:  # Faqat yangi kalitlar
                    self.api_keys[key_id] = APIKeyInfo(
                        service=key_info['service'],
                        key=key_info['key'],
                        encrypted=False,
                        active=key_info.get('active', True),
                        usage_count=key_info.get('usage_count', 0),
                        daily_limit=key_info.get('daily_limit', 1000),
                        daily_usage=key_info.get('daily_usage', 0)
                    )
            
            print("📋 JSON fayldan API kalitlar yuklandi")
            
        except Exception as e:
            print(f"❌ JSON fayldan yuklashda xato: {e}")
    
    def _extract_service_name(self, key: str) -> Optional[str]:
        """API kaliti nomidan servis nomini ajratish"""
        key = key.upper()
        
        if key.startswith('ONEINCH_'):
            return 'oneinch'
        elif key.startswith('ALCHEMY_'):
            return 'alchemy'
        elif key.startswith('HUGGINGFACE_'):
            return 'huggingface'
        elif key.startswith('CLAUDE_'):
            return 'claude'
        elif key.startswith('NEWS_'):
            return 'newsapi'
        elif key.startswith('PROPSHOT_'):
            return 'propshot'
        elif key.startswith('THEGRAPH_'):
            return 'thegraph'
        
        return None
    
    def get_api_key(self, service: str, key_id: Optional[str] = None) -> Optional[APIKeyInfo]:
        """API kalitini olish"""
        try:
            if key_id:
                # Aniq kalitni olish
                return self.api_keys.get(key_id)
            
            # Servis uchun birinchi active kalitni olish
            for key_id, key_info in self.api_keys.items():
                if key_info.service == service and key_info.active:
                    # Rate limit tekshirish
                    if self._check_rate_limit(key_info):
                        return key_info
            
            return None
            
        except Exception as e:
            print(f"❌ API kalitini olishda xato: {e}")
            return None
    
    def get_multiple_keys(self, service: str, count: int = 5) -> List[APIKeyInfo]:
        """Bir servis uchun ko'p kalitlarni olish (masalan, Gemini uchun 5 ta)"""
        try:
            keys = []
            for key_id, key_info in self.api_keys.items():
                if key_info.service == service and key_info.active:
                    if self._check_rate_limit(key_info):
                        keys.append(key_info)
                        if len(keys) >= count:
                            break
            
            return keys
            
        except Exception as e:
            print(f"❌ Ko'p kalitlarni olishda xato: {e}")
            return []
    
    def _check_rate_limit(self, key_info: APIKeyInfo) -> bool:
        """Rate limit tekshirish"""
        try:
            now = datetime.now()
            
            # Kunlik limit tekshirish
            if key_info.daily_usage >= key_info.daily_limit:
                return False
            
            # Rate limit reset vaqti tekshirish
            if key_info.rate_limit_reset and now < key_info.rate_limit_reset:
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ Rate limit tekshirishda xato: {e}")
            return False
    
    def update_key_usage(self, key_info: APIKeyInfo, reset_time: Optional[datetime] = None) -> None:
        """API kaliti ishlatilganini belgilash"""
        try:
            key_info.usage_count += 1
            key_info.daily_usage += 1
            key_info.last_used = datetime.now()
            
            if reset_time:
                key_info.rate_limit_reset = reset_time
            
            # Har 10 ishlatilgandan keyin saqlash
            if key_info.usage_count % 10 == 0:
                self.save_keys()
            
        except Exception as e:
            print(f"❌ Kaliti ishlatilganini belgilashda xato: {e}")
    
    def add_api_key(self, service: str, key: str, key_id: Optional[str] = None, 
                   daily_limit: int = 1000, encrypt: bool = True) -> bool:
        """Yangi API kalitini qo'shish"""
        try:
            if not key_id:
                key_id = f"{service}_{len([k for k in self.api_keys.keys() if k.startswith(service)]) + 1}"
            
            # Kalitni shifrlash
            encrypted_key = key
            if encrypt and self.cipher_suite:
                encrypted_key = self.cipher_suite.encrypt(key.encode()).decode()
            
            # APIKeyInfo yaratish
            self.api_keys[key_id] = APIKeyInfo(
                service=service,
                key=encrypted_key,
                encrypted=encrypt,
                active=True,
                daily_limit=daily_limit
            )
            
            # Saqlash
            self.save_keys()
            
            print(f"✅ API kaliti qo'shildi: {key_id}")
            return True
            
        except Exception as e:
            print(f"❌ API kalitini qo'shishda xato: {e}")
            return False
    
    def deactivate_key(self, key_id: str) -> bool:
        """API kalitini o'chirish"""
        try:
            if key_id in self.api_keys:
                self.api_keys[key_id].active = False
                self.save_keys()
                print(f"🔒 API kaliti o'chirildi: {key_id}")
                return True
            
            return False
            
        except Exception as e:
            print(f"❌ API kalitini o'chirishda xato: {e}")
            return False
    
    def get_service_config(self, service: str) -> Optional[APIServiceConfig]:
        """Servis konfiguratsiyasini olish"""
        return self.service_configs.get(service)
    
    def get_service_headers(self, service: str, api_key: str) -> Dict[str, str]:
        """Servis uchun headerlarni olish"""
        try:
            config = self.service_configs.get(service)
            if not config:
                return {}
            
            headers = config.headers.copy()
            for key, value in headers.items():
                if '{key}' in value:
                    headers[key] = value.replace('{key}', api_key)
            
            return headers
            
        except Exception as e:
            print(f"❌ Headerlarni olishda xato: {e}")
            return {}
    
    def save_keys(self) -> None:
        """API kalitlarni saqlash"""
        try:
            # Shifrlangan faylga saqlash
            if self.cipher_suite:
                encrypted_data = {}
                for key_id, key_info in self.api_keys.items():
                    if key_info.encrypted:
                        encrypted_data[key_id] = {
                            'service': key_info.service,
                            'key': key_info.key,  # Allaqachon shifrlangan
                            'active': key_info.active,
                            'usage_count': key_info.usage_count,
                            'daily_limit': key_info.daily_limit,
                            'daily_usage': key_info.daily_usage
                        }
                
                if encrypted_data:
                    with open(self.encrypted_keys_file, 'w', encoding='utf-8') as f:
                        json.dump(encrypted_data, f, indent=2)
            
            # Shifrlangan bo'lmagan kalitlarni oddiy faylga saqlash
            json_data = {}
            for key_id, key_info in self.api_keys.items():
                if not key_info.encrypted:
                    json_data[key_id] = {
                        'service': key_info.service,
                        'key': key_info.key,
                        'active': key_info.active,
                        'usage_count': key_info.usage_count,
                        'daily_limit': key_info.daily_limit,
                        'daily_usage': key_info.daily_usage
                    }
            
            if json_data:
                with open(self.keys_file, 'w', encoding='utf-8') as f:
                    json.dump(json_data, f, indent=2)
            
            print("💾 API kalitlar saqlandi")
            
        except Exception as e:
            print(f"❌ API kalitlarni saqlashda xato: {e}")
    
    def get_keys_status(self) -> Dict[str, Any]:
        """API kalitlar holatini olish"""
        try:
            status = {
                'total_keys': len(self.api_keys),
                'active_keys': sum(1 for k in self.api_keys.values() if k.active),
                'services': {},
                'daily_usage': {}
            }
            
            # Servis bo'yicha gruppalashtirish
            for key_id, key_info in self.api_keys.items():
                service = key_info.service
                if service not in status['services']:
                    status['services'][service] = {
                        'total': 0,
                        'active': 0,
                        'usage': 0
                    }
                
                status['services'][service]['total'] += 1
                if key_info.active:
                    status['services'][service]['active'] += 1
                status['services'][service]['usage'] += key_info.usage_count
                
                # Kunlik ishlatilish
                if service not in status['daily_usage']:
                    status['daily_usage'][service] = 0
                status['daily_usage'][service] += key_info.daily_usage
            
            return status
            
        except Exception as e:
            print(f"❌ Kalitlar holatini olishda xato: {e}")
            return {}
    
    def reset_daily_usage(self) -> None:
        """Kunlik ishlatilish statistikasini reset qilish"""
        try:
            for key_info in self.api_keys.values():
                key_info.daily_usage = 0
                key_info.rate_limit_reset = None
            
            self.save_keys()
            print("🔄 Kunlik ishlatilish statistikasi reset qilindi")
            
        except Exception as e:
            print(f"❌ Kunlik statistikani reset qilishda xato: {e}")


class APIKeyError(Exception):
    """API kaliti xatosi"""
    pass


# Global API key manager
api_key_manager = APIKeyManager()


def get_api_key_manager() -> APIKeyManager:
    """Global API key manager olish"""
    return api_key_manager


def get_api_key(service: str, key_id: Optional[str] = None) -> Optional[APIKeyInfo]:
    """API kalitini olish (shortcut)"""
    return api_key_manager.get_api_key(service, key_id)


def get_service_config(service: str) -> Optional[APIServiceConfig]:
    """Servis konfiguratsiyasini olish (shortcut)"""
    return api_key_manager.get_service_config(service)


if __name__ == "__main__":
    # Test uchun
    try:
        manager = get_api_key_manager()
        status = manager.get_keys_status()
        
        print("🔑 API Kalitlar Holati:")
        print(f"📊 Jami kalitlar: {status['total_keys']}")
        print(f"✅ Active kalitlar: {status['active_keys']}")
        print("\n📡 Servislar:")
        
        for service, info in status['services'].items():
            print(f"  {service}: {info['active']}/{info['total']} active, {info['usage']} ishlatilgan")
        
        print("\n📈 Kunlik ishlatilish:")
        for service, usage in status['daily_usage'].items():
            print(f"  {service}: {usage} so'rov")
        
    except Exception as e:
        print(f"❌ Test xatosi: {e}")

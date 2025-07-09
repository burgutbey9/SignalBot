"""
utils/fallback_manager.py - Fallback tizimi implementatsiyasi
API va servislar uchun fallback mexanizmi
"""

import asyncio
import time
from typing import List, Dict, Any, Optional, Callable, Union
from dataclasses import dataclass
from enum import Enum
from utils.logger import get_logger
from utils.retry_handler import RetryHandler, RetryConfig

logger = get_logger(__name__)

class ServiceStatus(Enum):
    """Servis holati"""
    ACTIVE = "active"          # Faol
    DEGRADED = "degraded"      # Sekinlashgan
    FAILED = "failed"          # Ishlamayapti
    UNKNOWN = "unknown"        # Noma'lum

@dataclass
class ServiceInfo:
    """Servis ma'lumotlari"""
    name: str
    priority: int = 0          # Pastroq raqam = yuqori muhimlik
    status: ServiceStatus = ServiceStatus.ACTIVE
    last_success: Optional[float] = None
    last_failure: Optional[float] = None
    failure_count: int = 0
    success_count: int = 0
    response_time: float = 0.0
    timeout: float = 30.0
    max_retries: int = 3
    is_enabled: bool = True
    health_check_url: Optional[str] = None
    
    def __post_init__(self):
        """Yaratilgandan keyin sozlash"""
        if self.last_success is None:
            self.last_success = time.time()

@dataclass
class FallbackResult:
    """Fallback natijasi"""
    success: bool
    data: Any = None
    error: Optional[str] = None
    service_used: Optional[str] = None
    fallback_level: int = 0
    response_time: float = 0.0
    retry_count: int = 0

class FallbackManager:
    """Fallback tizimi boshqaruvchisi"""
    
    def __init__(self, name: str = "FallbackManager"):
        self.name = name
        self.services: Dict[str, ServiceInfo] = {}
        self.fallback_chains: Dict[str, List[str]] = {}
        self.circuit_breaker_threshold = 5  # Xato soni chegarasi
        self.circuit_breaker_timeout = 300  # 5 daqiqa
        self.health_check_interval = 60     # 1 daqiqa
        self.stats: Dict[str, Dict] = {}
        logger.info(f"FallbackManager yaratildi: {name}")
    
    def register_service(self, service_info: ServiceInfo) -> None:
        """Servisni ro'yxatga olish"""
        try:
            self.services[service_info.name] = service_info
            self.stats[service_info.name] = {
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "average_response_time": 0.0,
                "last_health_check": None
            }
            logger.info(f"Servis ro'yxatga olindi: {service_info.name}")
        except Exception as e:
            logger.error(f"Servis ro'yxatga olishda xato: {e}")
    
    def register_fallback_chain(self, chain_name: str, service_names: List[str]) -> None:
        """Fallback zanjirini ro'yxatga olish"""
        try:
            # Servislarni prioritet bo'yicha saralash
            sorted_services = []
            for service_name in service_names:
                if service_name in self.services:
                    sorted_services.append(service_name)
                else:
                    logger.warning(f"Servis topilmadi: {service_name}")
            
            # Prioritet bo'yicha saralash
            sorted_services.sort(key=lambda x: self.services[x].priority)
            
            self.fallback_chains[chain_name] = sorted_services
            logger.info(f"Fallback zanjir ro'yxatga olindi: {chain_name} -> {sorted_services}")
        except Exception as e:
            logger.error(f"Fallback zanjir ro'yxatga olishda xato: {e}")
    
    def get_service_status(self, service_name: str) -> ServiceStatus:
        """Servis holatini olish"""
        try:
            if service_name not in self.services:
                return ServiceStatus.UNKNOWN
            
            service = self.services[service_name]
            
            # Agar servis o'chirilgan bo'lsa
            if not service.is_enabled:
                return ServiceStatus.FAILED
            
            # Circuit breaker tekshirish
            if service.failure_count >= self.circuit_breaker_threshold:
                if service.last_failure and (time.time() - service.last_failure) < self.circuit_breaker_timeout:
                    return ServiceStatus.FAILED
                else:
                    # Timeout tugagan, qayta urinish
                    service.failure_count = 0
                    service.status = ServiceStatus.ACTIVE
            
            return service.status
        except Exception as e:
            logger.error(f"Servis holatini olishda xato: {e}")
            return ServiceStatus.UNKNOWN
    
    def update_service_status(self, service_name: str, success: bool, 
                            response_time: float = 0.0, error: str = None) -> None:
        """Servis holatini yangilash"""
        try:
            if service_name not in self.services:
                logger.warning(f"Servis topilmadi: {service_name}")
                return
            
            service = self.services[service_name]
            stats = self.stats[service_name]
            
            # Statistikani yangilash
            stats["total_requests"] += 1
            
            if success:
                service.success_count += 1
                service.last_success = time.time()
                service.response_time = response_time
                service.failure_count = max(0, service.failure_count - 1)
                stats["successful_requests"] += 1
                
                # Holatni yaxshilash
                if service.status == ServiceStatus.FAILED:
                    service.status = ServiceStatus.DEGRADED
                elif service.status == ServiceStatus.DEGRADED and service.failure_count == 0:
                    service.status = ServiceStatus.ACTIVE
                    
            else:
                service.failure_count += 1
                service.last_failure = time.time()
                stats["failed_requests"] += 1
                
                # Holatni yomonlashtirish
                if service.failure_count >= self.circuit_breaker_threshold:
                    service.status = ServiceStatus.FAILED
                    logger.warning(f"Circuit breaker ochildi: {service_name}")
                elif service.failure_count >= 2:
                    service.status = ServiceStatus.DEGRADED
            
            # O'rtacha javob vaqtini hisoblash
            if stats["successful_requests"] > 0:
                old_avg = stats["average_response_time"]
                stats["average_response_time"] = (
                    (old_avg * (stats["successful_requests"] - 1) + response_time) / 
                    stats["successful_requests"]
                )
            
            logger.debug(f"Servis holati yangilandi: {service_name} -> {service.status.value}")
            
        except Exception as e:
            logger.error(f"Servis holatini yangilashda xato: {e}")
    
    async def execute_with_fallback(self, chain_name: str, operation: Callable, 
                                  *args, **kwargs) -> FallbackResult:
        """Fallback bilan operatsiyani bajarish"""
        try:
            if chain_name not in self.fallback_chains:
                logger.error(f"Fallback zanjir topilmadi: {chain_name}")
                return FallbackResult(
                    success=False,
                    error=f"Fallback zanjir topilmadi: {chain_name}"
                )
            
            services = self.fallback_chains[chain_name]
            last_error = None
            
            for level, service_name in enumerate(services):
                try:
                    # Servis holatini tekshirish
                    status = self.get_service_status(service_name)
                    if status == ServiceStatus.FAILED:
                        logger.warning(f"Servis ishlamayapti: {service_name}")
                        continue
                    
                    logger.info(f"Urinish {level + 1}: {service_name}")
                    
                    # Operatsiyani bajarish
                    start_time = time.time()
                    
                    if asyncio.iscoroutinefunction(operation):
                        result = await operation(service_name, *args, **kwargs)
                    else:
                        result = operation(service_name, *args, **kwargs)
                    
                    end_time = time.time()
                    response_time = end_time - start_time
                    
                    # Muvaffaqiyatli natija
                    self.update_service_status(service_name, True, response_time)
                    
                    logger.info(f"Muvaffaqiyat: {service_name} ({response_time:.2f}s)")
                    
                    return FallbackResult(
                        success=True,
                        data=result,
                        service_used=service_name,
                        fallback_level=level,
                        response_time=response_time
                    )
                    
                except Exception as e:
                    last_error = str(e)
                    response_time = time.time() - start_time if 'start_time' in locals() else 0
                    
                    self.update_service_status(service_name, False, response_time, last_error)
                    
                    logger.warning(f"Servis xato: {service_name} -> {e}")
                    
                    # Keyingi servisga o'tish
                    continue
            
            # Barcha servislar muvaffaqiyatsiz
            logger.error(f"Barcha servislar muvaffaqiyatsiz: {chain_name}")
            
            return FallbackResult(
                success=False,
                error=f"Barcha servislar muvaffaqiyatsiz: {last_error}",
                fallback_level=len(services)
            )
            
        except Exception as e:
            logger.error(f"Fallback bajarishda xato: {e}")
            return FallbackResult(
                success=False,
                error=f"Fallback bajarishda xato: {e}"
            )
    
    async def health_check(self, service_name: str) -> bool:
        """Servis salomatligini tekshirish"""
        try:
            if service_name not in self.services:
                return False
            
            service = self.services[service_name]
            
            # Oddiy ping test
            if service.health_check_url:
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        service.health_check_url,
                        timeout=aiohttp.ClientTimeout(total=10)
                    ) as response:
                        is_healthy = response.status == 200
            else:
                # URL bo'lmasa, holat asosida tekshirish
                is_healthy = service.status != ServiceStatus.FAILED
            
            self.stats[service_name]["last_health_check"] = time.time()
            
            if is_healthy:
                logger.debug(f"Servis sog'lom: {service_name}")
            else:
                logger.warning(f"Servis kasal: {service_name}")
            
            return is_healthy
            
        except Exception as e:
            logger.error(f"Health check xato: {service_name} -> {e}")
            return False
    
    async def health_check_all(self) -> Dict[str, bool]:
        """Barcha servislarni tekshirish"""
        results = {}
        
        for service_name in self.services:
            try:
                results[service_name] = await self.health_check(service_name)
            except Exception as e:
                logger.error(f"Health check xato: {service_name} -> {e}")
                results[service_name] = False
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Statistikani olish"""
        try:
            stats = {
                "services": {},
                "fallback_chains": self.fallback_chains,
                "circuit_breaker_threshold": self.circuit_breaker_threshold,
                "circuit_breaker_timeout": self.circuit_breaker_timeout
            }
            
            for service_name, service in self.services.items():
                service_stats = self.stats[service_name].copy()
                service_stats.update({
                    "status": service.status.value,
                    "priority": service.priority,
                    "failure_count": service.failure_count,
                    "success_count": service.success_count,
                    "is_enabled": service.is_enabled,
                    "last_success": service.last_success,
                    "last_failure": service.last_failure,
                    "response_time": service.response_time
                })
                stats["services"][service_name] = service_stats
            
            return stats
            
        except Exception as e:
            logger.error(f"Statistika olishda xato: {e}")
            return {}
    
    def reset_service(self, service_name: str) -> bool:
        """Servisni qayta tiklash"""
        try:
            if service_name not in self.services:
                return False
            
            service = self.services[service_name]
            service.failure_count = 0
            service.status = ServiceStatus.ACTIVE
            service.is_enabled = True
            
            logger.info(f"Servis qayta tiklandi: {service_name}")
            return True
            
        except Exception as e:
            logger.error(f"Servis qayta tiklashda xato: {e}")
            return False
    
    def disable_service(self, service_name: str) -> bool:
        """Servisni o'chirish"""
        try:
            if service_name not in self.services:
                return False
            
            self.services[service_name].is_enabled = False
            self.services[service_name].status = ServiceStatus.FAILED
            
            logger.info(f"Servis o'chirildi: {service_name}")
            return True
            
        except Exception as e:
            logger.error(f"Servis o'chirishda xato: {e}")
            return False
    
    def enable_service(self, service_name: str) -> bool:
        """Servisni yoqish"""
        try:
            if service_name not in self.services:
                return False
            
            self.services[service_name].is_enabled = True
            self.services[service_name].status = ServiceStatus.ACTIVE
            self.services[service_name].failure_count = 0
            
            logger.info(f"Servis yoqildi: {service_name}")
            return True
            
        except Exception as e:
            logger.error(f"Servis yoqishda xato: {e}")
            return False

# Loyiha uchun maxsus fallback managerlar
class OrderFlowFallbackManager(FallbackManager):
    """Order Flow uchun fallback manager"""
    
    def __init__(self):
        super().__init__("OrderFlowFallback")
        
        # Servislarni ro'yxatga olish
        self.register_service(ServiceInfo(
            name="oneinch",
            priority=1,
            timeout=30.0,
            max_retries=3
        ))
        
        self.register_service(ServiceInfo(
            name="thegraph",
            priority=2,
            timeout=45.0,
            max_retries=5
        ))
        
        self.register_service(ServiceInfo(
            name="alchemy",
            priority=3,
            timeout=60.0,
            max_retries=3
        ))
        
        # Fallback zanjirini ro'yxatga olish
        self.register_fallback_chain("order_flow", ["oneinch", "thegraph", "alchemy"])

class SentimentFallbackManager(FallbackManager):
    """Sentiment tahlil uchun fallback manager"""
    
    def __init__(self):
        super().__init__("SentimentFallback")
        
        # Servislarni ro'yxatga olish
        self.register_service(ServiceInfo(
            name="huggingface",
            priority=1,
            timeout=60.0,
            max_retries=2
        ))
        
        # Gemini API kalitlari
        for i in range(1, 6):
            self.register_service(ServiceInfo(
                name=f"gemini_{i}",
                priority=i + 1,
                timeout=45.0,
                max_retries=3
            ))
        
        self.register_service(ServiceInfo(
            name="claude",
            priority=10,
            timeout=60.0,
            max_retries=2
        ))
        
        self.register_service(ServiceInfo(
            name="local_nlp",
            priority=11,
            timeout=10.0,
            max_retries=1
        ))
        
        # Fallback zanjirini ro'yxatga olish
        services = ["huggingface"] + [f"gemini_{i}" for i in range(1, 6)] + ["claude", "local_nlp"]
        self.register_fallback_chain("sentiment", services)

class NewsFallbackManager(FallbackManager):
    """Yangiliklar uchun fallback manager"""
    
    def __init__(self):
        super().__init__("NewsFallback")
        
        # Servislarni ro'yxatga olish
        self.register_service(ServiceInfo(
            name="newsapi",
            priority=1,
            timeout=30.0,
            max_retries=3
        ))
        
        self.register_service(ServiceInfo(
            name="reddit",
            priority=2,
            timeout=45.0,
            max_retries=3
        ))
        
        self.register_service(ServiceInfo(
            name="claude",
            priority=3,
            timeout=60.0,
            max_retries=2
        ))
        
        # Fallback zanjirini ro'yxatga olish
        self.register_fallback_chain("news", ["newsapi", "reddit", "claude"])

# Foydalanish misoli
async def example_usage():
    """Fallback manager foydalanish misoli"""
    
    # Order Flow fallback manager
    order_flow_manager = OrderFlowFallbackManager()
    
    async def fetch_order_flow(service_name: str, token_address: str):
        """Order flow ma'lumotlarini olish"""
        logger.info(f"Order flow olish: {service_name} -> {token_address}")
        
        # Xato simulatsiyasi
        import random
        if random.random() < 0.3:  # 30% xato
            raise Exception(f"{service_name} xato simulatsiyasi")
        
        return {
            "service": service_name,
            "token": token_address,
            "price": 1.25,
            "volume": 1000000
        }
    
    try:
        # Fallback bilan bajarish
        result = await order_flow_manager.execute_with_fallback(
            "order_flow", 
            fetch_order_flow, 
            "0x1234567890abcdef"
        )
        
        if result.success:
            logger.info(f"Natija: {result.data}")
            logger.info(f"Ishlatilgan servis: {result.service_used}")
            logger.info(f"Fallback darajasi: {result.fallback_level}")
        else:
            logger.error(f"Xato: {result.error}")
            
    except Exception as e:
        logger.error(f"Umumiy xato: {e}")

# Test funksiya
async def test_fallback_manager():
    """Fallback manager testlari"""
    logger.info("=== Fallback Manager Testlari ===")
    
    # Asosiy test
    await example_usage()
    
    # Health check test
    manager = OrderFlowFallbackManager()
    health_results = await manager.health_check_all()
    logger.info(f"Health check natijalari: {health_results}")
    
    # Statistika test
    stats = manager.get_statistics()
    logger.info(f"Statistika: {stats}")
    
    logger.info("=== Testlar tugadi ===")

if __name__ == "__main__":
    # Testlarni ishga tushirish
    import asyncio
    asyncio.run(test_fallback_manager())

"""
Fallback tizimi va prioritet boshqaruvi
Bu modul barcha API clientlar uchun fallback ketma-ketlik va prioritetlarni boshqaradi
"""

import asyncio
import json
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import logging
from collections import defaultdict, deque

from utils.logger import get_logger

logger = get_logger(__name__)


class ServiceStatus(Enum):
    """Service holati enum"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class FallbackReason(Enum):
    """Fallback sabablari enum"""
    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"
    API_ERROR = "api_error"
    AUTHENTICATION_ERROR = "auth_error"
    NETWORK_ERROR = "network_error"
    SERVICE_UNAVAILABLE = "service_unavailable"
    QUOTA_EXCEEDED = "quota_exceeded"
    MANUAL_OVERRIDE = "manual_override"


@dataclass
class ServiceHealth:
    """Service sog'ligi haqida ma'lumot"""
    name: str
    status: ServiceStatus = ServiceStatus.UNKNOWN
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    consecutive_failures: int = 0
    success_rate: float = 0.0
    average_response_time: float = 0.0
    total_requests: int = 0
    failed_requests: int = 0
    last_error: Optional[str] = None
    downtime_start: Optional[datetime] = None
    uptime_percentage: float = 100.0


@dataclass
class FallbackRule:
    """Fallback qoidasi"""
    service_name: str
    priority: int
    max_retries: int
    retry_delay: float
    timeout: float
    health_check_interval: int
    conditions: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True


@dataclass
class FallbackAttempt:
    """Fallback urinishi ma'lumoti"""
    service_name: str
    reason: FallbackReason
    timestamp: datetime
    response_time: Optional[float] = None
    error_message: Optional[str] = None
    success: bool = False


class FallbackManager:
    """Fallback tizimi menejeri"""
    
    def __init__(self, config_path: str = "config/settings.json"):
        self.config_path = Path(config_path)
        self.services: Dict[str, ServiceHealth] = {}
        self.fallback_rules: Dict[str, List[FallbackRule]] = {}
        self.attempt_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.circuit_breakers: Dict[str, bool] = {}
        self.load_config()
        
        # Statistika uchun
        self.stats = {
            'total_fallbacks': 0,
            'successful_fallbacks': 0,
            'failed_fallbacks': 0,
            'most_used_fallback': None,
            'average_fallback_time': 0.0
        }
        
        logger.info("FallbackManager ishga tushirildi")
    
    def load_config(self) -> None:
        """Konfiguratsiya faylini yuklash"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Fallback qoidalarini yuklash
            fallback_orders = config.get('fallback_order', {})
            api_limits = config.get('api_limits', {})
            
            for category, order_config in fallback_orders.items():
                primary = order_config.get('primary')
                fallbacks = order_config.get('fallbacks', [])
                
                # Asosiy service
                if primary:
                    self._add_fallback_rule(category, primary, 1, api_limits)
                
                # Fallback services
                for i, service in enumerate(fallbacks, 2):
                    self._add_fallback_rule(category, service, i, api_limits)
            
            # Service health holatini boshlash
            self._initialize_service_health()
            
            logger.info(f"Fallback konfiguratsiya yuklandi: {len(self.fallback_rules)} kategoriya")
            
        except Exception as e:
            logger.error(f"Fallback konfiguratsiya yuklashda xato: {e}")
            raise
    
    def _add_fallback_rule(self, category: str, service: str, priority: int, api_limits: Dict) -> None:
        """Fallback qoidasi qo'shish"""
        if category not in self.fallback_rules:
            self.fallback_rules[category] = []
        
        service_config = api_limits.get(service, {})
        
        rule = FallbackRule(
            service_name=service,
            priority=priority,
            max_retries=service_config.get('max_retries', 3),
            retry_delay=service_config.get('retry_delay', 1),
            timeout=service_config.get('timeout', 30),
            health_check_interval=service_config.get('health_check_interval', 300),
            conditions=service_config.get('conditions', {}),
            enabled=service_config.get('enabled', True)
        )
        
        self.fallback_rules[category].append(rule)
        
        # Prioritet bo'yicha saralash
        self.fallback_rules[category].sort(key=lambda x: x.priority)
    
    def _initialize_service_health(self) -> None:
        """Service health holatini boshlash"""
        for category, rules in self.fallback_rules.items():
            for rule in rules:
                if rule.service_name not in self.services:
                    self.services[rule.service_name] = ServiceHealth(
                        name=rule.service_name,
                        status=ServiceStatus.UNKNOWN
                    )
                    self.circuit_breakers[rule.service_name] = False
    
    async def execute_with_fallback(
        self,
        category: str,
        operation: Callable,
        *args,
        **kwargs
    ) -> Tuple[bool, Any, str]:
        """
        Fallback bilan operatsiya bajarish
        
        Args:
            category: Fallback kategoriyasi (masalan: 'order_flow', 'sentiment')
            operation: Bajarilishi kerak bo'lgan operatsiya
            *args: Operatsiya argumentlari
            **kwargs: Operatsiya kalit argumentlari
            
        Returns:
            Tuple[bool, Any, str]: (success, result, used_service)
        """
        if category not in self.fallback_rules:
            logger.error(f"Fallback kategoriya topilmadi: {category}")
            return False, None, ""
        
        rules = self.fallback_rules[category]
        start_time = datetime.now()
        
        for rule in rules:
            if not rule.enabled:
                continue
                
            # Circuit breaker tekshirish
            if self.circuit_breakers.get(rule.service_name, False):
                logger.warning(f"Circuit breaker faol: {rule.service_name}")
                continue
            
            # Service health tekshirish
            service_health = self.services.get(rule.service_name)
            if service_health and service_health.status == ServiceStatus.UNHEALTHY:
                logger.warning(f"Service noto'g'ri: {rule.service_name}")
                continue
            
            # Operatsiyani bajarish
            try:
                result = await self._execute_with_retry(
                    rule, operation, *args, **kwargs
                )
                
                if result is not None:
                    # Muvaffaqiyat
                    self._record_success(rule.service_name)
                    self.stats['successful_fallbacks'] += 1
                    
                    if rule.priority > 1:
                        fallback_time = (datetime.now() - start_time).total_seconds()
                        logger.info(f"Fallback muvaffaqiyatli: {rule.service_name} ({fallback_time:.2f}s)")
                    
                    return True, result, rule.service_name
                    
            except Exception as e:
                # Xato
                self._record_failure(rule.service_name, str(e))
                logger.error(f"Service xatosi {rule.service_name}: {e}")
                
                # Attempt history
                attempt = FallbackAttempt(
                    service_name=rule.service_name,
                    reason=self._determine_failure_reason(e),
                    timestamp=datetime.now(),
                    error_message=str(e),
                    success=False
                )
                self.attempt_history[category].append(attempt)
                
                continue
        
        # Barcha fallback muvaffaqiyatsiz
        total_time = (datetime.now() - start_time).total_seconds()
        logger.error(f"Barcha fallback muvaffaqiyatsiz: {category} ({total_time:.2f}s)")
        
        self.stats['failed_fallbacks'] += 1
        return False, None, ""
    
    async def _execute_with_retry(
        self,
        rule: FallbackRule,
        operation: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Retry bilan operatsiya bajarish"""
        last_exception = None
        
        for attempt in range(rule.max_retries + 1):
            try:
                # Timeout bilan bajarish
                result = await asyncio.wait_for(
                    operation(*args, **kwargs),
                    timeout=rule.timeout
                )
                return result
                
            except asyncio.TimeoutError:
                last_exception = Exception(f"Timeout: {rule.timeout}s")
                logger.warning(f"Timeout: {rule.service_name} (attempt {attempt + 1})")
                
            except Exception as e:
                last_exception = e
                logger.warning(f"Xato: {rule.service_name} (attempt {attempt + 1}): {e}")
            
            # Keyingi urinish uchun kutish
            if attempt < rule.max_retries:
                await asyncio.sleep(rule.retry_delay * (2 ** attempt))  # Exponential backoff
        
        # Barcha urinishlar muvaffaqiyatsiz
        if last_exception:
            raise last_exception
        
        return None
    
    def _determine_failure_reason(self, exception: Exception) -> FallbackReason:
        """Xato sababini aniqlash"""
        error_str = str(exception).lower()
        
        if "timeout" in error_str:
            return FallbackReason.TIMEOUT
        elif "rate limit" in error_str or "too many requests" in error_str:
            return FallbackReason.RATE_LIMITED
        elif "authentication" in error_str or "unauthorized" in error_str:
            return FallbackReason.AUTHENTICATION_ERROR
        elif "network" in error_str or "connection" in error_str:
            return FallbackReason.NETWORK_ERROR
        elif "quota" in error_str or "exceeded" in error_str:
            return FallbackReason.QUOTA_EXCEEDED
        elif "service unavailable" in error_str or "503" in error_str:
            return FallbackReason.SERVICE_UNAVAILABLE
        else:
            return FallbackReason.API_ERROR
    
    def _record_success(self, service_name: str) -> None:
        """Muvaffaqiyatli operatsiyani qayd etish"""
        service = self.services.get(service_name)
        if service:
            service.last_success = datetime.now()
            service.consecutive_failures = 0
            service.total_requests += 1
            service.status = ServiceStatus.HEALTHY
            
            # Circuit breaker ochish
            self.circuit_breakers[service_name] = False
            
            # Success rate yangilash
            service.success_rate = ((service.total_requests - service.failed_requests) / 
                                  service.total_requests * 100)
            
            # Uptime yangilash
            if service.downtime_start:
                service.downtime_start = None
            service.uptime_percentage = min(100.0, service.uptime_percentage + 0.1)
    
    def _record_failure(self, service_name: str, error: str) -> None:
        """Muvaffaqiyatsiz operatsiyani qayd etish"""
        service = self.services.get(service_name)
        if service:
            service.last_failure = datetime.now()
            service.consecutive_failures += 1
            service.total_requests += 1
            service.failed_requests += 1
            service.last_error = error
            
            # Status yangilash
            if service.consecutive_failures >= 3:
                service.status = ServiceStatus.UNHEALTHY
                service.downtime_start = datetime.now()
                
                # Circuit breaker faollashtirish
                self.circuit_breakers[service_name] = True
                
                logger.error(f"Service noto'g'ri: {service_name} ({service.consecutive_failures} ketma-ket xato)")
            
            elif service.consecutive_failures >= 2:
                service.status = ServiceStatus.DEGRADED
            
            # Success rate yangilash
            service.success_rate = ((service.total_requests - service.failed_requests) / 
                                  service.total_requests * 100)
            
            # Uptime yangilash
            service.uptime_percentage = max(0.0, service.uptime_percentage - 1.0)
    
    def get_service_health(self, service_name: str) -> Optional[ServiceHealth]:
        """Service health ma'lumotini olish"""
        return self.services.get(service_name)
    
    def get_all_services_health(self) -> Dict[str, ServiceHealth]:
        """Barcha service health ma'lumotlarini olish"""
        return self.services.copy()
    
    def get_fallback_stats(self) -> Dict[str, Any]:
        """Fallback statistikalarini olish"""
        return {
            **self.stats,
            'service_count': len(self.services),
            'healthy_services': sum(1 for s in self.services.values() 
                                  if s.status == ServiceStatus.HEALTHY),
            'unhealthy_services': sum(1 for s in self.services.values() 
                                    if s.status == ServiceStatus.UNHEALTHY),
            'circuit_breakers_active': sum(1 for cb in self.circuit_breakers.values() if cb)
        }
    
    def get_category_priority(self, category: str) -> List[str]:
        """Kategoriya uchun prioritet ketma-ketligini olish"""
        if category not in self.fallback_rules:
            return []
        
        return [rule.service_name for rule in self.fallback_rules[category] 
                if rule.enabled]
    
    def override_service_status(self, service_name: str, status: ServiceStatus) -> None:
        """Service holatini qo'lda o'zgartirish"""
        if service_name in self.services:
            self.services[service_name].status = status
            
            # Circuit breaker boshqarish
            if status == ServiceStatus.HEALTHY:
                self.circuit_breakers[service_name] = False
                self.services[service_name].consecutive_failures = 0
            elif status == ServiceStatus.UNHEALTHY:
                self.circuit_breakers[service_name] = True
            
            logger.info(f"Service holati o'zgartirildi: {service_name} -> {status.value}")
    
    def disable_service(self, service_name: str) -> None:
        """Service ni o'chirish"""
        for category, rules in self.fallback_rules.items():
            for rule in rules:
                if rule.service_name == service_name:
                    rule.enabled = False
        
        logger.info(f"Service o'chirildi: {service_name}")
    
    def enable_service(self, service_name: str) -> None:
        """Service ni yoqish"""
        for category, rules in self.fallback_rules.items():
            for rule in rules:
                if rule.service_name == service_name:
                    rule.enabled = True
        
        logger.info(f"Service yoqildi: {service_name}")
    
    async def health_check(self, service_name: str) -> bool:
        """Service health check"""
        try:
            # Bu yerda service-specific health check logikasi bo'lishi kerak
            # Hozircha basic check
            service = self.services.get(service_name)
            if not service:
                return False
            
            # Oxirgi muvaffaqiyatli operatsiyadan 5 daqiqa o'tgan bo'lsa
            if service.last_success:
                time_since_success = datetime.now() - service.last_success
                if time_since_success > timedelta(minutes=5):
                    return False
            
            return service.status in [ServiceStatus.HEALTHY, ServiceStatus.DEGRADED]
            
        except Exception as e:
            logger.error(f"Health check xatosi {service_name}: {e}")
            return False
    
    async def run_health_checks(self) -> None:
        """Barcha service lar uchun health check"""
        tasks = []
        for service_name in self.services:
            task = asyncio.create_task(self.health_check(service_name))
            tasks.append((service_name, task))
        
        for service_name, task in tasks:
            try:
                is_healthy = await task
                if not is_healthy:
                    self._record_failure(service_name, "Health check failed")
            except Exception as e:
                self._record_failure(service_name, f"Health check error: {e}")
    
    def get_attempt_history(self, category: str, limit: int = 10) -> List[FallbackAttempt]:
        """Fallback urinishlar tarixini olish"""
        if category not in self.attempt_history:
            return []
        
        return list(self.attempt_history[category])[-limit:]
    
    def reset_circuit_breaker(self, service_name: str) -> None:
        """Circuit breaker ni qayta boshlash"""
        if service_name in self.circuit_breakers:
            self.circuit_breakers[service_name] = False
            if service_name in self.services:
                self.services[service_name].consecutive_failures = 0
                self.services[service_name].status = ServiceStatus.UNKNOWN
            
            logger.info(f"Circuit breaker qayta boshlandi: {service_name}")
    
    def get_uzbek_status_report(self) -> str:
        """O'zbekcha holat hisoboti"""
        healthy_count = sum(1 for s in self.services.values() 
                          if s.status == ServiceStatus.HEALTHY)
        total_count = len(self.services)
        
        report = f"""
📊 FALLBACK TIZIMI HISOBOTI
═══════════════════════════

🟢 Sog'lom servicelar: {healthy_count}/{total_count}
🔴 Ishlamayotgan servicelar: {sum(1 for s in self.services.values() if s.status == ServiceStatus.UNHEALTHY)}
🟡 Sekinlashgan servicelar: {sum(1 for s in self.services.values() if s.status == ServiceStatus.DEGRADED)}

📈 STATISTIKA:
• Jami fallback: {self.stats['total_fallbacks']}
• Muvaffaqiyatli: {self.stats['successful_fallbacks']}
• Muvaffaqiyatsiz: {self.stats['failed_fallbacks']}
• Circuit breaker faol: {sum(1 for cb in self.circuit_breakers.values() if cb)}

🔧 SERVICE HOLATI:
"""
        
        for service_name, service in self.services.items():
            status_icon = {
                ServiceStatus.HEALTHY: "🟢",
                ServiceStatus.DEGRADED: "🟡",
                ServiceStatus.UNHEALTHY: "🔴",
                ServiceStatus.UNKNOWN: "⚪"
            }.get(service.status, "⚪")
            
            report += f"{status_icon} {service_name}: {service.success_rate:.1f}% muvaffaqiyat\n"
        
        return report
    
    def __str__(self) -> str:
        return f"FallbackManager({len(self.services)} services, {len(self.fallback_rules)} categories)"
    
    def __repr__(self) -> str:
        return self.__str__()


# Global fallback manager instance
fallback_manager = FallbackManager()


async def get_fallback_manager() -> FallbackManager:
    """Global fallback manager olish"""
    return fallback_manager


# Utility functions
def create_fallback_decorator(category: str):
    """Fallback decorator yaratish"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            async def operation():
                return await func(*args, **kwargs)
            
            success, result, service = await fallback_manager.execute_with_fallback(
                category, operation
            )
            
            if not success:
                raise Exception(f"Barcha fallback muvaffaqiyatsiz: {category}")
            
            return result
        return wrapper
    return decorator


# Decorators
order_flow_fallback = create_fallback_decorator("order_flow")
sentiment_fallback = create_fallback_decorator("sentiment")
news_fallback = create_fallback_decorator("news")
market_data_fallback = create_fallback_decorator("market_data")

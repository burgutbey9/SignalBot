import asyncio
import time
from typing import Dict, Optional, List
from dataclasses import dataclass
from collections import deque
from utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class RateLimit:
    """Rate limit ma'lumotlari"""
    calls: int                    # Ruxsat etilgan so'rovlar soni
    period: int                   # Vaqt oralig'i (sekund)
    remaining: int = 0            # Qolgan so'rovlar soni
    reset_time: float = 0         # Qayta tiklanish vaqti
    burst_limit: int = 0          # Burst limit (ixtiyoriy)

@dataclass
class RateLimitStatus:
    """Rate limit holati"""
    is_allowed: bool
    remaining_calls: int
    reset_time: float
    wait_time: float = 0.0

class RateLimiter:
    """
    Rate limiting tizimi - API so'rovlarini cheklash uchun
    Token bucket algoritmi asosida ishlaydi
    """
    
    def __init__(self, calls: int, period: int, burst_limit: Optional[int] = None):
        """
        Rate limiter yaratish
        
        Args:
            calls: Ruxsat etilgan so'rovlar soni
            period: Vaqt oralig'i (sekund)
            burst_limit: Burst limit (agar None bo'lsa, calls ga teng)
        """
        self.calls = calls
        self.period = period
        self.burst_limit = burst_limit or calls
        self.requests: deque = deque()  # So'rovlar tarixini saqlash
        self.lock = asyncio.Lock()      # Thread safety uchun
        self.name = f"RateLimiter({calls}/{period}s)"
        
        logger.info(f"{self.name} yaratildi - burst_limit: {self.burst_limit}")
    
    async def wait(self) -> None:
        """
        Rate limit tekshirish va kerak bo'lsa kutish
        """
        async with self.lock:
            status = await self._check_rate_limit()
            
            if not status.is_allowed:
                if status.wait_time > 0:
                    logger.warning(f"{self.name} - {status.wait_time:.2f} sekund kutish kerak")
                    await asyncio.sleep(status.wait_time)
                    
                    # Qayta tekshirish
                    status = await self._check_rate_limit()
                    if not status.is_allowed:
                        logger.error(f"{self.name} - Kutishdan keyin ham ruxsat berilmadi")
                        raise Exception("Rate limit exceeded after waiting")
            
            # So'rovni ro'yxatga qo'shish
            self.requests.append(time.time())
            logger.debug(f"{self.name} - So'rov qo'shildi, qolgan: {status.remaining_calls}")
    
    async def _check_rate_limit(self) -> RateLimitStatus:
        """
        Rate limit holatini tekshirish
        """
        current_time = time.time()
        
        # Eski so'rovlarni o'chirish
        while self.requests and current_time - self.requests[0] >= self.period:
            self.requests.popleft()
        
        current_count = len(self.requests)
        remaining_calls = max(0, self.calls - current_count)
        
        # Agar limit oshib ketgan bo'lsa
        if current_count >= self.calls:
            if self.requests:
                # Eng eski so'rovdan keyingi kutish vaqti
                oldest_request = self.requests[0]
                wait_time = self.period - (current_time - oldest_request)
                reset_time = current_time + wait_time
                
                return RateLimitStatus(
                    is_allowed=False,
                    remaining_calls=0,
                    reset_time=reset_time,
                    wait_time=max(0, wait_time)
                )
            else:
                # Bu holat bo'lmasligi kerak, lekin xavfsizlik uchun
                return RateLimitStatus(
                    is_allowed=True,
                    remaining_calls=self.calls,
                    reset_time=current_time + self.period
                )
        
        # Ruxsat berilgan holat
        next_reset = current_time + self.period
        if self.requests:
            next_reset = self.requests[0] + self.period
        
        return RateLimitStatus(
            is_allowed=True,
            remaining_calls=remaining_calls,
            reset_time=next_reset
        )
    
    async def can_make_request(self) -> bool:
        """
        So'rov qilish mumkinligini tekshirish (so'rov qo'shmasdan)
        """
        async with self.lock:
            status = await self._check_rate_limit()
            return status.is_allowed
    
    async def get_status(self) -> RateLimitStatus:
        """
        Joriy rate limit holatini olish
        """
        async with self.lock:
            return await self._check_rate_limit()
    
    async def reset(self) -> None:
        """
        Rate limiter ni qayta boshlash
        """
        async with self.lock:
            self.requests.clear()
            logger.info(f"{self.name} - Qayta boshlandi")
    
    def __repr__(self) -> str:
        return f"RateLimiter(calls={self.calls}, period={self.period}s)"

class MultiRateLimiter:
    """
    Bir nechta rate limiter bilan ishlash
    Masalan: per-second, per-minute, per-hour limitlar
    """
    
    def __init__(self, limits: List[RateLimit]):
        """
        Multi rate limiter yaratish
        
        Args:
            limits: RateLimit obyektlari ro'yxati
        """
        self.limiters = [
            RateLimiter(limit.calls, limit.period, limit.burst_limit)
            for limit in limits
        ]
        self.name = f"MultiRateLimiter({len(self.limiters)} limiters)"
        
        logger.info(f"{self.name} yaratildi")
    
    async def wait(self) -> None:
        """
        Barcha rate limiterlar uchun kutish
        """
        wait_times = []
        
        # Barcha limiterlar uchun kutish vaqtini hisoblash
        for limiter in self.limiters:
            status = await limiter.get_status()
            if not status.is_allowed:
                wait_times.append(status.wait_time)
        
        # Eng uzun kutish vaqti
        if wait_times:
            max_wait = max(wait_times)
            if max_wait > 0:
                logger.warning(f"{self.name} - {max_wait:.2f} sekund kutish kerak")
                await asyncio.sleep(max_wait)
        
        # Barcha limiterlar uchun so'rov qo'shish
        for limiter in self.limiters:
            await limiter.wait()
    
    async def can_make_request(self) -> bool:
        """
        Barcha limiterlar uchun so'rov mumkinligini tekshirish
        """
        for limiter in self.limiters:
            if not await limiter.can_make_request():
                return False
        return True
    
    async def get_most_restrictive_status(self) -> RateLimitStatus:
        """
        Eng cheklangan limiter holatini olish
        """
        statuses = []
        for limiter in self.limiters:
            status = await limiter.get_status()
            statuses.append(status)
        
        # Eng cheklangan holatni topish
        if not statuses:
            return RateLimitStatus(is_allowed=True, remaining_calls=0, reset_time=0)
        
        # Ruxsat berilmagan holatlar orasidan eng tez tiklanadigan
        denied_statuses = [s for s in statuses if not s.is_allowed]
        if denied_statuses:
            return min(denied_statuses, key=lambda s: s.reset_time)
        
        # Ruxsat berilgan holatlar orasidan eng kam qolgani
        allowed_statuses = [s for s in statuses if s.is_allowed]
        return min(allowed_statuses, key=lambda s: s.remaining_calls)
    
    async def reset(self) -> None:
        """
        Barcha rate limiterlerni qayta boshlash
        """
        for limiter in self.limiters:
            await limiter.reset()
        logger.info(f"{self.name} - Barcha limiterlar qayta boshlandi")

class APIRateLimiter:
    """
    API-ga maxsus rate limiter
    Har xil endpoint uchun turli limitlar
    """
    
    def __init__(self, api_name: str, default_limits: Optional[List[RateLimit]] = None):
        """
        API rate limiter yaratish
        
        Args:
            api_name: API nomi
            default_limits: Standart limitlar
        """
        self.api_name = api_name
        self.endpoint_limiters: Dict[str, MultiRateLimiter] = {}
        self.default_limits = default_limits or [
            RateLimit(calls=100, period=60),  # 100 so'rov/minut
            RateLimit(calls=1000, period=3600)  # 1000 so'rov/soat
        ]
        
        logger.info(f"APIRateLimiter yaratildi - {api_name}")
    
    def add_endpoint_limits(self, endpoint: str, limits: List[RateLimit]) -> None:
        """
        Endpoint uchun maxsus limitlar qo'shish
        
        Args:
            endpoint: Endpoint nomi
            limits: Limitlar ro'yxati
        """
        self.endpoint_limiters[endpoint] = MultiRateLimiter(limits)
        logger.info(f"Endpoint limitlar qo'shildi - {endpoint}: {len(limits)} limit")
    
    async def wait_for_endpoint(self, endpoint: str) -> None:
        """
        Endpoint uchun kutish
        
        Args:
            endpoint: Endpoint nomi
        """
        # Endpoint limiter mavjudligini tekshirish
        if endpoint not in self.endpoint_limiters:
            self.endpoint_limiters[endpoint] = MultiRateLimiter(self.default_limits)
            logger.debug(f"Standart limitlar qo'llandi - {endpoint}")
        
        # Kutish
        await self.endpoint_limiters[endpoint].wait()
    
    async def can_make_request_to_endpoint(self, endpoint: str) -> bool:
        """
        Endpoint ga so'rov qilish mumkinligini tekshirish
        
        Args:
            endpoint: Endpoint nomi
            
        Returns:
            bool: So'rov mumkinligi
        """
        if endpoint not in self.endpoint_limiters:
            self.endpoint_limiters[endpoint] = MultiRateLimiter(self.default_limits)
        
        return await self.endpoint_limiters[endpoint].can_make_request()
    
    async def get_endpoint_status(self, endpoint: str) -> RateLimitStatus:
        """
        Endpoint holatini olish
        
        Args:
            endpoint: Endpoint nomi
            
        Returns:
            RateLimitStatus: Holat ma'lumotlari
        """
        if endpoint not in self.endpoint_limiters:
            self.endpoint_limiters[endpoint] = MultiRateLimiter(self.default_limits)
        
        return await self.endpoint_limiters[endpoint].get_most_restrictive_status()
    
    async def reset_endpoint(self, endpoint: str) -> None:
        """
        Endpoint limiterlarni qayta boshlash
        
        Args:
            endpoint: Endpoint nomi
        """
        if endpoint in self.endpoint_limiters:
            await self.endpoint_limiters[endpoint].reset()
    
    async def reset_all(self) -> None:
        """
        Barcha endpoint limiterlarni qayta boshlash
        """
        for endpoint, limiter in self.endpoint_limiters.items():
            await limiter.reset()
        logger.info(f"{self.api_name} - Barcha endpoint limiterlari qayta boshlandi")

class GlobalRateLimiter:
    """
    Global rate limiter - barcha API lar uchun
    """
    
    _instance = None
    _api_limiters: Dict[str, APIRateLimiter] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            logger.info("GlobalRateLimiter yaratildi")
        return cls._instance
    
    def get_api_limiter(self, api_name: str, default_limits: Optional[List[RateLimit]] = None) -> APIRateLimiter:
        """
        API limiter olish yoki yaratish
        
        Args:
            api_name: API nomi
            default_limits: Standart limitlar
            
        Returns:
            APIRateLimiter: API limiter
        """
        if api_name not in self._api_limiters:
            self._api_limiters[api_name] = APIRateLimiter(api_name, default_limits)
        
        return self._api_limiters[api_name]
    
    async def wait_for_api_endpoint(self, api_name: str, endpoint: str) -> None:
        """
        API endpoint uchun kutish
        
        Args:
            api_name: API nomi
            endpoint: Endpoint nomi
        """
        limiter = self.get_api_limiter(api_name)
        await limiter.wait_for_endpoint(endpoint)
    
    async def reset_api(self, api_name: str) -> None:
        """
        API limiterni qayta boshlash
        
        Args:
            api_name: API nomi
        """
        if api_name in self._api_limiters:
            await self._api_limiters[api_name].reset_all()
    
    async def reset_all_apis(self) -> None:
        """
        Barcha API limiterlarni qayta boshlash
        """
        for api_name, limiter in self._api_limiters.items():
            await limiter.reset_all()
        logger.info("Barcha API limiterlari qayta boshlandi")

# Standart rate limiter instancelari
def create_standard_limiters() -> Dict[str, APIRateLimiter]:
    """
    Standart API limiterlari yaratish
    
    Returns:
        Dict: API limiterlari
    """
    global_limiter = GlobalRateLimiter()
    
    # 1inch API limitlari
    oneinch_limits = [
        RateLimit(calls=10, period=1),    # 10 so'rov/sekund
        RateLimit(calls=100, period=60),  # 100 so'rov/minut
        RateLimit(calls=1000, period=3600) # 1000 so'rov/soat
    ]
    
    # Alchemy API limitlari
    alchemy_limits = [
        RateLimit(calls=25, period=1),     # 25 so'rov/sekund
        RateLimit(calls=300, period=60),   # 300 so'rov/minut
        RateLimit(calls=5000, period=3600) # 5000 so'rov/soat
    ]
    
    # HuggingFace API limitlari
    huggingface_limits = [
        RateLimit(calls=1000, period=3600)  # 1000 so'rov/soat
    ]
    
    # Gemini API limitlari
    gemini_limits = [
        RateLimit(calls=1, period=1),      # 1 so'rov/sekund
        RateLimit(calls=60, period=60),    # 60 so'rov/minut
        RateLimit(calls=1000, period=3600) # 1000 so'rov/soat
    ]
    
    # Claude API limitlari
    claude_limits = [
        RateLimit(calls=5, period=60),     # 5 so'rov/minut
        RateLimit(calls=20, period=3600)   # 20 so'rov/soat
    ]
    
    # Limiterlerni yaratish
    limiters = {
        'oneinch': global_limiter.get_api_limiter('oneinch', oneinch_limits),
        'alchemy': global_limiter.get_api_limiter('alchemy', alchemy_limits),
        'huggingface': global_limiter.get_api_limiter('huggingface', huggingface_limits),
        'gemini': global_limiter.get_api_limiter('gemini', gemini_limits),
        'claude': global_limiter.get_api_limiter('claude', claude_limits)
    }
    
    logger.info("Standart API limiterlari yaratildi")
    return limiters

# Test funksiyalari
async def test_rate_limiter():
    """Rate limiter test qilish"""
    print("Rate limiter test boshlandi...")
    
    # Oddiy rate limiter
    limiter = RateLimiter(calls=3, period=10)
    
    for i in range(5):
        print(f"So'rov {i+1}...")
        start_time = time.time()
        await limiter.wait()
        end_time = time.time()
        print(f"So'rov {i+1} bajarildi, kutish vaqti: {end_time - start_time:.2f}s")
    
    print("Test tugadi!")

if __name__ == "__main__":
    # Test ishga tushirish
    asyncio.run(test_rate_limiter())

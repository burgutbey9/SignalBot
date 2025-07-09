import asyncio
import aiohttp
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from utils.logger import get_logger
from utils.rate_limiter import RateLimiter
from utils.error_handler import handle_api_error
from utils.retry_handler import retry_async

logger = get_logger(__name__)

@dataclass
class ClaudeAPIResponse:
    """Claude API javob formati"""
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    rate_limit_remaining: int = 0
    usage: Optional[Dict] = None

@dataclass
class ClaudeMessage:
    """Claude xabar formati"""
    role: str  # "user" yoki "assistant"
    content: str

@dataclass
class ClaudeRequest:
    """Claude so'rov formati"""
    model: str
    messages: List[ClaudeMessage]
    max_tokens: int = 1000
    temperature: float = 0.7
    system: Optional[str] = None

class ClaudeClient:
    """Claude AI API client - sentiment tahlil va fallback uchun"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.anthropic.com/v1"
        self.rate_limiter = RateLimiter(calls=20, period=60)  # 20 so'rov/daqiqa
        self.session: Optional[aiohttp.ClientSession] = None
        self.model = "claude-3-haiku-20240307"  # Tezkor model
        
        logger.info("Claude AI client ishga tushirildi")
    
    async def __aenter__(self):
        """Async context manager kirish"""
        self.session = aiohttp.ClientSession(
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            },
            timeout=aiohttp.ClientTimeout(total=60)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager chiqish"""
        if self.session:
            await self.session.close()
    
    @retry_async(max_retries=2, delay=5)
    async def _make_request(self, endpoint: str, payload: Dict) -> ClaudeAPIResponse:
        """Asosiy so'rov yuborish methodi"""
        try:
            await self.rate_limiter.wait()
            
            if not self.session:
                raise ValueError("Session ochilmagan")
            
            url = f"{self.base_url}{endpoint}"
            
            async with self.session.post(url, json=payload) as response:
                response_data = await response.json()
                
                # Rate limit ma'lumotlarini olish
                rate_limit_remaining = int(response.headers.get('anthropic-ratelimit-requests-remaining', 0))
                
                if response.status == 200:
                    logger.info(f"Claude API muvaffaqiyatli: {endpoint}")
                    return ClaudeAPIResponse(
                        success=True,
                        data=response_data,
                        rate_limit_remaining=rate_limit_remaining,
                        usage=response_data.get('usage')
                    )
                elif response.status == 429:
                    # Rate limit tugagan
                    logger.warning("Claude API rate limit tugagan")
                    return ClaudeAPIResponse(
                        success=False,
                        error="Rate limit tugagan",
                        rate_limit_remaining=0
                    )
                else:
                    error_msg = response_data.get('error', {}).get('message', 'Noma\'lum xato')
                    logger.error(f"Claude API xato {response.status}: {error_msg}")
                    return ClaudeAPIResponse(
                        success=False,
                        error=error_msg,
                        rate_limit_remaining=rate_limit_remaining
                    )
                    
        except asyncio.TimeoutError:
            logger.error("Claude API timeout")
            return ClaudeAPIResponse(success=False, error="Timeout")
        except Exception as e:
            logger.error(f"Claude API so'rov xatosi: {e}")
            return ClaudeAPIResponse(success=False, error=str(e))
    
    async def analyze_sentiment(self, text: str, context: str = "crypto") -> ClaudeAPIResponse:
        """Sentiment tahlil qilish"""
        try:
            # O'zbekcha system prompt
            system_prompt = f"""
            Siz kripto bozori sentiment tahlilchisisiz. 
            Berilgan matnni tahlil qilib, quyidagi formatda javob bering:
            
            {{
                "sentiment": "bullish/bearish/neutral",
                "confidence": 0.85,
                "score": 0.7,
                "reasoning": "Tahlil sababi o'zbekcha",
                "key_phrases": ["muhim", "iboralar", "ro'yxati"],
                "market_impact": "high/medium/low"
            }}
            
            Kontekst: {context}
            Faqat JSON formatda javob bering.
            """
            
            messages = [
                ClaudeMessage(role="user", content=f"Matn tahlili: {text}")
            ]
            
            request = ClaudeRequest(
                model=self.model,
                messages=[{"role": msg.role, "content": msg.content} for msg in messages],
                max_tokens=500,
                temperature=0.3,
                system=system_prompt
            )
            
            response = await self._make_request("/messages", request.__dict__)
            
            if response.success:
                # Claude javobini JSON formatga o'tkazish
                content = response.data.get('content', [{}])[0].get('text', '')
                try:
                    sentiment_data = json.loads(content)
                    response.data = sentiment_data
                    logger.info(f"Sentiment tahlil muvaffaqiyatli: {sentiment_data.get('sentiment')}")
                except json.JSONDecodeError:
                    logger.error("Claude javobini JSON formatga o'tkazib bo'lmadi")
                    response.success = False
                    response.error = "JSON parse xatosi"
            
            return response
            
        except Exception as e:
            logger.error(f"Sentiment tahlil xatosi: {e}")
            return ClaudeAPIResponse(success=False, error=str(e))
    
    async def analyze_news(self, headlines: List[str], symbols: List[str] = None) -> ClaudeAPIResponse:
        """Yangiliklar tahlili"""
        try:
            symbols_str = ", ".join(symbols) if symbols else "umumiy kripto"
            
            system_prompt = f"""
            Siz kripto yangiliklar tahlilchisisiz.
            Berilgan sarlavhalarni tahlil qilib, quyidagi formatda javob bering:
            
            {{
                "overall_sentiment": "bullish/bearish/neutral",
                "confidence": 0.85,
                "market_impact": "high/medium/low",
                "affected_symbols": ["{symbols_str}"],
                "summary": "Qisqacha xulosani o'zbekcha yozing",
                "key_events": ["muhim", "hodisalar", "ro'yxati"],
                "recommendation": "hold/buy/sell/wait"
            }}
            
            Faqat JSON formatda javob bering.
            """
            
            headlines_text = "\n".join([f"- {headline}" for headline in headlines])
            
            messages = [
                ClaudeMessage(role="user", content=f"Yangiliklar tahlili:\n{headlines_text}")
            ]
            
            request = ClaudeRequest(
                model=self.model,
                messages=[{"role": msg.role, "content": msg.content} for msg in messages],
                max_tokens=800,
                temperature=0.4,
                system=system_prompt
            )
            
            response = await self._make_request("/messages", request.__dict__)
            
            if response.success:
                content = response.data.get('content', [{}])[0].get('text', '')
                try:
                    news_data = json.loads(content)
                    response.data = news_data
                    logger.info(f"Yangiliklar tahlil muvaffaqiyatli: {news_data.get('overall_sentiment')}")
                except json.JSONDecodeError:
                    logger.error("Claude yangiliklar javobini JSON formatga o'tkazib bo'lmadi")
                    response.success = False
                    response.error = "JSON parse xatosi"
            
            return response
            
        except Exception as e:
            logger.error(f"Yangiliklar tahlil xatosi: {e}")
            return ClaudeAPIResponse(success=False, error=str(e))
    
    async def generate_trading_signal(self, market_data: Dict, sentiment_data: Dict) -> ClaudeAPIResponse:
        """Trading signal yaratish"""
        try:
            system_prompt = """
            Siz professional trading signal generatorisiz.
            Berilgan market ma'lumotlar va sentiment tahlilni asosida signal yarating:
            
            {
                "action": "buy/sell/hold",
                "confidence": 0.85,
                "entry_price": 0.0,
                "stop_loss": 0.0,
                "take_profit": 0.0,
                "risk_reward": 2.5,
                "reasoning": "Signal sababi o'zbekcha",
                "timeframe": "1h/4h/1d",
                "urgency": "high/medium/low"
            }
            
            Faqat JSON formatda javob bering.
            """
            
            analysis_text = f"""
            Market ma'lumotlar: {json.dumps(market_data, indent=2)}
            Sentiment tahlil: {json.dumps(sentiment_data, indent=2)}
            """
            
            messages = [
                ClaudeMessage(role="user", content=f"Trading signal yaratish:\n{analysis_text}")
            ]
            
            request = ClaudeRequest(
                model=self.model,
                messages=[{"role": msg.role, "content": msg.content} for msg in messages],
                max_tokens=600,
                temperature=0.2,
                system=system_prompt
            )
            
            response = await self._make_request("/messages", request.__dict__)
            
            if response.success:
                content = response.data.get('content', [{}])[0].get('text', '')
                try:
                    signal_data = json.loads(content)
                    response.data = signal_data
                    logger.info(f"Trading signal yaratildi: {signal_data.get('action')}")
                except json.JSONDecodeError:
                    logger.error("Claude signal javobini JSON formatga o'tkazib bo'lmadi")
                    response.success = False
                    response.error = "JSON parse xatosi"
            
            return response
            
        except Exception as e:
            logger.error(f"Trading signal yaratish xatosi: {e}")
            return ClaudeAPIResponse(success=False, error=str(e))
    
    async def health_check(self) -> ClaudeAPIResponse:
        """Claude API holatini tekshirish"""
        try:
            test_messages = [
                ClaudeMessage(role="user", content="Test message")
            ]
            
            request = ClaudeRequest(
                model=self.model,
                messages=[{"role": msg.role, "content": msg.content} for msg in test_messages],
                max_tokens=50,
                temperature=0.1
            )
            
            response = await self._make_request("/messages", request.__dict__)
            
            if response.success:
                logger.info("Claude API health check muvaffaqiyatli")
            else:
                logger.error("Claude API health check xatolik")
            
            return response
            
        except Exception as e:
            logger.error(f"Claude API health check xatosi: {e}")
            return ClaudeAPIResponse(success=False, error=str(e))
    
    def get_rate_limit_status(self) -> Dict:
        """Rate limit holatini olish"""
        return {
            "remaining_calls": self.rate_limiter.get_remaining_calls(),
            "reset_time": self.rate_limiter.get_reset_time(),
            "calls_per_period": self.rate_limiter.calls,
            "period_seconds": self.rate_limiter.period
        }

# Fallback uchun Claude client wrapper
class ClaudeFallbackClient:
    """Claude client fallback bilan"""
    
    def __init__(self, api_key: str):
        self.client = ClaudeClient(api_key)
        self.is_available = True
        self.last_error = None
    
    async def __aenter__(self):
        await self.client.__aenter__()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.__aexit__(exc_type, exc_val, exc_tb)
    
    async def execute_with_fallback(self, operation: str, **kwargs) -> ClaudeAPIResponse:
        """Fallback bilan operatsiya bajarish"""
        try:
            if not self.is_available:
                logger.warning("Claude client mavjud emas")
                return ClaudeAPIResponse(success=False, error="Client mavjud emas")
            
            # Operatsiyani bajarish
            if hasattr(self.client, operation):
                result = await getattr(self.client, operation)(**kwargs)
                
                if not result.success:
                    self.last_error = result.error
                    if "rate limit" in result.error.lower():
                        self.is_available = False
                        logger.warning("Claude rate limit tugagan, client o'chirildi")
                
                return result
            else:
                logger.error(f"Claude clientda {operation} operatsiya mavjud emas")
                return ClaudeAPIResponse(success=False, error=f"Operatsiya mavjud emas: {operation}")
                
        except Exception as e:
            logger.error(f"Claude fallback xatosi: {e}")
            self.last_error = str(e)
            return ClaudeAPIResponse(success=False, error=str(e))
    
    def reset_availability(self):
        """Client holatini tiklash"""
        self.is_available = True
        self.last_error = None
        logger.info("Claude client holati tiklandi")

# Test funksiyasi
async def test_claude_client():
    """Claude client test qilish"""
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    api_key = os.getenv('CLAUDE_API_KEY')
    
    if not api_key:
        logger.error("CLAUDE_API_KEY topilmadi")
        return
    
    async with ClaudeFallbackClient(api_key) as client:
        # Health check
        print("=== Claude Health Check ===")
        health = await client.execute_with_fallback('health_check')
        print(f"Health: {health.success}")
        
        # Sentiment tahlil
        print("\n=== Sentiment Tahlil ===")
        sentiment = await client.execute_with_fallback(
            'analyze_sentiment',
            text="Bitcoin yangi rekord o'rnatdi! Narxi $70,000 dan oshdi.",
            context="crypto"
        )
        print(f"Sentiment: {sentiment.data if sentiment.success else sentiment.error}")
        
        # Yangiliklar tahlil
        print("\n=== Yangiliklar Tahlil ===")
        news = await client.execute_with_fallback(
            'analyze_news',
            headlines=[
                "Bitcoin ETF rasmiy tasdiqlandi",
                "Ethereum 2.0 yangilanishi muvaffaqiyatli",
                "Regulyatsiya yangiliklari ijobiy"
            ],
            symbols=["BTC", "ETH"]
        )
        print(f"News: {news.data if news.success else news.error}")
        
        # Rate limit holati
        print("\n=== Rate Limit Status ===")
        status = client.client.get_rate_limit_status()
        print(f"Rate limit: {status}")

if __name__ == "__main__":
    asyncio.run(test_claude_client())

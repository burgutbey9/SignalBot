import asyncio
import aiohttp
import json
import random
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime
import time

from utils.logger import get_logger
from utils.rate_limiter import RateLimiter
from utils.error_handler import handle_api_error
from utils.retry_handler import retry_async
from config.config import ConfigManager

logger = get_logger(__name__)

@dataclass
class GeminiResponse:
    """Gemini API javob strukturasi"""
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    confidence: float = 0.0
    model_name: str = "gemini-pro"
    processing_time: float = 0.0
    api_key_index: int = 0
    tokens_used: int = 0
    rate_limit_remaining: int = 0

@dataclass
class GeminiSentimentResult:
    """Gemini sentiment tahlil natijasi"""
    label: str  # POSITIVE, NEGATIVE, NEUTRAL
    score: float  # 0.0 - 1.0
    confidence: float
    reasoning: str  # Gemini ning sababi
    market_impact: str  # HIGH, MEDIUM, LOW
    keywords: List[str]  # Muhim kalit so'zlar
    emotions: Dict[str, float]  # Hissiy tahlil

@dataclass
class APIKeyStatus:
    """API key holati"""
    index: int
    key: str
    is_active: bool = True
    last_used: float = 0.0
    error_count: int = 0
    rate_limit_reset: float = 0.0
    daily_usage: int = 0
    max_daily_usage: int = 1000

class GeminiClient:
    """Gemini AI client - 5 ta API key bilan fallback tizimi"""
    
    def __init__(self, api_keys: List[str], config_manager: ConfigManager):
        if len(api_keys) != 5:
            raise ValueError("Gemini client uchun aynan 5 ta API key kerak")
        
        self.config = config_manager
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
        
        # API kalitlarni sozlash
        self.api_keys = [
            APIKeyStatus(index=i, key=key)
            for i, key in enumerate(api_keys)
        ]
        
        # Rate limiting - 60 so'rov / daqiqa har key uchun
        self.rate_limiters = {
            i: RateLimiter(calls=60, period=60)
            for i in range(5)
        }
        
        # Session
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Joriy ishlatilayotgan key
        self.current_key_index = 0
        
        # Modellar ro'yxati
        self.models = {
            "gemini-pro": "models/gemini-pro",
            "gemini-pro-vision": "models/gemini-pro-vision"
        }
        
        # Sentiment tahlil uchun promptlar
        self.prompts = {
            "sentiment": """
Quyidagi matnni moliyaviy sentiment jihatdan tahlil qiling va JSON formatda javob bering:

Matn: "{text}"

Javob formati:
{{
    "sentiment": "POSITIVE/NEGATIVE/NEUTRAL",
    "confidence": 0.0-1.0,
    "score": 0.0-1.0,
    "reasoning": "Nima uchun bunday sentiment",
    "market_impact": "HIGH/MEDIUM/LOW",
    "keywords": ["kalit", "sozlar"],
    "emotions": {{"fear": 0.0, "greed": 0.0, "optimism": 0.0}}
}}

Faqat JSON javob bering, boshqa matn yo'q.
""",
            "crypto_sentiment": """
Quyidagi kripto va moliyaviy matnni tahlil qiling va JSON formatda javob bering:

Matn: "{text}"

Kripto bozor kontekstida tahlil qiling va JSON formatda javob bering:
{{
    "sentiment": "BULLISH/BEARISH/NEUTRAL",
    "confidence": 0.0-1.0,
    "score": 0.0-1.0,
    "reasoning": "Nima uchun bunday sentiment",
    "market_impact": "HIGH/MEDIUM/LOW",
    "keywords": ["bitcoin", "pump", "dump"],
    "price_prediction": "UP/DOWN/SIDEWAYS",
    "emotions": {{"fear": 0.0, "greed": 0.0, "fomo": 0.0}}
}}

Faqat JSON javob bering.
""",
            "news_summary": """
Quyidagi yangilikni qisqacha xulosalang va moliyaviy ta'sirini baholang:

Matn: "{text}"

JSON formatda javob:
{{
    "summary": "Qisqacha xulosa",
    "sentiment": "POSITIVE/NEGATIVE/NEUTRAL",
    "market_relevance": "HIGH/MEDIUM/LOW",
    "key_points": ["muhim", "nuqtalar"],
    "potential_impact": "Bozorga ta'siri"
}}
"""
        }
        
        logger.info("Gemini client 5 ta API key bilan ishga tushirildi")
    
    async def __aenter__(self):
        """Async context manager kirish"""
        timeout = aiohttp.ClientTimeout(total=60)
        self.session = aiohttp.ClientSession(timeout=timeout)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager chiqish"""
        if self.session:
            await self.session.close()
    
    def _get_next_available_key(self) -> Optional[APIKeyStatus]:
        """Keyingi mavjud API kalitni topish"""
        current_time = time.time()
        
        # Faol kalitlarni saralash
        active_keys = [
            key for key in self.api_keys 
            if key.is_active and key.error_count < 5 and key.daily_usage < key.max_daily_usage
        ]
        
        if not active_keys:
            # Barcha kalitlar ishlamayotgan bo'lsa, eng kam xatolik bilan kalitni qaytarish
            logger.warning("Barcha Gemini API kalitlari ishlamayotgan")
            return min(self.api_keys, key=lambda x: x.error_count)
        
        # Eng kam ishlatilgan kalitni topish
        return min(active_keys, key=lambda x: x.last_used)
    
    def _update_key_status(self, key_index: int, success: bool, error_type: str = None):
        """API kalit holatini yangilash"""
        key = self.api_keys[key_index]
        key.last_used = time.time()
        
        if success:
            key.error_count = max(0, key.error_count - 1)  # Muvaffaqiyatli so'rov xatolarni kamaytiradi
            key.daily_usage += 1
        else:
            key.error_count += 1
            
            # 5 ta ketma-ket xato bo'lsa, kalitni vaqtincha o'chirish
            if key.error_count >= 5:
                key.is_active = False
                key.rate_limit_reset = time.time() + 300  # 5 daqiqaga o'chirish
                logger.warning(f"Gemini API kalit {key_index} vaqtincha o'chirildi")
            
            # Rate limit xatosi bo'lsa
            if error_type == "rate_limit":
                key.rate_limit_reset = time.time() + 60  # 1 daqiqaga o'chirish
        
        logger.debug(f"API kalit {key_index} holati yangilandi: xatolar={key.error_count}, faol={key.is_active}")
    
    def _reactivate_keys(self):
        """Vaqti kelgan kalitlarni qayta faollashtirish"""
        current_time = time.time()
        
        for key in self.api_keys:
            if not key.is_active and current_time > key.rate_limit_reset:
                key.is_active = True
                key.error_count = 0
                logger.info(f"Gemini API kalit {key.index} qayta faollashtirildi")
    
    @retry_async(max_retries=3, delay=2)
    async def _make_request(self, prompt: str, model: str = "gemini-pro") -> GeminiResponse:
        """Gemini API ga so'rov yuborish"""
        start_time = time.time()
        
        try:
            # O'chirilgan kalitlarni qayta faollashtirish
            self._reactivate_keys()
            
            # Mavjud kalitni topish
            key_status = self._get_next_available_key()
            if not key_status:
                return GeminiResponse(
                    success=False,
                    error="Barcha API kalitlar ishlamayotgan",
                    processing_time=time.time() - start_time
                )
            
            # Rate limiting
            await self.rate_limiters[key_status.index].wait()
            
            # So'rov URL
            url = f"{self.base_url}/{self.models[model]}:generateContent"
            
            # Headers
            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": key_status.key
            }
            
            # Payload
            payload = {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": prompt
                            }
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.7,
                    "topK": 40,
                    "topP": 0.95,
                    "maxOutputTokens": 1024,
                    "stopSequences": []
                },
                "safetySettings": [
                    {
                        "category": "HARM_CATEGORY_HARASSMENT",
                        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                    },
                    {
                        "category": "HARM_CATEGORY_HATE_SPEECH",
                        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                    },
                    {
                        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                    },
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                    }
                ]
            }
            
            async with self.session.post(url, headers=headers, json=payload) as response:
                processing_time = time.time() - start_time
                
                if response.status == 200:
                    data = await response.json()
                    
                    # Javobni qayta ishlash
                    if "candidates" in data and len(data["candidates"]) > 0:
                        content = data["candidates"][0]["content"]["parts"][0]["text"]
                        
                        # Token hisobi
                        tokens_used = len(content.split()) + len(prompt.split())
                        
                        # Muvaffaqiyatli so'rov
                        self._update_key_status(key_status.index, True)
                        
                        logger.info(f"Gemini API muvaffaqiyatli (kalit {key_status.index})")
                        
                        return GeminiResponse(
                            success=True,
                            data=content,
                            confidence=0.9,  # Gemini yuqori ishonch
                            model_name=model,
                            processing_time=processing_time,
                            api_key_index=key_status.index,
                            tokens_used=tokens_used
                        )
                    else:
                        error_msg = "Gemini javob bermadi yoki bo'sh javob"
                        self._update_key_status(key_status.index, False)
                        
                elif response.status == 429:
                    error_msg = "Rate limit oshib ketdi"
                    self._update_key_status(key_status.index, False, "rate_limit")
                    logger.warning(f"Gemini rate limit (kalit {key_status.index})")
                    
                elif response.status == 403:
                    error_msg = "API kalit yaroqsiz yoki ruxsat yo'q"
                    self._update_key_status(key_status.index, False, "forbidden")
                    logger.error(f"Gemini API kalit {key_status.index} yaroqsiz")
                    
                else:
                    error_text = await response.text()
                    error_msg = f"API xatosi: {response.status} - {error_text}"
                    self._update_key_status(key_status.index, False)
                    logger.error(f"Gemini API xatosi: {error_msg}")
                
                return GeminiResponse(
                    success=False,
                    error=error_msg,
                    model_name=model,
                    processing_time=processing_time,
                    api_key_index=key_status.index
                )
                
        except asyncio.TimeoutError:
            error_msg = "So'rov timeout bo'ldi"
            logger.error(f"Gemini timeout")
            return GeminiResponse(
                success=False,
                error=error_msg,
                processing_time=time.time() - start_time
            )
            
        except Exception as e:
            error_msg = f"Kutilmagan xato: {str(e)}"
            logger.error(f"Gemini kutilmagan xato: {error_msg}")
            return GeminiResponse(
                success=False,
                error=error_msg,
                processing_time=time.time() - start_time
            )
    
    def _parse_json_response(self, response_text: str) -> Optional[Dict]:
        """JSON javobni qayta ishlash"""
        try:
            # JSON ni topish
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}')
            
            if start_idx != -1 and end_idx != -1:
                json_str = response_text[start_idx:end_idx+1]
                return json.loads(json_str)
            
            return None
        except Exception as e:
            logger.error(f"JSON parsing xatosi: {e}")
            return None
    
    async def analyze_sentiment(self, text: str, analysis_type: str = "sentiment") -> GeminiSentimentResult:
        """Gemini bilan sentiment tahlil"""
        try:
            # Prompt tanlash
            prompt_template = self.prompts.get(analysis_type, self.prompts["sentiment"])
            prompt = prompt_template.format(text=text)
            
            # Gemini ga so'rov yuborish
            response = await self._make_request(prompt)
            
            if not response.success:
                logger.error(f"Gemini sentiment tahlil muvaffaqiyatsiz: {response.error}")
                return GeminiSentimentResult(
                    label="NEUTRAL",
                    score=0.5,
                    confidence=0.0,
                    reasoning="Gemini API ishlamadi",
                    market_impact="LOW",
                    keywords=[],
                    emotions={}
                )
            
            # JSON javobni qayta ishlash
            parsed_data = self._parse_json_response(response.data)
            
            if not parsed_data:
                logger.error("Gemini dan JSON javob kelmadi")
                return GeminiSentimentResult(
                    label="NEUTRAL",
                    score=0.5,
                    confidence=0.0,
                    reasoning="JSON parsing xatosi",
                    market_impact="LOW",
                    keywords=[],
                    emotions={}
                )
            
            # Natijani qayta ishlash
            result = self._process_sentiment_result(parsed_data)
            logger.info(f"Gemini sentiment tahlil muvaffaqiyatli: {result.label} ({result.score:.2f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Gemini sentiment tahlil xatosi: {e}")
            return GeminiSentimentResult(
                label="NEUTRAL",
                score=0.5,
                confidence=0.0,
                reasoning=f"Xato: {str(e)}",
                market_impact="LOW",
                keywords=[],
                emotions={}
            )
    
    def _process_sentiment_result(self, data: Dict) -> GeminiSentimentResult:
        """Gemini sentiment natijasini qayta ishlash"""
        try:
            # Sentiment ni normalizatsiya qilish
            sentiment = data.get("sentiment", "NEUTRAL").upper()
            if sentiment in ["BULLISH", "POSITIVE"]:
                normalized_sentiment = "POSITIVE"
            elif sentiment in ["BEARISH", "NEGATIVE"]:
                normalized_sentiment = "NEGATIVE"
            else:
                normalized_sentiment = "NEUTRAL"
            
            return GeminiSentimentResult(
                label=normalized_sentiment,
                score=float(data.get("score", 0.5)),
                confidence=float(data.get("confidence", 0.0)),
                reasoning=data.get("reasoning", ""),
                market_impact=data.get("market_impact", "MEDIUM"),
                keywords=data.get("keywords", []),
                emotions=data.get("emotions", {})
            )
            
        except Exception as e:
            logger.error(f"Gemini natijasini qayta ishlashda xato: {e}")
            return GeminiSentimentResult(
                label="NEUTRAL",
                score=0.5,
                confidence=0.0,
                reasoning="Natijani qayta ishlashda xato",
                market_impact="LOW",
                keywords=[],
                emotions={}
            )
    
    async def analyze_news(self, text: str) -> Dict[str, Any]:
        """Yangilikni tahlil qilish"""
        try:
            prompt = self.prompts["news_summary"].format(text=text)
            response = await self._make_request(prompt)
            
            if not response.success:
                return {
                    "summary": "Yangilik tahlil qilinmadi",
                    "sentiment": "NEUTRAL",
                    "market_relevance": "LOW",
                    "key_points": [],
                    "potential_impact": "Noma'lum"
                }
            
            parsed_data = self._parse_json_response(response.data)
            
            if parsed_data:
                return parsed_data
            else:
                return {
                    "summary": response.data[:200] + "...",
                    "sentiment": "NEUTRAL",
                    "market_relevance": "MEDIUM",
                    "key_points": [],
                    "potential_impact": "Gemini JSON berish muvaffaqiyatsiz"
                }
                
        except Exception as e:
            logger.error(f"Gemini yangilik tahlil xatosi: {e}")
            return {
                "summary": "Yangilik tahlil qilinmadi",
                "sentiment": "NEUTRAL",
                "market_relevance": "LOW",
                "key_points": [],
                "potential_impact": f"Xato: {str(e)}"
            }
    
    async def get_api_keys_status(self) -> Dict[str, Any]:
        """API kalitlar holatini olish"""
        try:
            status = []
            for key in self.api_keys:
                status.append({
                    "index": key.index,
                    "is_active": key.is_active,
                    "error_count": key.error_count,
                    "daily_usage": key.daily_usage,
                    "max_daily_usage": key.max_daily_usage,
                    "last_used": datetime.fromtimestamp(key.last_used).strftime("%H:%M:%S") if key.last_used > 0 else "Hech qachon",
                    "rate_limit_reset": datetime.fromtimestamp(key.rate_limit_reset).strftime("%H:%M:%S") if key.rate_limit_reset > time.time() else "Yo'q"
                })
            
            active_count = sum(1 for key in self.api_keys if key.is_active)
            
            return {
                "total_keys": len(self.api_keys),
                "active_keys": active_count,
                "inactive_keys": len(self.api_keys) - active_count,
                "keys_status": status
            }
            
        except Exception as e:
            logger.error(f"API kalitlar holatini olishda xato: {e}")
            return {"error": str(e)}
    
    async def health_check(self) -> Dict[str, Any]:
        """Barcha API kalitlarni tekshirish"""
        try:
            results = []
            test_text = "The market is performing well today"
            
            for i, key in enumerate(self.api_keys):
                if not key.is_active:
                    results.append({
                        "key_index": i,
                        "status": "inactive",
                        "error": "API kalit faol emas"
                    })
                    continue
                
                try:
                    # Test so'rov
                    old_current = self.current_key_index
                    self.current_key_index = i
                    
                    result = await self.analyze_sentiment(test_text)
                    
                    results.append({
                        "key_index": i,
                        "status": "healthy" if result.confidence > 0 else "unhealthy",
                        "confidence": result.confidence,
                        "response_time": time.time()
                    })
                    
                    self.current_key_index = old_current
                    
                except Exception as e:
                    results.append({
                        "key_index": i,
                        "status": "error",
                        "error": str(e)
                    })
            
            healthy_count = sum(1 for r in results if r["status"] == "healthy")
            
            return {
                "overall_status": "healthy" if healthy_count > 0 else "unhealthy",
                "healthy_keys": healthy_count,
                "total_keys": len(self.api_keys),
                "individual_results": results
            }
            
        except Exception as e:
            logger.error(f"Gemini health check xatosi: {e}")
            return {
                "overall_status": "error",
                "error": str(e)
            }


# Foydalanish namunasi
async def main():
    """Test funksiyasi"""
    config = ConfigManager()
    api_keys = [
        "key1", "key2", "key3", "key4", "key5"
    ]
    
    async with GeminiClient(api_keys, config) as client:
        # Sentiment tahlil
        text = "Bitcoin is showing strong bullish momentum with institutional adoption increasing"
        result = await client.analyze_sentiment(text, "crypto_sentiment")
        print(f"Sentiment: {result.label} ({result.score:.2f})")
        print(f"Reasoning: {result.reasoning}")
        print(f"Keywords: {result.keywords}")
        
        # Yangilik tahlil
        news_text = "Federal Reserve announces new monetary policy changes affecting crypto markets"
        news_result = await client.analyze_news(news_text)
        print(f"News summary: {news_result['summary']}")
        
        # API kalitlar holati
        status = await client.get_api_keys_status()
        print(f"Active keys: {status['active_keys']}/{status['total_keys']}")
        
        # Health check
        health = await client.health_check()
        print(f"Overall health: {health['overall_status']}")

if __name__ == "__main__":
    asyncio.run(main())

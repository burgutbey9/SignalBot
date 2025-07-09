import asyncio
import aiohttp
import json
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
class HuggingFaceResponse:
    """HuggingFace API javob strukturasi"""
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    confidence: float = 0.0
    model_name: str = ""
    processing_time: float = 0.0
    rate_limit_remaining: int = 0

@dataclass
class SentimentResult:
    """Sentiment tahlil natijasi"""
    label: str  # POSITIVE, NEGATIVE, NEUTRAL
    score: float  # 0.0 - 1.0
    confidence: float
    raw_scores: Dict[str, float]

@dataclass
class TextClassificationResult:
    """Matn klassifikatsiya natijasi"""
    predictions: List[Dict[str, Any]]
    top_prediction: Dict[str, Any]
    confidence: float

class HuggingFaceClient:
    """HuggingFace API client - AI sentiment tahlil uchun"""
    
    def __init__(self, api_key: str, config_manager: ConfigManager):
        self.api_key = api_key
        self.config = config_manager
        self.base_url = "https://api-inference.huggingface.co/models"
        
        # Rate limiting - 1000 so'rov / soat
        self.rate_limiter = RateLimiter(calls=1000, period=3600)
        
        # Session
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Modellar ro'yxati
        self.models = {
            "sentiment": {
                "primary": "cardiffnlp/twitter-roberta-base-sentiment-latest",
                "fallback": "nlptown/bert-base-multilingual-uncased-sentiment"
            },
            "emotion": {
                "primary": "j-hartmann/emotion-english-distilroberta-base",
                "fallback": "SamLowe/roberta-base-go_emotions"
            },
            "financial": {
                "primary": "ProsusAI/finbert",
                "fallback": "ahmedrachid/FinancialBERT-Sentiment-Analysis"
            },
            "crypto": {
                "primary": "ElKulako/cryptobert",
                "fallback": "cardiffnlp/twitter-roberta-base-sentiment-latest"
            }
        }
        
        logger.info("HuggingFace client ishga tushirildi")
    
    async def __aenter__(self):
        """Async context manager kirish"""
        timeout = aiohttp.ClientTimeout(total=60)
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager chiqish"""
        if self.session:
            await self.session.close()
    
    @retry_async(max_retries=3, delay=1)
    async def _make_request(self, model_name: str, payload: Dict) -> HuggingFaceResponse:
        """HuggingFace API ga so'rov yuborish"""
        start_time = time.time()
        
        try:
            await self.rate_limiter.wait()
            
            url = f"{self.base_url}/{model_name}"
            
            async with self.session.post(url, json=payload) as response:
                processing_time = time.time() - start_time
                
                # Rate limit info
                rate_limit_remaining = int(response.headers.get('x-ratelimit-remaining', 0))
                
                if response.status == 200:
                    data = await response.json()
                    
                    # Confidence hisoblash
                    confidence = self._calculate_confidence(data)
                    
                    logger.info(f"HuggingFace API muvaffaqiyatli: {model_name}")
                    return HuggingFaceResponse(
                        success=True,
                        data=data,
                        confidence=confidence,
                        model_name=model_name,
                        processing_time=processing_time,
                        rate_limit_remaining=rate_limit_remaining
                    )
                
                elif response.status == 429:
                    error_msg = "Rate limit oshib ketdi"
                    logger.warning(f"HuggingFace rate limit: {model_name}")
                    await asyncio.sleep(5)  # Rate limit da 5 soniya kutish
                    
                elif response.status == 503:
                    error_msg = "Model yuklanmoqda, biroz kuting"
                    logger.warning(f"HuggingFace model yuklanmoqda: {model_name}")
                    await asyncio.sleep(10)  # Model yuklanishi uchun 10 soniya
                    
                else:
                    error_text = await response.text()
                    error_msg = f"API xatosi: {response.status} - {error_text}"
                    logger.error(f"HuggingFace API xatosi: {error_msg}")
                
                return HuggingFaceResponse(
                    success=False,
                    error=error_msg,
                    model_name=model_name,
                    processing_time=processing_time,
                    rate_limit_remaining=rate_limit_remaining
                )
                
        except asyncio.TimeoutError:
            error_msg = "So'rov timeout bo'ldi"
            logger.error(f"HuggingFace timeout: {model_name}")
            return HuggingFaceResponse(
                success=False,
                error=error_msg,
                model_name=model_name,
                processing_time=time.time() - start_time
            )
            
        except Exception as e:
            error_msg = f"Kutilmagan xato: {str(e)}"
            logger.error(f"HuggingFace kutilmagan xato: {error_msg}")
            return HuggingFaceResponse(
                success=False,
                error=error_msg,
                model_name=model_name,
                processing_time=time.time() - start_time
            )
    
    def _calculate_confidence(self, data: Any) -> float:
        """API javobidan confidence hisoblash"""
        try:
            if isinstance(data, list) and len(data) > 0:
                if isinstance(data[0], list):
                    # Klassifikatsiya natijalari
                    max_score = max(item.get('score', 0) for item in data[0])
                    return max_score
                elif isinstance(data[0], dict) and 'score' in data[0]:
                    # Bitta natija
                    return data[0]['score']
            return 0.0
        except Exception:
            return 0.0
    
    async def analyze_sentiment(self, text: str, model_type: str = "sentiment") -> SentimentResult:
        """Matn sentiment tahlil qilish"""
        try:
            # Model tanlash
            models = self.models.get(model_type, self.models["sentiment"])
            model_name = models["primary"]
            
            payload = {
                "inputs": text,
                "options": {
                    "wait_for_model": True,
                    "use_cache": False
                }
            }
            
            # Asosiy model bilan sinash
            response = await self._make_request(model_name, payload)
            
            # Fallback model bilan sinash
            if not response.success:
                logger.warning(f"Asosiy model ishlamadi, fallback ishlatiladi: {models['fallback']}")
                model_name = models["fallback"]
                response = await self._make_request(model_name, payload)
            
            if not response.success:
                logger.error(f"Sentiment tahlil muvaffaqiyatsiz: {response.error}")
                return SentimentResult(
                    label="NEUTRAL",
                    score=0.5,
                    confidence=0.0,
                    raw_scores={}
                )
            
            # Natijani qayta ishlash
            result = self._process_sentiment_result(response.data)
            logger.info(f"Sentiment tahlil muvaffaqiyatli: {result.label} ({result.score:.2f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Sentiment tahlil xatosi: {e}")
            return SentimentResult(
                label="NEUTRAL",
                score=0.5,
                confidence=0.0,
                raw_scores={}
            )
    
    def _process_sentiment_result(self, data: Any) -> SentimentResult:
        """Sentiment natijasini qayta ishlash"""
        try:
            if isinstance(data, list) and len(data) > 0:
                predictions = data[0] if isinstance(data[0], list) else data
                
                # Eng yuqori skorli natijani topish
                top_prediction = max(predictions, key=lambda x: x.get('score', 0))
                
                # Label ni standartlashtirish
                label = self._normalize_sentiment_label(top_prediction['label'])
                
                # Raw scores yaratish
                raw_scores = {
                    pred['label']: pred['score'] 
                    for pred in predictions
                }
                
                return SentimentResult(
                    label=label,
                    score=top_prediction['score'],
                    confidence=top_prediction['score'],
                    raw_scores=raw_scores
                )
            
            return SentimentResult(
                label="NEUTRAL",
                score=0.5,
                confidence=0.0,
                raw_scores={}
            )
            
        except Exception as e:
            logger.error(f"Sentiment natijasini qayta ishlashda xato: {e}")
            return SentimentResult(
                label="NEUTRAL",
                score=0.5,
                confidence=0.0,
                raw_scores={}
            )
    
    def _normalize_sentiment_label(self, label: str) -> str:
        """Sentiment labelni standartlashtirish"""
        label_lower = label.lower()
        
        positive_labels = ['positive', 'pos', 'bullish', 'buy', 'good', 'happy', 'joy']
        negative_labels = ['negative', 'neg', 'bearish', 'sell', 'bad', 'sad', 'fear', 'anger']
        
        if any(pos in label_lower for pos in positive_labels):
            return "POSITIVE"
        elif any(neg in label_lower for neg in negative_labels):
            return "NEGATIVE"
        else:
            return "NEUTRAL"
    
    async def analyze_market_sentiment(self, texts: List[str]) -> Dict[str, Any]:
        """Bir nechta matnlar uchun bozor sentiment tahlil"""
        try:
            if not texts:
                return {
                    "overall_sentiment": "NEUTRAL",
                    "sentiment_score": 0.5,
                    "confidence": 0.0,
                    "individual_results": [],
                    "sentiment_distribution": {
                        "POSITIVE": 0,
                        "NEGATIVE": 0,
                        "NEUTRAL": 0
                    }
                }
            
            # Har bir matn uchun sentiment tahlil
            results = []
            for text in texts[:10]:  # Faqat birinchi 10 ta matn
                result = await analyze_sentiment(text, "financial")
                results.append(result)
                
                # Rate limit uchun kichik kutish
                await asyncio.sleep(0.1)
            
            # Umumiy sentiment hisoblash
            overall_result = self._calculate_overall_sentiment(results)
            
            logger.info(f"Bozor sentiment tahlil muvaffaqiyatli: {overall_result['overall_sentiment']}")
            return overall_result
            
        except Exception as e:
            logger.error(f"Bozor sentiment tahlil xatosi: {e}")
            return {
                "overall_sentiment": "NEUTRAL",
                "sentiment_score": 0.5,
                "confidence": 0.0,
                "individual_results": [],
                "sentiment_distribution": {
                    "POSITIVE": 0,
                    "NEGATIVE": 0,
                    "NEUTRAL": 0
                }
            }
    
    def _calculate_overall_sentiment(self, results: List[SentimentResult]) -> Dict[str, Any]:
        """Umumiy sentiment hisoblash"""
        try:
            if not results:
                return {
                    "overall_sentiment": "NEUTRAL",
                    "sentiment_score": 0.5,
                    "confidence": 0.0,
                    "individual_results": [],
                    "sentiment_distribution": {
                        "POSITIVE": 0,
                        "NEGATIVE": 0,
                        "NEUTRAL": 0
                    }
                }
            
            # Sentiment taqsimot hisoblash
            distribution = {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0}
            total_score = 0
            total_confidence = 0
            
            for result in results:
                distribution[result.label] += 1
                
                # Skorni -1 dan 1 gacha normallashtirish
                if result.label == "POSITIVE":
                    total_score += result.score
                elif result.label == "NEGATIVE":
                    total_score -= result.score
                
                total_confidence += result.confidence
            
            # Umumiy sentiment aniqlash
            if distribution["POSITIVE"] > distribution["NEGATIVE"]:
                overall_sentiment = "POSITIVE"
            elif distribution["NEGATIVE"] > distribution["POSITIVE"]:
                overall_sentiment = "NEGATIVE"
            else:
                overall_sentiment = "NEUTRAL"
            
            # O'rtacha skor va confidence
            avg_score = total_score / len(results)
            avg_confidence = total_confidence / len(results)
            
            # Skorni 0-1 oralig'iga keltirish
            normalized_score = (avg_score + 1) / 2
            
            return {
                "overall_sentiment": overall_sentiment,
                "sentiment_score": normalized_score,
                "confidence": avg_confidence,
                "individual_results": [
                    {
                        "label": r.label,
                        "score": r.score,
                        "confidence": r.confidence
                    }
                    for r in results
                ],
                "sentiment_distribution": distribution
            }
            
        except Exception as e:
            logger.error(f"Umumiy sentiment hisoblashda xato: {e}")
            return {
                "overall_sentiment": "NEUTRAL",
                "sentiment_score": 0.5,
                "confidence": 0.0,
                "individual_results": [],
                "sentiment_distribution": {
                    "POSITIVE": 0,
                    "NEGATIVE": 0,
                    "NEUTRAL": 0
                }
            }
    
    async def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Model haqida ma'lumot olish"""
        try:
            url = f"https://huggingface.co/api/models/{model_name}"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "name": data.get("modelId", model_name),
                        "downloads": data.get("downloads", 0),
                        "likes": data.get("likes", 0),
                        "library_name": data.get("library_name", "unknown"),
                        "tags": data.get("tags", [])
                    }
                else:
                    logger.warning(f"Model ma'lumotini olishda xato: {response.status}")
                    return {"name": model_name, "status": "ma'lumot topilmadi"}
                    
        except Exception as e:
            logger.error(f"Model ma'lumotini olishda xato: {e}")
            return {"name": model_name, "error": str(e)}
    
    async def health_check(self) -> Dict[str, Any]:
        """API holatini tekshirish"""
        try:
            test_text = "The market is looking good today"
            result = await self.analyze_sentiment(test_text)
            
            return {
                "status": "healthy" if result.confidence > 0 else "unhealthy",
                "response_time": time.time(),
                "test_result": {
                    "label": result.label,
                    "confidence": result.confidence
                }
            }
            
        except Exception as e:
            logger.error(f"Health check xatosi: {e}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }


# Foydalanish namunasi
async def main():
    """Test funksiyasi"""
    config = ConfigManager()
    api_key = "your_huggingface_api_key"
    
    async with HuggingFaceClient(api_key, config) as client:
        # Sentiment tahlil
        text = "Bitcoin is going to the moon! Great news for crypto investors!"
        result = await client.analyze_sentiment(text, "crypto")
        print(f"Sentiment: {result.label} ({result.score:.2f})")
        
        # Bozor sentiment
        texts = [
            "Bulls are back in the market",
            "Bearish sentiment continues",
            "Market is consolidating"
        ]
        market_result = await client.analyze_market_sentiment(texts)
        print(f"Market sentiment: {market_result['overall_sentiment']}")
        
        # Health check
        health = await client.health_check()
        print(f"Health: {health['status']}")

if __name__ == "__main__":
    asyncio.run(main())

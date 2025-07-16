import asyncio
import json
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import statistics
import numpy as np
from utils.logger import get_logger
from utils.error_handler import handle_processing_error
from utils.fallback_manager import FallbackManager
from api_clients.huggingface_client import HuggingFaceClient
from api_clients.gemini_client import GeminiClient
from api_clients.claude_client import ClaudeClient
from api_clients.news_client import NewsClient
from api_clients.reddit_client import RedditClient

logger = get_logger(__name__)

@dataclass
class SentimentScore:
    """Sentiment ball natijasi"""
    score: float  # -1.0 dan 1.0 gacha
    confidence: float  # 0.0 dan 1.0 gacha
    source: str
    timestamp: datetime
    raw_data: Optional[Dict] = None

@dataclass
class SentimentAnalysis:
    """Sentiment tahlil natijasi"""
    overall_score: float  # -1.0 dan 1.0 gacha
    confidence: float  # 0.0 dan 1.0 gacha
    sentiment_label: str  # "BULLISH", "BEARISH", "NEUTRAL"
    individual_scores: List[SentimentScore]
    news_sentiment: Optional[float] = None
    social_sentiment: Optional[float] = None
    ai_sentiment: Optional[float] = None
    signal_strength: str = "WEAK"  # "WEAK", "MODERATE", "STRONG"
    timestamp: datetime = None
    error: Optional[str] = None

@dataclass
class ProcessingResult:
    """Qayta ishlash natijasi"""
    success: bool
    data: Optional[SentimentAnalysis] = None
    confidence: float = 0.0
    error: Optional[str] = None

class SentimentAnalyzer:
    """Sentiment tahlil qiluvchi asosiy class"""
    
    def __init__(self):
        self.name = self.__class__.__name__
        logger.info(f"{self.name} ishga tushirildi")
        
        # Fallback manager
        self.fallback_manager = FallbackManager()
        
        # AI clientlar
        self.ai_clients = {
            'huggingface': HuggingFaceClient(),
            'gemini': GeminiClient(),
            'claude': ClaudeClient()
        }
        
        # Ma'lumot manbalari
        self.news_client = NewsClient()
        self.reddit_client = RedditClient()
        
        # Sentiment so'zlar lug'ati
        self.sentiment_keywords = {
            'bullish': ['buy', 'bull', 'up', 'moon', 'pump', 'rise', 'rally', 'breakout', 'strong', 'bullish', 'positive'],
            'bearish': ['sell', 'bear', 'down', 'dump', 'fall', 'crash', 'drop', 'bearish', 'negative', 'decline'],
            'neutral': ['hold', 'wait', 'sideways', 'consolidate', 'neutral', 'stable', 'range']
        }
        
        # Sentiment og'irliklari
        self.sentiment_weights = {
            'news': 0.4,
            'social': 0.3,
            'ai': 0.3
        }
        
        logger.info("Sentiment Analyzer muvaffaqiyatli ishga tushdi")

    async def analyze_sentiment(self, symbol: str, timeframe: str = "24h") -> ProcessingResult:
        """Asosiy sentiment tahlil methodi"""
        try:
            logger.info(f"Sentiment tahlil boshlandi - {symbol}")
            
            # Parallel ma'lumot yig'ish
            tasks = [
                self._analyze_news_sentiment(symbol, timeframe),
                self._analyze_social_sentiment(symbol, timeframe),
                self._analyze_ai_sentiment(symbol, timeframe)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Natijalarni qayta ishlash
            news_sentiment = results[0] if not isinstance(results[0], Exception) else None
            social_sentiment = results[1] if not isinstance(results[1], Exception) else None
            ai_sentiment = results[2] if not isinstance(results[2], Exception) else None
            
            # Umumiy sentiment hisoblash
            overall_sentiment = await self._calculate_overall_sentiment(
                news_sentiment, social_sentiment, ai_sentiment
            )
            
            return ProcessingResult(
                success=True,
                data=overall_sentiment,
                confidence=overall_sentiment.confidence
            )
            
        except Exception as e:
            logger.error(f"Sentiment tahlilida xato: {e}")
            return ProcessingResult(
                success=False,
                error=str(e)
            )

    async def _analyze_news_sentiment(self, symbol: str, timeframe: str) -> Optional[SentimentScore]:
        """Yangilik sentiment tahlili"""
        try:
            # Yangiliklar olish
            news_data = await self.news_client.get_crypto_news(symbol, timeframe)
            
            if not news_data or not news_data.get('articles'):
                logger.warning(f"Yangilik topilmadi - {symbol}")
                return None
            
            # Har bir yangilik uchun sentiment tahlil
            sentiment_scores = []
            
            for article in news_data['articles'][:10]:  # Faqat oxirgi 10 ta yangilik
                title = article.get('title', '')
                description = article.get('description', '')
                content = f"{title} {description}"
                
                # AI orqali sentiment tahlil
                ai_score = await self._get_ai_sentiment(content)
                
                if ai_score:
                    sentiment_scores.append(ai_score)
            
            if not sentiment_scores:
                return None
            
            # O'rtacha sentiment
            avg_score = statistics.mean(sentiment_scores)
            confidence = min(len(sentiment_scores) / 10, 1.0)  # Ko'proq yangilik - yuqori ishonch
            
            return SentimentScore(
                score=avg_score,
                confidence=confidence,
                source="news",
                timestamp=datetime.now(),
                raw_data={"article_count": len(sentiment_scores)}
            )
            
        except Exception as e:
            logger.error(f"Yangilik sentiment tahlilida xato: {e}")
            return None

    async def _analyze_social_sentiment(self, symbol: str, timeframe: str) -> Optional[SentimentScore]:
        """Ijtimoiy tarmoq sentiment tahlili"""
        try:
            # Reddit ma'lumotlari
            reddit_data = await self.reddit_client.get_crypto_discussions(symbol, timeframe)
            
            if not reddit_data:
                logger.warning(f"Reddit ma'lumotlari topilmadi - {symbol}")
                return None
            
            # Postlar va komentariyalar sentiment tahlili
            sentiment_scores = []
            
            for post in reddit_data[:20]:  # Faqat oxirgi 20 ta post
                content = post.get('title', '') + " " + post.get('text', '')
                
                # Keyword-based sentiment tahlil
                keyword_score = self._analyze_keywords(content)
                
                if keyword_score is not None:
                    sentiment_scores.append(keyword_score)
            
            if not sentiment_scores:
                return None
            
            # O'rtacha sentiment
            avg_score = statistics.mean(sentiment_scores)
            confidence = min(len(sentiment_scores) / 20, 1.0)
            
            return SentimentScore(
                score=avg_score,
                confidence=confidence,
                source="social",
                timestamp=datetime.now(),
                raw_data={"post_count": len(sentiment_scores)}
            )
            
        except Exception as e:
            logger.error(f"Ijtimoiy sentiment tahlilida xato: {e}")
            return None

    async def _analyze_ai_sentiment(self, symbol: str, timeframe: str) -> Optional[SentimentScore]:
        """AI sentiment tahlili"""
        try:
            # Market ma'lumotlari va yangiliklar asosida AI sentiment
            market_context = await self._get_market_context(symbol)
            
            if not market_context:
                return None
            
            # AI clientlar orqali sentiment tahlil
            ai_sentiment = await self._get_ai_sentiment_with_fallback(market_context)
            
            if ai_sentiment is None:
                return None
            
            return SentimentScore(
                score=ai_sentiment,
                confidence=0.8,  # AI sentiment yuqori ishonch
                source="ai",
                timestamp=datetime.now(),
                raw_data={"context_length": len(market_context)}
            )
            
        except Exception as e:
            logger.error(f"AI sentiment tahlilida xato: {e}")
            return None

    async def _get_ai_sentiment(self, text: str) -> Optional[float]:
        """Bitta matn uchun AI sentiment"""
        try:
            # HuggingFace asosiy
            hf_result = await self.ai_clients['huggingface'].analyze_sentiment(text)
            
            if hf_result and hf_result.get('success'):
                return self._normalize_sentiment_score(hf_result['data'])
            
            # Gemini fallback
            gemini_result = await self.ai_clients['gemini'].analyze_sentiment(text)
            
            if gemini_result and gemini_result.get('success'):
                return self._normalize_sentiment_score(gemini_result['data'])
            
            # Claude fallback
            claude_result = await self.ai_clients['claude'].analyze_sentiment(text)
            
            if claude_result and claude_result.get('success'):
                return self._normalize_sentiment_score(claude_result['data'])
            
            return None
            
        except Exception as e:
            logger.error(f"AI sentiment olishda xato: {e}")
            return None

    async def _get_ai_sentiment_with_fallback(self, context: str) -> Optional[float]:
        """Fallback bilan AI sentiment"""
        try:
            # Fallback ketma-ketligi: HuggingFace -> Gemini -> Claude
            fallback_order = ['huggingface', 'gemini', 'claude']
            
            for client_name in fallback_order:
                try:
                    client = self.ai_clients[client_name]
                    result = await client.analyze_market_sentiment(context)
                    
                    if result and result.get('success'):
                        sentiment_score = self._normalize_sentiment_score(result['data'])
                        
                        if sentiment_score is not None:
                            logger.info(f"AI sentiment olindi - {client_name}: {sentiment_score}")
                            return sentiment_score
                        
                except Exception as e:
                    logger.warning(f"{client_name} sentiment tahlilida xato: {e}")
                    continue
            
            logger.warning("Barcha AI clientlar sentiment tahlilida ishlamadi")
            return None
            
        except Exception as e:
            logger.error(f"AI sentiment fallback xatosi: {e}")
            return None

    def _analyze_keywords(self, text: str) -> Optional[float]:
        """Keyword-based sentiment tahlil"""
        try:
            if not text:
                return None
            
            text_lower = text.lower()
            
            # Har bir toifadagi so'zlar sonini hisoblash
            bullish_count = sum(1 for word in self.sentiment_keywords['bullish'] if word in text_lower)
            bearish_count = sum(1 for word in self.sentiment_keywords['bearish'] if word in text_lower)
            neutral_count = sum(1 for word in self.sentiment_keywords['neutral'] if word in text_lower)
            
            total_count = bullish_count + bearish_count + neutral_count
            
            if total_count == 0:
                return 0.0  # Neutral
            
            # Sentiment ball hisoblash
            bullish_score = bullish_count / total_count
            bearish_score = bearish_count / total_count
            neutral_score = neutral_count / total_count
            
            # Final sentiment (-1.0 dan 1.0 gacha)
            sentiment = bullish_score - bearish_score
            
            return max(-1.0, min(1.0, sentiment))
            
        except Exception as e:
            logger.error(f"Keyword sentiment tahlilida xato: {e}")
            return None

    def _normalize_sentiment_score(self, ai_result: Dict) -> Optional[float]:
        """AI natijasini normalizatsiya qilish"""
        try:
            if not ai_result:
                return None
            
            # Turli AI service formatlarini qo'llab-quvvatlash
            if 'sentiment' in ai_result:
                sentiment = ai_result['sentiment']
                
                if isinstance(sentiment, str):
                    # String formatdagi sentiment
                    if sentiment.lower() in ['positive', 'bullish']:
                        return 0.5
                    elif sentiment.lower() in ['negative', 'bearish']:
                        return -0.5
                    else:
                        return 0.0
                
                elif isinstance(sentiment, (int, float)):
                    # Raqamli sentiment
                    return max(-1.0, min(1.0, float(sentiment)))
            
            if 'score' in ai_result:
                score = ai_result['score']
                return max(-1.0, min(1.0, float(score)))
            
            if 'label' in ai_result:
                label = ai_result['label'].lower()
                confidence = ai_result.get('confidence', 0.5)
                
                if 'positive' in label:
                    return confidence
                elif 'negative' in label:
                    return -confidence
                else:
                    return 0.0
            
            return None
            
        except Exception as e:
            logger.error(f"Sentiment normalizatsiya xatosi: {e}")
            return None

    async def _get_market_context(self, symbol: str) -> Optional[str]:
        """Market kontekstini olish"""
        try:
            # Yangiliklar
            news_data = await self.news_client.get_crypto_news(symbol, "24h")
            
            context_parts = []
            
            if news_data and news_data.get('articles'):
                context_parts.append("Recent news:")
                for article in news_data['articles'][:5]:
                    title = article.get('title', '')
                    if title:
                        context_parts.append(f"- {title}")
            
            # Reddit ma'lumotlari
            reddit_data = await self.reddit_client.get_crypto_discussions(symbol, "24h")
            
            if reddit_data:
                context_parts.append("\nCommunity discussions:")
                for post in reddit_data[:3]:
                    title = post.get('title', '')
                    if title:
                        context_parts.append(f"- {title}")
            
            if not context_parts:
                return None
            
            # Context yaratish
            context = f"Market analysis for {symbol}:\n" + "\n".join(context_parts)
            
            # Uzunlik cheklash (AI limitlari uchun)
            if len(context) > 2000:
                context = context[:2000] + "..."
            
            return context
            
        except Exception as e:
            logger.error(f"Market kontekst olishda xato: {e}")
            return None

    async def _calculate_overall_sentiment(self, news_sentiment: Optional[SentimentScore], 
                                         social_sentiment: Optional[SentimentScore], 
                                         ai_sentiment: Optional[SentimentScore]) -> SentimentAnalysis:
        """Umumiy sentiment hisoblash"""
        try:
            individual_scores = []
            weighted_scores = []
            total_weight = 0
            
            # Har bir sentiment manbasi uchun
            if news_sentiment:
                individual_scores.append(news_sentiment)
                weighted_scores.append(news_sentiment.score * self.sentiment_weights['news'])
                total_weight += self.sentiment_weights['news']
            
            if social_sentiment:
                individual_scores.append(social_sentiment)
                weighted_scores.append(social_sentiment.score * self.sentiment_weights['social'])
                total_weight += self.sentiment_weights['social']
            
            if ai_sentiment:
                individual_scores.append(ai_sentiment)
                weighted_scores.append(ai_sentiment.score * self.sentiment_weights['ai'])
                total_weight += self.sentiment_weights['ai']
            
            # Umumiy sentiment
            if total_weight == 0:
                overall_score = 0.0
                confidence = 0.0
            else:
                overall_score = sum(weighted_scores) / total_weight
                confidence = min(total_weight, 1.0)
            
            # Sentiment label
            if overall_score > 0.3:
                sentiment_label = "BULLISH"
            elif overall_score < -0.3:
                sentiment_label = "BEARISH"
            else:
                sentiment_label = "NEUTRAL"
            
            # Signal kuchi
            if abs(overall_score) > 0.6 and confidence > 0.7:
                signal_strength = "STRONG"
            elif abs(overall_score) > 0.3 and confidence > 0.5:
                signal_strength = "MODERATE"
            else:
                signal_strength = "WEAK"
            
            return SentimentAnalysis(
                overall_score=overall_score,
                confidence=confidence,
                sentiment_label=sentiment_label,
                individual_scores=individual_scores,
                news_sentiment=news_sentiment.score if news_sentiment else None,
                social_sentiment=social_sentiment.score if social_sentiment else None,
                ai_sentiment=ai_sentiment.score if ai_sentiment else None,
                signal_strength=signal_strength,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Umumiy sentiment hisoblashda xato: {e}")
            return SentimentAnalysis(
                overall_score=0.0,
                confidence=0.0,
                sentiment_label="NEUTRAL",
                individual_scores=[],
                signal_strength="WEAK",
                timestamp=datetime.now(),
                error=str(e)
            )

    def validate_input(self, data: Dict) -> bool:
        """Kirish ma'lumotlarini tekshirish"""
        try:
            if not data:
                return False
            
            # Symbol majburiy
            if 'symbol' not in data or not data['symbol']:
                logger.error("Symbol ko'rsatilmagan")
                return False
            
            # Timeframe ixtiyoriy
            if 'timeframe' in data and data['timeframe'] not in ['1h', '4h', '24h', '7d']:
                logger.error(f"Noto'g'ri timeframe: {data['timeframe']}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Input validatsiya xatosi: {e}")
            return False

    async def process(self, data: Dict) -> ProcessingResult:
        """Asosiy qayta ishlash methodi"""
        try:
            # Input validatsiyasi
            if not self.validate_input(data):
                return ProcessingResult(
                    success=False,
                    error="Noto'g'ri input ma'lumotlari"
                )
            
            # Sentiment tahlil
            symbol = data['symbol']
            timeframe = data.get('timeframe', '24h')
            
            result = await self.analyze_sentiment(symbol, timeframe)
            
            if result.success:
                logger.info(f"Sentiment tahlil yakunlandi - {symbol}: {result.data.sentiment_label}")
            else:
                logger.error(f"Sentiment tahlil xatosi - {symbol}: {result.error}")
            
            return result
            
        except Exception as e:
            logger.error(f"Sentiment processor xatosi: {e}")
            return ProcessingResult(
                success=False,
                error=str(e)
            )

    async def get_sentiment_summary(self, symbol: str) -> Dict:
        """Sentiment xulosa olish"""
        try:
            result = await self.analyze_sentiment(symbol)
            
            if not result.success:
                return {
                    'success': False,
                    'error': result.error
                }
            
            sentiment = result.data
            
            return {
                'success': True,
                'symbol': symbol,
                'sentiment_label': sentiment.sentiment_label,
                'score': round(sentiment.overall_score, 3),
                'confidence': round(sentiment.confidence, 3),
                'signal_strength': sentiment.signal_strength,
                'sources': {
                    'news': sentiment.news_sentiment,
                    'social': sentiment.social_sentiment,
                    'ai': sentiment.ai_sentiment
                },
                'timestamp': sentiment.timestamp.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Sentiment xulosa xatosi: {e}")
            return {
                'success': False,
                'error': str(e)
            }

# Singleton instance
sentiment_analyzer = SentimentAnalyzer()

# Export
__all__ = ['SentimentAnalyzer', 'SentimentAnalysis', 'SentimentScore', 'sentiment_analyzer']

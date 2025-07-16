import asyncio
import aiohttp
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
from utils.logger import get_logger
from utils.rate_limiter import RateLimiter
from utils.error_handler import handle_api_error
from utils.retry_handler import retry_async
from config.config import APIConfig

logger = get_logger(__name__)

@dataclass
class NewsArticle:
    """Yangilik maqolasi ma'lumotlari"""
    title: str
    description: str
    url: str
    source: str
    published_at: str
    sentiment_score: float = 0.0
    relevance_score: float = 0.0
    keywords: List[str] = None
    
    def __post_init__(self):
        if self.keywords is None:
            self.keywords = []

@dataclass
class NewsResponse:
    """NewsAPI javob formati"""
    success: bool
    articles: List[NewsArticle] = None
    total_results: int = 0
    error: Optional[str] = None
    rate_limit_remaining: int = 0
    source: str = "NewsAPI"
    
    def __post_init__(self):
        if self.articles is None:
            self.articles = []

class NewsAPIClient:
    """NewsAPI bilan ishlash uchun mijoz"""
    
    def __init__(self, api_key: str, config: APIConfig):
        self.api_key = api_key
        self.base_url = "https://newsapi.org/v2"
        self.config = config
        self.rate_limiter = RateLimiter(calls=1000, period=86400)  # 1000 requests per day
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Kriptovalyuta bilan bog'liq kalit so'zlar
        self.crypto_keywords = [
            "bitcoin", "ethereum", "crypto", "cryptocurrency", "blockchain",
            "defi", "nft", "trading", "altcoin", "binance", "coinbase",
            "uniswap", "dex", "yield farming", "staking", "token"
        ]
        
        # Forex bilan bog'liq kalit so'zlar
        self.forex_keywords = [
            "forex", "currency", "usd", "eur", "gbp", "jpy", "aud",
            "fed", "ecb", "central bank", "interest rate", "inflation",
            "economic", "trade war", "recession", "gdp"
        ]
        
        logger.info("NewsAPI client yaratildi")
    
    async def __aenter__(self):
        """Async context manager kirish"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.timeout)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager chiqish"""
        if self.session:
            await self.session.close()
    
    @retry_async(max_retries=3, delay=2)
    async def make_request(self, endpoint: str, params: Dict = None) -> Dict:
        """Asosiy so'rov yuborish methodi"""
        try:
            await self.rate_limiter.wait()
            
            if not self.session:
                raise Exception("Session ochilmagan")
            
            url = f"{self.base_url}/{endpoint}"
            
            # Standart parametrlar
            default_params = {
                "apiKey": self.api_key,
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": 100
            }
            
            if params:
                default_params.update(params)
            
            async with self.session.get(url, params=default_params) as response:
                data = await response.json()
                
                if response.status == 200:
                    return {
                        "success": True,
                        "data": data,
                        "rate_limit_remaining": int(response.headers.get("X-RateLimit-Remaining", 0))
                    }
                elif response.status == 429:
                    logger.warning("NewsAPI rate limit oshib ketdi")
                    return {
                        "success": False,
                        "error": "Rate limit exceeded",
                        "rate_limit_remaining": 0
                    }
                else:
                    logger.error(f"NewsAPI xato: {response.status} - {data}")
                    return {
                        "success": False,
                        "error": f"HTTP {response.status}: {data.get('message', 'Unknown error')}"
                    }
                    
        except asyncio.TimeoutError:
            logger.error("NewsAPI so'rov vaqti tugadi")
            return {"success": False, "error": "Request timeout"}
        except Exception as e:
            logger.error(f"NewsAPI so'rov xatosi: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_crypto_news(self, 
                            hours_back: int = 24,
                            sources: List[str] = None) -> NewsResponse:
        """Kriptovalyuta yangiliklar olish"""
        try:
            # Sanani hisoblash
            from_date = (datetime.now() - timedelta(hours=hours_back)).strftime("%Y-%m-%dT%H:%M:%S")
            
            # Qidiruv so'zlari
            query = " OR ".join(self.crypto_keywords)
            
            params = {
                "q": query,
                "from": from_date,
                "sortBy": "publishedAt"
            }
            
            # Manba belgilash
            if sources:
                params["sources"] = ",".join(sources)
            
            result = await self.make_request("everything", params)
            
            if result["success"]:
                articles = []
                for article_data in result["data"].get("articles", []):
                    if self._is_relevant_article(article_data, self.crypto_keywords):
                        article = NewsArticle(
                            title=article_data.get("title", ""),
                            description=article_data.get("description", ""),
                            url=article_data.get("url", ""),
                            source=article_data.get("source", {}).get("name", ""),
                            published_at=article_data.get("publishedAt", ""),
                            keywords=self._extract_keywords(article_data, self.crypto_keywords)
                        )
                        articles.append(article)
                
                logger.info(f"Crypto yangiliklar olindi: {len(articles)} ta")
                return NewsResponse(
                    success=True,
                    articles=articles,
                    total_results=len(articles),
                    rate_limit_remaining=result.get("rate_limit_remaining", 0)
                )
            else:
                return NewsResponse(
                    success=False,
                    error=result.get("error", "Unknown error")
                )
                
        except Exception as e:
            logger.error(f"Crypto yangiliklar olishda xato: {e}")
            return NewsResponse(success=False, error=str(e))
    
    async def get_forex_news(self, 
                           hours_back: int = 24,
                           sources: List[str] = None) -> NewsResponse:
        """Forex yangiliklar olish"""
        try:
            # Sanani hisoblash
            from_date = (datetime.now() - timedelta(hours=hours_back)).strftime("%Y-%m-%dT%H:%M:%S")
            
            # Qidiruv so'zlari
            query = " OR ".join(self.forex_keywords)
            
            params = {
                "q": query,
                "from": from_date,
                "sortBy": "publishedAt"
            }
            
            # Manba belgilash
            if sources:
                params["sources"] = ",".join(sources)
            
            result = await self.make_request("everything", params)
            
            if result["success"]:
                articles = []
                for article_data in result["data"].get("articles", []):
                    if self._is_relevant_article(article_data, self.forex_keywords):
                        article = NewsArticle(
                            title=article_data.get("title", ""),
                            description=article_data.get("description", ""),
                            url=article_data.get("url", ""),
                            source=article_data.get("source", {}).get("name", ""),
                            published_at=article_data.get("publishedAt", ""),
                            keywords=self._extract_keywords(article_data, self.forex_keywords)
                        )
                        articles.append(article)
                
                logger.info(f"Forex yangiliklar olindi: {len(articles)} ta")
                return NewsResponse(
                    success=True,
                    articles=articles,
                    total_results=len(articles),
                    rate_limit_remaining=result.get("rate_limit_remaining", 0)
                )
            else:
                return NewsResponse(
                    success=False,
                    error=result.get("error", "Unknown error")
                )
                
        except Exception as e:
            logger.error(f"Forex yangiliklar olishda xato: {e}")
            return NewsResponse(success=False, error=str(e))
    
    async def get_market_news(self, 
                            query: str,
                            hours_back: int = 24,
                            sources: List[str] = None) -> NewsResponse:
        """Bozor yangiliklar olish (custom query)"""
        try:
            # Sanani hisoblash
            from_date = (datetime.now() - timedelta(hours=hours_back)).strftime("%Y-%m-%dT%H:%M:%S")
            
            params = {
                "q": query,
                "from": from_date,
                "sortBy": "publishedAt"
            }
            
            # Manba belgilash
            if sources:
                params["sources"] = ",".join(sources)
            
            result = await self.make_request("everything", params)
            
            if result["success"]:
                articles = []
                for article_data in result["data"].get("articles", []):
                    article = NewsArticle(
                        title=article_data.get("title", ""),
                        description=article_data.get("description", ""),
                        url=article_data.get("url", ""),
                        source=article_data.get("source", {}).get("name", ""),
                        published_at=article_data.get("publishedAt", ""),
                        keywords=self._extract_keywords(article_data, query.split())
                    )
                    articles.append(article)
                
                logger.info(f"Market yangiliklar olindi: {len(articles)} ta")
                return NewsResponse(
                    success=True,
                    articles=articles,
                    total_results=len(articles),
                    rate_limit_remaining=result.get("rate_limit_remaining", 0)
                )
            else:
                return NewsResponse(
                    success=False,
                    error=result.get("error", "Unknown error")
                )
                
        except Exception as e:
            logger.error(f"Market yangiliklar olishda xato: {e}")
            return NewsResponse(success=False, error=str(e))
    
    async def get_top_headlines(self, 
                              category: str = "business",
                              country: str = "us") -> NewsResponse:
        """Eng muhim yangiliklar olish"""
        try:
            params = {
                "category": category,
                "country": country
            }
            
            result = await self.make_request("top-headlines", params)
            
            if result["success"]:
                articles = []
                for article_data in result["data"].get("articles", []):
                    article = NewsArticle(
                        title=article_data.get("title", ""),
                        description=article_data.get("description", ""),
                        url=article_data.get("url", ""),
                        source=article_data.get("source", {}).get("name", ""),
                        published_at=article_data.get("publishedAt", "")
                    )
                    articles.append(article)
                
                logger.info(f"Top headlines olindi: {len(articles)} ta")
                return NewsResponse(
                    success=True,
                    articles=articles,
                    total_results=len(articles),
                    rate_limit_remaining=result.get("rate_limit_remaining", 0)
                )
            else:
                return NewsResponse(
                    success=False,
                    error=result.get("error", "Unknown error")
                )
                
        except Exception as e:
            logger.error(f"Top headlines olishda xato: {e}")
            return NewsResponse(success=False, error=str(e))
    
    def _is_relevant_article(self, article_data: Dict, keywords: List[str]) -> bool:
        """Maqolaning tegishli ekanligini tekshirish"""
        try:
            title = article_data.get("title", "").lower()
            description = article_data.get("description", "").lower()
            
            # Kalit so'zlar borligini tekshirish
            for keyword in keywords:
                if keyword.lower() in title or keyword.lower() in description:
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Maqola tegishliligini tekshirishda xato: {e}")
            return False
    
    def _extract_keywords(self, article_data: Dict, base_keywords: List[str]) -> List[str]:
        """Maqoladan kalit so'zlarni ajratish"""
        try:
            title = article_data.get("title", "").lower()
            description = article_data.get("description", "").lower()
            
            found_keywords = []
            
            for keyword in base_keywords:
                if keyword.lower() in title or keyword.lower() in description:
                    found_keywords.append(keyword)
            
            return found_keywords
            
        except Exception as e:
            logger.error(f"Kalit so'zlarni ajratishda xato: {e}")
            return []
    
    def _calculate_sentiment_score(self, article_data: Dict) -> float:
        """Maqolaning sentiment scorini hisoblash (oddiy usul)"""
        try:
            title = article_data.get("title", "").lower()
            description = article_data.get("description", "").lower()
            
            # Ijobiy so'zlar
            positive_words = [
                "rise", "gain", "bull", "up", "growth", "profit", "success",
                "breakthrough", "adoption", "rally", "surge", "boom"
            ]
            
            # Salbiy so'zlar
            negative_words = [
                "fall", "drop", "bear", "down", "loss", "crash", "decline",
                "fear", "risk", "hack", "ban", "regulation", "dump"
            ]
            
            positive_count = sum(1 for word in positive_words if word in title or word in description)
            negative_count = sum(1 for word in negative_words if word in title or word in description)
            
            if positive_count + negative_count == 0:
                return 0.0
            
            return (positive_count - negative_count) / (positive_count + negative_count)
            
        except Exception as e:
            logger.error(f"Sentiment score hisoblashda xato: {e}")
            return 0.0
    
    async def get_sources(self) -> List[Dict]:
        """Mavjud yangilik manbalarini olish"""
        try:
            result = await self.make_request("sources")
            
            if result["success"]:
                sources = result["data"].get("sources", [])
                logger.info(f"Manbalar olindi: {len(sources)} ta")
                return sources
            else:
                logger.error(f"Manbalar olishda xato: {result.get('error')}")
                return []
                
        except Exception as e:
            logger.error(f"Manbalar olishda xato: {e}")
            return []

class NewsFallbackManager:
    """NewsAPI uchun fallback boshqaruvchi"""
    
    def __init__(self, primary_client: NewsAPIClient):
        self.primary_client = primary_client
        self.fallback_methods = [
            self._get_general_market_news,
            self._get_cached_news,
            self._get_minimal_news
        ]
        
        logger.info("News fallback manager yaratildi")
    
    async def get_news_with_fallback(self, 
                                   news_type: str = "crypto",
                                   hours_back: int = 24) -> NewsResponse:
        """Fallback bilan yangiliklar olish"""
        try:
            # Asosiy NewsAPI orqali
            if news_type == "crypto":
                result = await self.primary_client.get_crypto_news(hours_back)
            elif news_type == "forex":
                result = await self.primary_client.get_forex_news(hours_back)
            else:
                result = await self.primary_client.get_market_news(news_type, hours_back)
            
            if result.success and result.articles:
                return result
            
            logger.warning("Asosiy NewsAPI ishlamadi, fallback ishlatiladi")
            
            # Fallback metodlari
            for i, fallback_method in enumerate(self.fallback_methods):
                try:
                    result = await fallback_method(news_type, hours_back)
                    if result.success:
                        logger.info(f"Fallback {i+1} muvaffaqiyatli")
                        return result
                except Exception as e:
                    logger.error(f"Fallback {i+1} xato: {e}")
                    continue
            
            logger.error("Barcha fallback metodlari ishlamadi")
            return NewsResponse(
                success=False,
                error="Barcha yangilik manbalar ishlamadi"
            )
            
        except Exception as e:
            logger.error(f"News fallback xato: {e}")
            return NewsResponse(success=False, error=str(e))
    
    async def _get_general_market_news(self, news_type: str, hours_back: int) -> NewsResponse:
        """Umumiy bozor yangiliklar (fallback 1)"""
        try:
            # Top headlines orqali
            result = await self.primary_client.get_top_headlines(category="business")
            
            if result.success:
                # Tegishli maqolalarni filtrlash
                filtered_articles = []
                for article in result.articles:
                    if news_type == "crypto":
                        if any(keyword in article.title.lower() or keyword in article.description.lower() 
                               for keyword in self.primary_client.crypto_keywords):
                            filtered_articles.append(article)
                    elif news_type == "forex":
                        if any(keyword in article.title.lower() or keyword in article.description.lower() 
                               for keyword in self.primary_client.forex_keywords):
                            filtered_articles.append(article)
                
                return NewsResponse(
                    success=True,
                    articles=filtered_articles,
                    total_results=len(filtered_articles),
                    source="NewsAPI-Headlines"
                )
            
            return NewsResponse(success=False, error="Headlines olishda xato")
            
        except Exception as e:
            logger.error(f"General market news xato: {e}")
            return NewsResponse(success=False, error=str(e))
    
    async def _get_cached_news(self, news_type: str, hours_back: int) -> NewsResponse:
        """Kesh qilingan yangiliklar (fallback 2)"""
        try:
            # Bu yerda kesh qilingan yangiliklar bo'lishi mumkin
            # Hozircha bo'sh qaytaramiz
            return NewsResponse(
                success=True,
                articles=[],
                total_results=0,
                source="Cache"
            )
            
        except Exception as e:
            logger.error(f"Cached news xato: {e}")
            return NewsResponse(success=False, error=str(e))
    
    async def _get_minimal_news(self, news_type: str, hours_back: int) -> NewsResponse:
        """Minimal yangiliklar (fallback 3)"""
        try:
            # Minimal ma'lumot bilan yangilik yaratish
            minimal_article = NewsArticle(
                title=f"Market yangiliklar ({news_type})",
                description=f"Yangilik servislari vaqtinchalik ishlamayapti",
                url="",
                source="System",
                published_at=datetime.now().isoformat(),
                keywords=[news_type]
            )
            
            return NewsResponse(
                success=True,
                articles=[minimal_article],
                total_results=1,
                source="System"
            )
            
        except Exception as e:
            logger.error(f"Minimal news xato: {e}")
            return NewsResponse(success=False, error=str(e))

# Sinov uchun
async def test_news_client():
    """NewsAPI client sinovi"""
    from config.api_keys import get_news_api_key
    from config.config import get_api_config
    
    try:
        api_key = get_news_api_key()
        config = get_api_config("news")
        
        async with NewsAPIClient(api_key, config) as client:
            # Crypto yangiliklar
            crypto_news = await client.get_crypto_news(hours_back=24)
            print(f"Crypto yangiliklar: {len(crypto_news.articles)} ta")
            
            # Forex yangiliklar
            forex_news = await client.get_forex_news(hours_back=24)
            print(f"Forex yangiliklar: {len(forex_news.articles)} ta")
            
            # Fallback testi
            fallback_manager = NewsFallbackManager(client)
            fallback_news = await fallback_manager.get_news_with_fallback("crypto")
            print(f"Fallback yangiliklar: {len(fallback_news.articles)} ta")
            
    except Exception as e:
        print(f"Test xato: {e}")

if __name__ == "__main__":
    asyncio.run(test_news_client())

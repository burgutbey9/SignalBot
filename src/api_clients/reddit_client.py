import asyncio
import aiohttp
import praw
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import re
from utils.logger import get_logger
from utils.rate_limiter import RateLimiter
from utils.error_handler import handle_api_error
from utils.retry_handler import retry_async

logger = get_logger(__name__)

@dataclass
class RedditPost:
    """Reddit post ma'lumotlari"""
    title: str
    content: str
    score: int
    upvote_ratio: float
    num_comments: int
    created_utc: datetime
    subreddit: str
    url: str
    author: str
    awards: int = 0
    sentiment_score: float = 0.0

@dataclass
class RedditComment:
    """Reddit comment ma'lumotlari"""
    body: str
    score: int
    created_utc: datetime
    author: str
    replies_count: int = 0
    sentiment_score: float = 0.0

@dataclass
class RedditAPIResponse:
    """Reddit API javob formati"""
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    rate_limit_remaining: int = 0
    total_posts: int = 0
    
class RedditClient:
    """Reddit API client - sentiment analysis uchun"""
    
    def __init__(self, client_id: str, client_secret: str, user_agent: str = "AI_OrderFlow_Bot/1.0"):
        self.client_id = client_id
        self.client_secret = client_secret
        self.user_agent = user_agent
        self.rate_limiter = RateLimiter(calls=60, period=60)  # 60 so'rov/minut
        self.reddit_client: Optional[praw.Reddit] = None
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Kriptovalyuta subreddit ro'yxati
        self.crypto_subreddits = [
            "CryptoCurrency",
            "CryptoMarkets", 
            "Bitcoin",
            "ethereum",
            "CryptoMoonShots",
            "CryptoNews",
            "defi",
            "NFT",
            "dogecoin",
            "cardano",
            "solana"
        ]
        
        # Kalit so'zlar pattern
        self.bullish_keywords = [
            r'\b(moon|pump|bull|bullish|rocket|surge|rally|breakout|ATH|all.time.high)\b',
            r'\b(buy|long|hodl|accumulate|invest|opportunity)\b',
            r'\b(🚀|📈|💎|🌙|🔥|💰|⬆️)\b'
        ]
        
        self.bearish_keywords = [
            r'\b(dump|bear|bearish|crash|fall|drop|decline|dip|correction)\b',
            r'\b(sell|short|exit|panic|fear|bubble|scam)\b',
            r'\b(📉|💀|⬇️|🩸|😱|🔻)\b'
        ]
        
        logger.info("Reddit client ishga tushirildi")

    async def __aenter__(self):
        """Async context manager kirish"""
        await self.initialize_client()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager chiqish"""
        await self.close()
    
    async def initialize_client(self):
        """Reddit client va session yaratish"""
        try:
            # PRAW client yaratish
            self.reddit_client = praw.Reddit(
                client_id=self.client_id,
                client_secret=self.client_secret,
                user_agent=self.user_agent,
                timeout=30
            )
            
            # HTTP session yaratish
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
                headers={
                    'User-Agent': self.user_agent,
                    'Content-Type': 'application/json'
                }
            )
            
            logger.info("Reddit client muvaffaqiyatli ishga tushirildi")
            
        except Exception as e:
            logger.error(f"Reddit client yaratishda xato: {e}")
            raise

    async def close(self):
        """Connectionlarni yopish"""
        if self.session:
            await self.session.close()
            self.session = None
        logger.info("Reddit client yopildi")

    @retry_async(max_retries=3, delay=2)
    async def get_crypto_sentiment(self, limit: int = 50) -> RedditAPIResponse:
        """
        Kriptovalyuta sentiment ma'lumotlarini olish
        
        Args:
            limit: Post sonini cheklash
            
        Returns:
            RedditAPIResponse: Sentiment ma'lumotlari
        """
        try:
            await self.rate_limiter.wait()
            
            if not self.reddit_client:
                await self.initialize_client()
            
            all_posts = []
            
            # Har bir subreddit dan postlarni olish
            for subreddit_name in self.crypto_subreddits:
                try:
                    subreddit = self.reddit_client.subreddit(subreddit_name)
                    
                    # Hot postlar olish
                    for submission in subreddit.hot(limit=limit//len(self.crypto_subreddits)):
                        post = RedditPost(
                            title=submission.title,
                            content=submission.selftext,
                            score=submission.score,
                            upvote_ratio=submission.upvote_ratio,
                            num_comments=submission.num_comments,
                            created_utc=datetime.fromtimestamp(submission.created_utc),
                            subreddit=subreddit_name,
                            url=submission.url,
                            author=str(submission.author) if submission.author else "[deleted]",
                            awards=submission.total_awards_received
                        )
                        
                        # Sentiment hisoblash
                        post.sentiment_score = self._calculate_sentiment(post.title + " " + post.content)
                        all_posts.append(post)
                        
                except Exception as e:
                    logger.warning(f"Subreddit {subreddit_name} dan ma'lumot olishda xato: {e}")
                    continue
            
            # Postlarni score bo'yicha saralash
            all_posts.sort(key=lambda x: x.score, reverse=True)
            
            logger.info(f"Reddit dan {len(all_posts)} ta post olindi")
            
            return RedditAPIResponse(
                success=True,
                data=all_posts,
                total_posts=len(all_posts)
            )
            
        except Exception as e:
            logger.error(f"Reddit sentiment olishda xato: {e}")
            return RedditAPIResponse(
                success=False,
                error=str(e)
            )

    @retry_async(max_retries=3, delay=2)
    async def get_token_mentions(self, token_symbol: str, limit: int = 100) -> RedditAPIResponse:
        """
        Muayyan token haqida mention larni olish
        
        Args:
            token_symbol: Token belgisi (masalan: BTC, ETH)
            limit: Mention sonini cheklash
            
        Returns:
            RedditAPIResponse: Token mentions
        """
        try:
            await self.rate_limiter.wait()
            
            if not self.reddit_client:
                await self.initialize_client()
            
            mentions = []
            search_terms = [token_symbol.upper(), token_symbol.lower(), f"${token_symbol.upper()}"]
            
            for subreddit_name in self.crypto_subreddits:
                try:
                    subreddit = self.reddit_client.subreddit(subreddit_name)
                    
                    # Har bir search term uchun qidirish
                    for term in search_terms:
                        for submission in subreddit.search(term, limit=limit//len(self.crypto_subreddits)//len(search_terms)):
                            # Faqat so'nggi 24 soat ichidagi postlar
                            if datetime.fromtimestamp(submission.created_utc) > datetime.now() - timedelta(hours=24):
                                post = RedditPost(
                                    title=submission.title,
                                    content=submission.selftext,
                                    score=submission.score,
                                    upvote_ratio=submission.upvote_ratio,
                                    num_comments=submission.num_comments,
                                    created_utc=datetime.fromtimestamp(submission.created_utc),
                                    subreddit=subreddit_name,
                                    url=submission.url,
                                    author=str(submission.author) if submission.author else "[deleted]",
                                    awards=submission.total_awards_received
                                )
                                
                                post.sentiment_score = self._calculate_sentiment(post.title + " " + post.content)
                                mentions.append(post)
                                
                except Exception as e:
                    logger.warning(f"Subreddit {subreddit_name} dan {token_symbol} qidirishda xato: {e}")
                    continue
            
            # Duplicate larni olib tashlash
            unique_mentions = []
            seen_urls = set()
            
            for mention in mentions:
                if mention.url not in seen_urls:
                    unique_mentions.append(mention)
                    seen_urls.add(mention.url)
            
            # Score bo'yicha saralash
            unique_mentions.sort(key=lambda x: x.score, reverse=True)
            
            logger.info(f"{token_symbol} uchun {len(unique_mentions)} ta mention topildi")
            
            return RedditAPIResponse(
                success=True,
                data=unique_mentions,
                total_posts=len(unique_mentions)
            )
            
        except Exception as e:
            logger.error(f"{token_symbol} mentions olishda xato: {e}")
            return RedditAPIResponse(
                success=False,
                error=str(e)
            )

    @retry_async(max_retries=3, delay=2)
    async def get_trending_crypto_topics(self, limit: int = 20) -> RedditAPIResponse:
        """
        Trending kripto mavzularni olish
        
        Args:
            limit: Mavzu sonini cheklash
            
        Returns:
            RedditAPIResponse: Trending mavzular
        """
        try:
            await self.rate_limiter.wait()
            
            if not self.reddit_client:
                await self.initialize_client()
            
            trending_posts = []
            
            # CryptoCurrency subreddit dan hot postlar
            subreddit = self.reddit_client.subreddit("CryptoCurrency")
            
            for submission in subreddit.hot(limit=limit):
                # Faqat oxirgi 6 soat ichidagi postlar
                if datetime.fromtimestamp(submission.created_utc) > datetime.now() - timedelta(hours=6):
                    post = RedditPost(
                        title=submission.title,
                        content=submission.selftext,
                        score=submission.score,
                        upvote_ratio=submission.upvote_ratio,
                        num_comments=submission.num_comments,
                        created_utc=datetime.fromtimestamp(submission.created_utc),
                        subreddit="CryptoCurrency",
                        url=submission.url,
                        author=str(submission.author) if submission.author else "[deleted]",
                        awards=submission.total_awards_received
                    )
                    
                    post.sentiment_score = self._calculate_sentiment(post.title + " " + post.content)
                    trending_posts.append(post)
            
            # Score va comment ratio bo'yicha saralash
            trending_posts.sort(key=lambda x: (x.score * x.num_comments), reverse=True)
            
            logger.info(f"{len(trending_posts)} ta trending post topildi")
            
            return RedditAPIResponse(
                success=True,
                data=trending_posts,
                total_posts=len(trending_posts)
            )
            
        except Exception as e:
            logger.error(f"Trending topics olishda xato: {e}")
            return RedditAPIResponse(
                success=False,
                error=str(e)
            )

    async def get_market_sentiment_score(self) -> Dict[str, float]:
        """
        Umumiy bozor sentiment score hisoblash
        
        Returns:
            Dict: Sentiment score ma'lumotlari
        """
        try:
            sentiment_response = await self.get_crypto_sentiment(limit=100)
            
            if not sentiment_response.success:
                return {
                    "overall_sentiment": 0.0,
                    "bullish_ratio": 0.0,
                    "bearish_ratio": 0.0,
                    "neutral_ratio": 0.0,
                    "total_posts": 0
                }
            
            posts = sentiment_response.data
            
            if not posts:
                return {
                    "overall_sentiment": 0.0,
                    "bullish_ratio": 0.0,
                    "bearish_ratio": 0.0,
                    "neutral_ratio": 0.0,
                    "total_posts": 0
                }
            
            # Sentiment hisobi
            bullish_count = sum(1 for post in posts if post.sentiment_score > 0.1)
            bearish_count = sum(1 for post in posts if post.sentiment_score < -0.1)
            neutral_count = len(posts) - bullish_count - bearish_count
            
            # Weighted sentiment (score va upvote_ratio ni hisobga olgan holda)
            weighted_sentiment = sum(
                post.sentiment_score * post.score * post.upvote_ratio 
                for post in posts
            ) / sum(post.score * post.upvote_ratio for post in posts)
            
            return {
                "overall_sentiment": weighted_sentiment,
                "bullish_ratio": bullish_count / len(posts),
                "bearish_ratio": bearish_count / len(posts),
                "neutral_ratio": neutral_count / len(posts),
                "total_posts": len(posts),
                "average_score": sum(post.score for post in posts) / len(posts),
                "average_upvote_ratio": sum(post.upvote_ratio for post in posts) / len(posts)
            }
            
        except Exception as e:
            logger.error(f"Market sentiment hisoblashda xato: {e}")
            return {
                "overall_sentiment": 0.0,
                "bullish_ratio": 0.0,
                "bearish_ratio": 0.0,
                "neutral_ratio": 0.0,
                "total_posts": 0
            }

    def _calculate_sentiment(self, text: str) -> float:
        """
        Matn sentiment hisoblash (sodda keyword asosida)
        
        Args:
            text: Tahlil qilinadigan matn
            
        Returns:
            float: Sentiment score (-1 dan 1 gacha)
        """
        if not text:
            return 0.0
        
        text = text.lower()
        bullish_score = 0
        bearish_score = 0
        
        # Bullish keywords hisoblash
        for pattern in self.bullish_keywords:
            matches = re.findall(pattern, text, re.IGNORECASE)
            bullish_score += len(matches)
        
        # Bearish keywords hisoblash
        for pattern in self.bearish_keywords:
            matches = re.findall(pattern, text, re.IGNORECASE)
            bearish_score += len(matches)
        
        # Sentiment score hisoblash
        total_sentiment_words = bullish_score + bearish_score
        
        if total_sentiment_words == 0:
            return 0.0
        
        # -1 dan 1 gacha normallash
        sentiment_score = (bullish_score - bearish_score) / max(total_sentiment_words, 1)
        
        return max(-1.0, min(1.0, sentiment_score))

    async def get_post_comments(self, post_url: str, limit: int = 20) -> List[RedditComment]:
        """
        Post commentlarini olish
        
        Args:
            post_url: Post URL
            limit: Comment sonini cheklash
            
        Returns:
            List[RedditComment]: Comment ro'yxati
        """
        try:
            await self.rate_limiter.wait()
            
            if not self.reddit_client:
                await self.initialize_client()
            
            submission = self.reddit_client.submission(url=post_url)
            submission.comments.replace_more(limit=0)
            
            comments = []
            
            for comment in submission.comments.list()[:limit]:
                if hasattr(comment, 'body') and comment.body != "[deleted]":
                    reddit_comment = RedditComment(
                        body=comment.body,
                        score=comment.score,
                        created_utc=datetime.fromtimestamp(comment.created_utc),
                        author=str(comment.author) if comment.author else "[deleted]",
                        replies_count=len(comment.replies) if hasattr(comment, 'replies') else 0
                    )
                    
                    reddit_comment.sentiment_score = self._calculate_sentiment(comment.body)
                    comments.append(reddit_comment)
            
            return comments
            
        except Exception as e:
            logger.error(f"Post commentlarini olishda xato: {e}")
            return []

    async def health_check(self) -> bool:
        """
        Reddit API health tekshirish
        
        Returns:
            bool: API sog'ligini ko'rsatadi
        """
        try:
            if not self.reddit_client:
                await self.initialize_client()
            
            # Sodda test so'rovi
            subreddit = self.reddit_client.subreddit("test")
            list(subreddit.hot(limit=1))
            
            logger.info("Reddit API sog'lom")
            return True
            
        except Exception as e:
            logger.error(f"Reddit API health check xatosi: {e}")
            return False

# Fallback manager bilan integratsiya
class FallbackRedditManager:
    """Reddit API fallback manager"""
    
    def __init__(self, api_configs: List[Dict]):
        self.clients = []
        
        for config in api_configs:
            client = RedditClient(
                client_id=config['client_id'],
                client_secret=config['client_secret'],
                user_agent=config.get('user_agent', 'AI_OrderFlow_Bot/1.0')
            )
            self.clients.append(client)
        
        self.current_client_index = 0
        logger.info(f"Reddit fallback manager {len(self.clients)} ta client bilan yaratildi")

    async def execute_with_fallback(self, operation: str, **kwargs) -> RedditAPIResponse:
        """
        Fallback bilan operatsiya bajarish
        
        Args:
            operation: Bajarilishi kerak bo'lgan operatsiya
            **kwargs: Operatsiya parametrlari
            
        Returns:
            RedditAPIResponse: Operatsiya natijasi
        """
        last_error = None
        
        for i, client in enumerate(self.clients):
            try:
                async with client:
                    result = await getattr(client, operation)(**kwargs)
                    
                    if result.success:
                        if i > 0:
                            logger.warning(f"Reddit fallback ishlatildi: client {i}")
                        return result
                    else:
                        last_error = result.error
                        
            except Exception as e:
                logger.error(f"Reddit client {i} ishlamadi: {e}")
                last_error = str(e)
                continue
        
        logger.error("Barcha Reddit clientlar ishlamadi")
        return RedditAPIResponse(
            success=False,
            error=f"Barcha fallback clientlar ishlamadi. Oxirgi xato: {last_error}"
        )

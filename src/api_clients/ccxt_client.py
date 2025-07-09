import asyncio
import ccxt.async_support as ccxt
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import pandas as pd
import json
import time
from utils.logger import get_logger
from utils.rate_limiter import RateLimiter
from utils.error_handler import handle_api_error
from utils.retry_handler import retry_async

logger = get_logger(__name__)

@dataclass
class CCXTResponse:
    """CCXT API javob formatі"""
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    exchange: Optional[str] = None
    symbol: Optional[str] = None
    
@dataclass
class MarketData:
    """Market ma'lumotlari"""
    symbol: str
    price: float
    volume_24h: float
    change_24h: float
    high_24h: float
    low_24h: float
    timestamp: datetime
    exchange: str
    
@dataclass
class OrderBookData:
    """Order book ma'lumotlari"""
    symbol: str
    bids: List[List[float]]  # [[price, amount], ...]
    asks: List[List[float]]  # [[price, amount], ...]
    timestamp: datetime
    exchange: str
    
@dataclass
class TradeData:
    """Trade ma'lumotlari"""
    symbol: str
    side: str  # 'buy' or 'sell'
    amount: float
    price: float
    timestamp: datetime
    exchange: str
    
@dataclass
class OHLCVData:
    """OHLCV ma'lumotlari"""
    symbol: str
    timeframe: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    timestamp: datetime
    exchange: str

class CCXTClient:
    """CCXT client - CEX market data olish uchun"""
    
    def __init__(self, exchanges: List[str] = None):
        self.exchanges = exchanges or ['gateio', 'kucoin', 'mexc']
        self.exchange_instances = {}
        self.rate_limiters = {}
        
        # Har exchange uchun rate limiter
        self.rate_limits = {
            'gateio': {'calls': 1000, 'period': 60},
            'kucoin': {'calls': 100, 'period': 10},
            'mexc': {'calls': 200, 'period': 60}
        }
        
        # Cache uchun
        self.cache = {}
        self.cache_ttl = {
            'ticker': 5,      # 5 sekund
            'orderbook': 2,   # 2 sekund
            'trades': 10,     # 10 sekund
            'ohlcv': 60       # 1 minut
        }
        
        self.initialized = False
        logger.info(f"CCXT client yaratildi. Exchanges: {self.exchanges}")
    
    async def initialize(self):
        """Exchange instancelarini yaratish"""
        if self.initialized:
            return
        
        for exchange_name in self.exchanges:
            try:
                # Exchange class olish
                exchange_class = getattr(ccxt, exchange_name)
                
                # Exchange instance yaratish
                exchange = exchange_class({
                    'apiKey': '',  # Verifikatsiyasiz foydalanish
                    'secret': '',
                    'timeout': 30000,
                    'enableRateLimit': True,
                    'sandbox': False
                })
                
                # Markets yuklash
                await exchange.load_markets()
                
                self.exchange_instances[exchange_name] = exchange
                
                # Rate limiter yaratish
                limits = self.rate_limits.get(exchange_name, {'calls': 100, 'period': 60})
                self.rate_limiters[exchange_name] = RateLimiter(
                    calls=limits['calls'],
                    period=limits['period']
                )
                
                logger.info(f"Exchange ishga tushdi: {exchange_name}")
                
            except Exception as e:
                logger.error(f"Exchange ishga tushirishda xato ({exchange_name}): {e}")
        
        self.initialized = True
        logger.info("CCXT client to'liq ishga tushdi")
    
    async def __aenter__(self):
        """Async context manager kirish"""
        await self.initialize()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager chiqish"""
        await self.close()
    
    async def close(self):
        """Barcha exchange connectionlarni yopish"""
        for exchange in self.exchange_instances.values():
            try:
                await exchange.close()
            except Exception as e:
                logger.error(f"Exchange yopishda xato: {e}")
        
        logger.info("CCXT client yopildi")
    
    def _get_cache_key(self, exchange: str, method: str, symbol: str = None, timeframe: str = None) -> str:
        """Cache kalitini yaratish"""
        parts = [exchange, method]
        if symbol:
            parts.append(symbol)
        if timeframe:
            parts.append(timeframe)
        return ":".join(parts)
    
    def _is_cache_valid(self, cache_key: str, cache_type: str) -> bool:
        """Cache amal qilish muddatini tekshirish"""
        if cache_key not in self.cache:
            return False
        
        cached_time = self.cache[cache_key].get("timestamp", 0)
        ttl = self.cache_ttl.get(cache_type, 30)
        return (time.time() - cached_time) < ttl
    
    @retry_async(max_retries=3, delay=1)
    async def _execute_exchange_method(self, exchange_name: str, method: str, 
                                     symbol: str = None, **kwargs) -> CCXTResponse:
        """Exchange methodini bajarish"""
        try:
            if not self.initialized:
                await self.initialize()
            
            if exchange_name not in self.exchange_instances:
                return CCXTResponse(
                    success=False,
                    error=f"Exchange topilmadi: {exchange_name}",
                    exchange=exchange_name
                )
            
            exchange = self.exchange_instances[exchange_name]
            rate_limiter = self.rate_limiters[exchange_name]
            
            # Rate limiting
            await rate_limiter.wait()
            
            # Cache tekshirish
            cache_key = self._get_cache_key(exchange_name, method, symbol, kwargs.get('timeframe'))
            if self._is_cache_valid(cache_key, method):
                logger.debug(f"Cache dan qaytarildi: {cache_key}")
                return CCXTResponse(
                    success=True,
                    data=self.cache[cache_key]["data"],
                    exchange=exchange_name,
                    symbol=symbol
                )
            
            # Method bajarish
            if hasattr(exchange, method):
                method_func = getattr(exchange, method)
                
                if symbol:
                    result = await method_func(symbol, **kwargs)
                else:
                    result = await method_func(**kwargs)
                
                # Cache ga saqlash
                self.cache[cache_key] = {
                    "data": result,
                    "timestamp": time.time()
                }
                
                logger.debug(f"CCXT method bajarildi: {exchange_name}.{method}")
                return CCXTResponse(
                    success=True,
                    data=result,
                    exchange=exchange_name,
                    symbol=symbol
                )
            else:
                return CCXTResponse(
                    success=False,
                    error=f"Method topilmadi: {method}",
                    exchange=exchange_name
                )
                
        except ccxt.BaseError as e:
            logger.error(f"CCXT xato ({exchange_name}): {e}")
            return CCXTResponse(
                success=False,
                error=str(e),
                exchange=exchange_name,
                symbol=symbol
            )
        except Exception as e:
            logger.error(f"Umumiy xato ({exchange_name}): {e}")
            return CCXTResponse(
                success=False,
                error=str(e),
                exchange=exchange_name,
                symbol=symbol
            )
    
    async def get_ticker(self, symbol: str, exchange: str = None) -> CCXTResponse:
        """Ticker ma'lumotini olish"""
        try:
            if exchange:
                exchanges = [exchange]
            else:
                exchanges = list(self.exchange_instances.keys())
            
            # Birinchi muvaffaqiyatli natijani qaytarish
            for exch in exchanges:
                response = await self._execute_exchange_method(exch, 'fetch_ticker', symbol)
                if response.success:
                    ticker_data = MarketData(
                        symbol=symbol,
                        price=response.data.get('last', 0),
                        volume_24h=response.data.get('baseVolume', 0),
                        change_24h=response.data.get('change', 0),
                        high_24h=response.data.get('high', 0),
                        low_24h=response.data.get('low', 0),
                        timestamp=datetime.now(),
                        exchange=exch
                    )
                    
                    logger.debug(f"Ticker olindi: {symbol} = ${ticker_data.price} ({exch})")
                    return CCXTResponse(
                        success=True,
                        data=ticker_data,
                        exchange=exch,
                        symbol=symbol
                    )
            
            return CCXTResponse(
                success=False,
                error="Hech bir exchange dan ticker olinmadi",
                symbol=symbol
            )
            
        except Exception as e:
            logger.error(f"Ticker olishda xato: {e}")
            return CCXTResponse(
                success=False,
                error=str(e),
                symbol=symbol
            )
    
    async def get_orderbook(self, symbol: str, limit: int = 20, exchange: str = None) -> CCXTResponse:
        """Order book ma'lumotini olish"""
        try:
            if exchange:
                exchanges = [exchange]
            else:
                exchanges = list(self.exchange_instances.keys())
            
            for exch in exchanges:
                response = await self._execute_exchange_method(
                    exch, 'fetch_order_book', symbol, limit=limit
                )
                
                if response.success:
                    orderbook_data = OrderBookData(
                        symbol=symbol,
                        bids=response.data.get('bids', []),
                        asks=response.data.get('asks', []),
                        timestamp=datetime.now(),
                        exchange=exch
                    )
                    
                    logger.debug(f"Order book olindi: {symbol} - {len(orderbook_data.bids)} bids, {len(orderbook_data.asks)} asks ({exch})")
                    return CCXTResponse(
                        success=True,
                        data=orderbook_data,
                        exchange=exch,
                        symbol=symbol
                    )
            
            return CCXTResponse(
                success=False,
                error="Hech bir exchange dan order book olinmadi",
                symbol=symbol
            )
            
        except Exception as e:
            logger.error(f"Order book olishda xato: {e}")
            return CCXTResponse(
                success=False,
                error=str(e),
                symbol=symbol
            )
    
    async def get_trades(self, symbol: str, limit: int = 100, exchange: str = None) -> CCXTResponse:
        """So'nggi tradelarni olish"""
        try:
            if exchange:
                exchanges = [exchange]
            else:
                exchanges = list(self.exchange_instances.keys())
            
            for exch in exchanges:
                response = await self._execute_exchange_method(
                    exch, 'fetch_trades', symbol, limit=limit
                )
                
                if response.success:
                    trades = []
                    for trade in response.data:
                        trade_data = TradeData(
                            symbol=symbol,
                            side=trade.get('side', 'unknown'),
                            amount=trade.get('amount', 0),
                            price=trade.get('price', 0),
                            timestamp=datetime.fromtimestamp(trade.get('timestamp', 0) / 1000),
                            exchange=exch
                        )
                        trades.append(trade_data)
                    
                    logger.debug(f"Tradlar olindi: {symbol} - {len(trades)} ta trade ({exch})")
                    return CCXTResponse(
                        success=True,
                        data=trades,
                        exchange=exch,
                        symbol=symbol
                    )
            
            return CCXTResponse(
                success=False,
                error="Hech bir exchange dan tradlar olinmadi",
                symbol=symbol
            )
            
        except Exception as e:
            logger.error(f"Tradlar olishda xato: {e}")
            return CCXTResponse(
                success=False,
                error=str(e),
                symbol=symbol
            )
    
    async def get_ohlcv(self, symbol: str, timeframe: str = '1m', 
                       limit: int = 100, exchange: str = None) -> CCXTResponse:
        """OHLCV ma'lumotlarini olish"""
        try:
            if exchange:
                exchanges = [exchange]
            else:
                exchanges = list(self.exchange_instances.keys())
            
            for exch in exchanges:
                response = await self._execute_exchange_method(
                    exch, 'fetch_ohlcv', symbol, timeframe=timeframe, limit=limit
                )
                
                if response.success:
                    ohlcv_data = []
                    for candle in response.data:
                        ohlcv = OHLCVData(
                            symbol=symbol,
                            timeframe=timeframe,
                            open=candle[1],
                            high=candle[2],
                            low=candle[3],
                            close=candle[4],
                            volume=candle[5],
                            timestamp=datetime.fromtimestamp(candle[0] / 1000),
                            exchange=exch
                        )
                        ohlcv_data.append(ohlcv)
                    
                    logger.debug(f"OHLCV olindi: {symbol} {timeframe} - {len(ohlcv_data)} ta candle ({exch})")
                    return CCXTResponse(
                        success=True,
                        data=ohlcv_data,
                        exchange=exch,
                        symbol=symbol
                    )
            
            return CCXTResponse(
                success=False,
                error="Hech bir exchange dan OHLCV olinmadi",
                symbol=symbol
            )
            
        except Exception as e:
            logger.error(f"OHLCV olishda xato: {e}")
            return CCXTResponse(
                success=False,
                error=str(e),
                symbol=symbol
            )
    
    async def get_markets(self, exchange: str = None) -> CCXTResponse:
        """Mavjud marketlarni olish"""
        try:
            if exchange:
                exchanges = [exchange]
            else:
                exchanges = list(self.exchange_instances.keys())
            
            all_markets = {}
            
            for exch in exchanges:
                try:
                    if exch in self.exchange_instances:
                        markets = self.exchange_instances[exch].markets
                        all_markets[exch] = list(markets.keys())
                        logger.debug(f"Marketlar olindi: {exch} - {len(markets)} ta market")
                except Exception as e:
                    logger.error(f"Market olishda xato ({exch}): {e}")
            
            return CCXTResponse(
                success=True,
                data=all_markets,
                exchange=exchange
            )
            
        except Exception as e:
            logger.error(f"Marketlar olishda xato: {e}")
            return CCXTResponse(
                success=False,
                error=str(e),
                exchange=exchange
            )
    
    async def get_top_symbols(self, base_currency: str = 'USDT', 
                            limit: int = 20, exchange: str = None) -> CCXTResponse:
        """Top symbollarni volume bo'yicha olish"""
        try:
            if exchange:
                exchanges = [exchange]
            else:
                exchanges = list(self.exchange_instances.keys())
            
            all_tickers = {}
            
            for exch in exchanges:
                try:
                    response = await self._execute_exchange_method(exch, 'fetch_tickers')
                    if response.success:
                        # Base currency bo'yicha filter
                        filtered_tickers = {}
                        for symbol, ticker in response.data.items():
                            if symbol.endswith(f'/{base_currency}'):
                                filtered_tickers[symbol] = ticker
                        
                        # Volume bo'yicha saralash
                        sorted_tickers = sorted(
                            filtered_tickers.items(),
                            key=lambda x: x[1].get('baseVolume', 0),
                            reverse=True
                        )
                        
                        all_tickers[exch] = sorted_tickers[:limit]
                        logger.debug(f"Top symbollar olindi: {exch} - {len(sorted_tickers[:limit])} ta symbol")
                        
                except Exception as e:
                    logger.error(f"Top symbollar olishda xato ({exch}): {e}")
            
            return CCXTResponse(
                success=True,
                data=all_tickers
            )
            
        except Exception as e:
            logger.error(f"Top symbollar olishda xato: {e}")
            return CCXTResponse(
                success=False,
                error=str(e)
            )
    
    async def get_price_comparison(self, symbol: str) -> CCXTResponse:
        """Bir nechta exchange da narx taqqoslash"""
        try:
            prices = {}
            
            tasks = []
            for exch in self.exchange_instances.keys():
                task = asyncio.create_task(self.get_ticker(symbol, exch))
                tasks.append((exch, task))
            
            for exch, task in tasks:
                try:
                    response = await task
                    if response.success:
                        prices[exch] = {
                            'price': response.data.price,
                            'volume': response.data.volume_24h,
                            'change': response.data.change_24h
                        }
                except Exception as e:
                    logger.error(f"Narx taqqoslashda xato ({exch}): {e}")
            
            if prices:
                # Eng yaxshi va eng yomon narxlarni topish
                sorted_prices = sorted(prices.items(), key=lambda x: x[1]['price'])
                best_price = sorted_prices[0] if sorted_prices else None
                worst_price = sorted_prices[-1] if sorted_prices else None
                
                comparison_data = {
                    'symbol': symbol,
                    'prices': prices,
                    'best_price': best_price,
                    'worst_price': worst_price,
                    'price_difference': worst_price[1]['price'] - best_price[1]['price'] if best_price and worst_price else 0,
                    'exchanges_count': len(prices)
                }
                
                logger.info(f"Narx taqqoslash: {symbol} - {len(prices)} ta exchange")
                return CCXTResponse(
                    success=True,
                    data=comparison_data,
                    symbol=symbol
                )
            
            return CCXTResponse(
                success=False,
                error="Hech bir exchange dan narx olinmadi",
                symbol=symbol
            )
            
        except Exception as e:
            logger.error(f"Narx taqqoslashda xato: {e}")
            return CCXTResponse(
                success=False,
                error=str(e),
                symbol=symbol
            )
    
    async def get_market_summary(self) -> CCXTResponse:
        """Umumiy bozor xulosasi"""
        try:
            summary = {
                'total_markets': 0,
                'active_exchanges': 0,
                'top_volumes': [],
                'timestamp': datetime.now().isoformat()
            }
            
            # Har exchange dan statistika
            for exch in self.exchange_instances.keys():
                try:
                    markets_response = await self.get_markets(exch)
                    if markets_response.success:
                        summary['total_markets'] += len(markets_response.data.get(exch, []))
                        summary['active_exchanges'] += 1
                        
                        # Top volume symbollarni olish
                        top_response = await self.get_top_symbols(limit=5, exchange=exch)
                        if top_response.success:
                            summary['top_volumes'].extend(top_response.data.get(exch, []))
                            
                except Exception as e:
                    logger.error(f"Market summary xato ({exch}): {e}")
            
            # Top volumelarni global saralash
            summary['top_volumes'].sort(key=lambda x: x[1].get('baseVolume', 0), reverse=True)
            summary['top_volumes'] = summary['top_volumes'][:10]
            
            logger.info(f"Market summary: {summary['active_exchanges']} exchanges, {summary['total_markets']} markets")
            return CCXTResponse(
                success=True,
                data=summary
            )
            
        except Exception as e:
            logger.error(f"Market summary xato: {e}")
            return CCXTResponse(
                success=False,
                error=str(e)
            )
    
    def clear_cache(self):
        """Cache ni tozalash"""
        self.cache.clear()
        logger.info("CCXT cache tozalandi")
    
    async def health_check(self) -> Dict[str, bool]:
        """Barcha exchangelar uchun health check"""
        health_status = {}
        
        for exch in self.exchange_instances.keys():
            try:
                response = await self._execute_exchange_method(exch, 'fetch_status')
                health_status[exch] = response.success
                logger.debug(f"Health check: {exch} - {'✅' if response.success else '❌'}")
            except Exception as e:
                health_status[exch] = False
                logger.error(f"Health check xato ({exch}): {e}")
        
        return health_status
    
    async def get_historical_data(self, symbol: str, timeframe: str = '1h', 
                                days: int = 30, exchange: str = None) -> CCXTResponse:
        """Tarixiy ma'lumotlarni olish va DataFrame ga aylantirish"""
        try:
            # Limit hisoblash
            timeframe_minutes = {
                '1m': 1, '5m': 5, '15m': 15, '30m': 30,
                '1h': 60, '4h': 240, '1d': 1440
            }
            
            minutes = timeframe_minutes.get(timeframe, 60)
            limit = (days * 24 * 60) // minutes
            
            response = await self.get_ohlcv(symbol, timeframe, limit, exchange)
            
            if response.success:
                # DataFrame yaratish
                df_data = []
                for ohlcv in response.data:
                    df_data.append({
                        'timestamp': ohlcv.timestamp,
                        'open': ohlcv.open,
                        'high': ohlcv.high,
                        'low': ohlcv.low,
                        'close': ohlcv.close,
                        'volume': ohlcv.volume
                    })
                
                df = pd.DataFrame(df_data)
                df.set_index('timestamp', inplace=True)
                
                historical_data = {
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'days': days,
                    'records_count': len(df),
                    'dataframe': df,
                    'exchange': response.exchange
                }
                
                logger.info(f"Tarixiy ma'lumot olindi: {symbol} {timeframe} - {len(df)} ta record")
                return CCXTResponse(
                    success=True,
                    data=historical_data,
                    exchange=response.exchange,
                    symbol=symbol
                )
            
            return response
            
        except Exception as e:
            logger.error(f"Tarixiy ma'lumot olishda xato: {e}")
            return CCXTResponse(
                success=False,
                error=str(e),
                symbol=symbol
            )

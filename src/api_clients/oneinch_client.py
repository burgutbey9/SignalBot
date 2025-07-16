import asyncio
import aiohttp
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import time
import json

# Utils import qilish (keyinroq yaratiladi)
# from utils.logger import get_logger
# from utils.rate_limiter import RateLimiter
# from utils.error_handler import handle_api_error
# from utils.retry_handler import retry_async

# Hozircha oddiy logger
import logging
logger = logging.getLogger(__name__)

@dataclass
class APIResponse:
    """API javob formati"""
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    rate_limit_remaining: int = 0
    timestamp: float = 0

@dataclass
class OrderFlowData:
    """Order Flow ma'lumotlari"""
    symbol: str
    buy_volume: float
    sell_volume: float
    net_flow: float
    price: float
    timestamp: float
    confidence: float = 0.0

class OneInchClient:
    """1inch API client - Order Flow ma'lumotlari olish uchun"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.1inch.io/v5.0"
        self.session: Optional[aiohttp.ClientSession] = None
        self.rate_limit = 100  # So'rov/minut
        self.last_request_time = 0
        self.request_count = 0
        
    async def __aenter__(self):
        """Async context manager kirish"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager chiqish"""
        if self.session:
            await self.session.close()
    
    async def _wait_for_rate_limit(self):
        """Rate limit kutish"""
        current_time = time.time()
        if current_time - self.last_request_time < 60:
            if self.request_count >= self.rate_limit:
                sleep_time = 60 - (current_time - self.last_request_time)
                logger.warning(f"Rate limit: {sleep_time:.1f}s kutish")
                await asyncio.sleep(sleep_time)
                self.request_count = 0
        else:
            self.request_count = 0
        
        self.last_request_time = current_time
        self.request_count += 1
    
    async def make_request(self, endpoint: str, params: Dict = None) -> APIResponse:
        """Asosiy so'rov yuborish methodi"""
        if not self.session:
            return APIResponse(success=False, error="Session ochilmagan")
            
        try:
            await self._wait_for_rate_limit()
            
            url = f"{self.base_url}/{endpoint}"
            
            async with self.session.get(url, params=params) as response:
                response_data = await response.json()
                
                if response.status == 200:
                    return APIResponse(
                        success=True,
                        data=response_data,
                        rate_limit_remaining=int(response.headers.get('X-RateLimit-Remaining', 0)),
                        timestamp=time.time()
                    )
                else:
                    error_msg = response_data.get('message', f'HTTP {response.status}')
                    logger.error(f"1inch API xatosi: {error_msg}")
                    return APIResponse(success=False, error=error_msg)
                    
        except asyncio.TimeoutError:
            logger.error("1inch API timeout")
            return APIResponse(success=False, error="Timeout")
        except Exception as e:
            logger.error(f"1inch API xatosi: {str(e)}")
            return APIResponse(success=False, error=str(e))
    
    async def get_tokens(self, chain_id: int = 1) -> APIResponse:
        """Token ro'yxatini olish"""
        endpoint = f"{chain_id}/tokens"
        return await self.make_request(endpoint)
    
    async def get_quote(self, from_token: str, to_token: str, amount: str, chain_id: int = 1) -> APIResponse:
        """Swap quote olish"""
        endpoint = f"{chain_id}/quote"
        params = {
            'fromTokenAddress': from_token,
            'toTokenAddress': to_token,
            'amount': amount
        }
        return await self.make_request(endpoint, params)
    
    async def get_swap_data(self, from_token: str, to_token: str, amount: str, 
                           from_address: str, slippage: float = 1.0, chain_id: int = 1) -> APIResponse:
        """Swap ma'lumotlarini olish"""
        endpoint = f"{chain_id}/swap"
        params = {
            'fromTokenAddress': from_token,
            'toTokenAddress': to_token,
            'amount': amount,
            'fromAddress': from_address,
            'slippage': slippage
        }
        return await self.make_request(endpoint, params)
    
    async def get_order_flow_data(self, symbol: str, timeframe: str = '1h') -> APIResponse:
        """Order Flow ma'lumotlarini olish va tahlil qilish"""
        try:
            # USDC va WETH manzillari (Ethereum)
            tokens = {
                'USDC': '0xA0b86a33E6441b3B33C35cF7E5E6E9B9B4EA2E6c',
                'WETH': '0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2',
                'USDT': '0xdAC17F958D2ee523a2206206994597C13D831ec7'
            }
            
            # Quote olish
            quote_response = await self.get_quote(
                from_token=tokens['USDC'],
                to_token=tokens['WETH'],
                amount='1000000'  # 1 USDC (6 decimals)
            )
            
            if not quote_response.success:
                return quote_response
            
            quote_data = quote_response.data
            
            # Order Flow ma'lumotlarini yaratish
            order_flow = OrderFlowData(
                symbol=symbol,
                buy_volume=float(quote_data.get('toTokenAmount', 0)) / 10**18,  # WETH decimals
                sell_volume=0,  # Haqiqiy implementatsiyada hisoblash kerak
                net_flow=0,     # Buy - Sell
                price=float(quote_data.get('fromTokenAmount', 0)) / float(quote_data.get('toTokenAmount', 1)),
                timestamp=time.time(),
                confidence=0.8  # 1inch ga ishonch darajasi
            )
            
            # Net flow hisoblash
            order_flow.net_flow = order_flow.buy_volume - order_flow.sell_volume
            
            return APIResponse(
                success=True,
                data=order_flow.__dict__,
                timestamp=time.time()
            )
            
        except Exception as e:
            logger.error(f"Order Flow ma'lumotlarini olishda xato: {str(e)}")
            return APIResponse(success=False, error=str(e))
    
    async def get_protocol_data(self, chain_id: int = 1) -> APIResponse:
        """Protocol ma'lumotlarini olish"""
        endpoint = f"{chain_id}/protocols"
        return await self.make_request(endpoint)
    
    async def health_check(self) -> bool:
        """API holatini tekshirish"""
        try:
            response = await self.get_tokens()
            return response.success
        except Exception:
            return False

# Sinov uchun
async def test_oneinch_client():
    """1inch client sinovi"""
    api_key = "YOUR_API_KEY"  # Haqiqiy API key kiriting
    
    async with OneInchClient(api_key) as client:
        # Health check
        if await client.health_check():
            print("✅ 1inch API ishlayapti")
        else:
            print("❌ 1inch API ishlamayapti")
        
        # Order Flow ma'lumotlarini olish
        order_flow = await client.get_order_flow_data("USDC/WETH")
        if order_flow.success:
            print(f"📊 Order Flow: {order_flow.data}")
        else:
            print(f"❌ Order Flow xatosi: {order_flow.error}")

if __name__ == "__main__":
    asyncio.run(test_oneinch_client())

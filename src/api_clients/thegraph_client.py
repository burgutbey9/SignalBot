import asyncio
import aiohttp
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import time
import json

# Utils import qilish (keyinroq yaratiladi)
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
class GraphQLQuery:
    """GraphQL so'rov formati"""
    query: str
    variables: Optional[Dict] = None
    operation_name: Optional[str] = None

class TheGraphClient:
    """The Graph API client - Uniswap va boshqa DEX ma'lumotlari uchun"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.base_url = "https://api.thegraph.com/subgraphs/name"
        self.session: Optional[aiohttp.ClientSession] = None
        self.rate_limit = 1000  # So'rov/minut
        self.last_request_time = 0
        self.request_count = 0
        
        # The Graph subgraph endpoints
        self.subgraphs = {
            'uniswap_v3': 'uniswap/uniswap-v3',
            'uniswap_v2': 'uniswap/uniswap-v2',
            'sushiswap': 'sushiswap/exchange',
            'curve': 'messari/curve-finance-ethereum',
            'balancer': 'balancer-labs/balancer-v2'
        }
        
    async def __aenter__(self):
        """Async context manager kirish"""
        headers = {'Content-Type': 'application/json'}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
            
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers=headers
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
    
    async def make_graphql_request(self, subgraph: str, query: GraphQLQuery) -> APIResponse:
        """GraphQL so'rov yuborish"""
        if not self.session:
            return APIResponse(success=False, error="Session ochilmagan")
            
        try:
            await self._wait_for_rate_limit()
            
            url = f"{self.base_url}/{subgraph}"
            
            payload = {
                'query': query.query,
                'variables': query.variables or {},
                'operationName': query.operation_name
            }
            
            async with self.session.post(url, json=payload) as response:
                response_data = await response.json()
                
                if response.status == 200:
                    if 'errors' in response_data:
                        error_msg = response_data['errors'][0].get('message', 'GraphQL xatosi')
                        logger.error(f"GraphQL xatosi: {error_msg}")
                        return APIResponse(success=False, error=error_msg)
                    
                    return APIResponse(
                        success=True,
                        data=response_data.get('data'),
                        timestamp=time.time()
                    )
                else:
                    error_msg = f'HTTP {response.status}'
                    logger.error(f"The Graph API xatosi: {error_msg}")
                    return APIResponse(success=False, error=error_msg)
                    
        except asyncio.TimeoutError:
            logger.error("The Graph API timeout")
            return APIResponse(success=False, error="Timeout")
        except Exception as e:
            logger.error(f"The Graph API xatosi: {str(e)}")
            return APIResponse(success=False, error=str(e))
    
    async def get_uniswap_v3_pools(self, token_addresses: List[str], limit: int = 10) -> APIResponse:
        """Uniswap V3 pools ma'lumotlarini olish"""
        query = GraphQLQuery(
            query="""
            query GetPools($tokenAddresses: [String!]!, $first: Int!) {
                pools(
                    first: $first,
                    orderBy: totalValueLockedUSD,
                    orderDirection: desc,
                    where: {
                        or: [
                            {token0_in: $tokenAddresses},
                            {token1_in: $tokenAddresses}
                        ]
                    }
                ) {
                    id
                    token0 {
                        id
                        symbol
                        name
                        decimals
                    }
                    token1 {
                        id
                        symbol
                        name
                        decimals
                    }
                    feeTier
                    liquidity
                    sqrtPrice
                    tick
                    token0Price
                    token1Price
                    volumeUSD
                    totalValueLockedUSD
                    createdAtTimestamp
                }
            }
            """,
            variables={
                'tokenAddresses': token_addresses,
                'first': limit
            }
        )
        
        return await self.make_graphql_request('uniswap/uniswap-v3', query)
    
    async def get_uniswap_swaps(self, pool_id: str, limit: int = 100) -> APIResponse:
        """Uniswap swap ma'lumotlarini olish"""
        query = GraphQLQuery(
            query="""
            query GetSwaps($poolId: String!, $first: Int!) {
                swaps(
                    first: $first,
                    orderBy: timestamp,
                    orderDirection: desc,
                    where: {pool: $poolId}
                ) {
                    id
                    timestamp
                    pool {
                        id
                        token0 {
                            symbol
                        }
                        token1 {
                            symbol
                        }
                    }
                    origin
                    amount0
                    amount1
                    amountUSD
                    sqrtPriceX96
                    tick
                    logIndex
                }
            }
            """,
            variables={
                'poolId': pool_id,
                'first': limit
            }
        )
        
        return await self.make_graphql_request('uniswap/uniswap-v3', query)
    
    async def get_token_price_data(self, token_address: str, hours: int = 24) -> APIResponse:
        """Token narx ma'lumotlarini olish"""
        timestamp_from = int(time.time()) - (hours * 3600)
        
        query = GraphQLQuery(
            query="""
            query GetTokenPriceData($tokenAddress: String!, $timestampFrom: Int!) {
                tokenHourDatas(
                    first: 1000,
                    orderBy: periodStartUnix,
                    orderDirection: desc,
                    where: {
                        token: $tokenAddress,
                        periodStartUnix_gte: $timestampFrom
                    }
                ) {
                    periodStartUnix
                    high
                    low
                    open
                    close
                    priceUSD
                    volume
                    volumeUSD
                    totalValueLocked
                    totalValueLockedUSD
                }
            }
            """,
            variables={
                'tokenAddress': token_address,
                'timestampFrom': timestamp_from
            }
        )
        
        return await self.make_graphql_request('uniswap/uniswap-v3', query)
    
    async def get_order_flow_data(self, symbol: str, timeframe: str = '1h') -> APIResponse:
        """Order Flow ma'lumotlarini olish va tahlil qilish"""
        try:
            # USDC va WETH manzillari (Ethereum)
            tokens = {
                'USDC': '0xa0b86a33e6441b3b33c35cf7e5e6e9b9b4ea2e6c',
                'WETH': '0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2',
                'USDT': '0xdac17f958d2ee523a2206206994597c13d831ec7'
            }
            
            # Pool ma'lumotlarini olish
            pools_response = await self.get_uniswap_v3_pools([tokens['USDC'], tokens['WETH']])
            
            if not pools_response.success:
                return pools_response
            
            pools = pools_response.data.get('pools', [])
            if not pools:
                return APIResponse(success=False, error="Pool topilmadi")
            
            # Eng katta TVL ga ega pool ni tanlash
            main_pool = max(pools, key=lambda p: float(p.get('totalValueLockedUSD', 0)))
            
            # Swap ma'lumotlarini olish
            swaps_response = await self.get_uniswap_swaps(main_pool['id'])
            
            if not swaps_response.success:
                return swaps_response
            
            swaps = swaps_response.data.get('swaps', [])
            
            # Order Flow tahlil qilish
            buy_volume = 0
            sell_volume = 0
            
            for swap in swaps:
                amount_usd = float(swap.get('amountUSD', 0))
                amount0 = float(swap.get('amount0', 0))
                
                # Amount0 musbat bo'lsa - sotish, manfiy bo'lsa - sotib olish
                if amount0 > 0:
                    sell_volume += amount_usd
                else:
                    buy_volume += amount_usd
            
            # Order Flow ma'lumotlarini yaratish
            order_flow_data = {
                'symbol': symbol,
                'buy_volume': buy_volume,
                'sell_volume': sell_volume,
                'net_flow': buy_volume - sell_volume,
                'price': float(main_pool.get('token0Price', 0)),
                'timestamp': time.time(),
                'confidence': 0.85,  # The Graph ga ishonch darajasi
                'pool_tvl': float(main_pool.get('totalValueLockedUSD', 0)),
                'pool_volume': float(main_pool.get('volumeUSD', 0)),
                'swap_count': len(swaps)
            }
            
            return APIResponse(
                success=True,
                data=order_flow_data,
                timestamp=time.time()
            )
            
        except Exception as e:
            logger.error(f"Order Flow ma'lumotlarini olishda xato: {str(e)}")
            return APIResponse(success=False, error=str(e))
    
    async def get_protocol_stats(self, protocol: str = 'uniswap_v3') -> APIResponse:
        """Protocol statistikalarini olish"""
        subgraph = self.subgraphs.get(protocol, 'uniswap/uniswap-v3')
        
        query = GraphQLQuery(
            query="""
            query GetProtocolStats {
                factories(first: 1) {
                    id
                    poolCount
                    totalVolumeUSD
                    totalValueLockedUSD
                    totalFeesUSD
                    owner
                }
            }
            """
        )
        
        return await self.make_graphql_request(subgraph, query)
    
    async def health_check(self) -> bool:
        """API holatini tekshirish"""
        try:
            response = await self.get_protocol_stats()
            return response.success
        except Exception:
            return False

# Sinov uchun
async def test_thegraph_client():
    """The Graph client sinovi"""
    async with TheGraphClient() as client:
        # Health check
        if await client.health_check():
            print("✅ The Graph API ishlayapti")
        else:
            print("❌ The Graph API ishlamayapti")
        
        # Order Flow ma'lumotlarini olish
        order_flow = await client.get_order_flow_data("USDC/WETH")
        if order_flow.success:
            print(f"📊 Order Flow: {order_flow.data}")
        else:
            print(f"❌ Order Flow xatosi: {order_flow.error}")

if __name__ == "__main__":
    asyncio.run(test_thegraph_client())

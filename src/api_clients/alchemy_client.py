import asyncio
import aiohttp
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import time
from utils.logger import get_logger
from utils.rate_limiter import RateLimiter
from utils.error_handler import handle_api_error
from utils.retry_handler import retry_async

logger = get_logger(__name__)

@dataclass
class AlchemyResponse:
    """Alchemy API javob formatі"""
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    rate_limit_remaining: int = 0
    network: Optional[str] = None
    
@dataclass
class TransactionData:
    """Transaksiya ma'lumotlari"""
    hash: str
    from_address: str
    to_address: str
    value: float
    gas_price: float
    gas_used: int
    block_number: int
    timestamp: datetime
    token_transfers: List[Dict] = None
    
@dataclass
class TokenTransfer:
    """Token transfer ma'lumotlari"""
    token_address: str
    from_address: str
    to_address: str
    value: float
    token_symbol: str
    token_name: str
    decimals: int
    
@dataclass
class WalletActivity:
    """Wallet faoliyati ma'lumotlari"""
    address: str
    balance: float
    token_balances: List[Dict]
    recent_transactions: List[TransactionData]
    nft_count: int = 0

class AlchemyClient:
    """Alchemy API client - on-chain ma'lumotlarni olish uchun"""
    
    def __init__(self, api_key: str, networks: List[str] = None):
        self.api_key = api_key
        self.networks = networks or ["eth", "polygon", "arbitrum", "optimism"]
        self.base_urls = {
            "eth": f"https://eth-mainnet.alchemyapi.io/v2/{api_key}",
            "polygon": f"https://polygon-mainnet.alchemyapi.io/v2/{api_key}",
            "arbitrum": f"https://arb-mainnet.alchemyapi.io/v2/{api_key}",
            "optimism": f"https://opt-mainnet.alchemyapi.io/v2/{api_key}"
        }
        
        # Rate limiter - Alchemy uchun 300 req/min
        self.rate_limiter = RateLimiter(calls=300, period=60)
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Cache uchun
        self.cache = {}
        self.cache_ttl = 30  # 30 sekund
        
        logger.info(f"Alchemy client ishga tushdi. Tarmoqlar: {self.networks}")
    
    async def __aenter__(self):
        """Async context manager kirish"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={
                "Content-Type": "application/json",
                "User-Agent": "AI-OrderFlow-Bot/1.0"
            }
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager chiqish"""
        if self.session:
            await self.session.close()
            
    def _get_cache_key(self, method: str, params: Dict) -> str:
        """Cache kalitini yaratish"""
        return f"{method}:{hash(str(sorted(params.items())))}"
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Cache amal qilish muddatini tekshirish"""
        if cache_key not in self.cache:
            return False
        
        cached_time = self.cache[cache_key].get("timestamp", 0)
        return (time.time() - cached_time) < self.cache_ttl
    
    @retry_async(max_retries=3, delay=1)
    async def _make_request(self, network: str, method: str, params: List = None) -> AlchemyResponse:
        """Asosiy so'rov yuborish methodi"""
        try:
            await self.rate_limiter.wait()
            
            # Cache tekshirish
            cache_key = self._get_cache_key(method, {"network": network, "params": params})
            if self._is_cache_valid(cache_key):
                logger.debug(f"Cache dan qaytarildi: {method}")
                return AlchemyResponse(
                    success=True,
                    data=self.cache[cache_key]["data"],
                    network=network
                )
            
            if network not in self.base_urls:
                return AlchemyResponse(
                    success=False,
                    error=f"Noma'lum tarmoq: {network}",
                    network=network
                )
            
            url = self.base_urls[network]
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": method,
                "params": params or []
            }
            
            async with self.session.post(url, json=payload) as response:
                response_data = await response.json()
                
                if response.status == 200:
                    if "error" in response_data:
                        logger.error(f"Alchemy API xatosi: {response_data['error']}")
                        return AlchemyResponse(
                            success=False,
                            error=response_data["error"].get("message", "Noma'lum xato"),
                            network=network
                        )
                    
                    result = response_data.get("result")
                    
                    # Cache ga saqlash
                    self.cache[cache_key] = {
                        "data": result,
                        "timestamp": time.time()
                    }
                    
                    logger.debug(f"Alchemy so'rov muvaffaqiyatli: {method}")
                    return AlchemyResponse(
                        success=True,
                        data=result,
                        network=network
                    )
                else:
                    logger.error(f"HTTP xato: {response.status}")
                    return AlchemyResponse(
                        success=False,
                        error=f"HTTP {response.status}",
                        network=network
                    )
                    
        except asyncio.TimeoutError:
            logger.error(f"Alchemy so'rov timeout: {method}")
            return AlchemyResponse(
                success=False,
                error="Timeout",
                network=network
            )
        except Exception as e:
            logger.error(f"Alchemy so'rov xatosi: {e}")
            return AlchemyResponse(
                success=False,
                error=str(e),
                network=network
            )
    
    async def get_latest_block(self, network: str = "eth") -> AlchemyResponse:
        """Eng so'nggi blok ma'lumotini olish"""
        try:
            response = await self._make_request(
                network=network,
                method="eth_getBlockByNumber",
                params=["latest", False]
            )
            
            if response.success and response.data:
                block_data = {
                    "number": int(response.data["number"], 16),
                    "timestamp": int(response.data["timestamp"], 16),
                    "transactions_count": len(response.data.get("transactions", [])),
                    "gas_used": int(response.data.get("gasUsed", "0x0"), 16),
                    "gas_limit": int(response.data.get("gasLimit", "0x0"), 16)
                }
                
                logger.info(f"So'nggi blok olindi: {block_data['number']} ({network})")
                return AlchemyResponse(
                    success=True,
                    data=block_data,
                    network=network
                )
            
            return response
            
        except Exception as e:
            logger.error(f"Blok ma'lumotini olishda xato: {e}")
            return AlchemyResponse(
                success=False,
                error=str(e),
                network=network
            )
    
    async def get_transaction_receipt(self, tx_hash: str, network: str = "eth") -> AlchemyResponse:
        """Transaksiya receiptini olish"""
        try:
            response = await self._make_request(
                network=network,
                method="eth_getTransactionReceipt",
                params=[tx_hash]
            )
            
            if response.success and response.data:
                receipt_data = {
                    "transaction_hash": response.data["transactionHash"],
                    "block_number": int(response.data["blockNumber"], 16),
                    "gas_used": int(response.data["gasUsed"], 16),
                    "status": int(response.data["status"], 16),
                    "logs": response.data.get("logs", []),
                    "from": response.data.get("from"),
                    "to": response.data.get("to")
                }
                
                logger.debug(f"Transaksiya receipt olindi: {tx_hash[:10]}...")
                return AlchemyResponse(
                    success=True,
                    data=receipt_data,
                    network=network
                )
            
            return response
            
        except Exception as e:
            logger.error(f"Transaksiya receipt olishda xato: {e}")
            return AlchemyResponse(
                success=False,
                error=str(e),
                network=network
            )
    
    async def get_wallet_balance(self, address: str, network: str = "eth") -> AlchemyResponse:
        """Wallet balansini olish"""
        try:
            response = await self._make_request(
                network=network,
                method="eth_getBalance",
                params=[address, "latest"]
            )
            
            if response.success and response.data:
                balance_wei = int(response.data, 16)
                balance_eth = balance_wei / 1e18
                
                balance_data = {
                    "address": address,
                    "balance_wei": balance_wei,
                    "balance_eth": balance_eth,
                    "network": network
                }
                
                logger.debug(f"Wallet balans olindi: {address[:10]}... = {balance_eth:.6f} ETH")
                return AlchemyResponse(
                    success=True,
                    data=balance_data,
                    network=network
                )
            
            return response
            
        except Exception as e:
            logger.error(f"Wallet balans olishda xato: {e}")
            return AlchemyResponse(
                success=False,
                error=str(e),
                network=network
            )
    
    async def get_token_balances(self, address: str, network: str = "eth") -> AlchemyResponse:
        """Token balanslarini olish - Alchemy Enhanced API"""
        try:
            # Alchemy Enhanced API uchun boshqa endpoint
            enhanced_url = f"https://eth-mainnet.alchemyapi.io/v2/{self.api_key}"
            if network != "eth":
                enhanced_url = self.base_urls[network].replace("/v2/", "/v2/")
            
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "alchemy_getTokenBalances",
                "params": [address]
            }
            
            async with self.session.post(enhanced_url, json=payload) as response:
                response_data = await response.json()
                
                if response.status == 200 and "result" in response_data:
                    token_balances = []
                    
                    for token in response_data["result"]["tokenBalances"]:
                        if int(token["tokenBalance"], 16) > 0:
                            token_balances.append({
                                "contract_address": token["contractAddress"],
                                "balance": int(token["tokenBalance"], 16),
                                "balance_formatted": int(token["tokenBalance"], 16) / 1e18
                            })
                    
                    balance_data = {
                        "address": address,
                        "token_count": len(token_balances),
                        "tokens": token_balances,
                        "network": network
                    }
                    
                    logger.debug(f"Token balanslar olindi: {address[:10]}... = {len(token_balances)} ta token")
                    return AlchemyResponse(
                        success=True,
                        data=balance_data,
                        network=network
                    )
                else:
                    logger.error(f"Token balans olishda xato: {response_data}")
                    return AlchemyResponse(
                        success=False,
                        error="Token balans olishda xato",
                        network=network
                    )
                    
        except Exception as e:
            logger.error(f"Token balans olishda xato: {e}")
            return AlchemyResponse(
                success=False,
                error=str(e),
                network=network
            )
    
    async def get_asset_transfers(self, address: str, network: str = "eth", 
                                category: List[str] = None, limit: int = 100) -> AlchemyResponse:
        """Asset transferlarini olish"""
        try:
            if category is None:
                category = ["external", "internal", "erc20", "erc721", "erc1155"]
            
            # Alchemy Enhanced API
            enhanced_url = self.base_urls[network]
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "alchemy_getAssetTransfers",
                "params": [{
                    "fromAddress": address,
                    "category": category,
                    "maxCount": hex(limit),
                    "order": "desc"
                }]
            }
            
            async with self.session.post(enhanced_url, json=payload) as response:
                response_data = await response.json()
                
                if response.status == 200 and "result" in response_data:
                    transfers = []
                    
                    for transfer in response_data["result"]["transfers"]:
                        transfer_data = {
                            "block_num": transfer.get("blockNum"),
                            "hash": transfer.get("hash"),
                            "from": transfer.get("from"),
                            "to": transfer.get("to"),
                            "value": transfer.get("value", 0),
                            "asset": transfer.get("asset"),
                            "category": transfer.get("category"),
                            "raw_contract": transfer.get("rawContract", {}),
                            "metadata": transfer.get("metadata", {})
                        }
                        transfers.append(transfer_data)
                    
                    transfer_data = {
                        "address": address,
                        "transfers_count": len(transfers),
                        "transfers": transfers,
                        "network": network
                    }
                    
                    logger.debug(f"Asset transferlar olindi: {address[:10]}... = {len(transfers)} ta transfer")
                    return AlchemyResponse(
                        success=True,
                        data=transfer_data,
                        network=network
                    )
                else:
                    logger.error(f"Asset transfer olishda xato: {response_data}")
                    return AlchemyResponse(
                        success=False,
                        error="Asset transfer olishda xato",
                        network=network
                    )
                    
        except Exception as e:
            logger.error(f"Asset transfer olishda xato: {e}")
            return AlchemyResponse(
                success=False,
                error=str(e),
                network=network
            )
    
    async def get_nft_metadata(self, contract_address: str, token_id: str, network: str = "eth") -> AlchemyResponse:
        """NFT metadata olish"""
        try:
            enhanced_url = self.base_urls[network]
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "alchemy_getNFTMetadata",
                "params": [contract_address, token_id]
            }
            
            async with self.session.post(enhanced_url, json=payload) as response:
                response_data = await response.json()
                
                if response.status == 200 and "result" in response_data:
                    nft_data = {
                        "contract_address": contract_address,
                        "token_id": token_id,
                        "metadata": response_data["result"],
                        "network": network
                    }
                    
                    logger.debug(f"NFT metadata olindi: {contract_address[:10]}...#{token_id}")
                    return AlchemyResponse(
                        success=True,
                        data=nft_data,
                        network=network
                    )
                else:
                    logger.error(f"NFT metadata olishda xato: {response_data}")
                    return AlchemyResponse(
                        success=False,
                        error="NFT metadata olishda xato",
                        network=network
                    )
                    
        except Exception as e:
            logger.error(f"NFT metadata olishda xato: {e}")
            return AlchemyResponse(
                success=False,
                error=str(e),
                network=network
            )
    
    async def get_logs(self, from_block: str, to_block: str, address: str = None, 
                      topics: List[str] = None, network: str = "eth") -> AlchemyResponse:
        """Event loglarini olish"""
        try:
            params = {
                "fromBlock": from_block,
                "toBlock": to_block
            }
            
            if address:
                params["address"] = address
            if topics:
                params["topics"] = topics
            
            response = await self._make_request(
                network=network,
                method="eth_getLogs",
                params=[params]
            )
            
            if response.success and response.data:
                logs_data = {
                    "logs_count": len(response.data),
                    "logs": response.data,
                    "from_block": from_block,
                    "to_block": to_block,
                    "network": network
                }
                
                logger.debug(f"Event loglar olindi: {len(response.data)} ta log")
                return AlchemyResponse(
                    success=True,
                    data=logs_data,
                    network=network
                )
            
            return response
            
        except Exception as e:
            logger.error(f"Event log olishda xato: {e}")
            return AlchemyResponse(
                success=False,
                error=str(e),
                network=network
            )
    
    async def get_whale_transactions(self, min_value_eth: float = 100.0, 
                                   network: str = "eth", limit: int = 50) -> AlchemyResponse:
        """Katta miqdordagi transaksiyalarni kuzatish (whale tracking)"""
        try:
            # So'nggi blokni olish
            latest_block_response = await self.get_latest_block(network)
            if not latest_block_response.success:
                return latest_block_response
            
            latest_block = latest_block_response.data["number"]
            from_block = latest_block - 10  # So'nggi 10 blok
            
            # Loglarni olish
            logs_response = await self.get_logs(
                from_block=hex(from_block),
                to_block="latest",
                network=network
            )
            
            if not logs_response.success:
                return logs_response
            
            whale_transactions = []
            min_value_wei = int(min_value_eth * 1e18)
            
            # Loglarni filtrlash
            for log in logs_response.data["logs"]:
                if len(log.get("data", "")) > 2:
                    try:
                        # Transfer event signature
                        if log.get("topics", [{}])[0] == "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef":
                            # ERC20 transfer
                            value = int(log["data"], 16)
                            if value >= min_value_wei:
                                whale_transactions.append({
                                    "type": "ERC20",
                                    "block": int(log["blockNumber"], 16),
                                    "hash": log["transactionHash"],
                                    "address": log["address"],
                                    "value_wei": value,
                                    "value_eth": value / 1e18,
                                    "topics": log["topics"]
                                })
                    except:
                        continue
            
            # Katta transaksiyalarni saralash
            whale_transactions.sort(key=lambda x: x["value_eth"], reverse=True)
            whale_transactions = whale_transactions[:limit]
            
            whale_data = {
                "whale_count": len(whale_transactions),
                "transactions": whale_transactions,
                "min_value_eth": min_value_eth,
                "blocks_scanned": 10,
                "network": network
            }
            
            logger.info(f"Whale transaksiyalar topildi: {len(whale_transactions)} ta ({network})")
            return AlchemyResponse(
                success=True,
                data=whale_data,
                network=network
            )
            
        except Exception as e:
            logger.error(f"Whale transaksiya kuzatishda xato: {e}")
            return AlchemyResponse(
                success=False,
                error=str(e),
                network=network
            )
    
    async def get_network_stats(self, network: str = "eth") -> AlchemyResponse:
        """Tarmoq statistikalarini olish"""
        try:
            # So'nggi blok
            latest_block_response = await self.get_latest_block(network)
            if not latest_block_response.success:
                return latest_block_response
            
            latest_block = latest_block_response.data
            
            # Gas price
            gas_price_response = await self._make_request(
                network=network,
                method="eth_gasPrice"
            )
            
            gas_price = 0
            if gas_price_response.success:
                gas_price = int(gas_price_response.data, 16) / 1e9  # Gwei
            
            # Pending transactions
            pending_response = await self._make_request(
                network=network,
                method="eth_getBlockTransactionCountByNumber",
                params=["pending"]
            )
            
            pending_count = 0
            if pending_response.success:
                pending_count = int(pending_response.data, 16)
            
            network_stats = {
                "network": network,
                "latest_block": latest_block["number"],
                "gas_price_gwei": gas_price,
                "gas_used_percentage": (latest_block["gas_used"] / latest_block["gas_limit"]) * 100,
                "pending_transactions": pending_count,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"Tarmoq statistikasi olindi: {network} - Blok: {latest_block['number']}")
            return AlchemyResponse(
                success=True,
                data=network_stats,
                network=network
            )
            
        except Exception as e:
            logger.error(f"Tarmoq statistikasi olishda xato: {e}")
            return AlchemyResponse(
                success=False,
                error=str(e),
                network=network
            )
    
    async def monitor_address(self, address: str, networks: List[str] = None) -> Dict[str, AlchemyResponse]:
        """Bir nechta tarmoqda address kuzatish"""
        networks = networks or self.networks
        results = {}
        
        tasks = []
        for network in networks:
            task = asyncio.create_task(self.get_wallet_balance(address, network))
            tasks.append((network, task))
        
        for network, task in tasks:
            try:
                result = await task
                results[network] = result
                logger.debug(f"Address monitoring: {network} - {result.success}")
            except Exception as e:
                logger.error(f"Address monitoring xatosi ({network}): {e}")
                results[network] = AlchemyResponse(
                    success=False,
                    error=str(e),
                    network=network
                )
        
        return results
    
    def clear_cache(self):
        """Cache ni tozalash"""
        self.cache.clear()
        logger.info("Alchemy cache tozalandi")
    
    async def health_check(self) -> Dict[str, bool]:
        """Barcha tarmoqlar uchun health check"""
        health_status = {}
        
        for network in self.networks:
            try:
                response = await self.get_latest_block(network)
                health_status[network] = response.success
                logger.debug(f"Health check: {network} - {'✅' if response.success else '❌'}")
            except Exception as e:
                health_status[network] = False
                logger.error(f"Health check xatosi ({network}): {e}")
        
        return health_status

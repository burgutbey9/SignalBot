import asyncio
import json
import aiohttp
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from utils.logger import get_logger
from utils.rate_limiter import RateLimiter
from utils.error_handler import handle_api_error
from utils.retry_handler import retry_async
from config.config import ConfigManager

logger = get_logger(__name__)

@dataclass
class PropShotPosition:
    """Propshot pozitsiya ma'lumotlari"""
    symbol: str
    side: str  # BUY/SELL
    lot_size: float
    entry_price: float
    stop_loss: float
    take_profit: float
    timestamp: datetime
    position_id: str
    status: str = "PENDING"
    pnl: float = 0.0
    
@dataclass
class PropShotAccount:
    """Propshot akavunt ma'lumotlari"""
    account_id: str
    balance: float
    equity: float
    margin: float
    free_margin: float
    margin_level: float
    max_daily_loss: float
    max_total_loss: float
    daily_loss: float
    total_loss: float
    max_lot_size: float
    max_daily_trades: int
    daily_trades: int
    is_active: bool = True
    
@dataclass
class PropShotTrade:
    """Propshot savdo buyrug'i"""
    symbol: str
    action: str  # BUY/SELL
    volume: float
    price: float = 0.0  # 0 = market price
    stop_loss: float = 0.0
    take_profit: float = 0.0
    comment: str = ""
    magic_number: int = 12345
    trade_type: str = "MARKET"  # MARKET/LIMIT/STOP
    expiration: Optional[datetime] = None
    
@dataclass
class PropShotResponse:
    """Propshot javob formati"""
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    error_code: Optional[int] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class PropShotConnector:
    """Propshot API bilan bog'lanish va savdo bajarish"""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.base_url = config.get_propshot_url()
        self.api_key = config.get_propshot_api_key()
        self.account_id = config.get_propshot_account_id()
        self.session: Optional[aiohttp.ClientSession] = None
        self.rate_limiter = RateLimiter(calls=100, period=60)
        self.is_connected = False
        self.last_heartbeat = datetime.now()
        
        # Propshot sozlamalari
        self.max_daily_loss = config.get_propshot_max_daily_loss()
        self.max_total_loss = config.get_propshot_max_total_loss()
        self.max_lot_size = config.get_propshot_max_lot_size()
        self.max_daily_trades = config.get_propshot_max_daily_trades()
        
        # Kuzatuv ma'lumotlari
        self.positions: Dict[str, PropShotPosition] = {}
        self.daily_trades = 0
        self.daily_loss = 0.0
        self.total_loss = 0.0
        
        logger.info("PropShotConnector ishga tushirildi")
    
    async def __aenter__(self):
        """Async context manager kirish"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
        )
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager chiqish"""
        await self.disconnect()
        if self.session:
            await self.session.close()
    
    @retry_async(max_retries=3, delay=2)
    async def connect(self) -> PropShotResponse:
        """Propshot serveriga ulanish"""
        try:
            logger.info("Propshot serveriga ulanish...")
            
            # Ulanish so'rovi
            connect_data = {
                'account_id': self.account_id,
                'api_version': '1.0',
                'timestamp': datetime.now().isoformat()
            }
            
            response = await self._make_request('POST', '/connect', connect_data)
            
            if response.success:
                self.is_connected = True
                self.last_heartbeat = datetime.now()
                logger.info("Propshot serveriga muvaffaqiyatli ulandi")
                
                # Akavunt ma'lumotlarini yangilash
                await self._update_account_info()
                
                return PropShotResponse(
                    success=True,
                    data={'connected': True, 'account_id': self.account_id}
                )
            else:
                self.is_connected = False
                logger.error(f"Propshot ulanishda xato: {response.error}")
                return response
                
        except Exception as e:
            logger.error(f"Propshot ulanishda xato: {e}")
            self.is_connected = False
            return PropShotResponse(success=False, error=str(e))
    
    async def disconnect(self) -> PropShotResponse:
        """Propshot serveridan uzilish"""
        try:
            if self.is_connected:
                logger.info("Propshot serveridan uzilish...")
                
                response = await self._make_request('POST', '/disconnect', {
                    'account_id': self.account_id,
                    'timestamp': datetime.now().isoformat()
                })
                
                self.is_connected = False
                logger.info("Propshot serveridan uzildi")
                return response
            
            return PropShotResponse(success=True, data={'disconnected': True})
            
        except Exception as e:
            logger.error(f"Propshot uzilishda xato: {e}")
            return PropShotResponse(success=False, error=str(e))
    
    async def send_trade_signal(self, trade: PropShotTrade) -> PropShotResponse:
        """Savdo signalini Propshot ga yuborish"""
        try:
            # Xavfsizlik tekshiruvlari
            risk_check = await self._check_risk_limits(trade)
            if not risk_check.success:
                logger.warning(f"Risk tekshiruvi muvaffaqiyatsiz: {risk_check.error}")
                return risk_check
            
            # Savdo ma'lumotlarini tayyorlash
            trade_data = {
                'account_id': self.account_id,
                'symbol': trade.symbol,
                'action': trade.action,
                'volume': trade.volume,
                'price': trade.price,
                'stop_loss': trade.stop_loss,
                'take_profit': trade.take_profit,
                'comment': trade.comment,
                'magic_number': trade.magic_number,
                'trade_type': trade.trade_type,
                'timestamp': datetime.now().isoformat()
            }
            
            if trade.expiration:
                trade_data['expiration'] = trade.expiration.isoformat()
            
            logger.info(f"Savdo signali yuborilmoqda: {trade.symbol} {trade.action} {trade.volume}")
            
            response = await self._make_request('POST', '/trade/order', trade_data)
            
            if response.success:
                # Savdo hisobini yangilash
                self.daily_trades += 1
                
                # Pozitsiyani saqlash
                if response.data and 'position_id' in response.data:
                    position = PropShotPosition(
                        symbol=trade.symbol,
                        side=trade.action,
                        lot_size=trade.volume,
                        entry_price=response.data.get('entry_price', trade.price),
                        stop_loss=trade.stop_loss,
                        take_profit=trade.take_profit,
                        timestamp=datetime.now(),
                        position_id=response.data['position_id']
                    )
                    self.positions[position.position_id] = position
                
                logger.info(f"Savdo muvaffaqiyatli yuborildi: {response.data}")
                return response
            else:
                logger.error(f"Savdo yuborishda xato: {response.error}")
                return response
                
        except Exception as e:
            logger.error(f"Savdo yuborishda xato: {e}")
            return PropShotResponse(success=False, error=str(e))
    
    async def get_account_info(self) -> PropShotResponse:
        """Akavunt ma'lumotlarini olish"""
        try:
            response = await self._make_request('GET', f'/account/{self.account_id}')
            
            if response.success and response.data:
                # Akavunt ma'lumotlarini yangilash
                account_data = response.data
                self.daily_loss = account_data.get('daily_loss', 0.0)
                self.total_loss = account_data.get('total_loss', 0.0)
                self.daily_trades = account_data.get('daily_trades', 0)
                
                account = PropShotAccount(
                    account_id=self.account_id,
                    balance=account_data.get('balance', 0.0),
                    equity=account_data.get('equity', 0.0),
                    margin=account_data.get('margin', 0.0),
                    free_margin=account_data.get('free_margin', 0.0),
                    margin_level=account_data.get('margin_level', 0.0),
                    max_daily_loss=self.max_daily_loss,
                    max_total_loss=self.max_total_loss,
                    daily_loss=self.daily_loss,
                    total_loss=self.total_loss,
                    max_lot_size=self.max_lot_size,
                    max_daily_trades=self.max_daily_trades,
                    daily_trades=self.daily_trades,
                    is_active=account_data.get('is_active', True)
                )
                
                return PropShotResponse(success=True, data=account)
            
            return response
            
        except Exception as e:
            logger.error(f"Akavunt ma'lumotlarini olishda xato: {e}")
            return PropShotResponse(success=False, error=str(e))
    
    async def get_positions(self) -> PropShotResponse:
        """Ochiq pozitsiyalarni olish"""
        try:
            response = await self._make_request('GET', f'/positions/{self.account_id}')
            
            if response.success and response.data:
                positions = []
                for pos_data in response.data:
                    position = PropShotPosition(
                        symbol=pos_data['symbol'],
                        side=pos_data['side'],
                        lot_size=pos_data['lot_size'],
                        entry_price=pos_data['entry_price'],
                        stop_loss=pos_data.get('stop_loss', 0.0),
                        take_profit=pos_data.get('take_profit', 0.0),
                        timestamp=datetime.fromisoformat(pos_data['timestamp']),
                        position_id=pos_data['position_id'],
                        status=pos_data.get('status', 'OPEN'),
                        pnl=pos_data.get('pnl', 0.0)
                    )
                    positions.append(position)
                    self.positions[position.position_id] = position
                
                return PropShotResponse(success=True, data=positions)
            
            return response
            
        except Exception as e:
            logger.error(f"Pozitsiyalarni olishda xato: {e}")
            return PropShotResponse(success=False, error=str(e))
    
    async def close_position(self, position_id: str) -> PropShotResponse:
        """Pozitsiyani yopish"""
        try:
            close_data = {
                'account_id': self.account_id,
                'position_id': position_id,
                'timestamp': datetime.now().isoformat()
            }
            
            response = await self._make_request('POST', '/trade/close', close_data)
            
            if response.success:
                # Pozitsiyani o'chirish
                if position_id in self.positions:
                    del self.positions[position_id]
                logger.info(f"Pozitsiya yopildi: {position_id}")
            
            return response
            
        except Exception as e:
            logger.error(f"Pozitsiyani yopishda xato: {e}")
            return PropShotResponse(success=False, error=str(e))
    
    async def heartbeat(self) -> PropShotResponse:
        """Ulanish holatini tekshirish"""
        try:
            if not self.is_connected:
                return PropShotResponse(success=False, error="Ulanish yo'q")
            
            heartbeat_data = {
                'account_id': self.account_id,
                'timestamp': datetime.now().isoformat()
            }
            
            response = await self._make_request('POST', '/heartbeat', heartbeat_data)
            
            if response.success:
                self.last_heartbeat = datetime.now()
                logger.debug("Heartbeat muvaffaqiyatli")
            else:
                logger.warning(f"Heartbeat xatosi: {response.error}")
                self.is_connected = False
            
            return response
            
        except Exception as e:
            logger.error(f"Heartbeat xatosi: {e}")
            self.is_connected = False
            return PropShotResponse(success=False, error=str(e))
    
    async def _make_request(self, method: str, endpoint: str, data: Dict = None) -> PropShotResponse:
        """Propshot API ga so'rov yuborish"""
        try:
            await self.rate_limiter.wait()
            
            if not self.session:
                raise Exception("Session ochilmagan")
            
            url = f"{self.base_url}{endpoint}"
            
            async with self.session.request(
                method=method,
                url=url,
                json=data if data else None
            ) as response:
                
                response_data = await response.json()
                
                if response.status == 200:
                    return PropShotResponse(
                        success=True,
                        data=response_data.get('data'),
                        timestamp=datetime.now()
                    )
                else:
                    return PropShotResponse(
                        success=False,
                        error=response_data.get('error', 'Noma\'lum xato'),
                        error_code=response.status
                    )
                    
        except Exception as e:
            logger.error(f"API so'rov xatosi: {e}")
            return PropShotResponse(success=False, error=str(e))
    
    async def _check_risk_limits(self, trade: PropShotTrade) -> PropShotResponse:
        """Risk limitlarini tekshirish"""
        try:
            # Kunlik savdo limitini tekshirish
            if self.daily_trades >= self.max_daily_trades:
                return PropShotResponse(
                    success=False, 
                    error=f"Kunlik savdo limiti oshib ketdi: {self.daily_trades}/{self.max_daily_trades}"
                )
            
            # Lot hajmi limitini tekshirish
            if trade.volume > self.max_lot_size:
                return PropShotResponse(
                    success=False,
                    error=f"Lot hajmi limiti oshib ketdi: {trade.volume}/{self.max_lot_size}"
                )
            
            # Kunlik yo'qotish limitini tekshirish
            if self.daily_loss >= self.max_daily_loss:
                return PropShotResponse(
                    success=False,
                    error=f"Kunlik yo'qotish limiti oshib ketdi: {self.daily_loss}/{self.max_daily_loss}"
                )
            
            # Umumiy yo'qotish limitini tekshirish
            if self.total_loss >= self.max_total_loss:
                return PropShotResponse(
                    success=False,
                    error=f"Umumiy yo'qotish limiti oshib ketdi: {self.total_loss}/{self.max_total_loss}"
                )
            
            # Akavunt ma'lumotlarini yangilash
            account_response = await self.get_account_info()
            if not account_response.success:
                return PropShotResponse(
                    success=False,
                    error="Akavunt ma'lumotlarini olib bo'lmadi"
                )
            
            account = account_response.data
            if not account.is_active:
                return PropShotResponse(
                    success=False,
                    error="Akavunt faol emas"
                )
            
            # Margin tekshiruvi
            if account.margin_level < 100:
                return PropShotResponse(
                    success=False,
                    error=f"Margin level past: {account.margin_level}%"
                )
            
            return PropShotResponse(success=True, data={'risk_check': 'passed'})
            
        except Exception as e:
            logger.error(f"Risk tekshiruvida xato: {e}")
            return PropShotResponse(success=False, error=str(e))
    
    async def _update_account_info(self):
        """Akavunt ma'lumotlarini yangilash"""
        try:
            account_response = await self.get_account_info()
            if account_response.success:
                logger.debug("Akavunt ma'lumotlari yangilandi")
            else:
                logger.warning(f"Akavunt ma'lumotlarini yangilashda xato: {account_response.error}")
        except Exception as e:
            logger.error(f"Akavunt ma'lumotlarini yangilashda xato: {e}")
    
    def get_connection_status(self) -> Dict:
        """Ulanish holatini qaytarish"""
        return {
            'connected': self.is_connected,
            'last_heartbeat': self.last_heartbeat.isoformat(),
            'account_id': self.account_id,
            'positions_count': len(self.positions),
            'daily_trades': self.daily_trades,
            'daily_loss': self.daily_loss,
            'total_loss': self.total_loss
        }
    
    def format_signal_for_propshot(self, signal_data: Dict) -> PropShotTrade:
        """AI signalini PropShotTrade formatiga o'tkazish"""
        return PropShotTrade(
            symbol=signal_data['symbol'],
            action=signal_data['action'],
            volume=signal_data['lot_size'],
            price=signal_data.get('price', 0.0),
            stop_loss=signal_data.get('stop_loss', 0.0),
            take_profit=signal_data.get('take_profit', 0.0),
            comment=f"AI Signal - {signal_data.get('confidence', 0)}%",
            magic_number=signal_data.get('magic_number', 12345),
            trade_type='MARKET'
        )

# Propshot ulanish testlari
async def test_propshot_connection():
    """Propshot ulanishini test qilish"""
    try:
        from config.config import ConfigManager
        config = ConfigManager()
        
        async with PropShotConnector(config) as connector:
            # Ulanish testi
            connect_result = await connector.connect()
            print(f"Ulanish: {connect_result.success}")
            
            # Akavunt ma'lumotlari testi
            account_result = await connector.get_account_info()
            print(f"Akavunt: {account_result.success}")
            
            # Pozitsiyalar testi
            positions_result = await connector.get_positions()
            print(f"Pozitsiyalar: {positions_result.success}")
            
            # Test signali
            test_trade = PropShotTrade(
                symbol='EURUSD',
                action='BUY',
                volume=0.01,
                stop_loss=1.0900,
                take_profit=1.1000,
                comment='Test signal'
            )
            
            # Risk tekshiruvi
            risk_result = await connector._check_risk_limits(test_trade)
            print(f"Risk tekshiruvi: {risk_result.success}")
            
            print("Test yakunlandi")
            
    except Exception as e:
        print(f"Test xatosi: {e}")

if __name__ == "__main__":
    # Test ishlatish
    asyncio.run(test_propshot_connection())

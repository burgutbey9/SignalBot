"""
MetaTrader 5 Bridge - MT5 bilan bog'lanish va savdo bajarish
Bu modul MT5 terminal bilan bog'lanadi va signal bo'yicha savdo bajaradi
"""

import asyncio
import MetaTrader5 as mt5
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import pandas as pd
from enum import Enum
import time

from utils.logger import get_logger
from utils.error_handler import handle_api_error
from utils.retry_handler import retry_async
from config.config import ConfigManager

logger = get_logger(__name__)

class OrderType(Enum):
    """Savdo turlari"""
    BUY = mt5.ORDER_TYPE_BUY
    SELL = mt5.ORDER_TYPE_SELL
    BUY_LIMIT = mt5.ORDER_TYPE_BUY_LIMIT
    SELL_LIMIT = mt5.ORDER_TYPE_SELL_LIMIT
    BUY_STOP = mt5.ORDER_TYPE_BUY_STOP
    SELL_STOP = mt5.ORDER_TYPE_SELL_STOP

class TradeResult(Enum):
    """Savdo natijasi"""
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"
    PENDING = "pending"

@dataclass
class TradeRequest:
    """Savdo so'rovi"""
    symbol: str
    action: str  # "buy" yoki "sell"
    volume: float
    price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    comment: str = "AI_Signal"
    magic: int = 12345
    deviation: int = 20
    type_time: int = mt5.ORDER_TIME_GTC
    type_filling: int = mt5.ORDER_FILLING_IOC

@dataclass
class TradeResponse:
    """Savdo javobi"""
    success: bool
    order_id: Optional[int] = None
    position_id: Optional[int] = None
    volume: float = 0.0
    price: float = 0.0
    error: Optional[str] = None
    retcode: Optional[int] = None
    comment: Optional[str] = None
    request_id: Optional[int] = None

@dataclass
class PositionInfo:
    """Pozitsiya ma'lumotlari"""
    ticket: int
    symbol: str
    type: int
    volume: float
    price_open: float
    price_current: float
    profit: float
    swap: float
    comment: str
    magic: int
    time: datetime

@dataclass
class AccountInfo:
    """Akavunt ma'lumotlari"""
    balance: float
    equity: float
    margin: float
    free_margin: float
    margin_level: float
    profit: float
    currency: str
    leverage: int
    name: str
    server: str
    company: str

class MT5Bridge:
    """MetaTrader 5 Bridge - MT5 bilan bog'lanish"""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.is_connected = False
        self.account_info = None
        self.symbols_info = {}
        self.last_tick_time = {}
        
        # MT5 ulanish sozlamalari
        self.login = config.get_mt5_login()
        self.password = config.get_mt5_password()
        self.server = config.get_mt5_server()
        
        logger.info("MT5Bridge ishga tushirildi")
    
    async def connect(self) -> bool:
        """MT5 ga ulanish"""
        try:
            logger.info("MT5 ga ulanishga harakat qilmoqda...")
            
            # MT5 ni ishga tushirish
            if not mt5.initialize():
                error = mt5.last_error()
                logger.error(f"MT5 ishga tushmadi: {error}")
                return False
            
            # Akavuntga ulanish
            if not mt5.login(self.login, self.password, self.server):
                error = mt5.last_error()
                logger.error(f"MT5 akavuntga ulanmadi: {error}")
                mt5.shutdown()
                return False
            
            # Akavunt ma'lumotlarini olish
            account_info = mt5.account_info()
            if account_info is None:
                logger.error("Akavunt ma'lumotlari olinmadi")
                return False
            
            self.account_info = AccountInfo(
                balance=account_info.balance,
                equity=account_info.equity,
                margin=account_info.margin,
                free_margin=account_info.margin_free,
                margin_level=account_info.margin_level,
                profit=account_info.profit,
                currency=account_info.currency,
                leverage=account_info.leverage,
                name=account_info.name,
                server=account_info.server,
                company=account_info.company
            )
            
            self.is_connected = True
            logger.info(f"MT5 ga muvaffaqiyatli ulandi: {self.account_info.name}")
            logger.info(f"Balans: {self.account_info.balance} {self.account_info.currency}")
            
            return True
            
        except Exception as e:
            logger.error(f"MT5 ulanishda xato: {e}")
            return False
    
    async def disconnect(self):
        """MT5 dan uzish"""
        try:
            if self.is_connected:
                mt5.shutdown()
                self.is_connected = False
                logger.info("MT5 dan uzildi")
        except Exception as e:
            logger.error(f"MT5 dan uzishda xato: {e}")
    
    async def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """Symbol ma'lumotlarini olish"""
        try:
            if not self.is_connected:
                await self.connect()
            
            # Symbol ma'lumotlarini olish
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                logger.error(f"Symbol ma'lumotlari topilmadi: {symbol}")
                return None
            
            info = {
                'symbol': symbol_info.name,
                'digits': symbol_info.digits,
                'point': symbol_info.point,
                'spread': symbol_info.spread,
                'min_lot': symbol_info.volume_min,
                'max_lot': symbol_info.volume_max,
                'lot_step': symbol_info.volume_step,
                'tick_value': symbol_info.tick_value,
                'tick_size': symbol_info.tick_size,
                'contract_size': symbol_info.trade_contract_size,
                'margin_initial': symbol_info.margin_initial,
                'trade_allowed': symbol_info.trade_mode == mt5.SYMBOL_TRADE_MODE_FULL
            }
            
            self.symbols_info[symbol] = info
            return info
            
        except Exception as e:
            logger.error(f"Symbol ma'lumotlarini olishda xato: {e}")
            return None
    
    async def get_tick(self, symbol: str) -> Optional[Dict]:
        """Symbol uchun so'nggi tick olish"""
        try:
            if not self.is_connected:
                await self.connect()
            
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                logger.error(f"Tick ma'lumotlari topilmadi: {symbol}")
                return None
            
            tick_data = {
                'symbol': symbol,
                'time': datetime.fromtimestamp(tick.time),
                'bid': tick.bid,
                'ask': tick.ask,
                'spread': tick.ask - tick.bid,
                'volume': tick.volume,
                'flags': tick.flags
            }
            
            self.last_tick_time[symbol] = tick_data['time']
            return tick_data
            
        except Exception as e:
            logger.error(f"Tick olishda xato: {e}")
            return None
    
    async def validate_trade_request(self, request: TradeRequest) -> Tuple[bool, str]:
        """Savdo so'rovini tekshirish"""
        try:
            # Symbol ma'lumotlarini olish
            symbol_info = await self.get_symbol_info(request.symbol)
            if not symbol_info:
                return False, f"Symbol topilmadi: {request.symbol}"
            
            # Savdo ruxsati tekshirish
            if not symbol_info['trade_allowed']:
                return False, f"Symbol uchun savdo ruxsat etilmagan: {request.symbol}"
            
            # Lot hajmi tekshirish
            if request.volume < symbol_info['min_lot']:
                return False, f"Lot hajmi juda kichik: {request.volume} < {symbol_info['min_lot']}"
            
            if request.volume > symbol_info['max_lot']:
                return False, f"Lot hajmi juda katta: {request.volume} > {symbol_info['max_lot']}"
            
            # Lot qadami tekshirish
            lot_step = symbol_info['lot_step']
            if round(request.volume / lot_step) != request.volume / lot_step:
                return False, f"Lot hajmi noto'g'ri qadam: {request.volume}, qadam: {lot_step}"
            
            # Margin tekshirish
            if self.account_info:
                required_margin = request.volume * symbol_info['margin_initial']
                if required_margin > self.account_info.free_margin:
                    return False, f"Margin yetarli emas: {required_margin} > {self.account_info.free_margin}"
            
            return True, "OK"
            
        except Exception as e:
            logger.error(f"Savdo so'rovini tekshirishda xato: {e}")
            return False, f"Tekshirishda xato: {e}"
    
    @retry_async(max_retries=3, delay=1)
    async def send_order(self, request: TradeRequest) -> TradeResponse:
        """Savdo so'rovi yuborish"""
        try:
            if not self.is_connected:
                await self.connect()
            
            # So'rovni tekshirish
            is_valid, validation_message = await self.validate_trade_request(request)
            if not is_valid:
                return TradeResponse(
                    success=False,
                    error=validation_message
                )
            
            # So'nggi narxni olish
            tick = await self.get_tick(request.symbol)
            if not tick:
                return TradeResponse(
                    success=False,
                    error=f"Tick ma'lumotlari olinmadi: {request.symbol}"
                )
            
            # Savdo turiga qarab narx belgilash
            if request.action.lower() == "buy":
                order_type = mt5.ORDER_TYPE_BUY
                price = tick['ask'] if request.price is None else request.price
            else:
                order_type = mt5.ORDER_TYPE_SELL
                price = tick['bid'] if request.price is None else request.price
            
            # Savdo so'rovi yaratish
            trade_request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": request.symbol,
                "volume": request.volume,
                "type": order_type,
                "price": price,
                "deviation": request.deviation,
                "magic": request.magic,
                "comment": request.comment,
                "type_time": request.type_time,
                "type_filling": request.type_filling,
            }
            
            # Stop Loss va Take Profit qo'shish
            if request.stop_loss:
                trade_request["sl"] = request.stop_loss
            if request.take_profit:
                trade_request["tp"] = request.take_profit
            
            logger.info(f"Savdo so'rovi yuborilmoqda: {trade_request}")
            
            # So'rovni yuborish
            result = mt5.order_send(trade_request)
            
            if result is None:
                error = mt5.last_error()
                logger.error(f"Savdo so'rovi yuborilmadi: {error}")
                return TradeResponse(
                    success=False,
                    error=f"MT5 xatosi: {error}"
                )
            
            # Natijani qayta ishlash
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"Savdo muvaffaqiyatli bajarildi: {result.order}")
                return TradeResponse(
                    success=True,
                    order_id=result.order,
                    position_id=result.deal,
                    volume=result.volume,
                    price=result.price,
                    comment=result.comment,
                    request_id=result.request_id
                )
            else:
                logger.error(f"Savdo bajarilmadi: {result.retcode} - {result.comment}")
                return TradeResponse(
                    success=False,
                    error=f"Retcode: {result.retcode}, Comment: {result.comment}",
                    retcode=result.retcode,
                    comment=result.comment
                )
                
        except Exception as e:
            logger.error(f"Savdo so'rovi yuborishda xato: {e}")
            return TradeResponse(
                success=False,
                error=f"Xato: {e}"
            )
    
    async def get_positions(self, symbol: str = None) -> List[PositionInfo]:
        """Ochiq pozitsiyalarni olish"""
        try:
            if not self.is_connected:
                await self.connect()
            
            if symbol:
                positions = mt5.positions_get(symbol=symbol)
            else:
                positions = mt5.positions_get()
            
            if positions is None:
                logger.warning("Hech qanday pozitsiya topilmadi")
                return []
            
            position_list = []
            for pos in positions:
                position_info = PositionInfo(
                    ticket=pos.ticket,
                    symbol=pos.symbol,
                    type=pos.type,
                    volume=pos.volume,
                    price_open=pos.price_open,
                    price_current=pos.price_current,
                    profit=pos.profit,
                    swap=pos.swap,
                    comment=pos.comment,
                    magic=pos.magic,
                    time=datetime.fromtimestamp(pos.time)
                )
                position_list.append(position_info)
            
            return position_list
            
        except Exception as e:
            logger.error(f"Pozitsiyalarni olishda xato: {e}")
            return []
    
    async def close_position(self, position_ticket: int, volume: float = None) -> TradeResponse:
        """Pozitsiyani yopish"""
        try:
            if not self.is_connected:
                await self.connect()
            
            # Pozitsiya ma'lumotlarini olish
            position = mt5.positions_get(ticket=position_ticket)
            if not position:
                return TradeResponse(
                    success=False,
                    error=f"Pozitsiya topilmadi: {position_ticket}"
                )
            
            pos = position[0]
            
            # Yopish hajmi
            close_volume = volume if volume else pos.volume
            
            # Yopish turi
            if pos.type == mt5.POSITION_TYPE_BUY:
                close_type = mt5.ORDER_TYPE_SELL
            else:
                close_type = mt5.ORDER_TYPE_BUY
            
            # So'nggi narxni olish
            tick = await self.get_tick(pos.symbol)
            if not tick:
                return TradeResponse(
                    success=False,
                    error=f"Tick ma'lumotlari olinmadi: {pos.symbol}"
                )
            
            # Yopish narxi
            if close_type == mt5.ORDER_TYPE_BUY:
                price = tick['ask']
            else:
                price = tick['bid']
            
            # Yopish so'rovi
            close_request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": pos.symbol,
                "volume": close_volume,
                "type": close_type,
                "position": position_ticket,
                "price": price,
                "deviation": 20,
                "magic": pos.magic,
                "comment": "AI_Close",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            logger.info(f"Pozitsiya yopilmoqda: {position_ticket}")
            
            # So'rovni yuborish
            result = mt5.order_send(close_request)
            
            if result is None:
                error = mt5.last_error()
                logger.error(f"Pozitsiya yopilmadi: {error}")
                return TradeResponse(
                    success=False,
                    error=f"MT5 xatosi: {error}"
                )
            
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"Pozitsiya muvaffaqiyatli yopildi: {position_ticket}")
                return TradeResponse(
                    success=True,
                    order_id=result.order,
                    position_id=result.deal,
                    volume=result.volume,
                    price=result.price,
                    comment=result.comment
                )
            else:
                logger.error(f"Pozitsiya yopilmadi: {result.retcode} - {result.comment}")
                return TradeResponse(
                    success=False,
                    error=f"Retcode: {result.retcode}, Comment: {result.comment}",
                    retcode=result.retcode,
                    comment=result.comment
                )
                
        except Exception as e:
            logger.error(f"Pozitsiyani yopishda xato: {e}")
            return TradeResponse(
                success=False,
                error=f"Xato: {e}"
            )
    
    async def modify_position(self, position_ticket: int, stop_loss: float = None, 
                            take_profit: float = None) -> TradeResponse:
        """Pozitsiyani o'zgartirish (SL/TP)"""
        try:
            if not self.is_connected:
                await self.connect()
            
            # Pozitsiya ma'lumotlarini olish
            position = mt5.positions_get(ticket=position_ticket)
            if not position:
                return TradeResponse(
                    success=False,
                    error=f"Pozitsiya topilmadi: {position_ticket}"
                )
            
            pos = position[0]
            
            # O'zgartirish so'rovi
            modify_request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": pos.symbol,
                "position": position_ticket,
                "magic": pos.magic,
                "comment": "AI_Modify"
            }
            
            if stop_loss is not None:
                modify_request["sl"] = stop_loss
            if take_profit is not None:
                modify_request["tp"] = take_profit
            
            logger.info(f"Pozitsiya o'zgartirilmoqda: {position_ticket}")
            
            # So'rovni yuborish
            result = mt5.order_send(modify_request)
            
            if result is None:
                error = mt5.last_error()
                logger.error(f"Pozitsiya o'zgartirilmadi: {error}")
                return TradeResponse(
                    success=False,
                    error=f"MT5 xatosi: {error}"
                )
            
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"Pozitsiya muvaffaqiyatli o'zgartirildi: {position_ticket}")
                return TradeResponse(
                    success=True,
                    order_id=result.order,
                    position_id=position_ticket,
                    comment=result.comment
                )
            else:
                logger.error(f"Pozitsiya o'zgartirilmadi: {result.retcode} - {result.comment}")
                return TradeResponse(
                    success=False,
                    error=f"Retcode: {result.retcode}, Comment: {result.comment}",
                    retcode=result.retcode,
                    comment=result.comment
                )
                
        except Exception as e:
            logger.error(f"Pozitsiyani o'zgartirishda xato: {e}")
            return TradeResponse(
                success=False,
                error=f"Xato: {e}"
            )
    
    async def get_account_info(self) -> Optional[AccountInfo]:
        """Akavunt ma'lumotlarini yangilash"""
        try:
            if not self.is_connected:
                await self.connect()
            
            account_info = mt5.account_info()
            if account_info is None:
                logger.error("Akavunt ma'lumotlari olinmadi")
                return None
            
            self.account_info = AccountInfo(
                balance=account_info.balance,
                equity=account_info.equity,
                margin=account_info.margin,
                free_margin=account_info.margin_free,
                margin_level=account_info.margin_level,
                profit=account_info.profit,
                currency=account_info.currency,
                leverage=account_info.leverage,
                name=account_info.name,
                server=account_info.server,
                company=account_info.company
            )
            
            return self.account_info
            
        except Exception as e:
            logger.error(f"Akavunt ma'lumotlarini olishda xato: {e}")
            return None
    
    async def get_historical_data(self, symbol: str, timeframe: str, 
                                count: int = 1000) -> Optional[pd.DataFrame]:
        """Tarixiy ma'lumotlarni olish"""
        try:
            if not self.is_connected:
                await self.connect()
            
            # Timeframe konvertatsiyasi
            mt5_timeframe = {
                "M1": mt5.TIMEFRAME_M1,
                "M5": mt5.TIMEFRAME_M5,
                "M15": mt5.TIMEFRAME_M15,
                "M30": mt5.TIMEFRAME_M30,
                "H1": mt5.TIMEFRAME_H1,
                "H4": mt5.TIMEFRAME_H4,
                "D1": mt5.TIMEFRAME_D1,
                "W1": mt5.TIMEFRAME_W1,
                "MN1": mt5.TIMEFRAME_MN1
            }.get(timeframe.upper(), mt5.TIMEFRAME_H1)
            
            # Tarixiy ma'lumotlarni olish
            rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, count)
            
            if rates is None:
                logger.error(f"Tarixiy ma'lumotlar olinmadi: {symbol}")
                return None
            
            # DataFrame ga konvertatsiya
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            logger.info(f"Tarixiy ma'lumotlar olindi: {symbol}, {len(df)} bars")
            return df
            
        except Exception as e:
            logger.error(f"Tarixiy ma'lumotlarni olishda xato: {e}")
            return None
    
    async def check_connection(self) -> bool:
        """Ulanish holatini tekshirish"""
        try:
            if not self.is_connected:
                return False
            
            # Terminal ma'lumotlarini olish
            terminal_info = mt5.terminal_info()
            if terminal_info is None:
                logger.warning("Terminal ma'lumotlari olinmadi")
                self.is_connected = False
                return False
            
            # Akavunt ma'lumotlarini tekshirish
            account_info = mt5.account_info()
            if account_info is None:
                logger.warning("Akavunt ma'lumotlari olinmadi")
                self.is_connected = False
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Ulanishni tekshirishda xato: {e}")
            self.is_connected = False
            return False
    
    async def get_margin_required(self, symbol: str, volume: float, 
                                order_type: str = "buy") -> Optional[float]:
        """Kerakli margin hisoblash"""
        try:
            if not self.is_connected:
                await self.connect()
            
            # Symbol ma'lumotlarini olish
            symbol_info = await self.get_symbol_info(symbol)
            if not symbol_info:
                return None
            
            # Tick ma'lumotlarini olish
            tick = await self.get_tick(symbol)
            if not tick:
                return None
            
            # Margin hisoblash
            if order_type.lower() == "buy":
                price = tick['ask']
            else:
                price = tick['bid']
            
            # Margin formula: Volume * Contract_Size * Price / Leverage
            margin_required = (volume * symbol_info['contract_size'] * price) / self.account_info.leverage
            
            return margin_required
            
        except Exception as e:
            logger.error(f"Margin hisoblashda xato: {e}")
            return None
    
    def __del__(self):
        """Obyekt yo'q qilinishida ulanishni uzish"""
        try:
            if self.is_connected:
                mt5.shutdown()
        except:
            pass

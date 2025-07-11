import asyncio
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json
from decimal import Decimal

from utils.logger import get_logger
from utils.error_handler import handle_processing_error
from utils.retry_handler import retry_async
from config.config import ConfigManager
from risk_management.risk_calculator import RiskCalculator
from risk_management.position_sizer import PositionSizer
from risk_management.trade_monitor import TradeMonitor
from trading.propshot_connector import PropshotConnector
from trading.mt5_bridge import MT5Bridge
from api_clients.telegram_client import TelegramClient
from database.db_manager import DatabaseManager

logger = get_logger(__name__)

class OrderType(Enum):
    """Savdo buyruqi turlari"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class OrderStatus(Enum):
    """Savdo buyruqi holatlari"""
    PENDING = "pending"
    EXECUTED = "executed"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    PARTIAL = "partial"

class ExecutionMode(Enum):
    """Bajarish rejimlari"""
    MANUAL = "manual"      # Qo'lda tasdiqlash
    AUTO = "auto"          # Avtomatik bajarish
    SEMI_AUTO = "semi_auto" # Yarim avtomatik

@dataclass
class TradeOrder:
    """Savdo buyrugi ma'lumotlari"""
    symbol: str
    action: str  # BUY/SELL
    order_type: OrderType
    lot_size: float
    price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    magic_number: int = 12345
    comment: str = "AI OrderFlow Bot"
    expiration: Optional[datetime] = None
    slippage: int = 3
    
    # Qo'shimcha ma'lumotlar
    signal_confidence: float = 0.0
    risk_percent: float = 0.0
    reason: str = ""
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class ExecutionResult:
    """Bajarish natijasi"""
    success: bool
    order_id: Optional[str] = None
    execution_price: Optional[float] = None
    execution_time: Optional[datetime] = None
    error: Optional[str] = None
    broker_response: Optional[Dict] = None
    latency_ms: int = 0

@dataclass
class ExecutionStats:
    """Bajarish statistikasi"""
    total_orders: int = 0
    successful_orders: int = 0
    failed_orders: int = 0
    cancelled_orders: int = 0
    average_latency_ms: float = 0.0
    slippage_stats: List[float] = field(default_factory=list)
    success_rate: float = 0.0

class ExecutionEngine:
    """Savdo bajarish mexanizmi"""
    
    def __init__(self):
        self.config = ConfigManager()
        self.risk_calculator = RiskCalculator()
        self.position_sizer = PositionSizer()
        self.trade_monitor = TradeMonitor()
        self.propshot_connector = PropshotConnector()
        self.mt5_bridge = MT5Bridge()
        self.telegram_client = TelegramClient()
        self.db_manager = DatabaseManager()
        
        # Bajarish sozlamalari
        self.execution_mode = ExecutionMode.MANUAL
        self.max_slippage = 5.0  # pips
        self.max_execution_time = 30  # seconds
        self.order_timeout = 300  # seconds
        
        # Statistika
        self.stats = ExecutionStats()
        self.pending_orders: Dict[str, TradeOrder] = {}
        self.executed_orders: Dict[str, ExecutionResult] = {}
        
        # Ishga tushirish
        self.is_running = False
        self.last_execution_time = 0
        
        logger.info("ExecutionEngine ishga tushirildi")
    
    async def initialize(self) -> bool:
        """Bajarish mexanizmini ishga tushirish"""
        try:
            logger.info("ExecutionEngine ishga tushirish boshlandi")
            
            # Komponentlarni ishga tushirish
            await self.propshot_connector.initialize()
            await self.mt5_bridge.initialize()
            await self.telegram_client.initialize()
            
            # Sozlamalarni yuklash
            await self._load_settings()
            
            # Statistikani yuklash
            await self._load_stats()
            
            self.is_running = True
            logger.info("ExecutionEngine muvaffaqiyatli ishga tushirildi")
            return True
            
        except Exception as e:
            logger.error(f"ExecutionEngine ishga tushirishda xato: {e}")
            return False
    
    async def shutdown(self) -> None:
        """Bajarish mexanizmini to'xtatish"""
        try:
            logger.info("ExecutionEngine to'xtatish boshlandi")
            
            self.is_running = False
            
            # Kutilayotgan buyruqlarni bekor qilish
            await self._cancel_pending_orders()
            
            # Statistikani saqlash
            await self._save_stats()
            
            # Komponentlarni to'xtatish
            await self.propshot_connector.shutdown()
            await self.mt5_bridge.shutdown()
            
            logger.info("ExecutionEngine to'xtatildi")
            
        except Exception as e:
            logger.error(f"ExecutionEngine to'xtatishda xato: {e}")
    
    async def execute_signal(self, signal_data: Dict) -> ExecutionResult:
        """Signalni bajarish"""
        try:
            logger.info(f"Signal bajarish boshlandi: {signal_data.get('symbol')}")
            
            # Signalni buyruqqa aylantirish
            order = await self._create_order_from_signal(signal_data)
            if not order:
                return ExecutionResult(
                    success=False,
                    error="Signalni buyruqqa aylantirish xatosi"
                )
            
            # Risk tekshiruvi
            risk_check = await self.risk_calculator.check_trade_risk(order.__dict__)
            if not risk_check['allowed']:
                error_msg = f"Risk tekshiruvi o'tmadi: {risk_check['reason']}"
                logger.warning(error_msg)
                
                # Telegram ga xabar yuborish
                await self._send_telegram_alert(
                    f"❌ SAVDO BEKOR QILINDI\n"
                    f"📊 {order.symbol}\n"
                    f"🚫 Sabab: {risk_check['reason']}\n"
                    f"⏰ Vaqt: {datetime.now().strftime('%H:%M:%S')}"
                )
                
                return ExecutionResult(success=False, error=error_msg)
            
            # Bajarish rejimiga qarab harakat qilish
            if self.execution_mode == ExecutionMode.MANUAL:
                return await self._execute_manual(order)
            elif self.execution_mode == ExecutionMode.AUTO:
                return await self._execute_auto(order)
            else:  # SEMI_AUTO
                return await self._execute_semi_auto(order)
                
        except Exception as e:
            logger.error(f"Signal bajarishda xato: {e}")
            return ExecutionResult(success=False, error=str(e))
    
    async def _create_order_from_signal(self, signal_data: Dict) -> Optional[TradeOrder]:
        """Signalni savdo buyrug'iga aylantirish"""
        try:
            # Asosiy ma'lumotlarni olish
            symbol = signal_data.get('symbol')
            action = signal_data.get('action')  # BUY/SELL
            confidence = signal_data.get('confidence', 0.0)
            
            if not symbol or not action:
                logger.error("Signal ma'lumotlari to'liq emas")
                return None
            
            # Narxni aniqlash
            current_price = await self._get_current_price(symbol)
            if not current_price:
                logger.error(f"Joriy narx olinmadi: {symbol}")
                return None
            
            # Lot hajmini hisoblash
            lot_size = await self.position_sizer.calculate_position_size(
                symbol=symbol,
                entry_price=current_price,
                stop_loss=signal_data.get('stop_loss'),
                risk_percent=signal_data.get('risk_percent', 0.02)
            )
            
            # Stop Loss va Take Profit ni hisoblash
            sl_price = signal_data.get('stop_loss')
            tp_price = signal_data.get('take_profit')
            
            if not sl_price or not tp_price:
                # Avtomatik hisoblash
                sl_tp = await self._calculate_sl_tp(symbol, current_price, action)
                sl_price = sl_tp.get('stop_loss', sl_price)
                tp_price = sl_tp.get('take_profit', tp_price)
            
            # Buyruq yaratish
            order = TradeOrder(
                symbol=symbol,
                action=action,
                order_type=OrderType.MARKET,
                lot_size=lot_size,
                price=current_price,
                stop_loss=sl_price,
                take_profit=tp_price,
                signal_confidence=confidence,
                risk_percent=signal_data.get('risk_percent', 0.02),
                reason=signal_data.get('reason', 'AI Signal')
            )
            
            logger.info(f"Buyruq yaratildi: {symbol} {action} {lot_size} lot")
            return order
            
        except Exception as e:
            logger.error(f"Buyruq yaratishda xato: {e}")
            return None
    
    async def _execute_manual(self, order: TradeOrder) -> ExecutionResult:
        """Qo'lda bajarish - foydalanuvchi tasdig'i kutiladi"""
        try:
            logger.info(f"Qo'lda bajarish rejimi: {order.symbol}")
            
            # Telegram orqali tasdiq so'rash
            message = await self._format_confirmation_message(order)
            sent_msg = await self.telegram_client.send_message_with_buttons(
                message=message,
                buttons=[
                    [{"text": "✅ TASDIQLASH", "callback_data": f"confirm_{order.symbol}"}],
                    [{"text": "❌ BEKOR QILISH", "callback_data": f"cancel_{order.symbol}"}]
                ]
            )
            
            if not sent_msg:
                return ExecutionResult(success=False, error="Telegram xabar yuborilmadi")
            
            # Kutilayotgan buyruqlar ro'yxatiga qo'shish
            order_id = f"manual_{int(time.time())}"
            self.pending_orders[order_id] = order
            
            # Tasdiq kutish (timeout bilan)
            result = await self._wait_for_confirmation(order_id)
            
            return result
            
        except Exception as e:
            logger.error(f"Qo'lda bajarishda xato: {e}")
            return ExecutionResult(success=False, error=str(e))
    
    async def _execute_auto(self, order: TradeOrder) -> ExecutionResult:
        """Avtomatik bajarish"""
        try:
            logger.info(f"Avtomatik bajarish: {order.symbol}")
            
            # Telegram ga xabar yuborish
            await self._send_telegram_alert(
                f"🚀 AVTOMATIK SAVDO BOSHLANDI\n"
                f"📊 {order.symbol} {order.action}\n"
                f"💰 Lot: {order.lot_size}\n"
                f"⏰ Vaqt: {datetime.now().strftime('%H:%M:%S')}"
            )
            
            # Savdoni bajarish
            result = await self._execute_trade(order)
            
            # Natijani yuborish
            if result.success:
                await self._send_telegram_alert(
                    f"✅ SAVDO MUVAFFAQIYATLI\n"
                    f"📊 {order.symbol} {order.action}\n"
                    f"💰 Lot: {order.lot_size}\n"
                    f"💵 Narx: {result.execution_price}\n"
                    f"🆔 ID: {result.order_id}\n"
                    f"⏰ Vaqt: {result.execution_time}"
                )
            else:
                await self._send_telegram_alert(
                    f"❌ SAVDO XATOSI\n"
                    f"📊 {order.symbol} {order.action}\n"
                    f"🚫 Xato: {result.error}\n"
                    f"⏰ Vaqt: {datetime.now().strftime('%H:%M:%S')}"
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Avtomatik bajarishda xato: {e}")
            return ExecutionResult(success=False, error=str(e))
    
    async def _execute_semi_auto(self, order: TradeOrder) -> ExecutionResult:
        """Yarim avtomatik bajarish"""
        try:
            logger.info(f"Yarim avtomatik bajarish: {order.symbol}")
            
            # Yuqori confidence bo'lsa avtomatik
            if order.signal_confidence >= 80.0:
                return await self._execute_auto(order)
            else:
                return await self._execute_manual(order)
                
        except Exception as e:
            logger.error(f"Yarim avtomatik bajarishda xato: {e}")
            return ExecutionResult(success=False, error=str(e))
    
    async def _execute_trade(self, order: TradeOrder) -> ExecutionResult:
        """Savdoni bajarish"""
        try:
            start_time = time.time()
            
            # Propshot orqali bajarish
            if await self.propshot_connector.is_connected():
                result = await self.propshot_connector.execute_trade(order)
                if result.success:
                    # Statistikani yangilash
                    self._update_stats(result, time.time() - start_time)
                    return result
            
            # MT5 orqali bajarish (fallback)
            if await self.mt5_bridge.is_connected():
                result = await self.mt5_bridge.execute_trade(order)
                if result.success:
                    # Statistikani yangilash
                    self._update_stats(result, time.time() - start_time)
                    return result
            
            return ExecutionResult(
                success=False,
                error="Hech qanday broker ulanmagan"
            )
            
        except Exception as e:
            logger.error(f"Savdo bajarishda xato: {e}")
            return ExecutionResult(success=False, error=str(e))
    
    async def _wait_for_confirmation(self, order_id: str) -> ExecutionResult:
        """Foydalanuvchi tasdig'ini kutish"""
        try:
            timeout = self.order_timeout
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                # Telegram callback larni tekshirish
                callback = await self.telegram_client.get_pending_callback()
                if callback:
                    if callback.startswith(f"confirm_{order_id}"):
                        # Tasdiqlandi - savdoni bajarish
                        order = self.pending_orders.pop(order_id, None)
                        if order:
                            return await self._execute_trade(order)
                    elif callback.startswith(f"cancel_{order_id}"):
                        # Bekor qilindi
                        self.pending_orders.pop(order_id, None)
                        return ExecutionResult(
                            success=False,
                            error="Foydalanuvchi tomonidan bekor qilindi"
                        )
                
                await asyncio.sleep(1)
            
            # Timeout
            self.pending_orders.pop(order_id, None)
            return ExecutionResult(
                success=False,
                error="Tasdiq kutish vaqti tugadi"
            )
            
        except Exception as e:
            logger.error(f"Tasdiq kutishda xato: {e}")
            return ExecutionResult(success=False, error=str(e))
    
    async def _get_current_price(self, symbol: str) -> Optional[float]:
        """Joriy narxni olish"""
        try:
            # Propshot dan narx olish
            if await self.propshot_connector.is_connected():
                price = await self.propshot_connector.get_current_price(symbol)
                if price:
                    return price
            
            # MT5 dan narx olish
            if await self.mt5_bridge.is_connected():
                price = await self.mt5_bridge.get_current_price(symbol)
                if price:
                    return price
            
            logger.warning(f"Joriy narx olinmadi: {symbol}")
            return None
            
        except Exception as e:
            logger.error(f"Narx olishda xato: {e}")
            return None
    
    async def _calculate_sl_tp(self, symbol: str, price: float, action: str) -> Dict:
        """Stop Loss va Take Profit ni hisoblash"""
        try:
            # ATR asosida hisoblash
            atr = await self._get_atr(symbol)
            if not atr:
                atr = price * 0.01  # 1% default
            
            if action == "BUY":
                sl = price - (atr * 2)
                tp = price + (atr * 3)
            else:  # SELL
                sl = price + (atr * 2)
                tp = price - (atr * 3)
            
            return {
                'stop_loss': round(sl, 5),
                'take_profit': round(tp, 5),
                'atr': atr
            }
            
        except Exception as e:
            logger.error(f"SL/TP hisoblashda xato: {e}")
            return {}
    
    async def _get_atr(self, symbol: str) -> Optional[float]:
        """ATR indikatorini olish"""
        try:
            # MT5 dan ATR olish
            if await self.mt5_bridge.is_connected():
                atr = await self.mt5_bridge.get_atr(symbol, period=14)
                if atr:
                    return atr
            
            return None
            
        except Exception as e:
            logger.error(f"ATR olishda xato: {e}")
            return None
    
    async def _format_confirmation_message(self, order: TradeOrder) -> str:
        """Tasdiq xabarini formatlash"""
        try:
            # Pips hisoblash
            if order.stop_loss and order.take_profit:
                if order.action == "BUY":
                    sl_pips = (order.price - order.stop_loss) * 10000
                    tp_pips = (order.take_profit - order.price) * 10000
                else:
                    sl_pips = (order.stop_loss - order.price) * 10000
                    tp_pips = (order.price - order.take_profit) * 10000
            else:
                sl_pips = tp_pips = 0
            
            message = f"""
📊 SAVDO TASDIG'I KERAK
════════════════════════
📈 Savdo: {order.action} {order.symbol}
💰 Narx: {order.price}
📊 Lot: {order.lot_size} lot
🛡️ Stop Loss: {order.stop_loss} ({sl_pips:.1f} pips)
🎯 Take Profit: {order.take_profit} ({tp_pips:.1f} pips)
⚡ Ishonch: {order.signal_confidence}%
🔥 Risk: {order.risk_percent}%
════════════════════════
📝 Sabab: {order.reason}
⏰ Vaqt: {order.created_at.strftime('%H:%M:%S')} (UZB)

❓ Savdoni bajarasizmi?
            """
            
            return message.strip()
            
        except Exception as e:
            logger.error(f"Xabar formatlashda xato: {e}")
            return "Xabar formatlashda xato"
    
    async def _send_telegram_alert(self, message: str) -> None:
        """Telegram ga ogohlantirish yuborish"""
        try:
            await self.telegram_client.send_message(message)
        except Exception as e:
            logger.error(f"Telegram xabar yuborishda xato: {e}")
    
    async def _cancel_pending_orders(self) -> None:
        """Kutilayotgan buyruqlarni bekor qilish"""
        try:
            if self.pending_orders:
                logger.info(f"Kutilayotgan {len(self.pending_orders)} buyruq bekor qilinmoqda")
                
                for order_id in list(self.pending_orders.keys()):
                    self.pending_orders.pop(order_id, None)
                
                await self._send_telegram_alert(
                    f"⚠️ TIZIM TO'XTATILDI\n"
                    f"📊 Kutilayotgan buyruqlar bekor qilindi\n"
                    f"⏰ Vaqt: {datetime.now().strftime('%H:%M:%S')}"
                )
                
        except Exception as e:
            logger.error(f"Buyruqlarni bekor qilishda xato: {e}")
    
    def _update_stats(self, result: ExecutionResult, latency: float) -> None:
        """Statistikani yangilash"""
        try:
            self.stats.total_orders += 1
            
            if result.success:
                self.stats.successful_orders += 1
            else:
                self.stats.failed_orders += 1
            
            # Latency statistikasi
            total_latency = self.stats.average_latency_ms * (self.stats.total_orders - 1)
            self.stats.average_latency_ms = (total_latency + latency * 1000) / self.stats.total_orders
            
            # Muvaffaqiyat darajasi
            if self.stats.total_orders > 0:
                self.stats.success_rate = (self.stats.successful_orders / self.stats.total_orders) * 100
            
            logger.info(f"Statistika yangilandi: {self.stats.success_rate:.1f}% muvaffaqiyat")
            
        except Exception as e:
            logger.error(f"Statistika yangilashda xato: {e}")
    
    async def _load_settings(self) -> None:
        """Sozlamalarni yuklash"""
        try:
            settings = self.config.get_trading_settings()
            
            self.execution_mode = ExecutionMode(settings.get('execution_mode', 'manual'))
            self.max_slippage = settings.get('max_slippage', 5.0)
            self.max_execution_time = settings.get('max_execution_time', 30)
            self.order_timeout = settings.get('order_timeout', 300)
            
            logger.info(f"Sozlamalar yuklandi: {self.execution_mode.value} rejim")
            
        except Exception as e:
            logger.error(f"Sozlamalar yuklashda xato: {e}")
    
    async def _load_stats(self) -> None:
        """Statistikani yuklash"""
        try:
            stats_data = await self.db_manager.get_execution_stats()
            if stats_data:
                self.stats = ExecutionStats(**stats_data)
                logger.info("Statistika yuklandi")
            
        except Exception as e:
            logger.error(f"Statistika yuklashda xato: {e}")
    
    async def _save_stats(self) -> None:
        """Statistikani saqlash"""
        try:
            await self.db_manager.save_execution_stats(self.stats.__dict__)
            logger.info("Statistika saqlandi")
            
        except Exception as e:
            logger.error(f"Statistika saqlashda xato: {e}")
    
    async def get_stats(self) -> ExecutionStats:
        """Statistikani olish"""
        return self.stats
    
    async def set_execution_mode(self, mode: ExecutionMode) -> None:
        """Bajarish rejimini o'rnatish"""
        self.execution_mode = mode
        logger.info(f"Bajarish rejimi o'zgartirildi: {mode.value}")
        
        await self._send_telegram_alert(
            f"⚙️ REJIM O'ZGARTIRILDI\n"
            f"📊 Yangi rejim: {mode.value}\n"
            f"⏰ Vaqt: {datetime.now().strftime('%H:%M:%S')}"
        )
    
    async def cancel_order(self, order_id: str) -> bool:
        """Buyruqni bekor qilish"""
        try:
            if order_id in self.pending_orders:
                self.pending_orders.pop(order_id)
                logger.info(f"Buyruq bekor qilindi: {order_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Buyruq bekor qilishda xato: {e}")
            return False
    
    async def get_pending_orders(self) -> Dict[str, TradeOrder]:
        """Kutilayotgan buyruqlarni olish"""
        return self.pending_orders.copy()
    
    async def is_market_open(self) -> bool:
        """Bozor ochiq ekanligini tekshirish"""
        try:
            # MT5 orqali bozor holatini tekshirish
            if await self.mt5_bridge.is_connected():
                return await self.mt5_bridge.is_market_open()
            
            # Propshot orqali tekshirish
            if await self.propshot_connector.is_connected():
                return await self.propshot_connector.is_market_open()
            
            return False
            
        except Exception as e:
            logger.error(f"Bozor holati tekshirishda xato: {e}")
            return False

import asyncio
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
from pathlib import Path

from utils.logger import get_logger
from utils.error_handler import handle_processing_error
from config.config import ConfigManager
from database.models import TradeRecord, AccountStatus

logger = get_logger(__name__)

class TradeStatus(Enum):
    """Trade holati"""
    PENDING = "pending"
    ACTIVE = "active"
    CLOSED = "closed"
    CANCELLED = "cancelled"
    FAILED = "failed"

class RiskLevel(Enum):
    """Risk darajasi"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class TradeMetrics:
    """Trade metrikalari"""
    trade_id: str
    symbol: str
    action: str  # BUY/SELL
    entry_price: float
    current_price: float
    lot_size: float
    profit_loss: float
    profit_loss_percent: float
    duration: int  # seconds
    status: TradeStatus
    risk_level: RiskLevel
    stop_loss: float
    take_profit: float
    trailing_stop: bool = False
    max_drawdown: float = 0.0
    max_profit: float = 0.0
    commission: float = 0.0

@dataclass
class DailyLimits:
    """Kunlik limitlar"""
    max_daily_loss: float = 0.025  # 2.5%
    max_total_loss: float = 0.05   # 5%
    max_lot_size: float = 0.5      # 0.5 lot
    max_daily_trades: int = 3      # 3 ta trade
    max_consecutive_losses: int = 3
    max_drawdown: float = 0.03     # 3%
    
    # Propshot qoidalari
    propshot_max_daily_loss: float = 0.02  # 2%
    propshot_max_total_loss: float = 0.04  # 4%
    propshot_max_lot_size: float = 0.3     # 0.3 lot
    propshot_max_daily_trades: int = 2     # 2 ta trade

@dataclass
class AccountMetrics:
    """Akavunt metrikalari"""
    balance: float
    equity: float
    margin: float
    free_margin: float
    margin_level: float
    daily_profit_loss: float
    total_profit_loss: float
    daily_trades_count: int
    total_trades_count: int
    win_rate: float
    consecutive_losses: int
    current_drawdown: float
    max_drawdown: float
    last_update: datetime = field(default_factory=datetime.now)

@dataclass
class MonitoringResult:
    """Monitoring natijasi"""
    success: bool
    can_trade: bool
    risk_level: RiskLevel
    active_trades: List[TradeMetrics]
    account_metrics: AccountMetrics
    limit_violations: List[str]
    warnings: List[str]
    recommendations: List[str]
    error: Optional[str] = None

class TradeMonitor:
    """Trade kuzatuv va limit monitoring"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        self.name = "TradeMonitor"
        self.active_trades: Dict[str, TradeMetrics] = {}
        self.account_metrics: Optional[AccountMetrics] = None
        self.daily_limits = DailyLimits()
        self.last_update = datetime.now()
        self.monitoring_active = False
        
        # Propshot rejimi
        self.propshot_mode = self.config.get_trading_config().get('propshot_enabled', False)
        
        logger.info(f"{self.name} ishga tushirildi")
        if self.propshot_mode:
            logger.info("Propshot rejimi yoqilgan - qattiq limitlar")
    
    async def start_monitoring(self) -> None:
        """Monitoring jarayonini boshlash"""
        try:
            self.monitoring_active = True
            logger.info("Trade monitoring boshlandi")
            
            # Boshlang'ich ma'lumotlarni yuklash
            await self._load_initial_data()
            
            # Monitoring tsikli
            while self.monitoring_active:
                await self._monitor_cycle()
                await asyncio.sleep(5)  # 5 soniya kutish
                
        except Exception as e:
            logger.error(f"Monitoring boshlashda xato: {e}")
            self.monitoring_active = False
    
    async def stop_monitoring(self) -> None:
        """Monitoring jarayonini to'xtatish"""
        self.monitoring_active = False
        logger.info("Trade monitoring to'xtatildi")
    
    async def _load_initial_data(self) -> None:
        """Boshlang'ich ma'lumotlarni yuklash"""
        try:
            # Faol tradelarni yuklash
            await self._load_active_trades()
            
            # Akavunt metrikalari
            await self._update_account_metrics()
            
            # Kunlik limitlarni yuklash
            await self._load_daily_limits()
            
            logger.info("Boshlang'ich ma'lumotlar yuklandi")
            
        except Exception as e:
            logger.error(f"Boshlang'ich ma'lumotlar yuklanmadi: {e}")
    
    async def _monitor_cycle(self) -> None:
        """Monitoring tsikli"""
        try:
            # Akavunt metrikalari yangilash
            await self._update_account_metrics()
            
            # Faol tradelarni yangilash
            await self._update_active_trades()
            
            # Limitlarni tekshirish
            await self._check_limits()
            
            # Risk baholash
            await self._assess_risk()
            
            # Ogohlantirish yuborish
            await self._send_alerts()
            
        except Exception as e:
            logger.error(f"Monitoring tsiklida xato: {e}")
    
    async def _load_active_trades(self) -> None:
        """Faol tradelarni yuklash"""
        try:
            # MT5 dan faol tradelarni olish
            # Bu qism MT5 bridge bilan bog'lanadi
            pass
            
        except Exception as e:
            logger.error(f"Faol tradelar yuklanmadi: {e}")
    
    async def _update_account_metrics(self) -> None:
        """Akavunt metrikalari yangilash"""
        try:
            # MT5 dan akavunt ma'lumotlarini olish
            # Bu qism MT5 bridge bilan bog'lanadi
            
            # Namunaviy ma'lumotlar
            self.account_metrics = AccountMetrics(
                balance=10000.0,
                equity=10000.0,
                margin=0.0,
                free_margin=10000.0,
                margin_level=0.0,
                daily_profit_loss=0.0,
                total_profit_loss=0.0,
                daily_trades_count=0,
                total_trades_count=0,
                win_rate=0.0,
                consecutive_losses=0,
                current_drawdown=0.0,
                max_drawdown=0.0
            )
            
        except Exception as e:
            logger.error(f"Akavunt metrikalari yangilanmadi: {e}")
    
    async def _update_active_trades(self) -> None:
        """Faol tradelarni yangilash"""
        try:
            current_time = datetime.now()
            
            for trade_id, trade in self.active_trades.items():
                # Trade ma'lumotlarini yangilash
                await self._update_trade_metrics(trade)
                
                # Trailing stop tekshirish
                if trade.trailing_stop:
                    await self._update_trailing_stop(trade)
                
                # Trade yopilishi tekshirish
                if await self._check_trade_close_conditions(trade):
                    await self._close_trade(trade_id, "auto_close")
                    
        except Exception as e:
            logger.error(f"Faol tradolar yangilanmadi: {e}")
    
    async def _update_trade_metrics(self, trade: TradeMetrics) -> None:
        """Trade metrikalari yangilash"""
        try:
            # MT5 dan joriy narxni olish
            # current_price = await self._get_current_price(trade.symbol)
            current_price = trade.current_price  # Placeholder
            
            # Foyda/zarar hisoblash
            if trade.action == "BUY":
                profit_loss = (current_price - trade.entry_price) * trade.lot_size * 100000
            else:  # SELL
                profit_loss = (trade.entry_price - current_price) * trade.lot_size * 100000
            
            trade.current_price = current_price
            trade.profit_loss = profit_loss
            trade.profit_loss_percent = (profit_loss / self.account_metrics.balance) * 100
            
            # Maksimal foyda/zarar yangilash
            trade.max_profit = max(trade.max_profit, profit_loss)
            trade.max_drawdown = min(trade.max_drawdown, profit_loss)
            
            # Risk darajasi
            trade.risk_level = self._calculate_trade_risk(trade)
            
        except Exception as e:
            logger.error(f"Trade metrikalari yangilanmadi: {e}")
    
    async def _update_trailing_stop(self, trade: TradeMetrics) -> None:
        """Trailing stop yangilash"""
        try:
            # Trailing stop logikasi
            if trade.profit_loss > 0:
                # Foydali trade uchun stop loss ni siljitish
                trailing_distance = 0.001  # 10 pips
                
                if trade.action == "BUY":
                    new_stop = trade.current_price - trailing_distance
                    if new_stop > trade.stop_loss:
                        trade.stop_loss = new_stop
                        logger.info(f"Trailing stop yangilandi: {trade.trade_id} - {new_stop}")
                else:  # SELL
                    new_stop = trade.current_price + trailing_distance
                    if new_stop < trade.stop_loss:
                        trade.stop_loss = new_stop
                        logger.info(f"Trailing stop yangilandi: {trade.trade_id} - {new_stop}")
                        
        except Exception as e:
            logger.error(f"Trailing stop yangilanmadi: {e}")
    
    async def _check_trade_close_conditions(self, trade: TradeMetrics) -> bool:
        """Trade yopilishi sharoitlari tekshirish"""
        try:
            # Stop Loss tekshirish
            if trade.action == "BUY" and trade.current_price <= trade.stop_loss:
                logger.warning(f"Stop Loss ishga tushdi: {trade.trade_id}")
                return True
            
            if trade.action == "SELL" and trade.current_price >= trade.stop_loss:
                logger.warning(f"Stop Loss ishga tushdi: {trade.trade_id}")
                return True
            
            # Take Profit tekshirish
            if trade.action == "BUY" and trade.current_price >= trade.take_profit:
                logger.info(f"Take Profit ishga tushdi: {trade.trade_id}")
                return True
            
            if trade.action == "SELL" and trade.current_price <= trade.take_profit:
                logger.info(f"Take Profit ishga tushdi: {trade.trade_id}")
                return True
            
            # Vaqt limiti (masalan, 24 soat)
            if trade.duration > 86400:  # 24 soat
                logger.warning(f"Vaqt limiti tugadi: {trade.trade_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Trade yopilishi tekshirilmadi: {e}")
            return False
    
    async def _close_trade(self, trade_id: str, reason: str) -> None:
        """Trade yopish"""
        try:
            if trade_id in self.active_trades:
                trade = self.active_trades[trade_id]
                trade.status = TradeStatus.CLOSED
                
                # MT5 ga yopish buyrug'i
                # await self._execute_close_order(trade)
                
                # Database ga yozish
                await self._save_trade_record(trade, reason)
                
                # Faol tradelardan o'chirish
                del self.active_trades[trade_id]
                
                logger.info(f"Trade yopildi: {trade_id} - {reason}")
                
        except Exception as e:
            logger.error(f"Trade yopilmadi: {e}")
    
    async def _check_limits(self) -> List[str]:
        """Limitlarni tekshirish"""
        violations = []
        
        try:
            limits = self.daily_limits.propshot_max_daily_loss if self.propshot_mode else self.daily_limits.max_daily_loss
            
            # Kunlik zarar limiti
            if abs(self.account_metrics.daily_profit_loss) > limits * self.account_metrics.balance:
                violations.append(f"Kunlik zarar limiti oshdi: {self.account_metrics.daily_profit_loss}")
            
            # Umumiy zarar limiti
            total_limit = self.daily_limits.propshot_max_total_loss if self.propshot_mode else self.daily_limits.max_total_loss
            if abs(self.account_metrics.total_profit_loss) > total_limit * self.account_metrics.balance:
                violations.append(f"Umumiy zarar limiti oshdi: {self.account_metrics.total_profit_loss}")
            
            # Kunlik tradelar soni
            trade_limit = self.daily_limits.propshot_max_daily_trades if self.propshot_mode else self.daily_limits.max_daily_trades
            if self.account_metrics.daily_trades_count >= trade_limit:
                violations.append(f"Kunlik tradelar soni oshdi: {self.account_metrics.daily_trades_count}")
            
            # Ketma-ket zararlar
            if self.account_metrics.consecutive_losses >= self.daily_limits.max_consecutive_losses:
                violations.append(f"Ketma-ket zararlar oshdi: {self.account_metrics.consecutive_losses}")
            
            # Maksimal lot size
            lot_limit = self.daily_limits.propshot_max_lot_size if self.propshot_mode else self.daily_limits.max_lot_size
            for trade in self.active_trades.values():
                if trade.lot_size > lot_limit:
                    violations.append(f"Lot size oshdi: {trade.lot_size}")
            
            if violations:
                logger.warning(f"Limit buzilishlari: {violations}")
                
        except Exception as e:
            logger.error(f"Limitlar tekshirilmadi: {e}")
        
        return violations
    
    async def _assess_risk(self) -> RiskLevel:
        """Risk darajasi baholash"""
        try:
            risk_score = 0
            
            # Akavunt metrikalari asosida
            if abs(self.account_metrics.daily_profit_loss) > 0.01 * self.account_metrics.balance:
                risk_score += 1
            
            if abs(self.account_metrics.total_profit_loss) > 0.02 * self.account_metrics.balance:
                risk_score += 2
            
            if self.account_metrics.consecutive_losses >= 2:
                risk_score += 1
            
            if self.account_metrics.margin_level < 200:
                risk_score += 2
            
            # Faol tradelar asosida
            high_risk_trades = sum(1 for trade in self.active_trades.values() 
                                 if trade.risk_level == RiskLevel.HIGH)
            risk_score += high_risk_trades
            
            # Risk darajasi
            if risk_score >= 4:
                return RiskLevel.CRITICAL
            elif risk_score >= 3:
                return RiskLevel.HIGH
            elif risk_score >= 2:
                return RiskLevel.MEDIUM
            else:
                return RiskLevel.LOW
                
        except Exception as e:
            logger.error(f"Risk baholanmadi: {e}")
            return RiskLevel.MEDIUM
    
    def _calculate_trade_risk(self, trade: TradeMetrics) -> RiskLevel:
        """Trade risk darajasi hisoblash"""
        try:
            risk_score = 0
            
            # Zarar foizi
            if abs(trade.profit_loss_percent) > 1.0:
                risk_score += 2
            elif abs(trade.profit_loss_percent) > 0.5:
                risk_score += 1
            
            # Lot size
            if trade.lot_size > 0.3:
                risk_score += 2
            elif trade.lot_size > 0.1:
                risk_score += 1
            
            # Davomiyligi
            if trade.duration > 3600:  # 1 soat
                risk_score += 1
            
            # Risk darajasi
            if risk_score >= 4:
                return RiskLevel.CRITICAL
            elif risk_score >= 3:
                return RiskLevel.HIGH
            elif risk_score >= 2:
                return RiskLevel.MEDIUM
            else:
                return RiskLevel.LOW
                
        except Exception as e:
            logger.error(f"Trade risk baholanmadi: {e}")
            return RiskLevel.MEDIUM
    
    async def _send_alerts(self) -> None:
        """Ogohlantirish yuborish"""
        try:
            # Kritik holatlar uchun
            violations = await self._check_limits()
            risk_level = await self._assess_risk()
            
            if violations or risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                await self._send_telegram_alert(violations, risk_level)
                
        except Exception as e:
            logger.error(f"Ogohlantirish yuborilmadi: {e}")
    
    async def _send_telegram_alert(self, violations: List[str], risk_level: RiskLevel) -> None:
        """Telegram orqali ogohlantirish"""
        try:
            message = f"""
🚨 TRADE MONITORING OGOHLANTIRISH
════════════════════════════════════
⚠️ Risk darajasi: {risk_level.value.upper()}
📊 Akavunt holati: {self.account_metrics.balance:.2f} USD
📈 Kunlik P/L: {self.account_metrics.daily_profit_loss:.2f} USD
📊 Umumiy P/L: {self.account_metrics.total_profit_loss:.2f} USD
🔢 Kunlik tradelar: {self.account_metrics.daily_trades_count}
🔥 Ketma-ket zararlar: {self.account_metrics.consecutive_losses}

🚫 LIMIT BUZILISHLARI:
{chr(10).join(f'• {v}' for v in violations)}

⏰ Vaqt: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
            
            # Telegram client ga yuborish
            # await self.telegram_client.send_message(message)
            
            logger.warning(f"Telegram ogohlantirish: {message}")
            
        except Exception as e:
            logger.error(f"Telegram ogohlantirish yuborilmadi: {e}")
    
    async def _save_trade_record(self, trade: TradeMetrics, close_reason: str) -> None:
        """Trade yozuvni saqlash"""
        try:
            trade_record = TradeRecord(
                trade_id=trade.trade_id,
                symbol=trade.symbol,
                action=trade.action,
                entry_price=trade.entry_price,
                close_price=trade.current_price,
                lot_size=trade.lot_size,
                profit_loss=trade.profit_loss,
                duration=trade.duration,
                close_reason=close_reason,
                timestamp=datetime.now()
            )
            
            # Database ga saqlash
            # await self.db_manager.save_trade_record(trade_record)
            
        except Exception as e:
            logger.error(f"Trade yozuvi saqlanmadi: {e}")
    
    async def _load_daily_limits(self) -> None:
        """Kunlik limitlarni yuklash"""
        try:
            trading_config = self.config.get_trading_config()
            
            if 'propshot_settings' in trading_config:
                propshot = trading_config['propshot_settings']
                self.daily_limits.propshot_max_daily_loss = propshot.get('max_daily_loss', 0.02)
                self.daily_limits.propshot_max_total_loss = propshot.get('max_total_loss', 0.04)
                self.daily_limits.propshot_max_lot_size = propshot.get('max_lot_size', 0.3)
                self.daily_limits.propshot_max_daily_trades = propshot.get('max_daily_trades', 2)
            
            logger.info("Kunlik limitlar yuklandi")
            
        except Exception as e:
            logger.error(f"Kunlik limitlar yuklanmadi: {e}")
    
    async def add_trade(self, trade: TradeMetrics) -> bool:
        """Yangi trade qo'shish"""
        try:
            # Limitlarni tekshirish
            violations = await self._check_limits()
            if violations:
                logger.warning(f"Trade rad etildi - limitlar: {violations}")
                return False
            
            # Trade qo'shish
            self.active_trades[trade.trade_id] = trade
            trade.status = TradeStatus.ACTIVE
            
            # Akavunt metrikalari yangilash
            if self.account_metrics:
                self.account_metrics.daily_trades_count += 1
                self.account_metrics.total_trades_count += 1
            
            logger.info(f"Trade qo'shildi: {trade.trade_id}")
            return True
            
        except Exception as e:
            logger.error(f"Trade qo'shilmadi: {e}")
            return False
    
    async def get_monitoring_status(self) -> MonitoringResult:
        """Monitoring holati"""
        try:
            violations = await self._check_limits()
            risk_level = await self._assess_risk()
            
            return MonitoringResult(
                success=True,
                can_trade=len(violations) == 0,
                risk_level=risk_level,
                active_trades=list(self.active_trades.values()),
                account_metrics=self.account_metrics,
                limit_violations=violations,
                warnings=[],
                recommendations=[]
            )
            
        except Exception as e:
            logger.error(f"Monitoring holati olinmadi: {e}")
            return MonitoringResult(
                success=False,
                can_trade=False,
                risk_level=RiskLevel.CRITICAL,
                active_trades=[],
                account_metrics=None,
                limit_violations=[],
                warnings=[],
                recommendations=[],
                error=str(e)
            )
    
    async def emergency_close_all(self) -> None:
        """Favqulodda barcha tradelarni yopish"""
        try:
            logger.critical("FAVQULODDA: Barcha tradelar yopilmoqda")
            
            for trade_id in list(self.active_trades.keys()):
                await self._close_trade(trade_id, "emergency_close")
            
            # Telegram orqali xabar
            await self._send_telegram_alert(
                ["FAVQULODDA: Barcha tradelar yopildi"],
                RiskLevel.CRITICAL
            )
            
        except Exception as e:
            logger.error(f"Favqulodda yopish amalga oshmadi: {e}")
    
    def __del__(self):
        """Destruktor"""
        if self.monitoring_active:
            logger.info("TradeMonitor to'xtatildi")

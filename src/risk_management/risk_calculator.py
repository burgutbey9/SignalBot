import asyncio
import json
import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
from utils.logger import get_logger
from utils.error_handler import handle_processing_error
from config.config import ConfigManager

logger = get_logger(__name__)

class RiskLevel(Enum):
    """Risk darajasi enum"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"

class PositionType(Enum):
    """Pozitsiya turi enum"""
    BUY = "buy"
    SELL = "sell"
    LONG = "long"
    SHORT = "short"

@dataclass
class RiskMetrics:
    """Risk ko'rsatkichlari"""
    max_risk_per_trade: float
    max_daily_loss: float
    max_total_loss: float
    current_daily_loss: float
    current_total_loss: float
    risk_level: RiskLevel
    allowed_position_size: float
    
@dataclass
class TradeRisk:
    """Savdo risk ma'lumotlari"""
    symbol: str
    position_type: PositionType
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    risk_amount: float
    risk_percent: float
    reward_ratio: float
    pip_value: float
    
@dataclass
class PropShotLimits:
    """Propshot limitlari"""
    max_daily_loss: float = 0.025  # 2.5%
    max_total_loss: float = 0.05   # 5%
    max_lot_size: float = 0.5      # 0.5 lot
    max_daily_trades: int = 3      # 3 ta savdo
    max_drawdown: float = 0.08     # 8%
    
class RiskCalculator:
    """Risk hisoblash va boshqaruv"""
    
    def __init__(self):
        self.config = ConfigManager()
        self.propshot_limits = PropShotLimits()
        self.daily_trades = []
        self.daily_loss = 0.0
        self.total_loss = 0.0
        self.account_balance = 10000.0  # Default balance
        logger.info("RiskCalculator ishga tushirildi")
    
    async def calculate_position_size(self, 
                                    symbol: str,
                                    entry_price: float,
                                    stop_loss: float,
                                    risk_percent: float = 0.02) -> Dict:
        """Pozitsiya hajmini hisoblash"""
        try:
            # Pip qiymatini hisoblash
            pip_value = await self._calculate_pip_value(symbol, entry_price)
            
            # Pips farqini hisoblash
            pips_distance = abs(entry_price - stop_loss) / pip_value
            
            # Risk miqdorini hisoblash
            risk_amount = self.account_balance * risk_percent
            
            # Pozitsiya hajmini hisoblash
            position_size = risk_amount / (pips_distance * pip_value)
            
            # Propshot limitlarini tekshirish
            max_allowed_size = await self._check_propshot_limits(position_size)
            
            # Yakuniy pozitsiya hajmi
            final_position_size = min(position_size, max_allowed_size)
            
            result = {
                "symbol": symbol,
                "position_size": final_position_size,
                "risk_amount": risk_amount,
                "risk_percent": risk_percent,
                "pip_value": pip_value,
                "pips_distance": pips_distance,
                "max_allowed_size": max_allowed_size,
                "is_within_limits": final_position_size == position_size
            }
            
            logger.info(f"Pozitsiya hajmi hisoblandi: {symbol} - {final_position_size} lot")
            return result
            
        except Exception as e:
            logger.error(f"Pozitsiya hajmi hisoblashda xato: {e}")
            return {"error": str(e)}
    
    async def calculate_trade_risk(self, 
                                 symbol: str,
                                 position_type: PositionType,
                                 entry_price: float,
                                 stop_loss: float,
                                 take_profit: float,
                                 position_size: float) -> TradeRisk:
        """Savdo riskini hisoblash"""
        try:
            # Pip qiymatini hisoblash
            pip_value = await self._calculate_pip_value(symbol, entry_price)
            
            # Risk miqdorini hisoblash
            risk_pips = abs(entry_price - stop_loss) / pip_value
            risk_amount = risk_pips * pip_value * position_size
            risk_percent = (risk_amount / self.account_balance) * 100
            
            # Reward ratio hisoblash
            reward_pips = abs(take_profit - entry_price) / pip_value
            reward_ratio = reward_pips / risk_pips if risk_pips > 0 else 0
            
            trade_risk = TradeRisk(
                symbol=symbol,
                position_type=position_type,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=position_size,
                risk_amount=risk_amount,
                risk_percent=risk_percent,
                reward_ratio=reward_ratio,
                pip_value=pip_value
            )
            
            logger.info(f"Savdo riski hisoblandi: {symbol} - Risk: {risk_percent:.2f}%")
            return trade_risk
            
        except Exception as e:
            logger.error(f"Savdo riski hisoblashda xato: {e}")
            raise
    
    async def check_trade_approval(self, trade_risk: TradeRisk) -> Dict:
        """Savdoni tasdiqlash yoki rad etish"""
        try:
            approval_result = {
                "approved": False,
                "reason": "",
                "risk_level": RiskLevel.LOW,
                "recommendations": []
            }
            
            # Propshot limitlarini tekshirish
            if not await self._check_daily_loss_limit(trade_risk.risk_amount):
                approval_result["reason"] = "Kunlik zarar limiti oshib ketdi"
                approval_result["risk_level"] = RiskLevel.EXTREME
                return approval_result
            
            if not await self._check_total_loss_limit(trade_risk.risk_amount):
                approval_result["reason"] = "Umumiy zarar limiti oshib ketdi"
                approval_result["risk_level"] = RiskLevel.EXTREME
                return approval_result
            
            if trade_risk.position_size > self.propshot_limits.max_lot_size:
                approval_result["reason"] = "Lot hajmi limiti oshib ketdi"
                approval_result["risk_level"] = RiskLevel.HIGH
                return approval_result
            
            if len(self.daily_trades) >= self.propshot_limits.max_daily_trades:
                approval_result["reason"] = "Kunlik savdo limiti oshib ketdi"
                approval_result["risk_level"] = RiskLevel.HIGH
                return approval_result
            
            # Risk darajasini baholash
            risk_level = await self._assess_risk_level(trade_risk)
            approval_result["risk_level"] = risk_level
            
            # Reward ratio tekshirish
            if trade_risk.reward_ratio < 1.5:
                approval_result["recommendations"].append(
                    "Reward ratio past (< 1.5). Take profit ni oshiring."
                )
            
            # Risk foizini tekshirish
            if trade_risk.risk_percent > 2.0:
                approval_result["recommendations"].append(
                    "Risk foizi yuqori (> 2%). Pozitsiya hajmini kamaytiring."
                )
            
            # Tasdiqlash
            if risk_level in [RiskLevel.LOW, RiskLevel.MEDIUM]:
                approval_result["approved"] = True
                approval_result["reason"] = "Savdo tasdiqlandi"
            else:
                approval_result["reason"] = "Risk darajasi juda yuqori"
            
            logger.info(f"Savdo tasdiqlanishi: {approval_result['approved']} - {approval_result['reason']}")
            return approval_result
            
        except Exception as e:
            logger.error(f"Savdo tasdiqlashda xato: {e}")
            return {"approved": False, "reason": f"Xato: {e}"}
    
    async def get_risk_metrics(self) -> RiskMetrics:
        """Joriy risk ko'rsatkichlarini olish"""
        try:
            # Kunlik zarar hisoblash
            today = datetime.now().date()
            daily_loss = sum(
                trade.get("loss", 0) for trade in self.daily_trades
                if trade.get("date", datetime.now().date()) == today
            )
            
            # Risk darajasini baholash
            risk_level = RiskLevel.LOW
            if daily_loss > self.propshot_limits.max_daily_loss * 0.5:
                risk_level = RiskLevel.MEDIUM
            if daily_loss > self.propshot_limits.max_daily_loss * 0.8:
                risk_level = RiskLevel.HIGH
            if daily_loss > self.propshot_limits.max_daily_loss:
                risk_level = RiskLevel.EXTREME
            
            # Ruxsat etilgan pozitsiya hajmini hisoblash
            remaining_daily_risk = self.propshot_limits.max_daily_loss - daily_loss
            allowed_position_size = min(
                self.propshot_limits.max_lot_size,
                remaining_daily_risk / 0.02  # 2% per trade
            )
            
            metrics = RiskMetrics(
                max_risk_per_trade=0.02,
                max_daily_loss=self.propshot_limits.max_daily_loss,
                max_total_loss=self.propshot_limits.max_total_loss,
                current_daily_loss=daily_loss,
                current_total_loss=self.total_loss,
                risk_level=risk_level,
                allowed_position_size=max(0, allowed_position_size)
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Risk ko'rsatkichlarini olishda xato: {e}")
            raise
    
    async def kelly_criterion(self, 
                            win_rate: float,
                            avg_win: float,
                            avg_loss: float) -> float:
        """Kelly kriteriyasi bo'yicha optimal pozitsiya hajmi"""
        try:
            if avg_loss == 0:
                return 0.0
            
            # Kelly formulasi: f* = (bp - q) / b
            # b = avg_win / avg_loss (odds)
            # p = win_rate
            # q = 1 - p
            
            b = avg_win / avg_loss
            p = win_rate
            q = 1 - p
            
            kelly_fraction = (b * p - q) / b
            
            # Kelly ni cheklash (max 25%)
            kelly_fraction = max(0, min(kelly_fraction, 0.25))
            
            logger.info(f"Kelly kriteriyasi: {kelly_fraction:.4f}")
            return kelly_fraction
            
        except Exception as e:
            logger.error(f"Kelly kriteriyasida xato: {e}")
            return 0.02  # Default 2%
    
    async def _calculate_pip_value(self, symbol: str, price: float) -> float:
        """Pip qiymatini hisoblash"""
        try:
            # Forex juftliklari uchun pip qiymati
            if "JPY" in symbol:
                return 0.01  # JPY juftliklari uchun
            elif "USD" in symbol:
                return 0.0001  # USD juftliklari uchun
            else:
                return 0.0001  # Default pip qiymati
                
        except Exception as e:
            logger.error(f"Pip qiymatini hisoblashda xato: {e}")
            return 0.0001
    
    async def _check_propshot_limits(self, position_size: float) -> float:
        """Propshot limitlarini tekshirish"""
        try:
            # Maksimal lot hajmi
            max_size = self.propshot_limits.max_lot_size
            
            # Kunlik savdo limitini tekshirish
            if len(self.daily_trades) >= self.propshot_limits.max_daily_trades:
                return 0.0
            
            # Kunlik zarar limitini tekshirish
            if self.daily_loss >= self.propshot_limits.max_daily_loss:
                return 0.0
            
            return min(position_size, max_size)
            
        except Exception as e:
            logger.error(f"Propshot limitlarini tekshirishda xato: {e}")
            return 0.0
    
    async def _check_daily_loss_limit(self, risk_amount: float) -> bool:
        """Kunlik zarar limitini tekshirish"""
        try:
            potential_loss = self.daily_loss + risk_amount
            return potential_loss <= self.propshot_limits.max_daily_loss * self.account_balance
            
        except Exception as e:
            logger.error(f"Kunlik zarar limitini tekshirishda xato: {e}")
            return False
    
    async def _check_total_loss_limit(self, risk_amount: float) -> bool:
        """Umumiy zarar limitini tekshirish"""
        try:
            potential_loss = self.total_loss + risk_amount
            return potential_loss <= self.propshot_limits.max_total_loss * self.account_balance
            
        except Exception as e:
            logger.error(f"Umumiy zarar limitini tekshirishda xato: {e}")
            return False
    
    async def _assess_risk_level(self, trade_risk: TradeRisk) -> RiskLevel:
        """Risk darajasini baholash"""
        try:
            # Risk foiziga asoslangan baholash
            if trade_risk.risk_percent <= 1.0:
                return RiskLevel.LOW
            elif trade_risk.risk_percent <= 2.0:
                return RiskLevel.MEDIUM
            elif trade_risk.risk_percent <= 3.0:
                return RiskLevel.HIGH
            else:
                return RiskLevel.EXTREME
                
        except Exception as e:
            logger.error(f"Risk darajasini baholashda xato: {e}")
            return RiskLevel.EXTREME
    
    async def update_daily_loss(self, loss_amount: float):
        """Kunlik zaralni yangilash"""
        try:
            self.daily_loss += loss_amount
            self.total_loss += loss_amount
            
            # Savdo tarixini yangilash
            self.daily_trades.append({
                "date": datetime.now().date(),
                "loss": loss_amount,
                "timestamp": datetime.now()
            })
            
            logger.info(f"Kunlik zarar yangilandi: {self.daily_loss:.2f}")
            
        except Exception as e:
            logger.error(f"Kunlik zaralni yangilashda xato: {e}")
    
    async def reset_daily_metrics(self):
        """Kunlik ko'rsatkichlarni qayta tiklash"""
        try:
            self.daily_loss = 0.0
            self.daily_trades = []
            logger.info("Kunlik ko'rsatkichlar qayta tiklandi")
            
        except Exception as e:
            logger.error(f"Kunlik ko'rsatkichlarni qayta tiklashda xato: {e}")
    
    async def get_risk_summary(self) -> Dict:
        """Risk xulosasini olish"""
        try:
            metrics = await self.get_risk_metrics()
            
            summary = {
                "account_balance": self.account_balance,
                "daily_loss": self.daily_loss,
                "daily_loss_percent": (self.daily_loss / self.account_balance) * 100,
                "total_loss": self.total_loss,
                "total_loss_percent": (self.total_loss / self.account_balance) * 100,
                "risk_level": metrics.risk_level.value,
                "daily_trades_count": len(self.daily_trades),
                "max_daily_trades": self.propshot_limits.max_daily_trades,
                "allowed_position_size": metrics.allowed_position_size,
                "max_lot_size": self.propshot_limits.max_lot_size,
                "limits": {
                    "max_daily_loss": self.propshot_limits.max_daily_loss,
                    "max_total_loss": self.propshot_limits.max_total_loss,
                    "max_lot_size": self.propshot_limits.max_lot_size,
                    "max_daily_trades": self.propshot_limits.max_daily_trades
                }
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Risk xulosasini olishda xato: {e}")
            return {"error": str(e)}

# Singleton instance
risk_calculator = RiskCalculator()

async def main():
    """Test funktsiyasi"""
    try:
        # Test ma'lumotlari
        symbol = "EURUSD"
        position_type = PositionType.BUY
        entry_price = 1.0500
        stop_loss = 1.0450
        take_profit = 1.0600
        position_size = 0.1
        
        # Savdo riskini hisoblash
        trade_risk = await risk_calculator.calculate_trade_risk(
            symbol, position_type, entry_price, stop_loss, take_profit, position_size
        )
        
        print(f"Savdo riski: {trade_risk}")
        
        # Tasdiqlash
        approval = await risk_calculator.check_trade_approval(trade_risk)
        print(f"Tasdiqlash: {approval}")
        
        # Risk xulosasi
        summary = await risk_calculator.get_risk_summary()
        print(f"Risk xulosasi: {summary}")
        
    except Exception as e:
        print(f"Test xatosi: {e}")

if __name__ == "__main__":
    asyncio.run(main())

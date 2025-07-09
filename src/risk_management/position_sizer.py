import asyncio
import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np
from utils.logger import get_logger
from utils.error_handler import handle_processing_error
from config.config import ConfigManager

logger = get_logger(__name__)

class PositionSizeMethod(Enum):
    """Position size hisoblash usullari"""
    FIXED_AMOUNT = "fixed_amount"
    FIXED_PERCENTAGE = "fixed_percentage"
    KELLY_CRITERION = "kelly_criterion"
    VOLATILITY_BASED = "volatility_based"
    ATR_BASED = "atr_based"
    MARTINGALE = "martingale"
    ANTI_MARTINGALE = "anti_martingale"

@dataclass
class PositionSizeInput:
    """Position size hisoblash uchun kerakli ma'lumotlar"""
    account_balance: float
    risk_percentage: float
    entry_price: float
    stop_loss: float
    take_profit: Optional[float] = None
    volatility: Optional[float] = None
    atr_value: Optional[float] = None
    win_rate: Optional[float] = None
    avg_win: Optional[float] = None
    avg_loss: Optional[float] = None
    consecutive_losses: int = 0
    consecutive_wins: int = 0
    currency_pair: str = "EURUSD"
    pip_value: float = 10.0  # USD hisobida 1 pip qiymati
    
@dataclass
class PositionSizeResult:
    """Position size hisoblash natijasi"""
    success: bool
    lot_size: float = 0.0
    position_value: float = 0.0
    risk_amount: float = 0.0
    reward_ratio: float = 0.0
    max_loss: float = 0.0
    expected_profit: float = 0.0
    method_used: str = ""
    confidence: float = 0.0
    warnings: List[str] = None
    error: Optional[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []

class PositionSizer:
    """Position size hisoblash va boshqaruv"""
    
    def __init__(self):
        self.name = self.__class__.__name__
        self.config = ConfigManager()
        self.min_lot_size = 0.01
        self.max_lot_size = 100.0
        self.lot_step = 0.01
        
        # Propshot limitlar
        self.propshot_limits = {
            'max_daily_loss': 0.025,  # 2.5%
            'max_total_loss': 0.05,   # 5%
            'max_lot_size': 0.5,      # 0.5 lot
            'max_daily_trades': 3     # 3 ta savdo
        }
        
        logger.info(f"{self.name} ishga tushirildi")
    
    async def calculate_position_size(
        self, 
        input_data: PositionSizeInput,
        method: PositionSizeMethod = PositionSizeMethod.KELLY_CRITERION
    ) -> PositionSizeResult:
        """Asosiy position size hisoblash methodi"""
        try:
            # Kirish ma'lumotlarini tekshirish
            if not self._validate_input(input_data):
                return PositionSizeResult(
                    success=False,
                    error="Noto'g'ri kirish ma'lumotlari"
                )
            
            # Usul bo'yicha hisoblash
            if method == PositionSizeMethod.FIXED_AMOUNT:
                result = await self._calculate_fixed_amount(input_data)
            elif method == PositionSizeMethod.FIXED_PERCENTAGE:
                result = await self._calculate_fixed_percentage(input_data)
            elif method == PositionSizeMethod.KELLY_CRITERION:
                result = await self._calculate_kelly_criterion(input_data)
            elif method == PositionSizeMethod.VOLATILITY_BASED:
                result = await self._calculate_volatility_based(input_data)
            elif method == PositionSizeMethod.ATR_BASED:
                result = await self._calculate_atr_based(input_data)
            elif method == PositionSizeMethod.MARTINGALE:
                result = await self._calculate_martingale(input_data)
            elif method == PositionSizeMethod.ANTI_MARTINGALE:
                result = await self._calculate_anti_martingale(input_data)
            else:
                result = await self._calculate_fixed_percentage(input_data)
            
            # Propshot limitlarini tekshirish
            result = await self._apply_propshot_limits(result, input_data)
            
            # Qo'shimcha validatsiya
            result = await self._validate_result(result, input_data)
            
            logger.info(f"Position size hisoblandi: {result.lot_size} lot, usul: {method.value}")
            return result
            
        except Exception as e:
            logger.error(f"Position size hisoblashda xato: {e}")
            return PositionSizeResult(
                success=False,
                error=str(e)
            )
    
    async def _calculate_fixed_amount(self, input_data: PositionSizeInput) -> PositionSizeResult:
        """Belgilangan miqdor bo'yicha hisoblash"""
        try:
            # Risk miqdorini hisoblash
            risk_amount = input_data.account_balance * (input_data.risk_percentage / 100)
            
            # Pip farqi
            pip_difference = abs(input_data.entry_price - input_data.stop_loss) * 10000
            
            # Lot size hisoblash
            lot_size = risk_amount / (pip_difference * input_data.pip_value)
            
            # Lot size ni to'g'rilash
            lot_size = self._round_lot_size(lot_size)
            
            # Natija qaytarish
            return PositionSizeResult(
                success=True,
                lot_size=lot_size,
                position_value=lot_size * 100000 * input_data.entry_price,
                risk_amount=risk_amount,
                max_loss=lot_size * pip_difference * input_data.pip_value,
                method_used="fixed_amount",
                confidence=0.8
            )
            
        except Exception as e:
            logger.error(f"Fixed amount hisoblashda xato: {e}")
            return PositionSizeResult(success=False, error=str(e))
    
    async def _calculate_fixed_percentage(self, input_data: PositionSizeInput) -> PositionSizeResult:
        """Belgilangan foiz bo'yicha hisoblash"""
        try:
            # Risk miqdorini hisoblash
            risk_amount = input_data.account_balance * (input_data.risk_percentage / 100)
            
            # Pip farqi
            pip_difference = abs(input_data.entry_price - input_data.stop_loss) * 10000
            
            if pip_difference == 0:
                return PositionSizeResult(success=False, error="Stop loss va entry price bir xil")
            
            # Lot size hisoblash
            lot_size = risk_amount / (pip_difference * input_data.pip_value)
            
            # Lot size ni to'g'rilash
            lot_size = self._round_lot_size(lot_size)
            
            # Reward ratio hisoblash
            reward_ratio = 0.0
            if input_data.take_profit:
                tp_pips = abs(input_data.take_profit - input_data.entry_price) * 10000
                reward_ratio = tp_pips / pip_difference if pip_difference > 0 else 0.0
            
            return PositionSizeResult(
                success=True,
                lot_size=lot_size,
                position_value=lot_size * 100000 * input_data.entry_price,
                risk_amount=risk_amount,
                reward_ratio=reward_ratio,
                max_loss=lot_size * pip_difference * input_data.pip_value,
                method_used="fixed_percentage",
                confidence=0.7
            )
            
        except Exception as e:
            logger.error(f"Fixed percentage hisoblashda xato: {e}")
            return PositionSizeResult(success=False, error=str(e))
    
    async def _calculate_kelly_criterion(self, input_data: PositionSizeInput) -> PositionSizeResult:
        """Kelly Criterion bo'yicha hisoblash"""
        try:
            # Kelly parametrlarini tekshirish
            if not all([input_data.win_rate, input_data.avg_win, input_data.avg_loss]):
                logger.warning("Kelly uchun yetarli ma'lumot yo'q, fixed percentage ishlatiladi")
                return await self._calculate_fixed_percentage(input_data)
            
            # Kelly formulasi: f = (bp - q) / b
            # f = optimal fraction
            # b = average win / average loss
            # p = win probability
            # q = loss probability (1 - p)
            
            win_rate = input_data.win_rate
            avg_win = input_data.avg_win
            avg_loss = abs(input_data.avg_loss)
            
            if avg_loss == 0:
                return PositionSizeResult(success=False, error="O'rtacha zarar nolga teng")
            
            b = avg_win / avg_loss
            p = win_rate / 100
            q = 1 - p
            
            # Kelly fraksiyasi
            kelly_fraction = (b * p - q) / b
            
            # Kelly fraksiyasini cheklash (maksimal 25%)
            kelly_fraction = max(0, min(kelly_fraction, 0.25))
            
            # Kelly fraksiyasini konservativ qilish (Kelly / 2)
            kelly_fraction = kelly_fraction / 2
            
            # Risk miqdorini hisoblash
            risk_amount = input_data.account_balance * kelly_fraction
            
            # Pip farqi
            pip_difference = abs(input_data.entry_price - input_data.stop_loss) * 10000
            
            if pip_difference == 0:
                return PositionSizeResult(success=False, error="Stop loss va entry price bir xil")
            
            # Lot size hisoblash
            lot_size = risk_amount / (pip_difference * input_data.pip_value)
            
            # Lot size ni to'g'rilash
            lot_size = self._round_lot_size(lot_size)
            
            # Kutilayotgan foyda
            expected_profit = lot_size * (p * avg_win - q * avg_loss) * input_data.pip_value
            
            return PositionSizeResult(
                success=True,
                lot_size=lot_size,
                position_value=lot_size * 100000 * input_data.entry_price,
                risk_amount=risk_amount,
                max_loss=lot_size * pip_difference * input_data.pip_value,
                expected_profit=expected_profit,
                method_used="kelly_criterion",
                confidence=0.9
            )
            
        except Exception as e:
            logger.error(f"Kelly criterion hisoblashda xato: {e}")
            return PositionSizeResult(success=False, error=str(e))
    
    async def _calculate_volatility_based(self, input_data: PositionSizeInput) -> PositionSizeResult:
        """Volatilitga asoslangan hisoblash"""
        try:
            if not input_data.volatility:
                logger.warning("Volatility ma'lumoti yo'q, fixed percentage ishlatiladi")
                return await self._calculate_fixed_percentage(input_data)
            
            # Volatilitga teskari proporsional position size
            volatility_factor = 1 / (1 + input_data.volatility)
            
            # Asosiy risk foizini volatilitga moslash
            adjusted_risk = input_data.risk_percentage * volatility_factor
            
            # Risk miqdorini hisoblash
            risk_amount = input_data.account_balance * (adjusted_risk / 100)
            
            # Pip farqi
            pip_difference = abs(input_data.entry_price - input_data.stop_loss) * 10000
            
            if pip_difference == 0:
                return PositionSizeResult(success=False, error="Stop loss va entry price bir xil")
            
            # Lot size hisoblash
            lot_size = risk_amount / (pip_difference * input_data.pip_value)
            
            # Lot size ni to'g'rilash
            lot_size = self._round_lot_size(lot_size)
            
            return PositionSizeResult(
                success=True,
                lot_size=lot_size,
                position_value=lot_size * 100000 * input_data.entry_price,
                risk_amount=risk_amount,
                max_loss=lot_size * pip_difference * input_data.pip_value,
                method_used="volatility_based",
                confidence=0.8,
                warnings=[f"Volatility bo'yicha moslashtirildi: {volatility_factor:.2f}"]
            )
            
        except Exception as e:
            logger.error(f"Volatility based hisoblashda xato: {e}")
            return PositionSizeResult(success=False, error=str(e))
    
    async def _calculate_atr_based(self, input_data: PositionSizeInput) -> PositionSizeResult:
        """ATR (Average True Range) asosida hisoblash"""
        try:
            if not input_data.atr_value:
                logger.warning("ATR ma'lumoti yo'q, fixed percentage ishlatiladi")
                return await self._calculate_fixed_percentage(input_data)
            
            # ATR ga asoslangan position size
            # ATR katta bo'lsa, position size kichik bo'ladi
            atr_factor = 1 / (1 + input_data.atr_value * 100)
            
            # Asosiy risk foizini ATR ga moslash
            adjusted_risk = input_data.risk_percentage * atr_factor
            
            # Risk miqdorini hisoblash
            risk_amount = input_data.account_balance * (adjusted_risk / 100)
            
            # ATR ni stop loss sifatida ishlatish
            atr_stop_loss = input_data.atr_value * 10000  # pips hisobida
            
            # Lot size hisoblash
            lot_size = risk_amount / (atr_stop_loss * input_data.pip_value)
            
            # Lot size ni to'g'rilash
            lot_size = self._round_lot_size(lot_size)
            
            return PositionSizeResult(
                success=True,
                lot_size=lot_size,
                position_value=lot_size * 100000 * input_data.entry_price,
                risk_amount=risk_amount,
                max_loss=lot_size * atr_stop_loss * input_data.pip_value,
                method_used="atr_based",
                confidence=0.85,
                warnings=[f"ATR asosida moslashtirildi: {atr_factor:.2f}"]
            )
            
        except Exception as e:
            logger.error(f"ATR based hisoblashda xato: {e}")
            return PositionSizeResult(success=False, error=str(e))
    
    async def _calculate_martingale(self, input_data: PositionSizeInput) -> PositionSizeResult:
        """Martingale strategiyasi bo'yicha hisoblash"""
        try:
            # Asosiy lot size
            base_result = await self._calculate_fixed_percentage(input_data)
            
            if not base_result.success:
                return base_result
            
            # Ketma-ket yo'qotishlar soniga qarab lot size ni oshirish
            multiplier = 1.0
            if input_data.consecutive_losses > 0:
                # Konservativ martingale (2 emas, 1.5 ko'paytirish)
                multiplier = 1.5 ** min(input_data.consecutive_losses, 3)  # Maksimal 3 marta
            
            # Lot size ni ko'paytirish
            lot_size = base_result.lot_size * multiplier
            
            # Lot size ni to'g'rilash
            lot_size = self._round_lot_size(lot_size)
            
            # Xavfni hisoblash
            pip_difference = abs(input_data.entry_price - input_data.stop_loss) * 10000
            max_loss = lot_size * pip_difference * input_data.pip_value
            
            warnings = []
            if multiplier > 1:
                warnings.append(f"Martingale: {multiplier:.1f}x ko'paytirildi")
            
            return PositionSizeResult(
                success=True,
                lot_size=lot_size,
                position_value=lot_size * 100000 * input_data.entry_price,
                risk_amount=max_loss,
                max_loss=max_loss,
                method_used="martingale",
                confidence=0.6,  # Martingale xavfli
                warnings=warnings
            )
            
        except Exception as e:
            logger.error(f"Martingale hisoblashda xato: {e}")
            return PositionSizeResult(success=False, error=str(e))
    
    async def _calculate_anti_martingale(self, input_data: PositionSizeInput) -> PositionSizeResult:
        """Anti-Martingale strategiyasi bo'yicha hisoblash"""
        try:
            # Asosiy lot size
            base_result = await self._calculate_fixed_percentage(input_data)
            
            if not base_result.success:
                return base_result
            
            # Ketma-ket yutishlar soniga qarab lot size ni oshirish
            multiplier = 1.0
            if input_data.consecutive_wins > 0:
                # Konservativ anti-martingale
                multiplier = 1.3 ** min(input_data.consecutive_wins, 3)  # Maksimal 3 marta
            
            # Lot size ni ko'paytirish
            lot_size = base_result.lot_size * multiplier
            
            # Lot size ni to'g'rilash
            lot_size = self._round_lot_size(lot_size)
            
            # Xavfni hisoblash
            pip_difference = abs(input_data.entry_price - input_data.stop_loss) * 10000
            max_loss = lot_size * pip_difference * input_data.pip_value
            
            warnings = []
            if multiplier > 1:
                warnings.append(f"Anti-Martingale: {multiplier:.1f}x ko'paytirildi")
            
            return PositionSizeResult(
                success=True,
                lot_size=lot_size,
                position_value=lot_size * 100000 * input_data.entry_price,
                risk_amount=max_loss,
                max_loss=max_loss,
                method_used="anti_martingale",
                confidence=0.7,
                warnings=warnings
            )
            
        except Exception as e:
            logger.error(f"Anti-Martingale hisoblashda xato: {e}")
            return PositionSizeResult(success=False, error=str(e))
    
    async def _apply_propshot_limits(self, result: PositionSizeResult, input_data: PositionSizeInput) -> PositionSizeResult:
        """Propshot limitlarini qo'llash"""
        try:
            if not result.success:
                return result
            
            original_lot_size = result.lot_size
            
            # Maksimal lot size tekshirish
            if result.lot_size > self.propshot_limits['max_lot_size']:
                result.lot_size = self.propshot_limits['max_lot_size']
                result.warnings.append(f"Propshot limit: Lot size {original_lot_size:.2f} dan {result.lot_size:.2f} ga tushirildi")
            
            # Maksimal kunlik zarar tekshirish
            daily_loss_limit = input_data.account_balance * self.propshot_limits['max_daily_loss']
            if result.max_loss > daily_loss_limit:
                # Lot size ni kamaytirishimiz kerak
                pip_difference = abs(input_data.entry_price - input_data.stop_loss) * 10000
                max_allowed_lot = daily_loss_limit / (pip_difference * input_data.pip_value)
                result.lot_size = min(result.lot_size, max_allowed_lot)
                result.warnings.append(f"Propshot limit: Kunlik zarar limiti uchun lot size kamaytirildi")
            
            # Lot size ni qayta to'g'rilash
            result.lot_size = self._round_lot_size(result.lot_size)
            
            # Yangi qiymatlarni qayta hisoblash
            pip_difference = abs(input_data.entry_price - input_data.stop_loss) * 10000
            result.position_value = result.lot_size * 100000 * input_data.entry_price
            result.max_loss = result.lot_size * pip_difference * input_data.pip_value
            result.risk_amount = result.max_loss
            
            return result
            
        except Exception as e:
            logger.error(f"Propshot limitlarini qo'llashda xato: {e}")
            result.warnings.append(f"Propshot limit xatosi: {e}")
            return result
    
    def _round_lot_size(self, lot_size: float) -> float:
        """Lot size ni to'g'ri qiymatga yaxlitlash"""
        try:
            # Minimal va maksimal qiymatlarni tekshirish
            lot_size = max(self.min_lot_size, min(lot_size, self.max_lot_size))
            
            # Lot step ga moslash
            lot_size = round(lot_size / self.lot_step) * self.lot_step
            
            # 2 ta o'nlik kasrga yaxlitlash
            return round(lot_size, 2)
            
        except Exception as e:
            logger.error(f"Lot size yaxlitlashda xato: {e}")
            return self.min_lot_size
    
    def _validate_input(self, input_data: PositionSizeInput) -> bool:
        """Kirish ma'lumotlarini tekshirish"""
        try:
            # Asosiy tekshiruvlar
            if input_data.account_balance <= 0:
                logger.error("Hisob balansi noldan kichik yoki teng")
                return False
            
            if input_data.risk_percentage <= 0 or input_data.risk_percentage > 100:
                logger.error("Risk foizi noto'g'ri")
                return False
            
            if input_data.entry_price <= 0:
                logger.error("Entry price noldan kichik yoki teng")
                return False
            
            if input_data.stop_loss <= 0:
                logger.error("Stop loss noldan kichik yoki teng")
                return False
            
            if input_data.entry_price == input_data.stop_loss:
                logger.error("Entry price va stop loss bir xil")
                return False
            
            # Pip value tekshirish
            if input_data.pip_value <= 0:
                logger.error("Pip value noldan kichik yoki teng")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Input validatsiyada xato: {e}")
            return False
    
    async def _validate_result(self, result: PositionSizeResult, input_data: PositionSizeInput) -> PositionSizeResult:
        """Natijani tekshirish"""
        try:
            if not result.success:
                return result
            
            # Lot size tekshirish
            if result.lot_size < self.min_lot_size:
                result.warnings.append(f"Lot size juda kichik: {result.lot_size}")
                result.lot_size = self.min_lot_size
            
            if result.lot_size > self.max_lot_size:
                result.warnings.append(f"Lot size juda katta: {result.lot_size}")
                result.lot_size = self.max_lot_size
            
            # Risk foizini tekshirish
            risk_percentage = (result.risk_amount / input_data.account_balance) * 100
            if risk_percentage > input_data.risk_percentage * 1.5:  # 50% dan ko'p farq
                result.warnings.append(f"Haqiqiy risk {risk_percentage:.1f}% - kutilgandan yuqori")
            
            # Position value tekshirish
            if result.position_value > input_data.account_balance * 100:  # 1:100 leverage
                result.warnings.append("Position value juda katta")
            
            return result
            
        except Exception as e:
            logger.error(f"Natija validatsiyada xato: {e}")
            result.warnings.append(f"Validatsiya xatosi: {e}")
            return result
    
    async def get_optimal_position_size(self, input_data: PositionSizeInput) -> PositionSizeResult:
        """Optimal position size ni topish"""
        try:
            # Turli usullar bilan hisoblash
            methods = [
                PositionSizeMethod.KELLY_CRITERION,
                PositionSizeMethod.VOLATILITY_BASED,
                PositionSizeMethod.ATR_BASED,
                PositionSizeMethod.FIXED_PERCENTAGE
            ]
            
            results = []
            for method in methods:
                result = await self.calculate_position_size(input_data, method)
                if result.success:
                    results.append(result)
            
            if not results:
                return PositionSizeResult(success=False, error="Hech qanday usul ishlamadi")
            
            # Eng yaxshi natijani tanlash (confidence bo'yicha)
            best_result = max(results, key=lambda r: r.confidence)
            
            # Optimal deb belgilash
            best_result.method_used = f"optimal_{best_result.method_used}"
            best_result.warnings.append("Optimal usul tanlandi")
            
            logger.info(f"Optimal position size: {best_result.lot_size} lot")
            return best_result
            
        except Exception as e:
            logger.error(f"Optimal position size topishda xato: {e}")
            return PositionSizeResult(success=False, error=str(e))
    
    async def calculate_risk_metrics(self, input_data: PositionSizeInput, result: PositionSizeResult) -> Dict:
        """Risk metrics hisoblash"""
        try:
            if not result.success:
                return {}
            
            # Asosiy risk metrics
            metrics = {
                'position_size_usd': result.position_value,
                'risk_amount_usd': result.risk_amount,
                'risk_percentage': (result.risk_amount / input_data.account_balance) * 100,
                'max_loss_usd': result.max_loss,
                'reward_to_risk_ratio': result.reward_ratio,
                'position_to_equity_ratio': (result.position_value / input_data.account_balance) * 100
            }
            
            # Qo'shimcha metrics
            if input_data.take_profit:
                tp_pips = abs(input_data.take_profit - input_data.entry_price) * 10000
                metrics['potential_profit_usd'] = result.lot_size * tp_pips * input_data.pip_value
                metrics['profit_percentage'] = (metrics['potential_profit_usd'] / input_data.account_balance) * 100
            
            # Leverage hisoblash
            if input_data.account_balance > 0:
                metrics['effective_leverage'] = result.position_value / input_data.account_balance
            
            return metrics
            
        except Exception as e:
            logger.error(f"Risk metrics hisoblashda xato: {e}")
            return {}

# Qo'shimcha utility funksiyalar
async def validate_position_size(lot_size: float, account_balance: float, max_risk: float = 0.02) -> bool:
    """Position size ni validatsiya qilish"""
    try:
        # Asosiy tekshiruvlar
        if lot_size <= 0:
            return False
        
        if lot_size > 100:  # Maksimal lot size
            return False
        
        # Risk tekshirish (taxminan)
        estimated_risk = lot_size * 1000  # Taxminan risk
        if estimated_risk > account_balance * max_risk:
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Position size validatsiyada xato: {e}")
        return False

async def get_recommended_lot_size(account_balance: float, risk_percentage: float = 2.0) -> float:
    """Tavsiya etilgan lot size"""
    try:
        # Konservativ hisoblash
        risk_amount = account_balance * (risk_percentage / 100)
        
        # O'rtacha pip value va stop loss
        avg_pip_value = 10.0
        avg_stop_loss_pips = 50.0
        
        # Lot size hisoblash
        lot_size = risk_amount / (avg_stop_loss_pips * avg_pip_value)
        
        # Yaxlitlash
        return round(max(0.01, min(lot_size, 1.0)), 2)
        
    except Exception as e:
        logger.error(f"Tavsiya etilgan lot size hisoblashda xato: {e}")
        return 0.01


# Test funksiyasi
async def test_position_sizer():
    """Position sizer test funksiyasi"""
    try:
        # Test ma'lumotlari
        test_input = PositionSizeInput(
            account_balance=10000.0,
            risk_percentage=2.0,
            entry_price=1.1000,
            stop_loss=1.0950,
            take_profit=1.1100,
            volatility=0.02,
            atr_value=0.0025,
            win_rate=65.0,
            avg_win=80.0,
            avg_loss=50.0,
            consecutive_losses=0,
            consecutive_wins=1,
            currency_pair="EURUSD",
            pip_value=10.0
        )
        
        # Position sizer yaratish
        sizer = PositionSizer()
        
        # Turli usullar bilan test
        methods = [
            PositionSizeMethod.FIXED_PERCENTAGE,
            PositionSizeMethod.KELLY_CRITERION,
            PositionSizeMethod.VOLATILITY_BASED,
            PositionSizeMethod.ATR_BASED
        ]
        
        print("=== POSITION SIZER TEST ===")
        print(f"Account Balance: ${test_input.account_balance:,.2f}")
        print(f"Risk Percentage: {test_input.risk_percentage}%")
        print(f"Entry Price: {test_input.entry_price}")
        print(f"Stop Loss: {test_input.stop_loss}")
        print(f"Take Profit: {test_input.take_profit}")
        print("-" * 50)
        
        for method in methods:
            result = await sizer.calculate_position_size(test_input, method)
            
            if result.success:
                print(f"\n{method.value.upper()}:")
                print(f"  Lot Size: {result.lot_size:.2f}")
                print(f"  Position Value: ${result.position_value:,.2f}")
                print(f"  Risk Amount: ${result.risk_amount:,.2f}")
                print(f"  Max Loss: ${result.max_loss:,.2f}")
                print(f"  Reward Ratio: {result.reward_ratio:.2f}")
                print(f"  Confidence: {result.confidence:.1%}")
                
                if result.warnings:
                    print(f"  Warnings: {', '.join(result.warnings)}")
                    
                # Risk metrics
                metrics = await sizer.calculate_risk_metrics(test_input, result)
                if metrics:
                    print(f"  Risk %: {metrics.get('risk_percentage', 0):.2f}%")
                    print(f"  Leverage: {metrics.get('effective_leverage', 0):.1f}x")
            else:
                print(f"\n{method.value.upper()}: XATO - {result.error}")
        
        # Optimal position size
        print("\n" + "="*50)
        optimal_result = await sizer.get_optimal_position_size(test_input)
        
        if optimal_result.success:
            print("OPTIMAL POSITION SIZE:")
            print(f"  Method: {optimal_result.method_used}")
            print(f"  Lot Size: {optimal_result.lot_size:.2f}")
            print(f"  Risk Amount: ${optimal_result.risk_amount:,.2f}")
            print(f"  Confidence: {optimal_result.confidence:.1%}")
        else:
            print(f"OPTIMAL XATO: {optimal_result.error}")
            
    except Exception as e:
        print(f"Test xatosi: {e}")


if __name__ == "__main__":
    # Test ishga tushirish
    asyncio.run(test_position_sizer())

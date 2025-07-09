"""
Risk Calculator - AI OrderFlow & Signal Bot

Risk hisoblash va boshqaruv tizimi. Har bir savdo uchun risk darajasini
aniqlaydi, stop loss va take profit hisoblaydi.

O'zbekcha: Bu modul savdo riskini hisoblash va nazorat qilish uchun
"""

import asyncio
import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import json

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

class MarketCondition(Enum):
    """Bozor holatini aniqlash"""
    TRENDING = "trending"
    RANGING = "ranging"
    VOLATILE = "volatile"
    CALM = "calm"

@dataclass
class RiskMetrics:
    """Risk ko'rsatkichlari"""
    account_balance: float
    risk_per_trade: float
    max_risk_percent: float
    daily_loss_limit: float
    current_drawdown: float
    open_positions: int
    total_exposure: float
    
    def __post_init__(self):
        """Validatsiya qilish"""
        if self.risk_per_trade <= 0 or self.risk_per_trade > 0.1:
            raise ValueError("Risk per trade 0.1% dan 10% gacha bo'lishi kerak")

@dataclass
class RiskCalculationResult:
    """Risk hisoblash natijasi"""
    success: bool
    risk_level: RiskLevel
    recommended_lot_size: float
    max_lot_size: float
    stop_loss_price: float
    take_profit_price: float
    risk_reward_ratio: float
    position_value: float
    pip_value: float
    confidence_score: float
    warnings: List[str]
    error: Optional[str] = None

class RiskCalculator:
    """Risk hisoblash va boshqaruv class"""
    
    def __init__(self, config_manager: ConfigManager):
        """
        Risk calculator init
        
        Args:
            config_manager: Konfiguratsiya boshqaruvchi
        """
        self.config = config_manager
        self.name = self.__class__.__name__
        logger.info(f"{self.name} ishga tushirildi")
        
        # Risk sozlamalari
        self.max_risk_per_trade = self.config.get('trading.max_risk_per_trade', 0.02)
        self.max_daily_loss = self.config.get('trading.max_daily_loss', 0.05)
        self.max_total_exposure = self.config.get('trading.max_total_exposure', 0.1)
        self.min_risk_reward = self.config.get('trading.min_risk_reward', 1.5)
        
        # Bozor holatini kuzatish
        self.market_volatility = {}
        self.recent_trades = []
        
        logger.info(f"Risk sozlamalari yuklandi: max_risk={self.max_risk_per_trade}")

    async def calculate_risk(self, 
                           symbol: str,
                           entry_price: float,
                           direction: str,
                           account_balance: float,
                           market_data: Dict) -> RiskCalculationResult:
        """
        Asosiy risk hisoblash methodi
        
        Args:
            symbol: Valyuta juftligi
            entry_price: Kirish narxi
            direction: 'buy' yoki 'sell'
            account_balance: Akkaunt balansi
            market_data: Bozor ma'lumotlari
            
        Returns:
            RiskCalculationResult: Risk hisoblash natijasi
        """
        try:
            logger.info(f"Risk hisoblash boshlandi: {symbol} - {direction}")
            
            # Kirish ma'lumotlarini validatsiya qilish
            if not self._validate_inputs(symbol, entry_price, direction, account_balance):
                return RiskCalculationResult(
                    success=False,
                    risk_level=RiskLevel.EXTREME,
                    recommended_lot_size=0,
                    max_lot_size=0,
                    stop_loss_price=0,
                    take_profit_price=0,
                    risk_reward_ratio=0,
                    position_value=0,
                    pip_value=0,
                    confidence_score=0,
                    warnings=[],
                    error="Noto'g'ri kirish ma'lumotlari"
                )
            
            # Bozor holatini aniqlash
            market_condition = await self._analyze_market_condition(symbol, market_data)
            
            # Risk darajasini aniqlash
            risk_level = await self._calculate_risk_level(symbol, market_condition, market_data)
            
            # Stop Loss va Take Profit hisoblash
            stop_loss_price, take_profit_price = await self._calculate_sl_tp(
                symbol, entry_price, direction, market_data, risk_level
            )
            
            # Pip qiymatini hisoblash
            pip_value = await self._calculate_pip_value(symbol, account_balance)
            
            # Lot hajmini hisoblash
            recommended_lot_size = await self._calculate_lot_size(
                account_balance, entry_price, stop_loss_price, pip_value, symbol
            )
            
            # Maksimal lot hajmi
            max_lot_size = await self._calculate_max_lot_size(
                account_balance, symbol, risk_level
            )
            
            # Risk/Reward ratio
            risk_reward_ratio = await self._calculate_risk_reward_ratio(
                entry_price, stop_loss_price, take_profit_price, direction
            )
            
            # Position qiymatini hisoblash
            position_value = recommended_lot_size * entry_price * 100000  # 100k = 1 lot
            
            # Ishonch darajasini hisoblash
            confidence_score = await self._calculate_confidence_score(
                risk_level, market_condition, risk_reward_ratio
            )
            
            # Ogohlantirishlar
            warnings = await self._generate_warnings(
                risk_level, recommended_lot_size, max_lot_size, 
                risk_reward_ratio, market_condition
            )
            
            result = RiskCalculationResult(
                success=True,
                risk_level=risk_level,
                recommended_lot_size=min(recommended_lot_size, max_lot_size),
                max_lot_size=max_lot_size,
                stop_loss_price=stop_loss_price,
                take_profit_price=take_profit_price,
                risk_reward_ratio=risk_reward_ratio,
                position_value=position_value,
                pip_value=pip_value,
                confidence_score=confidence_score,
                warnings=warnings
            )
            
            logger.info(f"Risk hisoblash tugadi: {symbol} - Risk: {risk_level.value}")
            return result
            
        except Exception as e:
            logger.error(f"Risk hisoblashda xato: {e}")
            return RiskCalculationResult(
                success=False,
                risk_level=RiskLevel.EXTREME,
                recommended_lot_size=0,
                max_lot_size=0,
                stop_loss_price=0,
                take_profit_price=0,
                risk_reward_ratio=0,
                position_value=0,
                pip_value=0,
                confidence_score=0,
                warnings=[],
                error=str(e)
            )

    async def _validate_inputs(self, symbol: str, entry_price: float, 
                             direction: str, account_balance: float) -> bool:
        """
        Kirish ma'lumotlarini validatsiya qilish
        
        Args:
            symbol: Valyuta juftligi
            entry_price: Kirish narxi
            direction: Savdo yo'nalishi
            account_balance: Akkaunt balansi
            
        Returns:
            bool: Validatsiya natijasi
        """
        try:
            # Symbol tekshirish
            if not symbol or len(symbol) < 6:
                logger.error(f"Noto'g'ri symbol: {symbol}")
                return False
            
            # Narx tekshirish
            if entry_price <= 0:
                logger.error(f"Noto'g'ri narx: {entry_price}")
                return False
            
            # Yo'nalish tekshirish
            if direction not in ['buy', 'sell']:
                logger.error(f"Noto'g'ri yo'nalish: {direction}")
                return False
            
            # Balans tekshirish
            if account_balance <= 0:
                logger.error(f"Noto'g'ri balans: {account_balance}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Validatsiya xatosi: {e}")
            return False

    async def _analyze_market_condition(self, symbol: str, market_data: Dict) -> MarketCondition:
        """
        Bozor holatini aniqlash
        
        Args:
            symbol: Valyuta juftligi
            market_data: Bozor ma'lumotlari
            
        Returns:
            MarketCondition: Bozor holati
        """
        try:
            # Volatillikni hisoblash
            volatility = market_data.get('volatility', 0)
            atr = market_data.get('atr', 0)
            volume = market_data.get('volume', 0)
            
            # Trend kuchini aniqlash
            trend_strength = market_data.get('trend_strength', 0)
            
            # Bozor holatini aniqlash
            if volatility > 0.02 and atr > 0.001:
                return MarketCondition.VOLATILE
            elif trend_strength > 0.7:
                return MarketCondition.TRENDING
            elif volatility < 0.005 and atr < 0.0005:
                return MarketCondition.CALM
            else:
                return MarketCondition.RANGING
                
        except Exception as e:
            logger.error(f"Bozor holatini aniqlashda xato: {e}")
            return MarketCondition.RANGING

    async def _calculate_risk_level(self, symbol: str, market_condition: MarketCondition, 
                                  market_data: Dict) -> RiskLevel:
        """
        Risk darajasini aniqlash
        
        Args:
            symbol: Valyuta juftligi
            market_condition: Bozor holati
            market_data: Bozor ma'lumotlari
            
        Returns:
            RiskLevel: Risk darajasi
        """
        try:
            risk_score = 0
            
            # Bozor holatiga qarab risk
            if market_condition == MarketCondition.VOLATILE:
                risk_score += 3
            elif market_condition == MarketCondition.TRENDING:
                risk_score += 1
            elif market_condition == MarketCondition.RANGING:
                risk_score += 2
            else:  # CALM
                risk_score += 0
            
            # Volatillikka qarab risk
            volatility = market_data.get('volatility', 0)
            if volatility > 0.03:
                risk_score += 2
            elif volatility > 0.015:
                risk_score += 1
            
            # Spread ga qarab risk
            spread = market_data.get('spread', 0)
            if spread > 0.0005:
                risk_score += 1
            
            # Likvidlikka qarab risk
            volume = market_data.get('volume', 0)
            if volume < 1000:
                risk_score += 1
            
            # Risk darajasini aniqlash
            if risk_score >= 5:
                return RiskLevel.EXTREME
            elif risk_score >= 3:
                return RiskLevel.HIGH
            elif risk_score >= 1:
                return RiskLevel.MEDIUM
            else:
                return RiskLevel.LOW
                
        except Exception as e:
            logger.error(f"Risk darajasini aniqlashda xato: {e}")
            return RiskLevel.HIGH

    async def _calculate_sl_tp(self, symbol: str, entry_price: float, direction: str,
                             market_data: Dict, risk_level: RiskLevel) -> Tuple[float, float]:
        """
        Stop Loss va Take Profit hisoblash
        
        Args:
            symbol: Valyuta juftligi
            entry_price: Kirish narxi
            direction: Savdo yo'nalishi
            market_data: Bozor ma'lumotlari
            risk_level: Risk darajasi
            
        Returns:
            Tuple[float, float]: Stop Loss va Take Profit narxlari
        """
        try:
            # ATR (Average True Range) olish
            atr = market_data.get('atr', 0.001)
            
            # Risk darajasiga qarab multiplier
            multipliers = {
                RiskLevel.LOW: {'sl': 1.5, 'tp': 3.0},
                RiskLevel.MEDIUM: {'sl': 2.0, 'tp': 3.5},
                RiskLevel.HIGH: {'sl': 2.5, 'tp': 4.0},
                RiskLevel.EXTREME: {'sl': 3.0, 'tp': 4.5}
            }
            
            sl_multiplier = multipliers[risk_level]['sl']
            tp_multiplier = multipliers[risk_level]['tp']
            
            # Stop Loss va Take Profit hisoblash
            if direction == 'buy':
                stop_loss = entry_price - (atr * sl_multiplier)
                take_profit = entry_price + (atr * tp_multiplier)
            else:  # sell
                stop_loss = entry_price + (atr * sl_multiplier)
                take_profit = entry_price - (atr * tp_multiplier)
            
            # Minimal qadam hisobga olish
            pip_size = 0.0001 if 'JPY' not in symbol else 0.01
            
            stop_loss = round(stop_loss / pip_size) * pip_size
            take_profit = round(take_profit / pip_size) * pip_size
            
            return stop_loss, take_profit
            
        except Exception as e:
            logger.error(f"SL/TP hisoblashda xato: {e}")
            # Default qiymatlar
            if direction == 'buy':
                return entry_price * 0.99, entry_price * 1.02
            else:
                return entry_price * 1.01, entry_price * 0.98

    async def _calculate_pip_value(self, symbol: str, account_balance: float) -> float:
        """
        Pip qiymatini hisoblash
        
        Args:
            symbol: Valyuta juftligi
            account_balance: Akkaunt balansi
            
        Returns:
            float: Pip qiymat
        """
        try:
            # Base currency USD hisobida
            if symbol.endswith('USD'):
                pip_value = 10  # $10 per pip for 1 lot
            elif symbol.startswith('USD'):
                # USD/JPY kabi
                pip_value = 8  # taxminan
            else:
                # Cross currency
                pip_value = 10  # default
            
            return pip_value
            
        except Exception as e:
            logger.error(f"Pip qiymat hisoblashda xato: {e}")
            return 10

    async def _calculate_lot_size(self, account_balance: float, entry_price: float,
                                stop_loss_price: float, pip_value: float, symbol: str) -> float:
        """
        Lot hajmini hisoblash
        
        Args:
            account_balance: Akkaunt balansi
            entry_price: Kirish narxi
            stop_loss_price: Stop Loss narxi
            pip_value: Pip qiymat
            symbol: Valyuta juftligi
            
        Returns:
            float: Tavsiya etilgan lot hajmi
        """
        try:
            # Risk miqdorini hisoblash
            risk_amount = account_balance * self.max_risk_per_trade
            
            # Stop Loss masofasini hisoblash (pip hisobida)
            pip_size = 0.0001 if 'JPY' not in symbol else 0.01
            sl_distance_pips = abs(entry_price - stop_loss_price) / pip_size
            
            # Lot hajmini hisoblash
            if sl_distance_pips > 0 and pip_value > 0:
                lot_size = risk_amount / (sl_distance_pips * pip_value)
            else:
                lot_size = 0.01  # minimal lot
            
            # Lot hajmini cheklash
            lot_size = max(0.01, min(lot_size, 5.0))
            
            # 0.01 ga yaxlitlash
            lot_size = round(lot_size, 2)
            
            return lot_size
            
        except Exception as e:
            logger.error(f"Lot hajmini hisoblashda xato: {e}")
            return 0.01

    async def _calculate_max_lot_size(self, account_balance: float, symbol: str,
                                    risk_level: RiskLevel) -> float:
        """
        Maksimal lot hajmini hisoblash
        
        Args:
            account_balance: Akkaunt balansi
            symbol: Valyuta juftligi
            risk_level: Risk darajasi
            
        Returns:
            float: Maksimal lot hajmi
        """
        try:
            # Risk darajasiga qarab maksimal lot
            max_lots = {
                RiskLevel.LOW: 1.0,
                RiskLevel.MEDIUM: 0.5,
                RiskLevel.HIGH: 0.25,
                RiskLevel.EXTREME: 0.1
            }
            
            base_max_lot = max_lots[risk_level]
            
            # Balansga qarab sozlash
            if account_balance < 1000:
                base_max_lot *= 0.5
            elif account_balance > 10000:
                base_max_lot *= 1.5
            
            return round(base_max_lot, 2)
            
        except Exception as e:
            logger.error(f"Max lot hisoblashda xato: {e}")
            return 0.1

    async def _calculate_risk_reward_ratio(self, entry_price: float, stop_loss_price: float,
                                         take_profit_price: float, direction: str) -> float:
        """
        Risk/Reward ratio hisoblash
        
        Args:
            entry_price: Kirish narxi
            stop_loss_price: Stop Loss narxi
            take_profit_price: Take Profit narxi
            direction: Savdo yo'nalishi
            
        Returns:
            float: Risk/Reward ratio
        """
        try:
            # Risk va reward masofasini hisoblash
            risk_distance = abs(entry_price - stop_loss_price)
            reward_distance = abs(take_profit_price - entry_price)
            
            if risk_distance > 0:
                ratio = reward_distance / risk_distance
            else:
                ratio = 0
            
            return round(ratio, 2)
            
        except Exception as e:
            logger.error(f"Risk/Reward ratio hisoblashda xato: {e}")
            return 0

    async def _calculate_confidence_score(self, risk_level: RiskLevel, 
                                        market_condition: MarketCondition,
                                        risk_reward_ratio: float) -> float:
        """
        Ishonch darajasini hisoblash
        
        Args:
            risk_level: Risk darajasi
            market_condition: Bozor holati
            risk_reward_ratio: Risk/Reward ratio
            
        Returns:
            float: Ishonch darajasi (0-100)
        """
        try:
            confidence = 50  # base confidence
            
            # Risk darajasiga qarab
            if risk_level == RiskLevel.LOW:
                confidence += 20
            elif risk_level == RiskLevel.MEDIUM:
                confidence += 10
            elif risk_level == RiskLevel.HIGH:
                confidence -= 10
            else:  # EXTREME
                confidence -= 20
            
            # Bozor holatiga qarab
            if market_condition == MarketCondition.TRENDING:
                confidence += 15
            elif market_condition == MarketCondition.CALM:
                confidence += 5
            elif market_condition == MarketCondition.VOLATILE:
                confidence -= 15
            
            # Risk/Reward ratio ga qarab
            if risk_reward_ratio >= 3.0:
                confidence += 15
            elif risk_reward_ratio >= 2.0:
                confidence += 10
            elif risk_reward_ratio >= 1.5:
                confidence += 5
            else:
                confidence -= 10
            
            # 0-100 oralig'ida cheklash
            confidence = max(0, min(100, confidence))
            
            return round(confidence, 1)
            
        except Exception as e:
            logger.error(f"Ishonch darajasini hisoblashda xato: {e}")
            return 50.0

    async def _generate_warnings(self, risk_level: RiskLevel, recommended_lot_size: float,
                               max_lot_size: float, risk_reward_ratio: float,
                               market_condition: MarketCondition) -> List[str]:
        """
        Ogohlantirishlar yaratish
        
        Args:
            risk_level: Risk darajasi
            recommended_lot_size: Tavsiya etilgan lot hajmi
            max_lot_size: Maksimal lot hajmi
            risk_reward_ratio: Risk/Reward ratio
            market_condition: Bozor holati
            
        Returns:
            List[str]: Ogohlantirishlar ro'yxati
        """
        warnings = []
        
        try:
            # Yuqori risk ogohlantirish
            if risk_level == RiskLevel.HIGH:
                warnings.append("⚠️ Yuqori risk darajasi - ehtiyotkorlik tavsiya etiladi")
            elif risk_level == RiskLevel.EXTREME:
                warnings.append("🚨 Haddan tashqari risk - savdo tavsiya etilmaydi")
            
            # Lot hajmi ogohlantirish
            if recommended_lot_size > max_lot_size:
                warnings.append(f"⚠️ Tavsiya etilgan lot hajmi ({recommended_lot_size}) maksimaldan katta")
            
            # Risk/Reward ratio ogohlantirish
            if risk_reward_ratio < self.min_risk_reward:
                warnings.append(f"⚠️ Risk/Reward ratio ({risk_reward_ratio}) juda past")
            
            # Bozor holati ogohlantirish
            if market_condition == MarketCondition.VOLATILE:
                warnings.append("⚠️ Bozor juda o'zgaruvchan - ehtiyotkorlik tavsiya etiladi")
            
            # Minimal lot ogohlantirish
            if recommended_lot_size < 0.01:
                warnings.append("⚠️ Juda kichik lot hajmi - savdo samarasiz bo'lishi mumkin")
            
        except Exception as e:
            logger.error(f"Ogohlantirishlar yaratishda xato: {e}")
            warnings.append("⚠️ Risk tahlilida xato yuz berdi")
        
        return warnings

    async def check_daily_risk_limits(self, current_losses: float, account_balance: float) -> bool:
        """
        Kunlik risk limitlarini tekshirish
        
        Args:
            current_losses: Joriy kunlik yo'qotishlar
            account_balance: Akkaunt balansi
            
        Returns:
            bool: Savdo davom etish mumkinmi
        """
        try:
            daily_loss_percent = abs(current_losses) / account_balance
            
            if daily_loss_percent >= self.max_daily_loss:
                logger.warning(f"Kunlik yo'qotish limiti oshdi: {daily_loss_percent:.2%}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Kunlik risk limitini tekshirishda xato: {e}")
            return False

    async def update_market_volatility(self, symbol: str, volatility: float) -> None:
        """
        Bozor volatilligini yangilash
        
        Args:
            symbol: Valyuta juftligi
            volatility: Volatillik qiymat
        """
        try:
            self.market_volatility[symbol] = {
                'volatility': volatility,
                'updated_at': datetime.now()
            }
            
            # Eski ma'lumotlarni tozalash (24 soatdan katta)
            cutoff_time = datetime.now() - timedelta(hours=24)
            self.market_volatility = {
                k: v for k, v in self.market_volatility.items()
                if v['updated_at'] > cutoff_time
            }
            
        except Exception as e:
            logger.error(f"Volatillikni yangilashda xato: {e}")

    async def get_risk_summary(self) -> Dict:
        """
        Risk xulosasi olish
        
        Returns:
            Dict: Risk xulosasi
        """
        try:
            return {
                'max_risk_per_trade': self.max_risk_per_trade,
                'max_daily_loss': self.max_daily_loss,
                'max_total_exposure': self.max_total_exposure,
                'min_risk_reward': self.min_risk_reward,
                'market_volatility_count': len(self.market_volatility),
                'recent_trades_count': len(self.recent_trades),
                'status': 'active'
            }
            
        except Exception as e:
            logger.error(f"Risk xulosasini olishda xato: {e}")
            return {'status': 'error', 'error': str(e)}

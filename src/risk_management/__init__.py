"""
Risk Management Module - Risklarni boshqarish moduli
==================================================

Bu modul savdo riskini boshqarish uchun barcha kerakli komponentlarni o'z ichiga oladi:
- Risk kalkulyatori (RiskCalculator)
- Position hajm hisoblash (PositionSizer)  
- Trade monitoring (TradeMonitor)
- Propshot uchun maxsus risk qoidalari

Author: AI OrderFlow Bot
Version: 1.0
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from utils.logger import get_logger

logger = get_logger(__name__)

# Risk Management komponentlarini import qilish
try:
    from .risk_calculator import (
        RiskCalculator,
        RiskResult,
        RiskLevel,
        RiskMetrics
    )
    logger.info("RiskCalculator muvaffaqiyatli import qilindi")
except ImportError as e:
    logger.error(f"RiskCalculator import xatosi: {e}")
    RiskCalculator = None

try:
    from .position_sizer import (
        PositionSizer,
        PositionSize,
        SizingMethod,
        PositionResult
    )
    logger.info("PositionSizer muvaffaqiyatli import qilindi")
except ImportError as e:
    logger.error(f"PositionSizer import xatosi: {e}")
    PositionSizer = None

try:
    from .trade_monitor import (
        TradeMonitor,
        TradeStatus,
        MonitorResult,
        TradeAlert
    )
    logger.info("TradeMonitor muvaffaqiyatli import qilindi")
except ImportError as e:
    logger.error(f"TradeMonitor import xatosi: {e}")
    TradeMonitor = None

# Risk Level enum
@dataclass
class RiskLevel:
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

# Risk Management konfiguratsiyasi
@dataclass
class RiskConfig:
    """Risk boshqaruv konfiguratsiyasi"""
    max_risk_per_trade: float = 0.02      # Har trade uchun maksimal risk (2%)
    max_daily_loss: float = 0.05          # Kunlik maksimal yo'qotish (5%)
    max_total_loss: float = 0.10          # Umumiy maksimal yo'qotish (10%)
    max_lot_size: float = 0.5             # Maksimal lot hajmi
    max_daily_trades: int = 3             # Kunlik maksimal trade soni
    
    # Propshot maxsus qoidalari
    propshot_max_daily_loss: float = 0.025  # Propshot kunlik limit (2.5%)
    propshot_max_total_loss: float = 0.05   # Propshot umumiy limit (5%)
    propshot_max_lot_size: float = 0.5      # Propshot maksimal lot
    propshot_max_daily_trades: int = 3      # Propshot kunlik trade limit
    
    # Stop Loss va Take Profit qoidalari
    min_stop_loss_pips: int = 10          # Minimal SL (10 pips)
    max_stop_loss_pips: int = 50          # Maksimal SL (50 pips)
    min_take_profit_pips: int = 15        # Minimal TP (15 pips)
    risk_reward_ratio: float = 1.5        # Risk/Reward nisbati

# Risk Management Manager - barcha komponentlarni birlashtirivchi
class RiskManager:
    """
    Risk Management Manager - barcha risk komponentlarini boshqaradi
    """
    
    def __init__(self, config: Optional[RiskConfig] = None):
        """
        Risk Manager initializatsiyasi
        
        Args:
            config: Risk konfiguratsiyasi
        """
        self.config = config or RiskConfig()
        self.risk_calculator = RiskCalculator(self.config) if RiskCalculator else None
        self.position_sizer = PositionSizer(self.config) if PositionSizer else None
        self.trade_monitor = TradeMonitor(self.config) if TradeMonitor else None
        
        logger.info("RiskManager ishga tushirildi")
        
        # Komponentlar mavjudligini tekshirish
        self._check_components()
    
    def _check_components(self) -> None:
        """Risk komponentlarining mavjudligini tekshirish"""
        components = {
            'RiskCalculator': self.risk_calculator,
            'PositionSizer': self.position_sizer,
            'TradeMonitor': self.trade_monitor
        }
        
        missing_components = []
        for name, component in components.items():
            if component is None:
                missing_components.append(name)
        
        if missing_components:
            logger.warning(f"Quyidagi komponentlar mavjud emas: {missing_components}")
        else:
            logger.info("Barcha risk komponentlari muvaffaqiyatli yuklandi")
    
    async def calculate_trade_risk(self, trade_data: Dict) -> Optional[Dict]:
        """
        Trade uchun risk hisoblash
        
        Args:
            trade_data: Trade ma'lumotlari
            
        Returns:
            Risk hisoblash natijalari
        """
        if not self.risk_calculator:
            logger.error("RiskCalculator mavjud emas")
            return None
        
        try:
            result = await self.risk_calculator.calculate_risk(trade_data)
            logger.info(f"Risk hisoblandi: {result}")
            return result
        except Exception as e:
            logger.error(f"Risk hisoblashda xato: {e}")
            return None
    
    async def calculate_position_size(self, trade_data: Dict) -> Optional[Dict]:
        """
        Position hajmini hisoblash
        
        Args:
            trade_data: Trade ma'lumotlari
            
        Returns:
            Position hajm hisoblash natijalari
        """
        if not self.position_sizer:
            logger.error("PositionSizer mavjud emas")
            return None
        
        try:
            result = await self.position_sizer.calculate_size(trade_data)
            logger.info(f"Position hajmi hisoblandi: {result}")
            return result
        except Exception as e:
            logger.error(f"Position hajm hisoblashda xato: {e}")
            return None
    
    async def monitor_trade(self, trade_data: Dict) -> Optional[Dict]:
        """
        Trade monitoring
        
        Args:
            trade_data: Trade ma'lumotlari
            
        Returns:
            Monitoring natijalari
        """
        if not self.trade_monitor:
            logger.error("TradeMonitor mavjud emas")
            return None
        
        try:
            result = await self.trade_monitor.monitor_trade(trade_data)
            logger.info(f"Trade monitoring: {result}")
            return result
        except Exception as e:
            logger.error(f"Trade monitoringda xato: {e}")
            return None
    
    async def validate_trade(self, trade_data: Dict) -> Dict:
        """
        Trade validatsiyasi - barcha risk tekshiruvlari
        
        Args:
            trade_data: Trade ma'lumotlari
            
        Returns:
            Validatsiya natijalari
        """
        try:
            validation_result = {
                'valid': True,
                'errors': [],
                'warnings': [],
                'risk_data': {},
                'position_data': {},
                'monitor_data': {}
            }
            
            # Risk hisoblash
            risk_result = await self.calculate_trade_risk(trade_data)
            if risk_result:
                validation_result['risk_data'] = risk_result
                if risk_result.get('risk_level') == RiskLevel.CRITICAL:
                    validation_result['valid'] = False
                    validation_result['errors'].append("Risk darajasi juda yuqori")
            
            # Position hajm hisoblash
            position_result = await self.calculate_position_size(trade_data)
            if position_result:
                validation_result['position_data'] = position_result
                if position_result.get('lot_size', 0) > self.config.max_lot_size:
                    validation_result['valid'] = False
                    validation_result['errors'].append("Lot hajmi juda katta")
            
            # Trade monitoring
            monitor_result = await self.monitor_trade(trade_data)
            if monitor_result:
                validation_result['monitor_data'] = monitor_result
                if monitor_result.get('daily_trades', 0) >= self.config.max_daily_trades:
                    validation_result['valid'] = False
                    validation_result['errors'].append("Kunlik trade limiti to'ldi")
            
            logger.info(f"Trade validatsiyasi: {validation_result}")
            return validation_result
            
        except Exception as e:
            logger.error(f"Trade validatsiyasida xato: {e}")
            return {
                'valid': False,
                'errors': [f"Validatsiya xatosi: {str(e)}"],
                'warnings': [],
                'risk_data': {},
                'position_data': {},
                'monitor_data': {}
            }
    
    def get_risk_limits(self) -> Dict:
        """
        Joriy risk limitlarini qaytarish
        
        Returns:
            Risk limitlar ma'lumotlari
        """
        return {
            'max_risk_per_trade': self.config.max_risk_per_trade,
            'max_daily_loss': self.config.max_daily_loss,
            'max_total_loss': self.config.max_total_loss,
            'max_lot_size': self.config.max_lot_size,
            'max_daily_trades': self.config.max_daily_trades,
            'propshot_limits': {
                'max_daily_loss': self.config.propshot_max_daily_loss,
                'max_total_loss': self.config.propshot_max_total_loss,
                'max_lot_size': self.config.propshot_max_lot_size,
                'max_daily_trades': self.config.propshot_max_daily_trades
            }
        }
    
    def is_healthy(self) -> bool:
        """
        Risk Manager sog'lomligini tekshirish
        
        Returns:
            True agar barcha komponentlar ishlayotgan bo'lsa
        """
        return all([
            self.risk_calculator is not None,
            self.position_sizer is not None,
            self.trade_monitor is not None
        ])

# Module exports
__all__ = [
    # Classes
    'RiskManager',
    'RiskConfig',
    'RiskLevel',
    
    # Components (agar mavjud bo'lsa)
    'RiskCalculator',
    'PositionSizer', 
    'TradeMonitor',
    
    # Data classes
    'RiskResult',
    'PositionSize',
    'TradeStatus',
    'RiskMetrics',
    'SizingMethod',
    'PositionResult',
    'MonitorResult',
    'TradeAlert'
]

# Module versiya va ma'lumotlari
__version__ = "1.0.0"
__author__ = "AI OrderFlow Bot"
__description__ = "Risk Management Module - Savdo riskini boshqarish moduli"

# Module initializatsiya
logger.info(f"Risk Management Module v{__version__} yuklandi")
logger.info("Mavjud komponentlar: RiskManager, RiskConfig, RiskLevel")

# Propshot uchun maxsus risk qoidalari
PROPSHOT_RISK_RULES = {
    "max_daily_loss": 0.025,        # 2.5% kunlik yo'qotish
    "max_total_loss": 0.05,         # 5% umumiy yo'qotish
    "max_lot_size": 0.5,            # 0.5 lot maksimal
    "max_daily_trades": 3,          # 3 ta kunlik trade
    "required_profit_factor": 1.2,  # Profit faktor kamida 1.2
    "max_drawdown": 0.03,           # 3% maksimal drawdown
    "risk_per_trade": 0.02          # 2% risk har trade
}

# Xato handling uchun yordamchi funksiya
def handle_risk_error(error: Exception, context: str) -> Dict:
    """
    Risk xatolarini qayta ishlash
    
    Args:
        error: Xato objekti
        context: Xato konteksti
        
    Returns:
        Xato ma'lumotlari
    """
    logger.error(f"Risk xatosi [{context}]: {error}")
    return {
        'success': False,
        'error': str(error),
        'context': context,
        'timestamp': logger.time.time()
    }

# Risk moduli muvaffaqiyatli yuklandi
logger.info("Risk Management Module to'liq yuklandi ✅")

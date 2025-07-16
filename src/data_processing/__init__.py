"""
Data Processing Module - Ma'lumot qayta ishlash moduli
========================================================

Bu modul AI OrderFlow & Signal Bot uchun barcha ma'lumot qayta ishlash 
komponentlarini o'z ichiga oladi:

- Order Flow Analyzer: DEX dan kelgan order flow ma'lumotlarini tahlil qilish
- Sentiment Analyzer: AI yordamida sentiment tahlil qilish  
- Market Analyzer: Bozor ma'lumotlarini tahlil qilish
- Signal Generator: Tahlil natijalariga asoslanib signal yaratish

Yaratuvchi: AI OrderFlow Bot Team
Sana: 2025
Til: Python 3.9+
"""

import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from utils.logger import get_logger

# Logger o'rnatish
logger = get_logger(__name__)

# Modul versiyasi
__version__ = "1.0.0"
__author__ = "AI OrderFlow Bot Team"

# Asosiy ma'lumot qayta ishlash sinflar
try:
    from .order_flow_analyzer import OrderFlowAnalyzer
    from .sentiment_analyzer import SentimentAnalyzer
    from .market_analyzer import MarketAnalyzer
    from .signal_generator import SignalGenerator
    
    logger.info("Ma'lumot qayta ishlash moduli muvaffaqiyatli yuklandi")
    
except ImportError as e:
    logger.warning(f"Ba'zi komponentlar yuklanmadi: {e}")
    # Fallback - bo'sh klasslar yaratish
    OrderFlowAnalyzer = None
    SentimentAnalyzer = None
    MarketAnalyzer = None
    SignalGenerator = None

# Asosiy ma'lumot tuzilmalari
@dataclass
class ProcessingResult:
    """Qayta ishlash natijasi"""
    success: bool
    data: Optional[Dict] = None
    confidence: float = 0.0
    error: Optional[str] = None
    timestamp: float = 0.0
    processing_time: float = 0.0

@dataclass
class MarketData:
    """Bozor ma'lumotlari tuzilmasi"""
    symbol: str
    price: float
    volume: float
    timestamp: float
    source: str
    additional_data: Optional[Dict] = None

@dataclass
class SignalData:
    """Signal ma'lumotlari tuzilmasi"""
    symbol: str
    action: str  # BUY, SELL, HOLD
    confidence: float
    price: float
    stop_loss: float
    take_profit: float
    risk_percent: float
    reason: str
    timestamp: float
    expiry_time: float

# Modul konfiguratsiyasi
DEFAULT_CONFIG = {
    "order_flow": {
        "min_volume_threshold": 10000,
        "time_window": 300,  # 5 daqiqa
        "confidence_threshold": 0.7
    },
    "sentiment": {
        "sources": ["news", "social", "ai"],
        "weight_distribution": {"news": 0.4, "social": 0.3, "ai": 0.3},
        "min_confidence": 0.6
    },
    "market": {
        "timeframes": ["1m", "5m", "15m", "1h"],
        "indicators": ["rsi", "macd", "ema", "volume"],
        "lookback_period": 100
    },
    "signal": {
        "min_confidence": 0.75,
        "max_risk_per_trade": 0.02,
        "signal_expiry": 300  # 5 daqiqa
    }
}

# Yardamchi funksiyalar
async def initialize_processors(config: Dict = None) -> Dict[str, Any]:
    """
    Barcha qayta ishlash komponentlarini ishga tushirish
    
    Args:
        config: Konfiguratsiya lug'ati
        
    Returns:
        Ishga tushirilgan komponentlar lug'ati
    """
    try:
        if config is None:
            config = DEFAULT_CONFIG
            
        processors = {}
        
        # Order Flow Analyzer ishga tushirish
        if OrderFlowAnalyzer is not None:
            processors['order_flow'] = OrderFlowAnalyzer(config.get('order_flow', {}))
            logger.info("Order Flow Analyzer ishga tushirildi")
        
        # Sentiment Analyzer ishga tushirish  
        if SentimentAnalyzer is not None:
            processors['sentiment'] = SentimentAnalyzer(config.get('sentiment', {}))
            logger.info("Sentiment Analyzer ishga tushirildi")
            
        # Market Analyzer ishga tushirish
        if MarketAnalyzer is not None:
            processors['market'] = MarketAnalyzer(config.get('market', {}))
            logger.info("Market Analyzer ishga tushirildi")
            
        # Signal Generator ishga tushirish
        if SignalGenerator is not None:
            processors['signal'] = SignalGenerator(config.get('signal', {}))
            logger.info("Signal Generator ishga tushirildi")
            
        logger.info(f"Jami {len(processors)} ta komponent ishga tushirildi")
        return processors
        
    except Exception as e:
        logger.error(f"Komponentlarni ishga tushirishda xato: {e}")
        raise

def validate_market_data(data: Dict) -> bool:
    """
    Bozor ma'lumotlarini tekshirish
    
    Args:
        data: Tekshiriladigan ma'lumotlar
        
    Returns:
        True - agar ma'lumotlar to'g'ri bo'lsa
    """
    try:
        required_fields = ['symbol', 'price', 'volume', 'timestamp']
        
        for field in required_fields:
            if field not in data:
                logger.error(f"Majburiy maydon yo'q: {field}")
                return False
                
        # Ma'lumot turlari tekshirish
        if not isinstance(data['price'], (int, float)) or data['price'] <= 0:
            logger.error("Noto'g'ri narx qiymati")
            return False
            
        if not isinstance(data['volume'], (int, float)) or data['volume'] <= 0:
            logger.error("Noto'g'ri hajm qiymati")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"Ma'lumot tekshirishda xato: {e}")
        return False

def create_processing_pipeline(processors: Dict[str, Any]) -> List[str]:
    """
    Qayta ishlash quvurini yaratish
    
    Args:
        processors: Qayta ishlash komponentlari
        
    Returns:
        Qayta ishlash ketma-ketligi
    """
    pipeline = []
    
    # Standart ketma-ketlik
    if 'order_flow' in processors:
        pipeline.append('order_flow')
    if 'market' in processors:
        pipeline.append('market')
    if 'sentiment' in processors:
        pipeline.append('sentiment')
    if 'signal' in processors:
        pipeline.append('signal')
        
    logger.info(f"Qayta ishlash quvuri yaratildi: {' -> '.join(pipeline)}")
    return pipeline

# Eksport qilinadigan elementlar
__all__ = [
    'ProcessingResult',
    'MarketData', 
    'SignalData',
    'OrderFlowAnalyzer',
    'SentimentAnalyzer',
    'MarketAnalyzer',
    'SignalGenerator',
    'initialize_processors',
    'validate_market_data',
    'create_processing_pipeline',
    'DEFAULT_CONFIG'
]

# Modul yuklanganda log yozish
logger.info(f"Data Processing moduli yuklandi (v{__version__})")
logger.info("Mavjud komponentlar:")
for component in __all__:
    if component in globals() and globals()[component] is not None:
        logger.info(f"  ✅ {component}")
    else:
        logger.warning(f"  ❌ {component} - mavjud emas")

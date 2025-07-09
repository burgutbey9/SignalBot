import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import ta
from utils.logger import get_logger
from utils.error_handler import handle_processing_error
from config.config import ConfigManager

logger = get_logger(__name__)

@dataclass
class MarketSignal:
    """Market signal ma'lumotlari"""
    symbol: str
    signal_type: str  # 'buy', 'sell', 'hold'
    strength: float  # 0-100
    confidence: float  # 0-100
    price: float
    timestamp: datetime
    indicators: Dict[str, Any]
    risk_level: str  # 'low', 'medium', 'high'
    
@dataclass
class MarketAnalysisResult:
    """Market tahlil natijasi"""
    success: bool
    signals: List[MarketSignal] = None
    market_sentiment: str = None  # 'bullish', 'bearish', 'neutral'
    volatility: float = 0.0
    trend: str = None  # 'up', 'down', 'sideways'
    support_levels: List[float] = None
    resistance_levels: List[float] = None
    error: Optional[str] = None
    analysis_time: datetime = None
    
@dataclass
class TechnicalIndicators:
    """Texnik indikatorlar"""
    rsi: float
    macd: float
    macd_signal: float
    bollinger_upper: float
    bollinger_lower: float
    ema_20: float
    ema_50: float
    sma_200: float
    stoch_k: float
    stoch_d: float
    williams_r: float
    volume_sma: float
    
class MarketAnalyzer:
    """Market ma'lumotlarini tahlil qilish"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        self.name = self.__class__.__name__
        self.min_data_points = 200  # Minimum ma'lumot nuqtalari
        self.timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']
        logger.info(f"{self.name} ishga tushirildi")
        
    async def analyze_market_data(self, 
                                symbol: str, 
                                ohlcv_data: pd.DataFrame,
                                timeframe: str = '5m') -> MarketAnalysisResult:
        """
        Market ma'lumotlarini to'liq tahlil qilish
        
        Args:
            symbol: Trading juftligi
            ohlcv_data: OHLCV ma'lumotlari
            timeframe: Vaqt oralig'i
            
        Returns:
            MarketAnalysisResult
        """
        try:
            logger.info(f"Market tahlil boshlandi: {symbol} ({timeframe})")
            
            # Ma'lumotlarni validatsiya qilish
            if not self._validate_market_data(ohlcv_data):
                return MarketAnalysisResult(
                    success=False,
                    error="Noto'g'ri market ma'lumotlari"
                )
            
            # Texnik indikatorlarni hisoblash
            indicators = await self._calculate_technical_indicators(ohlcv_data)
            
            # Trend tahlili
            trend = await self._analyze_trend(ohlcv_data, indicators)
            
            # Support va resistance levellar
            support_levels, resistance_levels = await self._find_support_resistance(ohlcv_data)
            
            # Volatillik hisoblash
            volatility = await self._calculate_volatility(ohlcv_data)
            
            # Market sentiment
            market_sentiment = await self._analyze_market_sentiment(indicators, trend)
            
            # Signallar generatsiya qilish
            signals = await self._generate_market_signals(
                symbol, ohlcv_data, indicators, trend, timeframe
            )
            
            result = MarketAnalysisResult(
                success=True,
                signals=signals,
                market_sentiment=market_sentiment,
                volatility=volatility,
                trend=trend,
                support_levels=support_levels,
                resistance_levels=resistance_levels,
                analysis_time=datetime.now()
            )
            
            logger.info(f"Market tahlil tugadi: {symbol}, signallar: {len(signals)}")
            return result
            
        except Exception as e:
            logger.error(f"Market tahlil xatosi ({symbol}): {e}")
            return MarketAnalysisResult(
                success=False,
                error=str(e),
                analysis_time=datetime.now()
            )
    
    async def _calculate_technical_indicators(self, df: pd.DataFrame) -> TechnicalIndicators:
        """
        Texnik indikatorlarni hisoblash
        
        Args:
            df: OHLCV ma'lumotlari
            
        Returns:
            TechnicalIndicators
        """
        try:
            # RSI (Relative Strength Index)
            rsi = ta.momentum.rsi(df['close'], window=14).iloc[-1]
            
            # MACD
            macd_line = ta.trend.macd(df['close']).iloc[-1]
            macd_signal = ta.trend.macd_signal(df['close']).iloc[-1]
            
            # Bollinger Bands
            bollinger_upper = ta.volatility.bollinger_hband(df['close']).iloc[-1]
            bollinger_lower = ta.volatility.bollinger_lband(df['close']).iloc[-1]
            
            # Moving Averages
            ema_20 = ta.trend.ema_indicator(df['close'], window=20).iloc[-1]
            ema_50 = ta.trend.ema_indicator(df['close'], window=50).iloc[-1]
            sma_200 = ta.trend.sma_indicator(df['close'], window=200).iloc[-1]
            
            # Stochastic Oscillator
            stoch_k = ta.momentum.stoch(df['high'], df['low'], df['close']).iloc[-1]
            stoch_d = ta.momentum.stoch_signal(df['high'], df['low'], df['close']).iloc[-1]
            
            # Williams %R
            williams_r = ta.momentum.williams_r(df['high'], df['low'], df['close']).iloc[-1]
            
            # Volume SMA
            volume_sma = ta.volume.volume_sma(df['close'], df['volume']).iloc[-1]
            
            return TechnicalIndicators(
                rsi=rsi,
                macd=macd_line,
                macd_signal=macd_signal,
                bollinger_upper=bollinger_upper,
                bollinger_lower=bollinger_lower,
                ema_20=ema_20,
                ema_50=ema_50,
                sma_200=sma_200,
                stoch_k=stoch_k,
                stoch_d=stoch_d,
                williams_r=williams_r,
                volume_sma=volume_sma
            )
            
        except Exception as e:
            logger.error(f"Texnik indikatorlar hisoblashda xato: {e}")
            raise
    
    async def _analyze_trend(self, df: pd.DataFrame, indicators: TechnicalIndicators) -> str:
        """
        Trend tahlili
        
        Args:
            df: OHLCV ma'lumotlari
            indicators: Texnik indikatorlar
            
        Returns:
            str: 'up', 'down', 'sideways'
        """
        try:
            current_price = df['close'].iloc[-1]
            
            # EMA trend tahlili
            ema_trend_score = 0
            if current_price > indicators.ema_20:
                ema_trend_score += 1
            if indicators.ema_20 > indicators.ema_50:
                ema_trend_score += 1
            if indicators.ema_50 > indicators.sma_200:
                ema_trend_score += 1
            
            # Narx harakati tahlili
            price_change_5 = (df['close'].iloc[-1] - df['close'].iloc[-5]) / df['close'].iloc[-5]
            price_change_20 = (df['close'].iloc[-1] - df['close'].iloc[-20]) / df['close'].iloc[-20]
            
            # Umumiy trend hisoblash
            if ema_trend_score >= 2 and price_change_5 > 0.001 and price_change_20 > 0.01:
                return 'up'
            elif ema_trend_score <= 1 and price_change_5 < -0.001 and price_change_20 < -0.01:
                return 'down'
            else:
                return 'sideways'
                
        except Exception as e:
            logger.error(f"Trend tahlilida xato: {e}")
            return 'sideways'
    
    async def _find_support_resistance(self, df: pd.DataFrame) -> Tuple[List[float], List[float]]:
        """
        Support va resistance levellarni topish
        
        Args:
            df: OHLCV ma'lumotlari
            
        Returns:
            Tuple[List[float], List[float]]: Support va resistance levellar
        """
        try:
            # Pivot nuqtalarni topish
            highs = df['high'].rolling(window=10, center=True).max()
            lows = df['low'].rolling(window=10, center=True).min()
            
            # Resistance levellar (maksimum nuqtalar)
            resistance_levels = []
            for i in range(10, len(df) - 10):
                if df['high'].iloc[i] == highs.iloc[i]:
                    resistance_levels.append(df['high'].iloc[i])
            
            # Support levellar (minimum nuqtalar)
            support_levels = []
            for i in range(10, len(df) - 10):
                if df['low'].iloc[i] == lows.iloc[i]:
                    support_levels.append(df['low'].iloc[i])
            
            # Eng muhim levellarni saqlash (oxirgi 5 ta)
            resistance_levels = sorted(list(set(resistance_levels)), reverse=True)[:5]
            support_levels = sorted(list(set(support_levels)), reverse=True)[:5]
            
            return support_levels, resistance_levels
            
        except Exception as e:
            logger.error(f"Support/Resistance topishda xato: {e}")
            return [], []
    
    async def _calculate_volatility(self, df: pd.DataFrame) -> float:
        """
        Volatillikni hisoblash
        
        Args:
            df: OHLCV ma'lumotlari
            
        Returns:
            float: Volatillik foizi
        """
        try:
            # 20 kunlik volatillik
            returns = df['close'].pct_change().dropna()
            volatility = returns.rolling(window=20).std().iloc[-1] * np.sqrt(1440)  # 1440 minut = 1 kun
            
            return volatility * 100  # Foiz ko'rinishida
            
        except Exception as e:
            logger.error(f"Volatillik hisoblashda xato: {e}")
            return 0.0
    
    async def _analyze_market_sentiment(self, indicators: TechnicalIndicators, trend: str) -> str:
        """
        Market sentimentini tahlil qilish
        
        Args:
            indicators: Texnik indikatorlar
            trend: Trend yo'nalishi
            
        Returns:
            str: 'bullish', 'bearish', 'neutral'
        """
        try:
            bullish_signals = 0
            bearish_signals = 0
            
            # RSI tahlili
            if indicators.rsi < 30:
                bullish_signals += 1  # Oversold
            elif indicators.rsi > 70:
                bearish_signals += 1  # Overbought
            
            # MACD tahlili
            if indicators.macd > indicators.macd_signal:
                bullish_signals += 1
            else:
                bearish_signals += 1
            
            # Stochastic tahlili
            if indicators.stoch_k < 20:
                bullish_signals += 1
            elif indicators.stoch_k > 80:
                bearish_signals += 1
            
            # Trend tahlili
            if trend == 'up':
                bullish_signals += 2
            elif trend == 'down':
                bearish_signals += 2
            
            # Natijani aniqlash
            if bullish_signals > bearish_signals:
                return 'bullish'
            elif bearish_signals > bullish_signals:
                return 'bearish'
            else:
                return 'neutral'
                
        except Exception as e:
            logger.error(f"Market sentiment tahlilida xato: {e}")
            return 'neutral'
    
    async def _generate_market_signals(self, 
                                     symbol: str,
                                     df: pd.DataFrame,
                                     indicators: TechnicalIndicators,
                                     trend: str,
                                     timeframe: str) -> List[MarketSignal]:
        """
        Market signallarini generatsiya qilish
        
        Args:
            symbol: Trading juftligi
            df: OHLCV ma'lumotlari
            indicators: Texnik indikatorlar
            trend: Trend yo'nalishi
            timeframe: Vaqt oralig'i
            
        Returns:
            List[MarketSignal]: Signallar ro'yxati
        """
        try:
            signals = []
            current_price = df['close'].iloc[-1]
            
            # RSI signal
            rsi_signal = await self._analyze_rsi_signal(indicators, current_price)
            if rsi_signal:
                signals.append(rsi_signal)
            
            # MACD signal
            macd_signal = await self._analyze_macd_signal(indicators, current_price)
            if macd_signal:
                signals.append(macd_signal)
            
            # Bollinger Bands signal
            bb_signal = await self._analyze_bollinger_signal(indicators, current_price)
            if bb_signal:
                signals.append(bb_signal)
            
            # EMA Crossover signal
            ema_signal = await self._analyze_ema_crossover(indicators, current_price)
            if ema_signal:
                signals.append(ema_signal)
            
            # Har signal uchun umumiy ma'lumotlarni qo'shish
            for signal in signals:
                signal.symbol = symbol
                signal.timestamp = datetime.now()
                signal.indicators = {
                    'rsi': indicators.rsi,
                    'macd': indicators.macd,
                    'trend': trend,
                    'timeframe': timeframe
                }
            
            return signals
            
        except Exception as e:
            logger.error(f"Signal generatsiyada xato: {e}")
            return []
    
    async def _analyze_rsi_signal(self, indicators: TechnicalIndicators, price: float) -> Optional[MarketSignal]:
        """RSI asosida signal tahlili"""
        try:
            if indicators.rsi < 30:
                return MarketSignal(
                    symbol="",
                    signal_type="buy",
                    strength=min(100, (30 - indicators.rsi) * 3),
                    confidence=75,
                    price=price,
                    timestamp=datetime.now(),
                    indicators={},
                    risk_level="medium"
                )
            elif indicators.rsi > 70:
                return MarketSignal(
                    symbol="",
                    signal_type="sell",
                    strength=min(100, (indicators.rsi - 70) * 3),
                    confidence=75,
                    price=price,
                    timestamp=datetime.now(),
                    indicators={},
                    risk_level="medium"
                )
            return None
        except Exception as e:
            logger.error(f"RSI signal tahlilida xato: {e}")
            return None
    
    async def _analyze_macd_signal(self, indicators: TechnicalIndicators, price: float) -> Optional[MarketSignal]:
        """MACD asosida signal tahlili"""
        try:
            if indicators.macd > indicators.macd_signal and indicators.macd > 0:
                return MarketSignal(
                    symbol="",
                    signal_type="buy",
                    strength=min(100, abs(indicators.macd - indicators.macd_signal) * 1000),
                    confidence=80,
                    price=price,
                    timestamp=datetime.now(),
                    indicators={},
                    risk_level="low"
                )
            elif indicators.macd < indicators.macd_signal and indicators.macd < 0:
                return MarketSignal(
                    symbol="",
                    signal_type="sell",
                    strength=min(100, abs(indicators.macd - indicators.macd_signal) * 1000),
                    confidence=80,
                    price=price,
                    timestamp=datetime.now(),
                    indicators={},
                    risk_level="low"
                )
            return None
        except Exception as e:
            logger.error(f"MACD signal tahlilida xato: {e}")
            return None
    
    async def _analyze_bollinger_signal(self, indicators: TechnicalIndicators, price: float) -> Optional[MarketSignal]:
        """Bollinger Bands asosida signal tahlili"""
        try:
            if price <= indicators.bollinger_lower:
                return MarketSignal(
                    symbol="",
                    signal_type="buy",
                    strength=min(100, (indicators.bollinger_lower - price) / price * 1000),
                    confidence=70,
                    price=price,
                    timestamp=datetime.now(),
                    indicators={},
                    risk_level="medium"
                )
            elif price >= indicators.bollinger_upper:
                return MarketSignal(
                    symbol="",
                    signal_type="sell",
                    strength=min(100, (price - indicators.bollinger_upper) / price * 1000),
                    confidence=70,
                    price=price,
                    timestamp=datetime.now(),
                    indicators={},
                    risk_level="medium"
                )
            return None
        except Exception as e:
            logger.error(f"Bollinger signal tahlilida xato: {e}")
            return None
    
    async def _analyze_ema_crossover(self, indicators: TechnicalIndicators, price: float) -> Optional[MarketSignal]:
        """EMA Crossover asosida signal tahlili"""
        try:
            if indicators.ema_20 > indicators.ema_50 and price > indicators.ema_20:
                return MarketSignal(
                    symbol="",
                    signal_type="buy",
                    strength=min(100, (indicators.ema_20 - indicators.ema_50) / indicators.ema_50 * 1000),
                    confidence=85,
                    price=price,
                    timestamp=datetime.now(),
                    indicators={},
                    risk_level="low"
                )
            elif indicators.ema_20 < indicators.ema_50 and price < indicators.ema_20:
                return MarketSignal(
                    symbol="",
                    signal_type="sell",
                    strength=min(100, (indicators.ema_50 - indicators.ema_20) / indicators.ema_20 * 1000),
                    confidence=85,
                    price=price,
                    timestamp=datetime.now(),
                    indicators={},
                    risk_level="low"
                )
            return None
        except Exception as e:
            logger.error(f"EMA crossover tahlilida xato: {e}")
            return None
    
    def _validate_market_data(self, df: pd.DataFrame) -> bool:
        """
        Market ma'lumotlarini validatsiya qilish
        
        Args:
            df: OHLCV ma'lumotlari
            
        Returns:
            bool: Ma'lumotlar to'g'ri yoki yo'q
        """
        try:
            # Asosiy ustunlarni tekshirish
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_columns):
                logger.error("Kerakli ustunlar mavjud emas")
                return False
            
            # Minimum ma'lumot nuqtalarini tekshirish
            if len(df) < self.min_data_points:
                logger.error(f"Yetarli ma'lumot yo'q: {len(df)} < {self.min_data_points}")
                return False
            
            # NaN qiymatlarni tekshirish
            if df[required_columns].isnull().any().any():
                logger.error("NaN qiymatlar mavjud")
                return False
            
            # Musbat qiymatlarni tekshirish
            if (df[required_columns] <= 0).any().any():
                logger.error("Nolga yoki salbiy qiymatlar mavjud")
                return False
            
            # OHLC mantiqini tekshirish
            if not (df['high'] >= df['low']).all():
                logger.error("High < Low qiymatlar mavjud")
                return False
            
            if not (df['high'] >= df['open']).all():
                logger.error("High < Open qiymatlar mavjud")
                return False
            
            if not (df['high'] >= df['close']).all():
                logger.error("High < Close qiymatlar mavjud")
                return False
            
            if not (df['low'] <= df['open']).all():
                logger.error("Low > Open qiymatlar mavjud")
                return False
            
            if not (df['low'] <= df['close']).all():
                logger.error("Low > Close qiymatlar mavjud")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Ma'lumotlarni validatsiyada xato: {e}")
            return False
    
    async def get_market_summary(self, analysis_result: MarketAnalysisResult) -> Dict[str, Any]:
        """
        Market tahlil xulosasi
        
        Args:
            analysis_result: Tahlil natijasi
            
        Returns:
            Dict: Market xulosasi
        """
        try:
            if not analysis_result.success:
                return {
                    'success': False,
                    'error': analysis_result.error
                }
            
            # Signallarni guruhlash
            buy_signals = [s for s in analysis_result.signals if s.signal_type == 'buy']
            sell_signals = [s for s in analysis_result.signals if s.signal_type == 'sell']
            
            # O'rtacha kuch va ishonch
            avg_buy_strength = np.mean([s.strength for s in buy_signals]) if buy_signals else 0
            avg_sell_strength = np.mean([s.strength for s in sell_signals]) if sell_signals else 0
            
            avg_buy_confidence = np.mean([s.confidence for s in buy_signals]) if buy_signals else 0
            avg_sell_confidence = np.mean([s.confidence for s in sell_signals]) if sell_signals else 0
            
            # Umumiy tavsiya
            recommendation = "hold"
            if len(buy_signals) > len(sell_signals) and avg_buy_strength > 60:
                recommendation = "buy"
            elif len(sell_signals) > len(buy_signals) and avg_sell_strength > 60:
                recommendation = "sell"
            
            return {
                'success': True,
                'recommendation': recommendation,
                'market_sentiment': analysis_result.market_sentiment,
                'trend': analysis_result.trend,
                'volatility': round(analysis_result.volatility, 2),
                'signals_count': {
                    'buy': len(buy_signals),
                    'sell': len(sell_signals),
                    'total': len(analysis_result.signals)
                },
                'signal_strength': {
                    'buy': round(avg_buy_strength, 2),
                    'sell': round(avg_sell_strength, 2)
                },
                'confidence': {
                    'buy': round(avg_buy_confidence, 2),
                    'sell': round(avg_sell_confidence, 2)
                },
                'support_levels': analysis_result.support_levels,
                'resistance_levels': analysis_result.resistance_levels,
                'analysis_time': analysis_result.analysis_time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
        except Exception as e:
            logger.error(f"Market xulosasi yaratishda xato: {e}")
            return {
                'success': False,
                'error': str(e)
            }

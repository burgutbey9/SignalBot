import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
import pandas as pd

from utils.logger import get_logger
from utils.error_handler import handle_processing_error
from utils.fallback_manager import FallbackManager
from config.config import ConfigManager
from data_processing.order_flow_analyzer import OrderFlowAnalyzer
from data_processing.sentiment_analyzer import SentimentAnalyzer
from data_processing.market_analyzer import MarketAnalyzer
from risk_management.risk_calculator import RiskCalculator

logger = get_logger(__name__)

class SignalType(Enum):
    """Signal turlari"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    CLOSE = "CLOSE"

class SignalStrength(Enum):
    """Signal kuchi"""
    WEAK = "WEAK"
    MEDIUM = "MEDIUM"
    STRONG = "STRONG"
    VERY_STRONG = "VERY_STRONG"

class MarketCondition(Enum):
    """Market holati"""
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    SIDEWAYS = "SIDEWAYS"
    VOLATILE = "VOLATILE"

@dataclass
class SignalData:
    """Signal ma'lumotlari"""
    symbol: str
    action: SignalType
    price: float
    lot_size: float
    stop_loss: float
    take_profit: float
    confidence: float
    risk_percent: float
    strength: SignalStrength
    market_condition: MarketCondition
    timeframe: str
    timestamp: datetime
    reason: str
    sl_pips: float
    tp_pips: float
    expected_profit: float
    risk_reward_ratio: float
    account: str
    strategy_name: str
    
    # Tahlil ma'lumotlari
    order_flow_score: float
    sentiment_score: float
    market_score: float
    technical_score: float
    
    # Qo'shimcha ma'lumotlar
    news_impact: str
    volume_profile: Dict
    support_resistance: Dict
    trend_direction: str
    
    def to_dict(self) -> Dict:
        """Dictionary formatiga o'tkazish"""
        return asdict(self)

@dataclass
class SignalGeneratorConfig:
    """Signal Generator konfiguratsiyasi"""
    min_confidence: float = 0.65
    max_daily_signals: int = 10
    min_risk_reward: float = 1.5
    max_risk_per_trade: float = 0.02
    signal_expiry_minutes: int = 30
    required_confirmations: int = 3
    
    # Og'irlik koeffitsientlari
    order_flow_weight: float = 0.35
    sentiment_weight: float = 0.25
    market_weight: float = 0.25
    technical_weight: float = 0.15
    
    # Filtr parametrlari
    min_volume_threshold: float = 100000  # USD
    max_spread_pips: float = 2.0
    volatility_filter: bool = True
    news_filter: bool = True

class SignalGenerator:
    """AI Signal Generator - asosiy signal yaratish tizimi"""
    
    def __init__(self, config_manager: ConfigManager):
        """Signal Generator initsializatsiya"""
        self.config = config_manager
        self.signal_config = SignalGeneratorConfig()
        
        # Analyzer instancelari
        self.order_flow_analyzer = OrderFlowAnalyzer(config_manager)
        self.sentiment_analyzer = SentimentAnalyzer(config_manager)
        self.market_analyzer = MarketAnalyzer(config_manager)
        self.risk_calculator = RiskCalculator(config_manager)
        
        # Fallback manager
        self.fallback_manager = FallbackManager()
        
        # Signal tarixi
        self.signal_history: List[SignalData] = []
        self.daily_signal_count = 0
        self.last_reset_date = datetime.now().date()
        
        # Active signallar
        self.active_signals: Dict[str, SignalData] = {}
        
        logger.info("Signal Generator ishga tushirildi")
    
    async def generate_signal(self, symbol: str, timeframe: str = "1h") -> Optional[SignalData]:
        """
        Asosiy signal yaratish methodi
        
        Args:
            symbol: Valyuta juftligi (masalan: EURUSD)
            timeframe: Vaqt oralig'i (1m, 5m, 15m, 1h, 4h, 1d)
            
        Returns:
            SignalData yoki None
        """
        try:
            logger.info(f"Signal yaratish boshlandi: {symbol} - {timeframe}")
            
            # Kunlik signal limitini tekshirish
            if not self._check_daily_limit():
                logger.warning(f"Kunlik signal limiti ({self.signal_config.max_daily_signals}) tugadi")
                return None
            
            # Market holatini tekshirish
            if not await self._check_market_conditions(symbol):
                logger.info(f"Market sharoitlari signal yaratish uchun mos emas: {symbol}")
                return None
            
            # Ma'lumotlarni parallel ravishda olish
            analysis_results = await self._collect_analysis_data(symbol, timeframe)
            
            if not analysis_results:
                logger.error(f"Tahlil ma'lumotlari olinmadi: {symbol}")
                return None
            
            # Signal yaratish
            signal = await self._create_signal(symbol, timeframe, analysis_results)
            
            if signal:
                # Signal validatsiyasi
                if await self._validate_signal(signal):
                    # Signal saqlash
                    await self._save_signal(signal)
                    logger.info(f"Signal muvaffaqiyatli yaratildi: {symbol} - {signal.action.value}")
                    return signal
                else:
                    logger.warning(f"Signal validatsiyadan o'tmadi: {symbol}")
                    return None
            else:
                logger.info(f"Signal yaratilmadi: {symbol} - yetarli signal yo'q")
                return None
                
        except Exception as e:
            logger.error(f"Signal yaratishda xato: {symbol} - {e}")
            return None
    
    async def _collect_analysis_data(self, symbol: str, timeframe: str) -> Optional[Dict]:
        """Barcha tahlil ma'lumotlarini yig'ish"""
        try:
            # Parallel ravishda ma'lumotlar olish
            tasks = [
                self.order_flow_analyzer.analyze_order_flow(symbol, timeframe),
                self.sentiment_analyzer.analyze_sentiment(symbol),
                self.market_analyzer.analyze_market_data(symbol, timeframe),
                self._get_technical_analysis(symbol, timeframe)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Natijalarni tekshirish
            order_flow_data = results[0] if not isinstance(results[0], Exception) else None
            sentiment_data = results[1] if not isinstance(results[1], Exception) else None
            market_data = results[2] if not isinstance(results[2], Exception) else None
            technical_data = results[3] if not isinstance(results[3], Exception) else None
            
            if not any([order_flow_data, sentiment_data, market_data, technical_data]):
                logger.error(f"Hech qanday tahlil ma'lumoti olinmadi: {symbol}")
                return None
            
            return {
                "order_flow": order_flow_data,
                "sentiment": sentiment_data,
                "market": market_data,
                "technical": technical_data
            }
            
        except Exception as e:
            logger.error(f"Tahlil ma'lumotlarini yig'ishda xato: {symbol} - {e}")
            return None
    
    async def _create_signal(self, symbol: str, timeframe: str, analysis_data: Dict) -> Optional[SignalData]:
        """Signal yaratish asosiy logikasi"""
        try:
            # Scorelartni hisoblash
            scores = self._calculate_scores(analysis_data)
            
            # Umumiy confidence hisoblash
            confidence = self._calculate_confidence(scores)
            
            # Minimum confidence tekshirish
            if confidence < self.signal_config.min_confidence:
                logger.info(f"Yetarli ishonch yo'q: {confidence:.2f} < {self.signal_config.min_confidence}")
                return None
            
            # Signal yo'nalishini aniqlash
            signal_direction = self._determine_signal_direction(scores)
            
            if signal_direction == SignalType.HOLD:
                logger.info(f"Signal yo'nalishi aniqlanmadi: {symbol}")
                return None
            
            # Narx ma'lumotlarini olish
            current_price = await self._get_current_price(symbol)
            if not current_price:
                logger.error(f"Joriy narx olinmadi: {symbol}")
                return None
            
            # Risk hisoblash
            risk_data = await self.risk_calculator.calculate_risk(
                symbol, signal_direction, current_price, confidence
            )
            
            if not risk_data:
                logger.error(f"Risk hisoblash xatosi: {symbol}")
                return None
            
            # Signal kuchini aniqlash
            signal_strength = self._determine_signal_strength(confidence, scores)
            
            # Market holatini aniqlash
            market_condition = self._determine_market_condition(analysis_data)
            
            # Signal ma'lumotlarini yaratish
            signal = SignalData(
                symbol=symbol,
                action=signal_direction,
                price=current_price,
                lot_size=risk_data.lot_size,
                stop_loss=risk_data.stop_loss,
                take_profit=risk_data.take_profit,
                confidence=confidence,
                risk_percent=risk_data.risk_percent,
                strength=signal_strength,
                market_condition=market_condition,
                timeframe=timeframe,
                timestamp=datetime.now(),
                reason=self._generate_signal_reason(analysis_data, scores),
                sl_pips=risk_data.sl_pips,
                tp_pips=risk_data.tp_pips,
                expected_profit=risk_data.expected_profit,
                risk_reward_ratio=risk_data.risk_reward_ratio,
                account=self.config.get_setting("trading.account_name", "Main"),
                strategy_name="AI_OrderFlow_Strategy",
                
                # Tahlil scorelari
                order_flow_score=scores.get("order_flow", 0),
                sentiment_score=scores.get("sentiment", 0),
                market_score=scores.get("market", 0),
                technical_score=scores.get("technical", 0),
                
                # Qo'shimcha ma'lumotlar
                news_impact=analysis_data.get("sentiment", {}).get("news_impact", "LOW"),
                volume_profile=analysis_data.get("market", {}).get("volume_profile", {}),
                support_resistance=analysis_data.get("technical", {}).get("support_resistance", {}),
                trend_direction=analysis_data.get("technical", {}).get("trend_direction", "NEUTRAL")
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Signal yaratishda xato: {symbol} - {e}")
            return None
    
    def _calculate_scores(self, analysis_data: Dict) -> Dict[str, float]:
        """Har bir tahlil uchun score hisoblash"""
        scores = {}
        
        # Order Flow score
        if analysis_data.get("order_flow"):
            of_data = analysis_data["order_flow"]
            scores["order_flow"] = self._calculate_order_flow_score(of_data)
        else:
            scores["order_flow"] = 0.0
        
        # Sentiment score
        if analysis_data.get("sentiment"):
            sent_data = analysis_data["sentiment"]
            scores["sentiment"] = self._calculate_sentiment_score(sent_data)
        else:
            scores["sentiment"] = 0.0
        
        # Market score
        if analysis_data.get("market"):
            market_data = analysis_data["market"]
            scores["market"] = self._calculate_market_score(market_data)
        else:
            scores["market"] = 0.0
        
        # Technical score
        if analysis_data.get("technical"):
            tech_data = analysis_data["technical"]
            scores["technical"] = self._calculate_technical_score(tech_data)
        else:
            scores["technical"] = 0.0
        
        return scores
    
    def _calculate_order_flow_score(self, order_flow_data: Dict) -> float:
        """Order Flow score hisoblash"""
        try:
            # Buy/Sell pressure
            buy_pressure = order_flow_data.get("buy_pressure", 0)
            sell_pressure = order_flow_data.get("sell_pressure", 0)
            
            # Volume profili
            volume_delta = order_flow_data.get("volume_delta", 0)
            
            # Bid/Ask spread
            spread_score = order_flow_data.get("spread_score", 0)
            
            # Weighted score
            score = (buy_pressure - sell_pressure) * 0.4 + volume_delta * 0.4 + spread_score * 0.2
            
            # Normalize qilish (-1, 1)
            return max(-1, min(1, score))
            
        except Exception as e:
            logger.error(f"Order Flow score hisoblashda xato: {e}")
            return 0.0
    
    def _calculate_sentiment_score(self, sentiment_data: Dict) -> float:
        """Sentiment score hisoblash"""
        try:
            # News sentiment
            news_score = sentiment_data.get("news_sentiment", 0)
            
            # Social media sentiment
            social_score = sentiment_data.get("social_sentiment", 0)
            
            # Market sentiment
            market_sentiment = sentiment_data.get("market_sentiment", 0)
            
            # Weighted score
            score = news_score * 0.4 + social_score * 0.3 + market_sentiment * 0.3
            
            # Normalize qilish (-1, 1)
            return max(-1, min(1, score))
            
        except Exception as e:
            logger.error(f"Sentiment score hisoblashda xato: {e}")
            return 0.0
    
    def _calculate_market_score(self, market_data: Dict) -> float:
        """Market score hisoblash"""
        try:
            # Volume profili
            volume_score = market_data.get("volume_score", 0)
            
            # Volatility
            volatility_score = market_data.get("volatility_score", 0)
            
            # Liquidity
            liquidity_score = market_data.get("liquidity_score", 0)
            
            # Weighted score
            score = volume_score * 0.4 + volatility_score * 0.3 + liquidity_score * 0.3
            
            # Normalize qilish (-1, 1)
            return max(-1, min(1, score))
            
        except Exception as e:
            logger.error(f"Market score hisoblashda xato: {e}")
            return 0.0
    
    def _calculate_technical_score(self, technical_data: Dict) -> float:
        """Technical score hisoblash"""
        try:
            # Trend ko'rsatkichlari
            trend_score = technical_data.get("trend_score", 0)
            
            # Momentum
            momentum_score = technical_data.get("momentum_score", 0)
            
            # Support/Resistance
            sr_score = technical_data.get("support_resistance_score", 0)
            
            # Weighted score
            score = trend_score * 0.4 + momentum_score * 0.3 + sr_score * 0.3
            
            # Normalize qilish (-1, 1)
            return max(-1, min(1, score))
            
        except Exception as e:
            logger.error(f"Technical score hisoblashda xato: {e}")
            return 0.0
    
    def _calculate_confidence(self, scores: Dict[str, float]) -> float:
        """Umumiy confidence hisoblash"""
        try:
            # Weighted average
            weighted_sum = (
                scores.get("order_flow", 0) * self.signal_config.order_flow_weight +
                scores.get("sentiment", 0) * self.signal_config.sentiment_weight +
                scores.get("market", 0) * self.signal_config.market_weight +
                scores.get("technical", 0) * self.signal_config.technical_weight
            )
            
            # Absolute qiymat va 0-1 orasida normalize qilish
            confidence = abs(weighted_sum)
            
            # Confirmation bonus (ko'p tahlil bir yo'nalishda bo'lsa)
            confirmations = sum(1 for score in scores.values() if abs(score) > 0.3)
            confirmation_bonus = confirmations * 0.1
            
            final_confidence = min(1.0, confidence + confirmation_bonus)
            
            logger.info(f"Confidence hisoblandi: {final_confidence:.3f}")
            return final_confidence
            
        except Exception as e:
            logger.error(f"Confidence hisoblashda xato: {e}")
            return 0.0
    
    def _determine_signal_direction(self, scores: Dict[str, float]) -> SignalType:
        """Signal yo'nalishini aniqlash"""
        try:
            # Weighted average
            weighted_sum = (
                scores.get("order_flow", 0) * self.signal_config.order_flow_weight +
                scores.get("sentiment", 0) * self.signal_config.sentiment_weight +
                scores.get("market", 0) * self.signal_config.market_weight +
                scores.get("technical", 0) * self.signal_config.technical_weight
            )
            
            # Threshold
            threshold = 0.2
            
            if weighted_sum > threshold:
                return SignalType.BUY
            elif weighted_sum < -threshold:
                return SignalType.SELL
            else:
                return SignalType.HOLD
                
        except Exception as e:
            logger.error(f"Signal yo'nalishini aniqlashda xato: {e}")
            return SignalType.HOLD
    
    def _determine_signal_strength(self, confidence: float, scores: Dict[str, float]) -> SignalStrength:
        """Signal kuchini aniqlash"""
        try:
            # Confidence asosida
            if confidence >= 0.9:
                return SignalStrength.VERY_STRONG
            elif confidence >= 0.8:
                return SignalStrength.STRONG
            elif confidence >= 0.7:
                return SignalStrength.MEDIUM
            else:
                return SignalStrength.WEAK
                
        except Exception as e:
            logger.error(f"Signal kuchini aniqlashda xato: {e}")
            return SignalStrength.WEAK
    
    def _determine_market_condition(self, analysis_data: Dict) -> MarketCondition:
        """Market holatini aniqlash"""
        try:
            # Volatility tekshirish
            volatility = analysis_data.get("market", {}).get("volatility", 0)
            
            # Trend yo'nalishi
            trend = analysis_data.get("technical", {}).get("trend_direction", "NEUTRAL")
            
            if volatility > 0.7:
                return MarketCondition.VOLATILE
            elif trend == "BULLISH":
                return MarketCondition.BULLISH
            elif trend == "BEARISH":
                return MarketCondition.BEARISH
            else:
                return MarketCondition.SIDEWAYS
                
        except Exception as e:
            logger.error(f"Market holatini aniqlashda xato: {e}")
            return MarketCondition.SIDEWAYS
    
    def _generate_signal_reason(self, analysis_data: Dict, scores: Dict[str, float]) -> str:
        """Signal sababini yaratish"""
        try:
            reasons = []
            
            # Order Flow
            if abs(scores.get("order_flow", 0)) > 0.3:
                if scores["order_flow"] > 0:
                    reasons.append("Kuchli sotib olish bosimi")
                else:
                    reasons.append("Kuchli sotish bosimi")
            
            # Sentiment
            if abs(scores.get("sentiment", 0)) > 0.3:
                if scores["sentiment"] > 0:
                    reasons.append("Ijobiy sentiment")
                else:
                    reasons.append("Salbiy sentiment")
            
            # Market
            if abs(scores.get("market", 0)) > 0.3:
                if scores["market"] > 0:
                    reasons.append("Kuchli market signali")
                else:
                    reasons.append("Zaif market signali")
            
            # Technical
            if abs(scores.get("technical", 0)) > 0.3:
                if scores["technical"] > 0:
                    reasons.append("Texnik ko'rsatkichlar ijobiy")
                else:
                    reasons.append("Texnik ko'rsatkichlar salbiy")
            
            if not reasons:
                return "AI tahlil asosida"
            
            return " | ".join(reasons)
            
        except Exception as e:
            logger.error(f"Signal sababini yaratishda xato: {e}")
            return "AI tahlil asosida"
    
    async def _get_technical_analysis(self, symbol: str, timeframe: str) -> Optional[Dict]:
        """Technical analysis ma'lumotlarini olish"""
        try:
            # Bu yerda technical analysis logikasi bo'ladi
            # Hozircha mock data
            return {
                "trend_score": 0.3,
                "momentum_score": 0.2,
                "support_resistance_score": 0.1,
                "trend_direction": "BULLISH",
                "support_resistance": {
                    "support": 1.0800,
                    "resistance": 1.0900
                }
            }
            
        except Exception as e:
            logger.error(f"Technical analysis olishda xato: {symbol} - {e}")
            return None
    
    async def _get_current_price(self, symbol: str) -> Optional[float]:
        """Joriy narxni olish"""
        try:
            # Bu yerda narx olish logikasi bo'ladi
            # Hozircha mock data
            if symbol == "EURUSD":
                return 1.0850
            elif symbol == "GBPUSD":
                return 1.2750
            elif symbol == "USDJPY":
                return 148.50
            else:
                return 1.0000
                
        except Exception as e:
            logger.error(f"Joriy narx olishda xato: {symbol} - {e}")
            return None
    
    async def _check_market_conditions(self, symbol: str) -> bool:
        """Market sharoitlarini tekshirish"""
        try:
            # Spread tekshirish
            spread = await self._get_spread(symbol)
            if spread and spread > self.signal_config.max_spread_pips:
                logger.warning(f"Spread juda katta: {spread} pips")
                return False
            
            # Volume tekshirish
            volume = await self._get_volume(symbol)
            if volume and volume < self.signal_config.min_volume_threshold:
                logger.warning(f"Volume juda kam: {volume}")
                return False
            
            # Trading session tekshirish
            if not self._is_trading_session():
                logger.info("Trading session emas")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Market sharoitlarini tekshirishda xato: {symbol} - {e}")
            return False
    
    async def _get_spread(self, symbol: str) -> Optional[float]:
        """Spread olish"""
        try:
            # Mock data
            return 1.5  # pips
        except Exception as e:
            logger.error(f"Spread olishda xato: {symbol} - {e}")
            return None
    
    async def _get_volume(self, symbol: str) -> Optional[float]:
        """Volume olish"""
        try:
            # Mock data
            return 150000  # USD
        except Exception as e:
            logger.error(f"Volume olishda xato: {symbol} - {e}")
            return None
    
    def _is_trading_session(self) -> bool:
        """Trading session tekshirish"""
        try:
            # UZB vaqti bilan 07:00-19:30 orasida
            now = datetime.now()
            start_time = now.replace(hour=7, minute=0, second=0, microsecond=0)
            end_time = now.replace(hour=19, minute=30, second=0, microsecond=0)
            
            return start_time <= now <= end_time
            
        except Exception as e:
            logger.error(f"Trading session tekshirishda xato: {e}")
            return True
    
    def _check_daily_limit(self) -> bool:
        """Kunlik signal limitini tekshirish"""
        try:
            today = datetime.now().date()
            
            # Kun o'zgargan bo'lsa reset qilish
            if today != self.last_reset_date:
                self.daily_signal_count = 0
                self.last_reset_date = today
            
            return self.daily_signal_count < self.signal_config.max_daily_signals
            
        except Exception as e:
            logger.error(f"Kunlik limit tekshirishda xato: {e}")
            return False
    
    async def _validate_signal(self, signal: SignalData) -> bool:
        """Signal validatsiyasi"""
        try:
            # Risk/Reward ratio tekshirish
            if signal.risk_reward_ratio < self.signal_config.min_risk_reward:
                logger.warning(f"Risk/Reward ratio juda kichik: {signal.risk_reward_ratio}")
                return False
            
            # Risk foizi tekshirish
            if signal.risk_percent > self.signal_config.max_risk_per_trade:
                logger.warning(f"Risk foizi juda katta: {signal.risk_percent}")
                return False
            
            # Price validation
            if signal.price <= 0:
                logger.error(f"Noto'g'ri narx: {signal.price}")
                return False
            
            # Stop Loss va Take Profit tekshirish
            if signal.action == SignalType.BUY:
                if signal.stop_loss >= signal.price or signal.take_profit <= signal.price:
                    logger.error("BUY signal uchun SL/TP noto'g'ri")
                    return False
            elif signal.action == SignalType.SELL:
                if signal.stop_loss <= signal.price or signal.take_profit >= signal.price:
                    logger.error("SELL signal uchun SL/TP noto'g'ri")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Signal validatsiyasida xato: {e}")
            return False
    
    async def _save_signal(self, signal: SignalData) -> None:
        """Signalni saqlash"""
        try:
            # Tarixga qo'shish
            self.signal_history.append(signal)
            
            # Active signallarga qo'shish
            self.active_signals[signal.symbol] = signal
            
            # Kunlik hisobni oshirish
            self.daily_signal_count += 1
            
            # Database ga saqlash (keyingi versiyada)
            # await self.db_manager.save_signal(signal)
            
            logger.info(f"Signal saqlandi: {signal.symbol} - {signal.action.value}")
            
        except Exception as e:
            logger.error(f"Signal saqlashda xato: {e}")
    
    async def get_active_signals(self) -> List[SignalData]:
        """Faol signallar ro'yxati"""
        try:
            # Eski signallarni tozalash
            current_time = datetime.now()
            expired_symbols = []
            
            for symbol, signal in self.active_signals.items():
                if (current_time - signal.timestamp).total_seconds() > (self.signal_config.signal_expiry_minutes * 60):
                    expired_symbols.append(symbol)
            
            for symbol in expired_symbols:
                del self.active_signals[symbol]
                logger.info(f"Eski signal o'chirildi: {symbol}")
            
            return list(self.active_signals.values())
            
        except Exception as e:
            logger.error(f"Faol signallar olishda xato: {e}")
            return []
    
    async def close_signal(self, symbol: str) -> bool:
        """Signalni yopish"""
        try:
            if symbol in self.active_signals:
                del self.active_signals[symbol]
                logger.info(f"Signal yopildi: {symbol}")
                return True
            else:
                logger.warning(f"Signal topilmadi: {symbol}")
                return False
                
        except Exception as e:
            logger.error(f"Signal yopishda xato: {symbol} - {e}")
            return False
    
    def get_signal_statistics(self) -> Dict:
        """Signal statistikasi"""
        try:
            total_signals = len(self.signal_history)
            if total_signals == 0:
                return {"total_signals": 0}
            
            # Signal turlari
            buy_signals = sum(1 for s in self.signal_history if s.action == SignalType.BUY)
            sell_signals = sum(1 for s in self.signal_history if s.action == SignalType.SELL)
            
            # O'rtacha confidence
            avg_confidence = sum(s.confidence for s in self.signal_history) / total_signals
            
            # Bugungi signallar
            today = datetime.now().date()
            today_signals = sum(1 for s in self.signal_history if s.timestamp.date() == today)
            
            # Signal kuchlari
            strong_signals = sum(1 for s in self.signal_history 
                               if s.strength in [SignalStrength.STRONG, SignalStrength.VERY_STRONG])
            
            # O'rtacha risk/reward
            avg_risk_reward = sum(s.risk_reward_ratio for s in self.signal_history) / total_signals
            
            return {
                "total_signals": total_signals,
                "buy_signals": buy_signals,
                "sell_signals": sell_signals,
                "today_signals": today_signals,
                "daily_limit": self.signal_config.max_daily_signals,
                "avg_confidence": round(avg_confidence, 3),
                "strong_signals": strong_signals,
                "avg_risk_reward": round(avg_risk_reward, 2),
                "active_signals": len(self.active_signals)
            }
            
        except Exception as e:
            logger.error(f"Signal statistikasida xato: {e}")
            return {"error": str(e)}
    
    async def update_signal_config(self, new_config: Dict) -> bool:
        """Signal konfiguratsiyasini yangilash"""
        try:
            # Konfiguratsiya yangilash
            for key, value in new_config.items():
                if hasattr(self.signal_config, key):
                    setattr(self.signal_config, key, value)
                    logger.info(f"Konfiguratsiya yangilandi: {key} = {value}")
            
            return True
            
        except Exception as e:
            logger.error(f"Konfiguratsiya yangilashda xato: {e}")
            return False
    
    async def backtest_strategy(self, symbol: str, start_date: datetime, end_date: datetime) -> Dict:
        """Strategiya backtest qilish"""
        try:
            logger.info(f"Backtest boshlandi: {symbol} ({start_date} - {end_date})")
            
            # Bu yerda backtest logikasi bo'ladi
            # Tarixiy ma'lumotlarni olish
            # Signallarni generate qilish
            # P&L hisoblash
            
            # Hozircha mock data
            mock_results = {
                "total_trades": 25,
                "winning_trades": 15,
                "losing_trades": 10,
                "win_rate": 0.6,
                "total_profit": 1250.0,
                "total_loss": -750.0,
                "net_profit": 500.0,
                "profit_factor": 1.67,
                "max_drawdown": -200.0,
                "avg_win": 83.33,
                "avg_loss": -75.0,
                "risk_reward_ratio": 1.11,
                "sharpe_ratio": 1.25,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "symbol": symbol
            }
            
            logger.info(f"Backtest tugadi: {symbol} - Net P&L: {mock_results['net_profit']}")
            return mock_results
            
        except Exception as e:
            logger.error(f"Backtest xatosi: {symbol} - {e}")
            return {"error": str(e)}
    
    async def optimize_parameters(self, symbol: str, optimization_period: int = 30) -> Dict:
        """Signal parametrlarini optimizatsiya qilish"""
        try:
            logger.info(f"Parametr optimizatsiya boshlandi: {symbol}")
            
            # Bu yerda optimization logikasi bo'ladi
            # Turli parametrlarni sinash
            # Eng yaxshi natija beruvchi parametrlarni topish
            
            # Hozircha mock data
            optimized_params = {
                "min_confidence": 0.68,
                "order_flow_weight": 0.38,
                "sentiment_weight": 0.27,
                "market_weight": 0.22,
                "technical_weight": 0.13,
                "min_risk_reward": 1.6,
                "max_risk_per_trade": 0.018,
                "optimization_score": 0.85,
                "backtest_period": optimization_period,
                "symbol": symbol
            }
            
            logger.info(f"Parametr optimizatsiya tugadi: {symbol} - Score: {optimized_params['optimization_score']}")
            return optimized_params
            
        except Exception as e:
            logger.error(f"Parametr optimizatsiyasida xato: {symbol} - {e}")
            return {"error": str(e)}
    
    async def get_signal_performance(self, symbol: str = None, days: int = 7) -> Dict:
        """Signal performance tahlili"""
        try:
            # Ma'lum muddatdagi signallarni olish
            cutoff_date = datetime.now() - timedelta(days=days)
            
            if symbol:
                filtered_signals = [s for s in self.signal_history 
                                  if s.symbol == symbol and s.timestamp >= cutoff_date]
            else:
                filtered_signals = [s for s in self.signal_history 
                                  if s.timestamp >= cutoff_date]
            
            if not filtered_signals:
                return {"message": "Ma'lumot topilmadi"}
            
            # Performance metriklari
            total_signals = len(filtered_signals)
            
            # Signal turlari bo'yicha
            buy_count = sum(1 for s in filtered_signals if s.action == SignalType.BUY)
            sell_count = sum(1 for s in filtered_signals if s.action == SignalType.SELL)
            
            # Confidence statistikasi
            avg_confidence = sum(s.confidence for s in filtered_signals) / total_signals
            max_confidence = max(s.confidence for s in filtered_signals)
            min_confidence = min(s.confidence for s in filtered_signals)
            
            # Risk/Reward statistikasi
            avg_risk_reward = sum(s.risk_reward_ratio for s in filtered_signals) / total_signals
            
            # Kuch bo'yicha taqsimot
            strength_distribution = {}
            for strength in SignalStrength:
                count = sum(1 for s in filtered_signals if s.strength == strength)
                strength_distribution[strength.value] = count
            
            # Market condition bo'yicha
            market_distribution = {}
            for condition in MarketCondition:
                count = sum(1 for s in filtered_signals if s.market_condition == condition)
                market_distribution[condition.value] = count
            
            # Kunlik taqsimot
            daily_distribution = {}
            for signal in filtered_signals:
                date_str = signal.timestamp.strftime("%Y-%m-%d")
                daily_distribution[date_str] = daily_distribution.get(date_str, 0) + 1
            
            return {
                "period_days": days,
                "symbol": symbol or "ALL",
                "total_signals": total_signals,
                "buy_signals": buy_count,
                "sell_signals": sell_count,
                "confidence_stats": {
                    "average": round(avg_confidence, 3),
                    "maximum": round(max_confidence, 3),
                    "minimum": round(min_confidence, 3)
                },
                "avg_risk_reward": round(avg_risk_reward, 2),
                "strength_distribution": strength_distribution,
                "market_distribution": market_distribution,
                "daily_distribution": daily_distribution
            }
            
        except Exception as e:
            logger.error(f"Performance tahlilida xato: {e}")
            return {"error": str(e)}
    
    async def generate_batch_signals(self, symbols: List[str], timeframe: str = "1h") -> List[SignalData]:
        """Bir nechta symbol uchun signallar yaratish"""
        try:
            logger.info(f"Batch signal yaratish boshlandi: {len(symbols)} symbols")
            
            # Parallel ravishda signallar yaratish
            tasks = [self.generate_signal(symbol, timeframe) for symbol in symbols]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Muvaffaqiyatli signallarni ajratish
            valid_signals = []
            for i, result in enumerate(results):
                if isinstance(result, SignalData):
                    valid_signals.append(result)
                elif isinstance(result, Exception):
                    logger.error(f"Batch signal xatosi {symbols[i]}: {result}")
            
            logger.info(f"Batch signal tugadi: {len(valid_signals)}/{len(symbols)} muvaffaqiyatli")
            return valid_signals
            
        except Exception as e:
            logger.error(f"Batch signal yaratishda xato: {e}")
            return []
    
    async def cleanup_old_signals(self, days: int = 30) -> int:
        """Eski signallarni tozalash"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            # Eski signallarni topish
            old_signals = [s for s in self.signal_history if s.timestamp < cutoff_date]
            old_count = len(old_signals)
            
            # Eski signallarni o'chirish
            self.signal_history = [s for s in self.signal_history if s.timestamp >= cutoff_date]
            
            logger.info(f"Eski signallar tozalandi: {old_count} ta signal o'chirildi")
            return old_count
            
        except Exception as e:
            logger.error(f"Eski signallarni tozalashda xato: {e}")
            return 0
    
    async def export_signals(self, format: str = "json", symbol: str = None, days: int = 7) -> Optional[str]:
        """Signallarni eksport qilish"""
        try:
            # Ma'lum muddatdagi signallarni olish
            cutoff_date = datetime.now() - timedelta(days=days)
            
            if symbol:
                filtered_signals = [s for s in self.signal_history 
                                  if s.symbol == symbol and s.timestamp >= cutoff_date]
            else:
                filtered_signals = [s for s in self.signal_history 
                                  if s.timestamp >= cutoff_date]
            
            if not filtered_signals:
                return None
            
            # Format bo'yicha eksport
            if format.lower() == "json":
                export_data = {
                    "export_date": datetime.now().isoformat(),
                    "period_days": days,
                    "symbol": symbol or "ALL",
                    "total_signals": len(filtered_signals),
                    "signals": [s.to_dict() for s in filtered_signals]
                }
                
                return json.dumps(export_data, indent=2, default=str)
            
            elif format.lower() == "csv":
                # CSV format uchun
                import csv
                import io
                
                output = io.StringIO()
                writer = csv.writer(output)
                
                # Header
                writer.writerow([
                    "Timestamp", "Symbol", "Action", "Price", "Lot Size",
                    "Stop Loss", "Take Profit", "Confidence", "Risk %",
                    "Strength", "Market Condition", "Reason"
                ])
                
                # Data
                for signal in filtered_signals:
                    writer.writerow([
                        signal.timestamp.isoformat(),
                        signal.symbol,
                        signal.action.value,
                        signal.price,
                        signal.lot_size,
                        signal.stop_loss,
                        signal.take_profit,
                        signal.confidence,
                        signal.risk_percent,
                        signal.strength.value,
                        signal.market_condition.value,
                        signal.reason
                    ])
                
                return output.getvalue()
            
            else:
                logger.error(f"Noto'g'ri format: {format}")
                return None
                
        except Exception as e:
            logger.error(f"Signal eksport qilishda xato: {e}")
            return None
    
    def __repr__(self) -> str:
        """String representation"""
        return f"SignalGenerator(active_signals={len(self.active_signals)}, total_signals={len(self.signal_history)})"

# Utility functions
async def create_signal_generator(config_path: str = "config/settings.json") -> SignalGenerator:
    """Signal Generator yaratish"""
    try:
        config_manager = ConfigManager(config_path)
        signal_generator = SignalGenerator(config_manager)
        
        logger.info("Signal Generator muvaffaqiyatli yaratildi")
        return signal_generator
        
    except Exception as e:
        logger.error(f"Signal Generator yaratishda xato: {e}")
        raise

async def test_signal_generation():
    """Test function - signal generation"""
    try:
        # Test uchun signal generator yaratish
        config_manager = ConfigManager()
        generator = SignalGenerator(config_manager)
        
        # Test symbollar
        test_symbols = ["EURUSD", "GBPUSD", "USDJPY"]
        
        logger.info("Signal generation test boshlandi")
        
        # Har bir symbol uchun signal yaratish
        for symbol in test_symbols:
            signal = await generator.generate_signal(symbol, "1h")
            
            if signal:
                logger.info(f"Test signal: {symbol} - {signal.action.value} - {signal.confidence:.3f}")
            else:
                logger.warning(f"Test signal yaratilmadi: {symbol}")
        
        # Statistika
        stats = generator.get_signal_statistics()
        logger.info(f"Test statistikasi: {stats}")
        
        logger.info("Signal generation test tugadi")
        
    except Exception as e:
        logger.error(f"Test xatosi: {e}")

if __name__ == "__main__":
    """Test va debug uchun"""
    import asyncio
    
    async def main():
        await test_signal_generation()
    
    asyncio.run(main())

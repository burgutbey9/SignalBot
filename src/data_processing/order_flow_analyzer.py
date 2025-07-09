import asyncio
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from enum import Enum
from collections import defaultdict
import statistics

from utils.logger import get_logger
from utils.error_handler import handle_processing_error
from utils.retry_handler import retry_async
from config.config import ConfigManager

logger = get_logger(__name__)

class OrderType(Enum):
    """Savdo buyruqlari turlari"""
    BUY = "BUY"
    SELL = "SELL"
    LIMIT_BUY = "LIMIT_BUY"
    LIMIT_SELL = "LIMIT_SELL"
    MARKET_BUY = "MARKET_BUY"
    MARKET_SELL = "MARKET_SELL"

class FlowSignal(Enum):
    """Order Flow signallari"""
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    NEUTRAL = "NEUTRAL"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"

@dataclass
class OrderFlowData:
    """Order Flow ma'lumotlari"""
    timestamp: datetime
    symbol: str
    price: float
    volume: float
    side: str  # buy/sell
    order_type: OrderType
    block_size: float = 0.0
    gas_price: float = 0.0
    transaction_hash: str = ""
    wallet_address: str = ""
    
@dataclass
class FlowMetrics:
    """Order Flow metriklari"""
    buy_volume: float = 0.0
    sell_volume: float = 0.0
    buy_orders: int = 0
    sell_orders: int = 0
    large_buy_orders: int = 0
    large_sell_orders: int = 0
    average_buy_size: float = 0.0
    average_sell_size: float = 0.0
    volume_imbalance: float = 0.0
    price_impact: float = 0.0
    whale_activity: float = 0.0
    
@dataclass
class FlowPattern:
    """Order Flow pattern"""
    pattern_type: str
    strength: float
    confidence: float
    direction: str
    duration: int  # minutes
    volume_ratio: float
    price_range: Tuple[float, float]
    
@dataclass
class OrderFlowAnalysis:
    """Order Flow tahlil natijasi"""
    success: bool
    symbol: str
    timestamp: datetime
    signal: FlowSignal
    confidence: float
    metrics: FlowMetrics
    patterns: List[FlowPattern] = field(default_factory=list)
    large_orders: List[OrderFlowData] = field(default_factory=list)
    price_levels: Dict[str, float] = field(default_factory=dict)
    market_sentiment: str = "NEUTRAL"
    recommendation: str = ""
    error: Optional[str] = None

class OrderFlowAnalyzer:
    """Order Flow tahlil qiluvchi"""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.name = "OrderFlowAnalyzer"
        self.order_history: List[OrderFlowData] = []
        self.analysis_window = 30  # minutes
        self.whale_threshold = 100000  # USD
        self.large_order_threshold = 50000  # USD
        
        # Pattern parametrlari
        self.imbalance_threshold = 0.7  # 70% imbalance
        self.volume_spike_threshold = 2.0  # 2x average volume
        self.price_impact_threshold = 0.005  # 0.5% price impact
        
        logger.info(f"{self.name} ishga tushirildi")
    
    async def analyze_order_flow(self, orders: List[Dict]) -> OrderFlowAnalysis:
        """Order Flow tahlil qilish"""
        try:
            if not orders:
                return OrderFlowAnalysis(
                    success=False,
                    symbol="",
                    timestamp=datetime.now(),
                    signal=FlowSignal.NEUTRAL,
                    confidence=0.0,
                    metrics=FlowMetrics(),
                    error="Bo'sh order ma'lumotlari"
                )
            
            # Ma'lumotlarni qayta ishlash
            flow_data = await self._process_order_data(orders)
            
            if not flow_data:
                return OrderFlowAnalysis(
                    success=False,
                    symbol="",
                    timestamp=datetime.now(),
                    signal=FlowSignal.NEUTRAL,
                    confidence=0.0,
                    metrics=FlowMetrics(),
                    error="Order ma'lumotlarini qayta ishlashda xato"
                )
            
            symbol = flow_data[0].symbol
            
            # Metriklari hisoblash
            metrics = await self._calculate_metrics(flow_data)
            
            # Pattern aniqlash
            patterns = await self._detect_patterns(flow_data, metrics)
            
            # Katta buyruqlarni topish
            large_orders = await self._find_large_orders(flow_data)
            
            # Muhim narx sathlarini aniqlash
            price_levels = await self._identify_price_levels(flow_data)
            
            # Signal yaratish
            signal, confidence = await self._generate_signal(metrics, patterns, large_orders)
            
            # Market sentiment aniqlash
            market_sentiment = await self._analyze_market_sentiment(metrics, patterns)
            
            # Tavsiya yaratish
            recommendation = await self._generate_recommendation(signal, confidence, metrics)
            
            logger.info(f"Order Flow tahlil tugallandi - {symbol}: {signal.value} ({confidence:.2f}%)")
            
            return OrderFlowAnalysis(
                success=True,
                symbol=symbol,
                timestamp=datetime.now(),
                signal=signal,
                confidence=confidence,
                metrics=metrics,
                patterns=patterns,
                large_orders=large_orders,
                price_levels=price_levels,
                market_sentiment=market_sentiment,
                recommendation=recommendation
            )
            
        except Exception as e:
            logger.error(f"Order Flow tahlilida xato: {e}")
            return OrderFlowAnalysis(
                success=False,
                symbol="",
                timestamp=datetime.now(),
                signal=FlowSignal.NEUTRAL,
                confidence=0.0,
                metrics=FlowMetrics(),
                error=str(e)
            )
    
    async def _process_order_data(self, orders: List[Dict]) -> List[OrderFlowData]:
        """Order ma'lumotlarini qayta ishlash"""
        try:
            flow_data = []
            
            for order in orders:
                try:
                    # Ma'lumotlarni validatsiya qilish
                    if not self._validate_order_data(order):
                        continue
                    
                    # OrderFlowData yaratish
                    flow_order = OrderFlowData(
                        timestamp=datetime.fromisoformat(order.get('timestamp', datetime.now().isoformat())),
                        symbol=order.get('symbol', ''),
                        price=float(order.get('price', 0)),
                        volume=float(order.get('volume', 0)),
                        side=order.get('side', '').lower(),
                        order_type=OrderType(order.get('type', 'MARKET_BUY')),
                        block_size=float(order.get('block_size', 0)),
                        gas_price=float(order.get('gas_price', 0)),
                        transaction_hash=order.get('tx_hash', ''),
                        wallet_address=order.get('wallet', '')
                    )
                    
                    flow_data.append(flow_order)
                    
                except Exception as e:
                    logger.warning(f"Order ma'lumotini qayta ishlashda xato: {e}")
                    continue
            
            # Vaqt bo'yicha saralash
            flow_data.sort(key=lambda x: x.timestamp)
            
            # Eski ma'lumotlarni olib tashlash
            cutoff_time = datetime.now() - timedelta(minutes=self.analysis_window)
            flow_data = [order for order in flow_data if order.timestamp > cutoff_time]
            
            logger.info(f"Qayta ishlangan orderlar: {len(flow_data)}")
            return flow_data
            
        except Exception as e:
            logger.error(f"Order ma'lumotlarini qayta ishlashda xato: {e}")
            return []
    
    def _validate_order_data(self, order: Dict) -> bool:
        """Order ma'lumotlarini validatsiya qilish"""
        required_fields = ['price', 'volume', 'side', 'symbol']
        
        for field in required_fields:
            if field not in order:
                return False
            
            if field in ['price', 'volume'] and float(order[field]) <= 0:
                return False
        
        if order['side'].lower() not in ['buy', 'sell']:
            return False
            
        return True
    
    async def _calculate_metrics(self, flow_data: List[OrderFlowData]) -> FlowMetrics:
        """Order Flow metriklari hisoblash"""
        try:
            metrics = FlowMetrics()
            
            if not flow_data:
                return metrics
            
            buy_orders = [order for order in flow_data if order.side == 'buy']
            sell_orders = [order for order in flow_data if order.side == 'sell']
            
            # Asosiy metrikalar
            metrics.buy_orders = len(buy_orders)
            metrics.sell_orders = len(sell_orders)
            
            metrics.buy_volume = sum(order.volume * order.price for order in buy_orders)
            metrics.sell_volume = sum(order.volume * order.price for order in sell_orders)
            
            # O'rtacha hajmlar
            if buy_orders:
                metrics.average_buy_size = metrics.buy_volume / len(buy_orders)
            if sell_orders:
                metrics.average_sell_size = metrics.sell_volume / len(sell_orders)
            
            # Katta buyruqlar
            metrics.large_buy_orders = len([order for order in buy_orders 
                                          if order.volume * order.price > self.large_order_threshold])
            metrics.large_sell_orders = len([order for order in sell_orders 
                                           if order.volume * order.price > self.large_order_threshold])
            
            # Volume imbalance
            total_volume = metrics.buy_volume + metrics.sell_volume
            if total_volume > 0:
                metrics.volume_imbalance = (metrics.buy_volume - metrics.sell_volume) / total_volume
            
            # Price impact hisoblash
            if len(flow_data) > 1:
                first_price = flow_data[0].price
                last_price = flow_data[-1].price
                metrics.price_impact = (last_price - first_price) / first_price
            
            # Whale activity
            whale_orders = [order for order in flow_data 
                           if order.volume * order.price > self.whale_threshold]
            metrics.whale_activity = len(whale_orders) / len(flow_data) if flow_data else 0
            
            return metrics
            
        except Exception as e:
            logger.error(f"Metrikalar hisoblashda xato: {e}")
            return FlowMetrics()
    
    async def _detect_patterns(self, flow_data: List[OrderFlowData], metrics: FlowMetrics) -> List[FlowPattern]:
        """Order Flow patternlarini aniqlash"""
        try:
            patterns = []
            
            # Volume imbalance pattern
            if abs(metrics.volume_imbalance) > self.imbalance_threshold:
                direction = "BULLISH" if metrics.volume_imbalance > 0 else "BEARISH"
                strength = abs(metrics.volume_imbalance)
                
                patterns.append(FlowPattern(
                    pattern_type="VOLUME_IMBALANCE",
                    strength=strength,
                    confidence=min(strength * 100, 95),
                    direction=direction,
                    duration=self.analysis_window,
                    volume_ratio=metrics.buy_volume / metrics.sell_volume if metrics.sell_volume > 0 else float('inf'),
                    price_range=(min(order.price for order in flow_data), 
                               max(order.price for order in flow_data))
                ))
            
            # Whale activity pattern
            if metrics.whale_activity > 0.1:  # 10% whale activity
                patterns.append(FlowPattern(
                    pattern_type="WHALE_ACTIVITY",
                    strength=metrics.whale_activity,
                    confidence=min(metrics.whale_activity * 500, 90),
                    direction="STRONG_MOVE",
                    duration=self.analysis_window,
                    volume_ratio=metrics.buy_volume / metrics.sell_volume if metrics.sell_volume > 0 else 1.0,
                    price_range=(min(order.price for order in flow_data), 
                               max(order.price for order in flow_data))
                ))
            
            # Price impact pattern
            if abs(metrics.price_impact) > self.price_impact_threshold:
                direction = "BULLISH" if metrics.price_impact > 0 else "BEARISH"
                
                patterns.append(FlowPattern(
                    pattern_type="PRICE_IMPACT",
                    strength=abs(metrics.price_impact),
                    confidence=min(abs(metrics.price_impact) * 10000, 85),
                    direction=direction,
                    duration=self.analysis_window,
                    volume_ratio=metrics.buy_volume / metrics.sell_volume if metrics.sell_volume > 0 else 1.0,
                    price_range=(min(order.price for order in flow_data), 
                               max(order.price for order in flow_data))
                ))
            
            # Large order clustering pattern
            if metrics.large_buy_orders > 3 or metrics.large_sell_orders > 3:
                direction = "BULLISH" if metrics.large_buy_orders > metrics.large_sell_orders else "BEARISH"
                strength = max(metrics.large_buy_orders, metrics.large_sell_orders) / len(flow_data)
                
                patterns.append(FlowPattern(
                    pattern_type="LARGE_ORDER_CLUSTER",
                    strength=strength,
                    confidence=min(strength * 200, 80),
                    direction=direction,
                    duration=self.analysis_window,
                    volume_ratio=metrics.buy_volume / metrics.sell_volume if metrics.sell_volume > 0 else 1.0,
                    price_range=(min(order.price for order in flow_data), 
                               max(order.price for order in flow_data))
                ))
            
            logger.info(f"Aniqlangan patternlar: {len(patterns)}")
            return patterns
            
        except Exception as e:
            logger.error(f"Pattern aniqlashda xato: {e}")
            return []
    
    async def _find_large_orders(self, flow_data: List[OrderFlowData]) -> List[OrderFlowData]:
        """Katta buyruqlarni topish"""
        try:
            large_orders = []
            
            for order in flow_data:
                order_value = order.volume * order.price
                if order_value > self.large_order_threshold:
                    large_orders.append(order)
            
            # Hajm bo'yicha saralash
            large_orders.sort(key=lambda x: x.volume * x.price, reverse=True)
            
            logger.info(f"Katta buyruqlar topildi: {len(large_orders)}")
            return large_orders[:10]  # Eng katta 10 ta
            
        except Exception as e:
            logger.error(f"Katta buyruqlarni topishda xato: {e}")
            return []
    
    async def _identify_price_levels(self, flow_data: List[OrderFlowData]) -> Dict[str, float]:
        """Muhim narx sathlarini aniqlash"""
        try:
            price_levels = {}
            
            if not flow_data:
                return price_levels
            
            prices = [order.price for order in flow_data]
            volumes = [order.volume for order in flow_data]
            
            # Support va resistance sathlar
            price_levels['support'] = min(prices)
            price_levels['resistance'] = max(prices)
            
            # Volume ağırlıklı o'rtacha narx (VWAP)
            total_volume = sum(volumes)
            if total_volume > 0:
                vwap = sum(order.price * order.volume for order in flow_data) / total_volume
                price_levels['vwap'] = vwap
            
            # Eng ko'p savdo bo'lgan narx sathi
            price_counts = defaultdict(float)
            for order in flow_data:
                price_bucket = round(order.price, 2)
                price_counts[price_bucket] += order.volume
            
            if price_counts:
                highest_volume_price = max(price_counts.items(), key=lambda x: x[1])[0]
                price_levels['high_volume_node'] = highest_volume_price
            
            # Joriy narx
            price_levels['current'] = flow_data[-1].price if flow_data else 0
            
            return price_levels
            
        except Exception as e:
            logger.error(f"Narx sathlarini aniqlashda xato: {e}")
            return {}
    
    async def _generate_signal(self, metrics: FlowMetrics, patterns: List[FlowPattern], 
                             large_orders: List[OrderFlowData]) -> Tuple[FlowSignal, float]:
        """Order Flow signali yaratish"""
        try:
            signal_score = 0
            total_weight = 0
            
            # Volume imbalance ta'siri
            if abs(metrics.volume_imbalance) > 0.1:
                weight = 0.3
                score = metrics.volume_imbalance * 100
                signal_score += score * weight
                total_weight += weight
            
            # Pattern ta'siri
            for pattern in patterns:
                weight = 0.2
                score = pattern.strength * 100
                if pattern.direction == "BEARISH":
                    score *= -1
                signal_score += score * weight
                total_weight += weight
            
            # Whale activity ta'siri
            if metrics.whale_activity > 0.05:
                weight = 0.25
                # Whale orderlarning yo'nalishini aniqlash
                whale_orders = [order for order in large_orders 
                               if order.volume * order.price > self.whale_threshold]
                if whale_orders:
                    buy_whales = sum(1 for order in whale_orders if order.side == 'buy')
                    sell_whales = sum(1 for order in whale_orders if order.side == 'sell')
                    
                    if buy_whales > sell_whales:
                        score = metrics.whale_activity * 100
                    else:
                        score = -metrics.whale_activity * 100
                    
                    signal_score += score * weight
                    total_weight += weight
            
            # Price impact ta'siri
            if abs(metrics.price_impact) > 0.001:
                weight = 0.15
                score = metrics.price_impact * 10000
                signal_score += score * weight
                total_weight += weight
            
            # Large order ratio ta'siri
            if metrics.large_buy_orders > 0 or metrics.large_sell_orders > 0:
                weight = 0.1
                large_ratio = (metrics.large_buy_orders - metrics.large_sell_orders) / max(1, metrics.large_buy_orders + metrics.large_sell_orders)
                score = large_ratio * 100
                signal_score += score * weight
                total_weight += weight
            
            # Normalizatsiya
            if total_weight > 0:
                final_score = signal_score / total_weight
            else:
                final_score = 0
            
            # Signal va confidence aniqlash
            confidence = min(abs(final_score), 95)
            
            if final_score > 60:
                signal = FlowSignal.STRONG_BUY
            elif final_score > 20:
                signal = FlowSignal.BUY
            elif final_score < -60:
                signal = FlowSignal.STRONG_SELL
            elif final_score < -20:
                signal = FlowSignal.SELL
            else:
                signal = FlowSignal.NEUTRAL
            
            logger.info(f"Signal yaratildi: {signal.value} (score: {final_score:.2f}, confidence: {confidence:.2f}%)")
            return signal, confidence
            
        except Exception as e:
            logger.error(f"Signal yaratishda xato: {e}")
            return FlowSignal.NEUTRAL, 0.0
    
    async def _analyze_market_sentiment(self, metrics: FlowMetrics, patterns: List[FlowPattern]) -> str:
        """Market sentimentini tahlil qilish"""
        try:
            sentiment_score = 0
            
            # Volume imbalance sentiment
            if abs(metrics.volume_imbalance) > 0.3:
                if metrics.volume_imbalance > 0:
                    sentiment_score += 2
                else:
                    sentiment_score -= 2
            
            # Pattern sentiment
            for pattern in patterns:
                if pattern.direction == "BULLISH":
                    sentiment_score += 1
                elif pattern.direction == "BEARISH":
                    sentiment_score -= 1
            
            # Whale activity sentiment
            if metrics.whale_activity > 0.1:
                if metrics.buy_volume > metrics.sell_volume:
                    sentiment_score += 1
                else:
                    sentiment_score -= 1
            
            # Sentiment aniqlash
            if sentiment_score >= 3:
                return "VERY_BULLISH"
            elif sentiment_score >= 1:
                return "BULLISH"
            elif sentiment_score <= -3:
                return "VERY_BEARISH"
            elif sentiment_score <= -1:
                return "BEARISH"
            else:
                return "NEUTRAL"
                
        except Exception as e:
            logger.error(f"Market sentiment tahlilida xato: {e}")
            return "NEUTRAL"
    
    async def _generate_recommendation(self, signal: FlowSignal, confidence: float, 
                                     metrics: FlowMetrics) -> str:
        """Savdo tavsiyasi yaratish"""
        try:
            if confidence < 50:
                return "Signalni kuzatib turing, ishonch darajasi past"
            
            if signal == FlowSignal.STRONG_BUY:
                return f"Kuchli BUY signal - Volume imbalance: {metrics.volume_imbalance:.2f}, Whale activity: {metrics.whale_activity:.2f}"
            elif signal == FlowSignal.BUY:
                return f"BUY signal - Buyerlar ustun, volume imbalance: {metrics.volume_imbalance:.2f}"
            elif signal == FlowSignal.STRONG_SELL:
                return f"Kuchli SELL signal - Volume imbalance: {metrics.volume_imbalance:.2f}, Whale activity: {metrics.whale_activity:.2f}"
            elif signal == FlowSignal.SELL:
                return f"SELL signal - Sellerlar ustun, volume imbalance: {metrics.volume_imbalance:.2f}"
            else:
                return "Neutral holat - Kuchli signal yo'q"
                
        except Exception as e:
            logger.error(f"Tavsiya yaratishda xato: {e}")
            return "Tavsiya yaratishda xato"
    
    async def get_realtime_flow_metrics(self, symbol: str) -> Dict[str, Any]:
        """Real-time Order Flow metriklari"""
        try:
            # So'nggi orderlarni filter qilish
            recent_orders = [order for order in self.order_history 
                           if order.symbol == symbol and 
                           order.timestamp > datetime.now() - timedelta(minutes=5)]
            
            if not recent_orders:
                return {"error": "So'nggi orderlar topilmadi"}
            
            metrics = await self._calculate_metrics(recent_orders)
            
            return {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "buy_sell_ratio": metrics.buy_volume / metrics.sell_volume if metrics.sell_volume > 0 else float('inf'),
                "volume_imbalance": metrics.volume_imbalance,
                "whale_activity": metrics.whale_activity,
                "large_orders": metrics.large_buy_orders + metrics.large_sell_orders,
                "price_impact": metrics.price_impact,
                "total_orders": len(recent_orders)
            }
            
        except Exception as e:
            logger.error(f"Real-time metrikalar olishda xato: {e}")
            return {"error": str(e)}
    
    def add_order_to_history(self, order: OrderFlowData) -> None:
        """Order tarixiga qo'shish"""
        try:
            self.order_history.append(order)
            
            # Eski orderlarni tozalash
            cutoff_time = datetime.now() - timedelta(hours=1)
            self.order_history = [order for order in self.order_history 
                                if order.timestamp > cutoff_time]
            
        except Exception as e:
            logger.error(f"Order tarixiga qo'shishda xato: {e}")
    
    def get_order_history(self, symbol: str = None, limit: int = 100) -> List[OrderFlowData]:
        """Order tarixini olish"""
        try:
            if symbol:
                filtered_orders = [order for order in self.order_history if order.symbol == symbol]
            else:
                filtered_orders = self.order_history
            
            return filtered_orders[-limit:]
            
        except Exception as e:
            logger.error(f"Order tarixini olishda xato: {e}")
            return []

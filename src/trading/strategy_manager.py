import asyncio
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import ta
from utils.logger import get_logger
from utils.error_handler import handle_processing_error
from config.config import ConfigManager
from risk_management.risk_calculator import RiskCalculator
from risk_management.position_sizer import PositionSizer
from database.db_manager import DatabaseManager

logger = get_logger(__name__)

class StrategyType(Enum):
    """Strategiya turlari"""
    ORDERFLOW_MOMENTUM = "orderflow_momentum"
    SENTIMENT_REVERSAL = "sentiment_reversal"
    COMBINED_SIGNAL = "combined_signal"
    BREAKOUT_SCALP = "breakout_scalp"
    MEAN_REVERSION = "mean_reversion"

class SignalStrength(Enum):
    """Signal kuchi"""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"

class TradeDirection(Enum):
    """Savdo yo'nalishi"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"

@dataclass
class StrategyParameters:
    """Strategiya parametrlari"""
    name: str
    type: StrategyType
    risk_per_trade: float = 0.02
    max_daily_risk: float = 0.05
    min_confidence: float = 0.75
    max_lot_size: float = 0.5
    stop_loss_pips: int = 20
    take_profit_pips: int = 40
    trailing_stop: bool = True
    max_trades_per_day: int = 3
    active_hours: Tuple[int, int] = (7, 19)  # UZB vaqti
    symbols: List[str] = None
    
    def __post_init__(self):
        if self.symbols is None:
            self.symbols = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"]

@dataclass
class TradingSignal:
    """Trading signal modeli"""
    id: str
    timestamp: datetime
    symbol: str
    direction: TradeDirection
    entry_price: float
    stop_loss: float
    take_profit: float
    lot_size: float
    confidence: float
    signal_strength: SignalStrength
    strategy_name: str
    reason: str
    risk_percent: float
    reward_risk_ratio: float
    valid_until: datetime
    market_conditions: Dict[str, Any]

@dataclass
class StrategyPerformance:
    """Strategiya performance metriklari"""
    strategy_name: str
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_profit: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    total_pnl: float = 0.0
    sharpe_ratio: float = 0.0
    last_updated: datetime = None

class StrategyManager:
    """Trading strategiyalarni boshqarish class"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        self.risk_calculator = RiskCalculator()
        self.position_sizer = PositionSizer()
        self.db_manager = DatabaseManager()
        self.strategies: Dict[str, StrategyParameters] = {}
        self.performance_metrics: Dict[str, StrategyPerformance] = {}
        self.active_signals: Dict[str, TradingSignal] = {}
        
        # Propshot qoidalari
        self.propshot_limits = {
            'max_daily_loss': 0.025,  # 2.5% kunlik maksimal zarar
            'max_total_loss': 0.05,   # 5% umumiy maksimal zarar
            'max_lot_size': 0.5,      # 0.5 lot maksimal hajm
            'max_daily_trades': 3,    # Kuniga 3 ta savdo
            'max_drawdown': 0.03,     # 3% maksimal drawdown
            'risk_per_trade': 0.02    # Savdo uchun 2% risk
        }
        
        self.daily_stats = {
            'trades_count': 0,
            'daily_pnl': 0.0,
            'max_drawdown': 0.0,
            'last_reset': datetime.now().date()
        }
        
        logger.info("StrategyManager ishga tushirildi")
    
    async def initialize(self) -> None:
        """Strategiya manager ni ishga tushirish"""
        try:
            await self.load_strategies()
            await self.load_performance_metrics()
            await self.reset_daily_stats_if_needed()
            logger.info("StrategyManager muvaffaqiyatli ishga tushdi")
        except Exception as e:
            logger.error(f"Strategiya optimizatsiya qilishda xato: {e}")
    
    async def get_active_signals(self) -> Dict[str, TradingSignal]:
        """Aktiv signallarni olish"""
        try:
            # Muddati o'tgan signallarni o'chirish
            current_time = datetime.now()
            expired_signals = []
            
            for signal_id, signal in self.active_signals.items():
                if signal.valid_until < current_time:
                    expired_signals.append(signal_id)
            
            for signal_id in expired_signals:
                del self.active_signals[signal_id]
                logger.info(f"Muddati o'tgan signal o'chirildi: {signal_id}")
            
            return self.active_signals.copy()
            
        except Exception as e:
            logger.error(f"Aktiv signallarni olishda xato: {e}")
            return {}
    
    async def cancel_signal(self, signal_id: str) -> bool:
        """Signalni bekor qilish"""
        try:
            if signal_id in self.active_signals:
                del self.active_signals[signal_id]
                await self.db_manager.cancel_signal(signal_id)
                logger.info(f"Signal bekor qilindi: {signal_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Signal bekor qilishda xato: {e}")
            return False
    
    async def update_strategy_parameters(self, strategy_name: str, 
                                       new_params: Dict) -> bool:
        """Strategiya parametrlarini yangilash"""
        try:
            if strategy_name not in self.strategies:
                logger.error(f"Strategiya topilmadi: {strategy_name}")
                return False
            
            strategy = self.strategies[strategy_name]
            
            # Parametrlarni yangilash
            for key, value in new_params.items():
                if hasattr(strategy, key):
                    setattr(strategy, key, value)
                    logger.info(f"Strategiya parametri yangilandi: {key} = {value}")
            
            # Propshot limitlarini tekshirish
            if hasattr(strategy, 'risk_per_trade'):
                strategy.risk_per_trade = min(
                    strategy.risk_per_trade, 
                    self.propshot_limits['risk_per_trade']
                )
            
            if hasattr(strategy, 'max_lot_size'):
                strategy.max_lot_size = min(
                    strategy.max_lot_size,
                    self.propshot_limits['max_lot_size']
                )
            
            # Saqlash
            await self.save_strategies()
            
            logger.info(f"Strategiya parametrlari yangilandi: {strategy_name}")
            return True
            
        except Exception as e:
            logger.error(f"Strategiya parametrlarini yangilashda xato: {e}")
            return False
    
    async def get_daily_stats(self) -> Dict[str, Any]:
        """Kunlik statistikalarni olish"""
        try:
            await self.reset_daily_stats_if_needed()
            
            stats = self.daily_stats.copy()
            
            # Qo'shimcha hisoblashlar
            stats['remaining_trades'] = max(
                0, 
                self.propshot_limits['max_daily_trades'] - stats['trades_count']
            )
            
            stats['remaining_risk'] = max(
                0,
                self.propshot_limits['max_daily_loss'] - abs(stats['daily_pnl'])
            )
            
            stats['can_trade'] = await self.check_propshot_limits()
            
            return stats
            
        except Exception as e:
            logger.error(f"Kunlik statistikalarni olishda xato: {e}")
            return {}
    
    async def validate_signal_quality(self, signal: TradingSignal) -> Tuple[bool, str]:
        """Signal sifatini tekshirish"""
        try:
            # Asosiy tekshiruvlar
            if signal.confidence < 0.5:
                return False, "Confidence juda past"
            
            if signal.lot_size <= 0:
                return False, "Lot size noto'g'ri"
            
            if signal.lot_size > self.propshot_limits['max_lot_size']:
                return False, f"Lot size juda katta: {signal.lot_size}"
            
            # Risk/Reward ratio tekshirish
            if signal.reward_risk_ratio < 1.0:
                return False, "Risk/Reward ratio juda past"
            
            # Market conditions tekshirish
            if signal.market_conditions.get('volatility') == 'high':
                if signal.confidence < 0.8:
                    return False, "Yuqori volatiliteda confidence kam"
            
            # Vaqt tekshirish
            if signal.valid_until < datetime.now():
                return False, "Signal muddati o'tgan"
            
            # Symbol tekshirish
            if signal.symbol not in self.strategies[signal.strategy_name].symbols:
                return False, f"Symbol ruxsat etilmagan: {signal.symbol}"
            
            return True, "Signal sifatli"
            
        except Exception as e:
            logger.error(f"Signal sifatini tekshirishda xato: {e}")
            return False, "Tekshirish xatosi"
    
    async def get_strategy_summary(self) -> Dict[str, Any]:
        """Strategiyalar xulosasi"""
        try:
            summary = {
                'total_strategies': len(self.strategies),
                'active_signals': len(self.active_signals),
                'daily_stats': await self.get_daily_stats(),
                'performances': {},
                'propshot_limits': self.propshot_limits
            }
            
            # Har strategiya uchun qisqacha ma'lumot
            for strategy_name, strategy in self.strategies.items():
                perf = self.performance_metrics.get(strategy_name)
                
                strategy_summary = {
                    'type': strategy.type.value,
                    'risk_per_trade': strategy.risk_per_trade,
                    'min_confidence': strategy.min_confidence,
                    'active': strategy_name in [s.strategy_name for s in self.active_signals.values()]
                }
                
                if perf:
                    strategy_summary.update({
                        'total_trades': perf.total_trades,
                        'win_rate': perf.win_rate,
                        'profit_factor': perf.profit_factor,
                        'total_pnl': perf.total_pnl
                    })
                
                summary['performances'][strategy_name] = strategy_summary
            
            return summary
            
        except Exception as e:
            logger.error(f"Strategiyalar xulosasini olishda xato: {e}")
            return {}
    
    async def emergency_stop(self) -> None:
        """Favqulodda to'xtatish"""
        try:
            # Barcha aktiv signallarni bekor qilish
            signal_ids = list(self.active_signals.keys())
            for signal_id in signal_ids:
                await self.cancel_signal(signal_id)
            
            # Kunlik limitlarni zero qilish
            self.daily_stats['trades_count'] = self.propshot_limits['max_daily_trades']
            self.daily_stats['daily_pnl'] = -self.propshot_limits['max_daily_loss']
            
            logger.critical("FAVQULODDA TO'XTATISH AMALGA OSHIRILDI!")
            
        except Exception as e:
            logger.error(f"Favqulodda to'xtatishda xato: {e}")
    
    async def cleanup(self) -> None:
        """Resurslarni tozalash"""
        try:
            # Barcha aktiv signallarni saqlash
            for signal in self.active_signals.values():
                await self.db_manager.save_signal(asdict(signal))
            
            # Performance metriklari saqlash
            for perf in self.performance_metrics.values():
                await self.db_manager.save_performance_metrics(asdict(perf))
            
            # Strategiyalar saqlash
            await self.save_strategies()
            
            logger.info("StrategyManager tozalandi")
            
        except Exception as e:
            logger.error(f"StrategyManager tozalashda xato: {e}")
    
    def __str__(self) -> str:
        """String representation"""
        return f"StrategyManager(strategies={len(self.strategies)}, active_signals={len(self.active_signals)})"
    
    def __repr__(self) -> str:
        """Representation"""
        return self.__str__()

# Strategiya factory function
async def create_strategy_manager(config_manager: ConfigManager) -> StrategyManager:
    """StrategyManager yaratish va ishga tushirish"""
    try:
        manager = StrategyManager(config_manager)
        await manager.initialize()
        return manager
    except Exception as e:
        logger.error(f"StrategyManager yaratishda xato: {e}")
        raise

# Test funksiya
async def test_strategy_manager():
    """StrategyManager test qilish"""
    try:
        from config.config import ConfigManager
        
        config = ConfigManager()
        manager = await create_strategy_manager(config)
        
        # Test market data
        market_data = {
            'ohlcv': [
                {'open': 1.1000, 'high': 1.1050, 'low': 1.0950, 'close': 1.1020, 'volume': 1000},
                {'open': 1.1020, 'high': 1.1080, 'low': 1.1010, 'close': 1.1060, 'volume': 1200},
                {'open': 1.1060, 'high': 1.1100, 'low': 1.1040, 'close': 1.1080, 'volume': 1500}
            ]
        }
        
        # Test orderflow data
        orderflow_data = {
            'buy_volume': 800,
            'sell_volume': 400,
            'net_flow': 400
        }
        
        # Test sentiment data
        sentiment_data = {
            'sentiment_score': 0.85,
            'confidence': 0.9
        }
        
        # Signal yaratish
        signal = await manager.generate_signal(
            'EURUSD', 'orderflow_momentum', 
            market_data, orderflow_data, sentiment_data
        )
        
        if signal:
            print(f"Test signal yaratildi: {signal.id}")
            print(f"Direction: {signal.direction.value}")
            print(f"Confidence: {signal.confidence:.2f}")
            print(f"Entry: {signal.entry_price:.5f}")
            print(f"SL: {signal.stop_loss:.5f}")
            print(f"TP: {signal.take_profit:.5f}")
        else:
            print("Test signal yaratilmadi")
        
        # Performance summary
        summary = await manager.get_strategy_summary()
        print(f"Strategiyalar soni: {summary['total_strategies']}")
        print(f"Aktiv signallar: {summary['active_signals']}")
        
        # Cleanup
        await manager.cleanup()
        
        logger.info("StrategyManager test muvaffaqiyatli")
        
    except Exception as e:
        logger.error(f"StrategyManager test xatosi: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(test_strategy_manager())
            logger.error(f"StrategyManager ishga tushishda xato: {e}")
            raise
    
    async def load_strategies(self) -> None:
        """Strategiyalarni yuklash"""
        try:
            # Default strategiyalar
            default_strategies = {
                'orderflow_momentum': StrategyParameters(
                    name='OrderFlow Momentum',
                    type=StrategyType.ORDERFLOW_MOMENTUM,
                    risk_per_trade=0.015,
                    min_confidence=0.80,
                    stop_loss_pips=15,
                    take_profit_pips=30
                ),
                'sentiment_reversal': StrategyParameters(
                    name='Sentiment Reversal',
                    type=StrategyType.SENTIMENT_REVERSAL,
                    risk_per_trade=0.020,
                    min_confidence=0.75,
                    stop_loss_pips=25,
                    take_profit_pips=50
                ),
                'combined_signal': StrategyParameters(
                    name='Combined Signal',
                    type=StrategyType.COMBINED_SIGNAL,
                    risk_per_trade=0.025,
                    min_confidence=0.85,
                    stop_loss_pips=20,
                    take_profit_pips=40
                )
            }
            
            # Database dan strategiyalarni yuklash
            saved_strategies = await self.db_manager.get_strategies()
            
            for strategy_data in saved_strategies:
                strategy = StrategyParameters(**strategy_data)
                self.strategies[strategy.name] = strategy
            
            # Agar database bo'sh bo'lsa, default strategiyalarni yuklash
            if not self.strategies:
                self.strategies = default_strategies
                await self.save_strategies()
            
            logger.info(f"{len(self.strategies)} ta strategiya yuklandi")
            
        except Exception as e:
            logger.error(f"Strategiyalar yuklashda xato: {e}")
            raise
    
    async def save_strategies(self) -> None:
        """Strategiyalarni saqlash"""
        try:
            strategies_data = []
            for strategy in self.strategies.values():
                strategies_data.append(asdict(strategy))
            
            await self.db_manager.save_strategies(strategies_data)
            logger.info("Strategiyalar saqlandi")
            
        except Exception as e:
            logger.error(f"Strategiyalar saqlashda xato: {e}")
            raise
    
    async def load_performance_metrics(self) -> None:
        """Performance metriklari yuklash"""
        try:
            performance_data = await self.db_manager.get_performance_metrics()
            
            for perf_data in performance_data:
                performance = StrategyPerformance(**perf_data)
                self.performance_metrics[performance.strategy_name] = performance
            
            logger.info(f"{len(self.performance_metrics)} ta performance metriki yuklandi")
            
        except Exception as e:
            logger.error(f"Performance metriklari yuklashda xato: {e}")
            raise
    
    async def reset_daily_stats_if_needed(self) -> None:
        """Kun boshida statistikalarni reset qilish"""
        try:
            today = datetime.now().date()
            
            if self.daily_stats['last_reset'] != today:
                self.daily_stats = {
                    'trades_count': 0,
                    'daily_pnl': 0.0,
                    'max_drawdown': 0.0,
                    'last_reset': today
                }
                logger.info("Kunlik statistikalar reset qilindi")
            
        except Exception as e:
            logger.error(f"Kunlik statistikalar reset qilishda xato: {e}")
    
    async def analyze_market_conditions(self, symbol: str, 
                                      market_data: Dict) -> Dict[str, Any]:
        """Bozor sharoitlarini tahlil qilish"""
        try:
            if not market_data or 'ohlcv' not in market_data:
                return {'trend': 'unknown', 'volatility': 'normal', 'volume': 'normal'}
            
            df = pd.DataFrame(market_data['ohlcv'])
            
            # Trend tahlili
            sma_20 = ta.trend.sma_indicator(df['close'], window=20)
            sma_50 = ta.trend.sma_indicator(df['close'], window=50)
            
            if sma_20.iloc[-1] > sma_50.iloc[-1]:
                trend = 'bullish'
            elif sma_20.iloc[-1] < sma_50.iloc[-1]:
                trend = 'bearish'
            else:
                trend = 'sideways'
            
            # Volatilite tahlili
            atr = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
            current_atr = atr.iloc[-1]
            avg_atr = atr.mean()
            
            if current_atr > avg_atr * 1.5:
                volatility = 'high'
            elif current_atr < avg_atr * 0.7:
                volatility = 'low'
            else:
                volatility = 'normal'
            
            # Volume tahlili
            if 'volume' in df.columns:
                current_volume = df['volume'].iloc[-1]
                avg_volume = df['volume'].mean()
                
                if current_volume > avg_volume * 1.5:
                    volume = 'high'
                elif current_volume < avg_volume * 0.7:
                    volume = 'low'
                else:
                    volume = 'normal'
            else:
                volume = 'normal'
            
            conditions = {
                'trend': trend,
                'volatility': volatility,
                'volume': volume,
                'atr': current_atr,
                'price': df['close'].iloc[-1],
                'timestamp': datetime.now()
            }
            
            return conditions
            
        except Exception as e:
            logger.error(f"Bozor sharoitlarini tahlil qilishda xato: {e}")
            return {'trend': 'unknown', 'volatility': 'normal', 'volume': 'normal'}
    
    async def generate_signal(self, symbol: str, strategy_name: str,
                            market_data: Dict, orderflow_data: Dict,
                            sentiment_data: Dict) -> Optional[TradingSignal]:
        """Signal generatsiya qilish"""
        try:
            # Propshot limitlarini tekshirish
            if not await self.check_propshot_limits():
                logger.warning("Propshot limitlari oshib ketdi - signal yaratilmadi")
                return None
            
            if strategy_name not in self.strategies:
                logger.error(f"Strategiya topilmadi: {strategy_name}")
                return None
            
            strategy = self.strategies[strategy_name]
            
            # Bozor sharoitlarini tahlil qilish
            market_conditions = await self.analyze_market_conditions(symbol, market_data)
            
            # Strategiya turiga qarab signal yaratish
            signal = None
            
            if strategy.type == StrategyType.ORDERFLOW_MOMENTUM:
                signal = await self._generate_orderflow_momentum_signal(
                    symbol, strategy, market_data, orderflow_data, market_conditions
                )
            elif strategy.type == StrategyType.SENTIMENT_REVERSAL:
                signal = await self._generate_sentiment_reversal_signal(
                    symbol, strategy, market_data, sentiment_data, market_conditions
                )
            elif strategy.type == StrategyType.COMBINED_SIGNAL:
                signal = await self._generate_combined_signal(
                    symbol, strategy, market_data, orderflow_data, 
                    sentiment_data, market_conditions
                )
            
            if signal and signal.confidence >= strategy.min_confidence:
                # Signal ID yaratish
                signal.id = f"{symbol}_{strategy_name}_{int(datetime.now().timestamp())}"
                
                # Signal saqlash
                self.active_signals[signal.id] = signal
                
                # Database ga saqlash
                await self.db_manager.save_signal(asdict(signal))
                
                logger.info(f"Signal yaratildi: {signal.id}, Confidence: {signal.confidence:.2f}")
                return signal
            
            return None
            
        except Exception as e:
            logger.error(f"Signal generatsiya qilishda xato: {e}")
            return None
    
    async def _generate_orderflow_momentum_signal(self, symbol: str, 
                                                strategy: StrategyParameters,
                                                market_data: Dict,
                                                orderflow_data: Dict,
                                                market_conditions: Dict) -> Optional[TradingSignal]:
        """OrderFlow Momentum signal yaratish"""
        try:
            if not orderflow_data or 'buy_volume' not in orderflow_data:
                return None
            
            buy_volume = orderflow_data.get('buy_volume', 0)
            sell_volume = orderflow_data.get('sell_volume', 0)
            total_volume = buy_volume + sell_volume
            
            if total_volume == 0:
                return None
            
            # Volume imbalance hisoblash
            volume_imbalance = (buy_volume - sell_volume) / total_volume
            
            # Momentum kuchi
            momentum_strength = abs(volume_imbalance)
            
            # Signal yo'nalishi
            if volume_imbalance > 0.3:  # 30% dan ko'p buy volume
                direction = TradeDirection.BUY
                confidence = min(0.95, 0.6 + momentum_strength * 0.4)
            elif volume_imbalance < -0.3:  # 30% dan ko'p sell volume
                direction = TradeDirection.SELL
                confidence = min(0.95, 0.6 + momentum_strength * 0.4)
            else:
                return None
            
            # Current price
            current_price = market_conditions.get('price', 0)
            if current_price == 0:
                return None
            
            # Stop loss va take profit hisoblash
            pip_value = 0.0001 if 'JPY' not in symbol else 0.01
            
            if direction == TradeDirection.BUY:
                stop_loss = current_price - (strategy.stop_loss_pips * pip_value)
                take_profit = current_price + (strategy.take_profit_pips * pip_value)
            else:
                stop_loss = current_price + (strategy.stop_loss_pips * pip_value)
                take_profit = current_price - (strategy.take_profit_pips * pip_value)
            
            # Position size hisoblash
            risk_amount = strategy.risk_per_trade
            lot_size = await self.position_sizer.calculate_position_size(
                symbol, current_price, stop_loss, risk_amount
            )
            
            # Propshot limitlarini tekshirish
            lot_size = min(lot_size, self.propshot_limits['max_lot_size'])
            
            # Reward/Risk ratio
            risk_pips = strategy.stop_loss_pips
            reward_pips = strategy.take_profit_pips
            reward_risk_ratio = reward_pips / risk_pips if risk_pips > 0 else 0
            
            # Signal strength
            if momentum_strength > 0.7:
                signal_strength = SignalStrength.VERY_STRONG
            elif momentum_strength > 0.5:
                signal_strength = SignalStrength.STRONG
            elif momentum_strength > 0.3:
                signal_strength = SignalStrength.MODERATE
            else:
                signal_strength = SignalStrength.WEAK
            
            # Reason yaratish
            reason = f"OrderFlow Momentum: {volume_imbalance:.2f} imbalance, {momentum_strength:.2f} kuch"
            
            # Signal yaratish
            signal = TradingSignal(
                id="",  # ID keyinroq qo'yiladi
                timestamp=datetime.now(),
                symbol=symbol,
                direction=direction,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                lot_size=lot_size,
                confidence=confidence,
                signal_strength=signal_strength,
                strategy_name=strategy.name,
                reason=reason,
                risk_percent=risk_amount * 100,
                reward_risk_ratio=reward_risk_ratio,
                valid_until=datetime.now() + timedelta(minutes=30),
                market_conditions=market_conditions
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"OrderFlow Momentum signal yaratishda xato: {e}")
            return None
    
    async def _generate_sentiment_reversal_signal(self, symbol: str,
                                                strategy: StrategyParameters,
                                                market_data: Dict,
                                                sentiment_data: Dict,
                                                market_conditions: Dict) -> Optional[TradingSignal]:
        """Sentiment Reversal signal yaratish"""
        try:
            if not sentiment_data or 'sentiment_score' not in sentiment_data:
                return None
            
            sentiment_score = sentiment_data.get('sentiment_score', 0)
            sentiment_confidence = sentiment_data.get('confidence', 0)
            
            # Extreme sentiment detection
            if sentiment_score > 0.8:  # Juda bullish - sell signal
                direction = TradeDirection.SELL
                confidence = min(0.95, 0.6 + (sentiment_score - 0.8) * 2)
            elif sentiment_score < -0.8:  # Juda bearish - buy signal
                direction = TradeDirection.BUY
                confidence = min(0.95, 0.6 + abs(sentiment_score + 0.8) * 2)
            else:
                return None
            
            # Sentiment confidence ni hisobga olish
            confidence = confidence * sentiment_confidence
            
            # Current price
            current_price = market_conditions.get('price', 0)
            if current_price == 0:
                return None
            
            # Stop loss va take profit hisoblash
            pip_value = 0.0001 if 'JPY' not in symbol else 0.01
            
            if direction == TradeDirection.BUY:
                stop_loss = current_price - (strategy.stop_loss_pips * pip_value)
                take_profit = current_price + (strategy.take_profit_pips * pip_value)
            else:
                stop_loss = current_price + (strategy.stop_loss_pips * pip_value)
                take_profit = current_price - (strategy.take_profit_pips * pip_value)
            
            # Position size hisoblash
            risk_amount = strategy.risk_per_trade
            lot_size = await self.position_sizer.calculate_position_size(
                symbol, current_price, stop_loss, risk_amount
            )
            
            # Propshot limitlarini tekshirish
            lot_size = min(lot_size, self.propshot_limits['max_lot_size'])
            
            # Reward/Risk ratio
            reward_risk_ratio = strategy.take_profit_pips / strategy.stop_loss_pips
            
            # Signal strength
            extreme_level = abs(sentiment_score)
            if extreme_level > 0.9:
                signal_strength = SignalStrength.VERY_STRONG
            elif extreme_level > 0.85:
                signal_strength = SignalStrength.STRONG
            else:
                signal_strength = SignalStrength.MODERATE
            
            # Reason yaratish
            reason = f"Sentiment Reversal: {sentiment_score:.2f} haddan tashqari sentiment"
            
            # Signal yaratish
            signal = TradingSignal(
                id="",
                timestamp=datetime.now(),
                symbol=symbol,
                direction=direction,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                lot_size=lot_size,
                confidence=confidence,
                signal_strength=signal_strength,
                strategy_name=strategy.name,
                reason=reason,
                risk_percent=risk_amount * 100,
                reward_risk_ratio=reward_risk_ratio,
                valid_until=datetime.now() + timedelta(minutes=45),
                market_conditions=market_conditions
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Sentiment Reversal signal yaratishda xato: {e}")
            return None
    
    async def _generate_combined_signal(self, symbol: str,
                                      strategy: StrategyParameters,
                                      market_data: Dict,
                                      orderflow_data: Dict,
                                      sentiment_data: Dict,
                                      market_conditions: Dict) -> Optional[TradingSignal]:
        """Combined signal yaratish"""
        try:
            # OrderFlow signal
            orderflow_signal = await self._generate_orderflow_momentum_signal(
                symbol, strategy, market_data, orderflow_data, market_conditions
            )
            
            # Sentiment signal
            sentiment_signal = await self._generate_sentiment_reversal_signal(
                symbol, strategy, market_data, sentiment_data, market_conditions
            )
            
            # Ikkala signal ham bo'lishi kerak
            if not orderflow_signal or not sentiment_signal:
                return None
            
            # Signallar bir xil yo'nalishda bo'lishi kerak
            if orderflow_signal.direction != sentiment_signal.direction:
                return None
            
            # Combined confidence
            combined_confidence = (orderflow_signal.confidence + sentiment_signal.confidence) / 2
            combined_confidence = min(0.95, combined_confidence * 1.1)  # Bonus
            
            # Eng yaxshi parametrlarni olish
            lot_size = min(orderflow_signal.lot_size, sentiment_signal.lot_size)
            lot_size = min(lot_size, self.propshot_limits['max_lot_size'])
            
            # Signal strength
            signal_strength = SignalStrength.VERY_STRONG
            
            # Reason yaratish
            reason = f"Combined Signal: OrderFlow + Sentiment bir yo'nalishda"
            
            # Signal yaratish
            signal = TradingSignal(
                id="",
                timestamp=datetime.now(),
                symbol=symbol,
                direction=orderflow_signal.direction,
                entry_price=orderflow_signal.entry_price,
                stop_loss=orderflow_signal.stop_loss,
                take_profit=orderflow_signal.take_profit,
                lot_size=lot_size,
                confidence=combined_confidence,
                signal_strength=signal_strength,
                strategy_name=strategy.name,
                reason=reason,
                risk_percent=strategy.risk_per_trade * 100,
                reward_risk_ratio=orderflow_signal.reward_risk_ratio,
                valid_until=datetime.now() + timedelta(hours=1),
                market_conditions=market_conditions
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Combined signal yaratishda xato: {e}")
            return None
    
    async def check_propshot_limits(self) -> bool:
        """Propshot limitlarini tekshirish"""
        try:
            # Kunlik savdo sonini tekshirish
            if self.daily_stats['trades_count'] >= self.propshot_limits['max_daily_trades']:
                logger.warning(f"Kunlik savdo limiti oshib ketdi: {self.daily_stats['trades_count']}")
                return False
            
            # Kunlik zarar limitini tekshirish
            if self.daily_stats['daily_pnl'] <= -self.propshot_limits['max_daily_loss']:
                logger.warning(f"Kunlik zarar limiti oshib ketdi: {self.daily_stats['daily_pnl']}")
                return False
            
            # Drawdown limitini tekshirish
            if self.daily_stats['max_drawdown'] >= self.propshot_limits['max_drawdown']:
                logger.warning(f"Drawdown limiti oshib ketdi: {self.daily_stats['max_drawdown']}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Propshot limitlarini tekshirishda xato: {e}")
            return False
    
    async def update_performance(self, signal_id: str, pnl: float, 
                               is_winning: bool) -> None:
        """Performance metriklari yangilash"""
        try:
            if signal_id not in self.active_signals:
                return
            
            signal = self.active_signals[signal_id]
            strategy_name = signal.strategy_name
            
            # Performance metriklari olish yoki yaratish
            if strategy_name not in self.performance_metrics:
                self.performance_metrics[strategy_name] = StrategyPerformance(
                    strategy_name=strategy_name,
                    last_updated=datetime.now()
                )
            
            perf = self.performance_metrics[strategy_name]
            
            # Metriklari yangilash
            perf.total_trades += 1
            perf.total_pnl += pnl
            
            if is_winning:
                perf.winning_trades += 1
                if perf.winning_trades == 1:
                    perf.avg_profit = pnl
                else:
                    perf.avg_profit = (perf.avg_profit + pnl) / 2
            else:
                perf.losing_trades += 1
                if perf.losing_trades == 1:
                    perf.avg_loss = pnl
                else:
                    perf.avg_loss = (perf.avg_loss + pnl) / 2
            
            # Win rate hisoblash
            perf.win_rate = perf.winning_trades / perf.total_trades if perf.total_trades > 0 else 0
            
            # Profit factor hisoblash
            total_profit = perf.avg_profit * perf.winning_trades
            total_loss = abs(perf.avg_loss * perf.losing_trades)
            perf.profit_factor = total_profit / total_loss if total_loss > 0 else 0
            
            perf.last_updated = datetime.now()
            
            # Kunlik statistikalarni yangilash
            self.daily_stats['trades_count'] += 1
            self.daily_stats['daily_pnl'] += pnl
            
            # Drawdown hisoblash
            if pnl < 0:
                self.daily_stats['max_drawdown'] = max(
                    self.daily_stats['max_drawdown'], abs(pnl)
                )
            
            # Database ga saqlash
            await self.db_manager.save_performance_metrics(asdict(perf))
            
            # Active signals dan o'chirish
            del self.active_signals[signal_id]
            
            logger.info(f"Performance yangilandi: {strategy_name}, PnL: {pnl:.2f}")
            
        except Exception as e:
            logger.error(f"Performance yangilashda xato: {e}")
    
    async def get_strategy_performance(self, strategy_name: str) -> Optional[StrategyPerformance]:
        """Strategiya performance metriklari olish"""
        try:
            return self.performance_metrics.get(strategy_name)
        except Exception as e:
            logger.error(f"Performance metriklari olishda xato: {e}")
            return None
    
    async def get_all_performances(self) -> Dict[str, StrategyPerformance]:
        """Barcha strategiyalar performance metriklari"""
        try:
            return self.performance_metrics.copy()
        except Exception as e:
            logger.error(f"Barcha performance metriklari olishda xato: {e}")
            return {}
    
    async def optimize_strategy(self, strategy_name: str, 
                               backtest_results: Dict) -> None:
        """Strategiya optimizatsiya qilish"""
        try:
            if strategy_name not in self.strategies:
                return
            
            strategy = self.strategies[strategy_name]
            
            # Backtest natijalariga qarab parametrlarni optimizatsiya
            if backtest_results.get('win_rate', 0) < 0.6:
                # Win rate past bo'lsa, confidence threshold oshirish
                strategy.min_confidence = min(0.9, strategy.min_confidence + 0.05)
            
            if backtest_results.get('profit_factor', 0) < 1.5:
                # Profit factor past bo'lsa, risk/reward ratio yaxshilash
                strategy.take_profit_pips = int(strategy.take_profit_pips * 1.2)
            
            if backtest_results.get('max_drawdown', 0) > 0.05:
                # Drawdown ko'p bo'lsa, risk kamaytirish
                strategy.risk_per_trade = max(0.01, strategy.risk_per_trade * 0.8)
            
            # Strategiya saqlash
            await self.save_strategies()
            
            logger.info(f"Strategiya optimizatsiya qilindi: {strategy_name}")
            
        except Exception as

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
from pathlib import Path
import logging
from enum import Enum

# O'zbekcha: Utils import qilish
from utils.logger import get_logger
from utils.error_handler import handle_processing_error
from risk_management.risk_calculator import RiskCalculator
from risk_management.position_sizer import PositionSizer
from database.models import BacktestResult, Trade, Signal

logger = get_logger(__name__)

class OrderType(Enum):
    """Savdo buyruq turlari"""
    BUY = "BUY"
    SELL = "SELL"
    
class OrderStatus(Enum):
    """Buyruq holati"""
    PENDING = "PENDING"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"

@dataclass
class BacktestOrder:
    """Backtest buyruq modeli"""
    id: str
    symbol: str
    order_type: OrderType
    quantity: float
    price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    status: OrderStatus = OrderStatus.PENDING
    fill_price: Optional[float] = None
    fill_time: Optional[datetime] = None
    commission: float = 0.0
    
    def __post_init__(self):
        """Buyruq validatsiyasi"""
        if self.quantity <= 0:
            raise ValueError("Miqdor musbat bo'lishi kerak")
        if self.price <= 0:
            raise ValueError("Narx musbat bo'lishi kerak")

@dataclass
class BacktestPosition:
    """Backtest pozitsiya modeli"""
    symbol: str
    quantity: float
    entry_price: float
    entry_time: datetime
    order_type: OrderType
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    
    def update_pnl(self, current_price: float) -> None:
        """PnL yangilash"""
        self.current_price = current_price
        if self.order_type == OrderType.BUY:
            self.unrealized_pnl = (current_price - self.entry_price) * self.quantity
        else:
            self.unrealized_pnl = (self.entry_price - current_price) * self.quantity

@dataclass
class BacktestSettings:
    """Backtest sozlamalari"""
    start_date: datetime
    end_date: datetime
    initial_capital: float = 10000.0
    commission: float = 0.0001  # 0.01%
    slippage: float = 0.0001    # 0.01%
    max_risk_per_trade: float = 0.02  # 2%
    max_daily_loss: float = 0.05      # 5%
    max_open_positions: int = 5
    min_bars_between_trades: int = 1
    
    def __post_init__(self):
        """Sozlamalar validatsiyasi"""
        if self.start_date >= self.end_date:
            raise ValueError("Boshlanish sanasi tugash sanasidan oldin bo'lishi kerak")
        if self.initial_capital <= 0:
            raise ValueError("Boshlang'ich kapital musbat bo'lishi kerak")

@dataclass
class BacktestMetrics:
    """Backtest natijalar metrikalari"""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    profit_factor: float = 0.0
    average_win: float = 0.0
    average_loss: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    start_capital: float = 0.0
    end_capital: float = 0.0
    total_return: float = 0.0
    annual_return: float = 0.0
    volatility: float = 0.0
    
    def calculate_metrics(self, trades: List[Dict], daily_returns: List[float]) -> None:
        """Metrikalari hisoblash"""
        if not trades:
            return
            
        # O'zbekcha: Asosiy metrikalar
        self.total_trades = len(trades)
        winning_trades = [t for t in trades if t['pnl'] > 0]
        losing_trades = [t for t in trades if t['pnl'] < 0]
        
        self.winning_trades = len(winning_trades)
        self.losing_trades = len(losing_trades)
        self.win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0
        
        # PnL hisoblash
        self.total_pnl = sum(t['pnl'] for t in trades)
        
        # O'rtacha foyda va zarar
        if winning_trades:
            self.average_win = sum(t['pnl'] for t in winning_trades) / len(winning_trades)
        if losing_trades:
            self.average_loss = sum(t['pnl'] for t in losing_trades) / len(losing_trades)
        
        # Profit factor
        gross_profit = sum(t['pnl'] for t in winning_trades) if winning_trades else 0
        gross_loss = abs(sum(t['pnl'] for t in losing_trades)) if losing_trades else 0
        self.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Ketma-ket g'alabalar va mag'lubiyatlar
        self._calculate_consecutive_trades(trades)
        
        # Sharpe va Sortino ratios
        if daily_returns:
            self._calculate_risk_metrics(daily_returns)

    def _calculate_consecutive_trades(self, trades: List[Dict]) -> None:
        """Ketma-ket g'alabalar va mag'lubiyatlarni hisoblash"""
        current_wins = 0
        current_losses = 0
        max_wins = 0
        max_losses = 0
        
        for trade in trades:
            if trade['pnl'] > 0:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            elif trade['pnl'] < 0:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)
                
        self.max_consecutive_wins = max_wins
        self.max_consecutive_losses = max_losses

    def _calculate_risk_metrics(self, daily_returns: List[float]) -> None:
        """Risk metrikalari hisoblash"""
        if not daily_returns:
            return
            
        returns_array = np.array(daily_returns)
        
        # Sharpe ratio
        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array)
        self.sharpe_ratio = mean_return / std_return if std_return > 0 else 0
        
        # Sortino ratio
        negative_returns = returns_array[returns_array < 0]
        downside_deviation = np.std(negative_returns) if len(negative_returns) > 0 else 0
        self.sortino_ratio = mean_return / downside_deviation if downside_deviation > 0 else 0
        
        # Volatility
        self.volatility = std_return * np.sqrt(252)  # Annualized

class BacktestEngine:
    """Backtest tizimi asosiy klassi"""
    
    def __init__(self, settings: BacktestSettings):
        """Backtest engine initialize qilish"""
        self.settings = settings
        self.risk_calculator = RiskCalculator()
        self.position_sizer = PositionSizer()
        
        # O'zbekcha: Backtest holati
        self.current_capital = settings.initial_capital
        self.positions: Dict[str, BacktestPosition] = {}
        self.orders: List[BacktestOrder] = []
        self.trades: List[Dict] = []
        self.daily_returns: List[float] = []
        self.equity_curve: List[Dict] = []
        self.current_date: Optional[datetime] = None
        
        # Statistika
        self.total_commission = 0.0
        self.total_slippage = 0.0
        self.max_drawdown = 0.0
        self.peak_equity = settings.initial_capital
        
        logger.info("BacktestEngine ishga tushirildi")

    async def run_backtest(self, signals: List[Dict], market_data: pd.DataFrame) -> BacktestMetrics:
        """Asosiy backtest ishga tushirish"""
        try:
            logger.info("Backtest boshlandi")
            
            # Ma'lumotlarni validatsiya qilish
            self._validate_inputs(signals, market_data)
            
            # Backtest sikli
            for index, row in market_data.iterrows():
                self.current_date = row['timestamp']
                current_prices = self._extract_prices(row)
                
                # Joriy pozitsiyalarni yangilash
                await self._update_positions(current_prices)
                
                # Buyruqlarni bajarish
                await self._execute_orders(current_prices)
                
                # Signallarni qayta ishlash
                daily_signals = self._get_daily_signals(signals, self.current_date)
                for signal in daily_signals:
                    await self._process_signal(signal, current_prices)
                
                # Kunlik hisobotni yangilash
                await self._update_daily_stats(current_prices)
                
                # Risk monitoring
                await self._monitor_risk()
            
            # Natijalarni hisoblash
            metrics = self._calculate_final_metrics()
            
            logger.info(f"Backtest tugadi. Jami savdolar: {metrics.total_trades}")
            return metrics
            
        except Exception as e:
            logger.error(f"Backtest xatosi: {e}")
            raise

    def _validate_inputs(self, signals: List[Dict], market_data: pd.DataFrame) -> None:
        """Kirish ma'lumotlarini validatsiya qilish"""
        if not signals:
            raise ValueError("Signallar bo'sh")
        
        if market_data.empty:
            raise ValueError("Market ma'lumotlari bo'sh")
        
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in market_data.columns]
        
        if missing_columns:
            raise ValueError(f"Yetishmayotgan ustunlar: {missing_columns}")

    def _extract_prices(self, row: pd.Series) -> Dict[str, float]:
        """Narxlarni ajratib olish"""
        return {
            'open': row['open'],
            'high': row['high'],
            'low': row['low'],
            'close': row['close'],
            'volume': row['volume']
        }

    async def _update_positions(self, current_prices: Dict[str, float]) -> None:
        """Pozitsiyalarni yangilash"""
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                position.update_pnl(current_prices[symbol])
                
                # Stop loss va take profit tekshirish
                await self._check_exit_conditions(position, current_prices[symbol])

    async def _check_exit_conditions(self, position: BacktestPosition, current_price: float) -> None:
        """Chiqish shartlarini tekshirish"""
        should_exit = False
        exit_reason = ""
        
        # Stop loss tekshirish
        if position.stop_loss:
            if position.order_type == OrderType.BUY and current_price <= position.stop_loss:
                should_exit = True
                exit_reason = "Stop Loss"
            elif position.order_type == OrderType.SELL and current_price >= position.stop_loss:
                should_exit = True
                exit_reason = "Stop Loss"
        
        # Take profit tekshirish
        if position.take_profit:
            if position.order_type == OrderType.BUY and current_price >= position.take_profit:
                should_exit = True
                exit_reason = "Take Profit"
            elif position.order_type == OrderType.SELL and current_price <= position.take_profit:
                should_exit = True
                exit_reason = "Take Profit"
        
        if should_exit:
            await self._close_position(position, current_price, exit_reason)

    async def _close_position(self, position: BacktestPosition, exit_price: float, reason: str) -> None:
        """Pozitsiyani yopish"""
        try:
            # Commission va slippage hisoblash
            commission = exit_price * position.quantity * self.settings.commission
            slippage = exit_price * position.quantity * self.settings.slippage
            
            # PnL hisoblash
            if position.order_type == OrderType.BUY:
                pnl = (exit_price - position.entry_price) * position.quantity
            else:
                pnl = (position.entry_price - exit_price) * position.quantity
            
            # Xarajatlarni ayirish
            net_pnl = pnl - commission - slippage
            
            # Savdo yozuvi
            trade = {
                'symbol': position.symbol,
                'entry_time': position.entry_time,
                'exit_time': self.current_date,
                'entry_price': position.entry_price,
                'exit_price': exit_price,
                'quantity': position.quantity,
                'order_type': position.order_type.value,
                'pnl': net_pnl,
                'commission': commission,
                'slippage': slippage,
                'reason': reason
            }
            
            self.trades.append(trade)
            
            # Kapitalni yangilash
            self.current_capital += net_pnl
            self.total_commission += commission
            self.total_slippage += slippage
            
            # Pozitsiyani o'chirish
            del self.positions[position.symbol]
            
            logger.info(f"Pozitsiya yopildi: {position.symbol}, PnL: {net_pnl:.2f}, Sabab: {reason}")
            
        except Exception as e:
            logger.error(f"Pozitsiya yopishda xato: {e}")

    async def _execute_orders(self, current_prices: Dict[str, float]) -> None:
        """Buyruqlarni bajarish"""
        executed_orders = []
        
        for order in self.orders:
            if order.status == OrderStatus.PENDING:
                # Buyruq bajarilish shartlarini tekshirish
                if order.symbol in current_prices:
                    current_price = current_prices[order.symbol]
                    
                    # Market order - darhol bajarish
                    if self._should_execute_order(order, current_price):
                        await self._execute_order(order, current_price)
                        executed_orders.append(order)
        
        # Bajarilgan buyruqlarni o'chirish
        self.orders = [o for o in self.orders if o not in executed_orders]

    def _should_execute_order(self, order: BacktestOrder, current_price: float) -> bool:
        """Buyruq bajarilishi kerakligini tekshirish"""
        # Market order uchun - har doim bajarish
        return True

    async def _execute_order(self, order: BacktestOrder, current_price: float) -> None:
        """Buyruqni bajarish"""
        try:
            # Slippage qo'shish
            fill_price = current_price * (1 + self.settings.slippage)
            
            # Commission hisoblash
            commission = fill_price * order.quantity * self.settings.commission
            
            # Pozitsiya yaratish
            position = BacktestPosition(
                symbol=order.symbol,
                quantity=order.quantity,
                entry_price=fill_price,
                entry_time=self.current_date,
                order_type=order.order_type,
                stop_loss=order.stop_loss,
                take_profit=order.take_profit
            )
            
            self.positions[order.symbol] = position
            
            # Kapitalni yangilash
            self.current_capital -= commission
            self.total_commission += commission
            
            # Buyruq holatini yangilash
            order.status = OrderStatus.FILLED
            order.fill_price = fill_price
            order.fill_time = self.current_date
            order.commission = commission
            
            logger.info(f"Buyruq bajarildi: {order.symbol}, Narx: {fill_price:.4f}")
            
        except Exception as e:
            logger.error(f"Buyruq bajarishda xato: {e}")
            order.status = OrderStatus.REJECTED

    def _get_daily_signals(self, signals: List[Dict], current_date: datetime) -> List[Dict]:
        """Kunlik signallarni olish"""
        daily_signals = []
        
        for signal in signals:
            signal_date = signal.get('timestamp')
            if isinstance(signal_date, str):
                signal_date = datetime.fromisoformat(signal_date)
            
            if signal_date.date() == current_date.date():
                daily_signals.append(signal)
        
        return daily_signals

    async def _process_signal(self, signal: Dict, current_prices: Dict[str, float]) -> None:
        """Signalni qayta ishlash"""
        try:
            symbol = signal.get('symbol')
            action = signal.get('action', '').upper()
            confidence = signal.get('confidence', 0.0)
            
            # Signal validatsiyasi
            if not symbol or action not in ['BUY', 'SELL']:
                return
            
            if symbol not in current_prices:
                return
            
            # Minimum confidence tekshirish
            if confidence < 0.6:  # 60% dan kam ishonch
                return
            
            # Mavjud pozitsiyani tekshirish
            if symbol in self.positions:
                return  # Allaqachon pozitsiya mavjud
            
            # Maksimal pozitsiyalar soni
            if len(self.positions) >= self.settings.max_open_positions:
                return
            
            # Position size hisoblash
            current_price = current_prices[symbol]
            position_size = await self._calculate_position_size(
                symbol, current_price, signal
            )
            
            if position_size <= 0:
                return
            
            # Buyruq yaratish
            order = BacktestOrder(
                id=f"{symbol}_{self.current_date.timestamp()}",
                symbol=symbol,
                order_type=OrderType.BUY if action == 'BUY' else OrderType.SELL,
                quantity=position_size,
                price=current_price,
                stop_loss=signal.get('stop_loss'),
                take_profit=signal.get('take_profit')
            )
            
            self.orders.append(order)
            
        except Exception as e:
            logger.error(f"Signal qayta ishlashda xato: {e}")

    async def _calculate_position_size(self, symbol: str, price: float, signal: Dict) -> float:
        """Pozitsiya hajmini hisoblash"""
        try:
            # Risk percentage asosida
            risk_amount = self.current_capital * self.settings.max_risk_per_trade
            
            # Stop loss masofasi
            stop_loss = signal.get('stop_loss')
            if not stop_loss:
                # Default stop loss (2%)
                stop_loss = price * 0.98 if signal.get('action') == 'BUY' else price * 1.02
            
            # Risk per share
            risk_per_share = abs(price - stop_loss)
            
            if risk_per_share <= 0:
                return 0
            
            # Position size
            position_size = risk_amount / risk_per_share
            
            # Maksimal pozitsiya hajmi (kapitalning 10%)
            max_position_value = self.current_capital * 0.1
            max_position_size = max_position_value / price
            
            return min(position_size, max_position_size)
            
        except Exception as e:
            logger.error(f"Pozitsiya hajmi hisoblashda xato: {e}")
            return 0

    async def _update_daily_stats(self, current_prices: Dict[str, float]) -> None:
        """Kunlik statistikani yangilash"""
        try:
            # Joriy equity hisoblash
            current_equity = self.current_capital
            
            # Ochiq pozitsiyalar PnL
            for position in self.positions.values():
                if position.symbol in current_prices:
                    position.update_pnl(current_prices[position.symbol])
                    current_equity += position.unrealized_pnl
            
            # Equity curve yangilash
            self.equity_curve.append({
                'date': self.current_date,
                'equity': current_equity,
                'capital': self.current_capital,
                'unrealized_pnl': sum(p.unrealized_pnl for p in self.positions.values())
            })
            
            # Drawdown hisoblash
            if current_equity > self.peak_equity:
                self.peak_equity = current_equity
            
            drawdown = (self.peak_equity - current_equity) / self.peak_equity
            self.max_drawdown = max(self.max_drawdown, drawdown)
            
        except Exception as e:
            logger.error(f"Kunlik statistika yangilashda xato: {e}")

    async def _monitor_risk(self) -> None:
        """Risk monitoringi"""
        try:
            # Kunlik zarar tekshirish
            current_equity = self.current_capital + sum(p.unrealized_pnl for p in self.positions.values())
            daily_loss = (self.settings.initial_capital - current_equity) / self.settings.initial_capital
            
            if daily_loss > self.settings.max_daily_loss:
                logger.warning(f"Kunlik maksimal zarar oshib ketdi: {daily_loss:.2%}")
                # Barcha pozitsiyalarni yopish
                await self._close_all_positions("Risk limit reached")
                
        except Exception as e:
            logger.error(f"Risk monitoringda xato: {e}")

    async def _close_all_positions(self, reason: str) -> None:
        """Barcha pozitsiyalarni yopish"""
        positions_to_close = list(self.positions.values())
        
        for position in positions_to_close:
            # Joriy narx olish (close narxni ishlatish)
            current_price = position.current_price or position.entry_price
            await self._close_position(position, current_price, reason)

    def _calculate_final_metrics(self) -> BacktestMetrics:
        """Yakuniy metrikalari hisoblash"""
        metrics = BacktestMetrics()
        
        # Asosiy ma'lumotlar
        metrics.start_capital = self.settings.initial_capital
        metrics.end_capital = self.current_capital
        metrics.total_return = (self.current_capital - self.settings.initial_capital) / self.settings.initial_capital
        
        # Kunlik returnlar hisoblash
        daily_returns = []
        if len(self.equity_curve) > 1:
            for i in range(1, len(self.equity_curve)):
                prev_equity = self.equity_curve[i-1]['equity']
                curr_equity = self.equity_curve[i]['equity']
                daily_return = (curr_equity - prev_equity) / prev_equity
                daily_returns.append(daily_return)
        
        self.daily_returns = daily_returns
        
        # Metrikalari hisoblash
        metrics.calculate_metrics(self.trades, daily_returns)
        
        # Drawdown
        metrics.max_drawdown = self.max_drawdown
        
        # Yillik return
        if len(self.equity_curve) > 0:
            days = (self.settings.end_date - self.settings.start_date).days
            if days > 0:
                metrics.annual_return = (1 + metrics.total_return) ** (365 / days) - 1
        
        return metrics

    def save_results(self, metrics: BacktestMetrics, filepath: str) -> None:
        """Natijalarni saqlash"""
        try:
            results = {
                'settings': {
                    'start_date': self.settings.start_date.isoformat(),
                    'end_date': self.settings.end_date.isoformat(),
                    'initial_capital': self.settings.initial_capital,
                    'commission': self.settings.commission,
                    'slippage': self.settings.slippage,
                    'max_risk_per_trade': self.settings.max_risk_per_trade
                },
                'metrics': {
                    'total_trades': metrics.total_trades,
                    'win_rate': metrics.win_rate,
                    'total_pnl': metrics.total_pnl,
                    'total_return': metrics.total_return,
                    'annual_return': metrics.annual_return,
                    'max_drawdown': metrics.max_drawdown,
                    'sharpe_ratio': metrics.sharpe_ratio,
                    'profit_factor': metrics.profit_factor
                },
                'trades': self.trades,
                'equity_curve': [
                    {
                        'date': point['date'].isoformat(),
                        'equity': point['equity'],
                        'capital': point['capital'],
                        'unrealized_pnl': point['unrealized_pnl']
                    } for point in self.equity_curve
                ]
            }
            
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Backtest natijalar saqlandi: {filepath}")
            
        except Exception as e:
            logger.error(f"Natijalar saqlashda xato: {e}")

# O'zbekcha: Yordamchi funksiyalar
async def run_simple_backtest(
    signals: List[Dict],
    market_data: pd.DataFrame,
    initial_capital: float = 10000.0,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
) -> BacktestMetrics:
    """Oddiy backtest ishga tushirish"""
    
    if not start_date:
        start_date = market_data['timestamp'].min()
    if not end_date:
        end_date = market_data['timestamp'].max()
    
    settings = BacktestSettings(
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital
    )
    
    engine = BacktestEngine(settings)
    return await engine.run_backtest(signals, market_data)

# Test funksiyasi
async def test_backtest_engine():
    """Backtest engine test qilish"""
    # Test ma'lumotlari
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    
    # Fake market data
    market_data = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.uniform(100, 200, len(dates)),
        'high': np.random.uniform(150, 250, len(dates)),
        'low': np.random.uniform(50, 150, len(dates)),
        'close': np.random.uniform(100, 200, len(dates)),
        'volume': np.random.randint(1000, 10000, len(dates))
    })
    
    # Test signals
    signals = [
        {
            'timestamp': dates[10],
            'symbol': 'EURUSD',
            'action': 'BUY',
            'confidence': 0.8,
            'stop_loss': 95.0,
            'take_profit': 105.0
        },
        {
            'timestamp': dates[50],
            'symbol': 'EURUSD',
            'action': 'SELL',
            'confidence': 0.7,
            'stop_loss': 205.0,
            'take_profit': 195.0
        }
    ]
    
    # Backtest ishga tushirish
    metrics = await run_simple_backtest(signals, market_data)
    
    print(f"Backtest natijalar:")
    print(f"Jami savdolar: {metrics.total_trades}")
    print(f"G'alaba foizi: {metrics.win_rate:.2%}")
    print(f"Jami PnL: {metrics.total_pnl:.2f}")
    print(f"Maksimal drawdown: {metrics.max_drawdown:.2%}")
    
    return metrics

if __name__ == "__main__":
    # Test ishga tushirish
    asyncio.run(test_backtest_engine())

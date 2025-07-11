import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from decimal import Decimal
import pandas as pd
from utils.logger import get_logger
from utils.error_handler import handle_processing_error
from risk_management.risk_calculator import RiskCalculator
from risk_management.position_sizer import PositionSizer
from database.db_manager import DatabaseManager

logger = get_logger(__name__)

@dataclass
class Position:
    """Pozitsiya ma'lumotlari"""
    symbol: str
    side: str  # BUY/SELL
    entry_price: float
    quantity: float
    stop_loss: float
    take_profit: float
    entry_time: datetime
    status: str = "OPEN"  # OPEN/CLOSED/PARTIAL
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    commission: float = 0.0
    swap: float = 0.0
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    exit_reason: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Pozitsiyani dict formatiga o'tkazish"""
        return asdict(self)

@dataclass
class Portfolio:
    """Portfolio ma'lumotlari"""
    balance: float
    equity: float
    margin: float
    free_margin: float
    margin_level: float
    unrealized_pnl: float
    realized_pnl: float
    daily_pnl: float
    drawdown: float
    max_drawdown: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    last_update: datetime
    
    def to_dict(self) -> Dict:
        """Portfolioni dict formatiga o'tkazish"""
        return asdict(self)

@dataclass
class PortfolioStats:
    """Portfolio statistika"""
    total_return: float
    daily_return: float
    weekly_return: float
    monthly_return: float
    sharpe_ratio: float
    max_drawdown: float
    current_drawdown: float
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    consecutive_wins: int
    consecutive_losses: int
    
    def to_dict(self) -> Dict:
        """Statistikani dict formatiga o'tkazish"""
        return asdict(self)

class PortfolioManager:
    """Portfolio boshqaruv tizimi"""
    
    def __init__(self, db_manager: DatabaseManager):
        """Portfolio manager yaratish"""
        self.name = self.__class__.__name__
        self.db_manager = db_manager
        self.risk_calculator = RiskCalculator()
        self.position_sizer = PositionSizer()
        
        # Portfolio ma'lumotlari
        self.portfolio = Portfolio(
            balance=0.0,
            equity=0.0,
            margin=0.0,
            free_margin=0.0,
            margin_level=0.0,
            unrealized_pnl=0.0,
            realized_pnl=0.0,
            daily_pnl=0.0,
            drawdown=0.0,
            max_drawdown=0.0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            profit_factor=0.0,
            last_update=datetime.now()
        )
        
        # Ochiq pozitsiyalar
        self.open_positions: Dict[str, Position] = {}
        
        # Yopilgan pozitsiyalar tarixi
        self.position_history: List[Position] = []
        
        # PnL tarixi
        self.pnl_history: List[Dict] = []
        
        # Maksimal balans (drawdown hisoblash uchun)
        self.max_balance = 0.0
        
        logger.info(f"{self.name} ishga tushirildi")
    
    async def initialize(self, initial_balance: float) -> bool:
        """Portfolio ni boshlang'ich holatga keltirish"""
        try:
            logger.info(f"Portfolio boshlang'ich balans: {initial_balance}")
            
            # Boshlang'ich balansni o'rnatish
            self.portfolio.balance = initial_balance
            self.portfolio.equity = initial_balance
            self.portfolio.free_margin = initial_balance
            self.portfolio.margin_level = 0.0
            self.max_balance = initial_balance
            
            # Database dan avvalgi ma'lumotlarni yuklash
            await self._load_portfolio_data()
            
            # Portfolio holatini saqlash
            await self._save_portfolio_state()
            
            logger.info("Portfolio muvaffaqiyatli boshlang'ich holatga keltirildi")
            return True
            
        except Exception as e:
            logger.error(f"Portfolio boshlang'ich holatga keltirishda xato: {e}")
            return False
    
    async def add_position(self, position: Position) -> bool:
        """Yangi pozitsiya qo'shish"""
        try:
            # Pozitsiya validatsiyasi
            if not self._validate_position(position):
                logger.warning(f"Pozitsiya validatsiyasidan o'tmadi: {position.symbol}")
                return False
            
            # Pozitsiyani ochiq pozitsiyalar ro'yxatiga qo'shish
            position_id = f"{position.symbol}_{position.side}_{position.entry_time.timestamp()}"
            self.open_positions[position_id] = position
            
            # Portfolio statistikasini yangilash
            await self._update_portfolio_stats()
            
            # Database ga saqlash
            await self._save_position_to_db(position)
            
            logger.info(f"Yangi pozitsiya qo'shildi: {position.symbol} {position.side}")
            return True
            
        except Exception as e:
            logger.error(f"Pozitsiya qo'shishda xato: {e}")
            return False
    
    async def update_position(self, position_id: str, current_price: float) -> bool:
        """Pozitsiya holatini yangilash"""
        try:
            if position_id not in self.open_positions:
                logger.warning(f"Pozitsiya topilmadi: {position_id}")
                return False
            
            position = self.open_positions[position_id]
            position.current_price = current_price
            
            # Unrealized PnL hisoblash
            position.unrealized_pnl = self._calculate_unrealized_pnl(position)
            
            # Stop Loss yoki Take Profit tekshirish
            if self._should_close_position(position):
                await self.close_position(position_id, current_price, "SL/TP")
            
            # Portfolio statistikasini yangilash
            await self._update_portfolio_stats()
            
            return True
            
        except Exception as e:
            logger.error(f"Pozitsiya yangilashda xato: {e}")
            return False
    
    async def close_position(self, position_id: str, exit_price: float, exit_reason: str) -> bool:
        """Pozitsiyani yopish"""
        try:
            if position_id not in self.open_positions:
                logger.warning(f"Pozitsiya topilmadi: {position_id}")
                return False
            
            position = self.open_positions[position_id]
            
            # Pozitsiyani yopish
            position.exit_price = exit_price
            position.exit_time = datetime.now()
            position.exit_reason = exit_reason
            position.status = "CLOSED"
            
            # Realized PnL hisoblash
            position.realized_pnl = self._calculate_realized_pnl(position)
            
            # Portfolio balansini yangilash
            self.portfolio.balance += position.realized_pnl
            self.portfolio.realized_pnl += position.realized_pnl
            
            # Pozitsiyani tariхga ko'chirish
            self.position_history.append(position)
            del self.open_positions[position_id]
            
            # Portfolio statistikasini yangilash
            await self._update_portfolio_stats()
            
            # Database ga saqlash
            await self._save_closed_position_to_db(position)
            
            logger.info(f"Pozitsiya yopildi: {position.symbol} {position.side}, PnL: {position.realized_pnl}")
            return True
            
        except Exception as e:
            logger.error(f"Pozitsiya yopishda xato: {e}")
            return False
    
    async def get_portfolio_stats(self) -> PortfolioStats:
        """Portfolio statistikasini olish"""
        try:
            # Asosiy statistikalar
            total_return = ((self.portfolio.equity - self.max_balance) / self.max_balance) * 100
            
            # Kunlik return
            daily_return = (self.portfolio.daily_pnl / self.portfolio.balance) * 100
            
            # Haftalik va oylik return
            weekly_return = await self._calculate_period_return(7)
            monthly_return = await self._calculate_period_return(30)
            
            # Sharpe ratio
            sharpe_ratio = await self._calculate_sharpe_ratio()
            
            # Drawdown
            current_drawdown = ((self.max_balance - self.portfolio.equity) / self.max_balance) * 100
            max_drawdown = self.portfolio.max_drawdown
            
            # Trade statistikalar
            win_rate = self.portfolio.win_rate
            profit_factor = self.portfolio.profit_factor
            
            # O'rtacha win/loss
            avg_win, avg_loss = await self._calculate_avg_win_loss()
            
            # Eng katta win/loss
            largest_win, largest_loss = await self._calculate_largest_win_loss()
            
            # Ketma-ket win/loss
            consecutive_wins, consecutive_losses = await self._calculate_consecutive_trades()
            
            return PortfolioStats(
                total_return=total_return,
                daily_return=daily_return,
                weekly_return=weekly_return,
                monthly_return=monthly_return,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                current_drawdown=current_drawdown,
                win_rate=win_rate,
                profit_factor=profit_factor,
                avg_win=avg_win,
                avg_loss=avg_loss,
                largest_win=largest_win,
                largest_loss=largest_loss,
                consecutive_wins=consecutive_wins,
                consecutive_losses=consecutive_losses
            )
            
        except Exception as e:
            logger.error(f"Portfolio statistika olishda xato: {e}")
            return PortfolioStats(**{k: 0.0 for k in PortfolioStats.__dataclass_fields__})
    
    async def check_risk_limits(self) -> Dict[str, bool]:
        """Risk limitlarini tekshirish"""
        try:
            # Kunlik yo'qotish limiti
            daily_loss_limit = self.portfolio.balance * 0.05  # 5%
            daily_loss_exceeded = abs(self.portfolio.daily_pnl) > daily_loss_limit
            
            # Umumiy drawdown limiti
            max_drawdown_limit = 0.10  # 10%
            current_drawdown = ((self.max_balance - self.portfolio.equity) / self.max_balance)
            drawdown_exceeded = current_drawdown > max_drawdown_limit
            
            # Margin level
            margin_level_low = self.portfolio.margin_level < 100.0
            
            # Ochiq pozitsiyalar soni
            max_positions = 5
            too_many_positions = len(self.open_positions) > max_positions
            
            return {
                "daily_loss_exceeded": daily_loss_exceeded,
                "drawdown_exceeded": drawdown_exceeded,
                "margin_level_low": margin_level_low,
                "too_many_positions": too_many_positions
            }
            
        except Exception as e:
            logger.error(f"Risk limitlarini tekshirishda xato: {e}")
            return {}
    
    async def get_position_sizing_advice(self, symbol: str, risk_percent: float) -> Dict:
        """Pozitsiya hajmi maslahatini olish"""
        try:
            # Risk hisoblash
            risk_amount = self.portfolio.balance * (risk_percent / 100)
            
            # Pozitsiya hajmi hisoblash
            position_size = await self.position_sizer.calculate_position_size(
                balance=self.portfolio.balance,
                risk_amount=risk_amount,
                symbol=symbol
            )
            
            return {
                "risk_amount": risk_amount,
                "position_size": position_size,
                "risk_percent": risk_percent,
                "max_lot_size": 0.5  # Propshot limiti
            }
            
        except Exception as e:
            logger.error(f"Pozitsiya hajmi maslahatini olishda xato: {e}")
            return {}
    
    def _validate_position(self, position: Position) -> bool:
        """Pozitsiya validatsiyasi"""
        try:
            # Asosiy validatsiyalar
            if not position.symbol:
                return False
            
            if position.side not in ["BUY", "SELL"]:
                return False
            
            if position.entry_price <= 0:
                return False
            
            if position.quantity <= 0:
                return False
            
            # Stop loss va take profit validatsiyasi
            if position.side == "BUY":
                if position.stop_loss >= position.entry_price:
                    return False
                if position.take_profit <= position.entry_price:
                    return False
            else:  # SELL
                if position.stop_loss <= position.entry_price:
                    return False
                if position.take_profit >= position.entry_price:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Pozitsiya validatsiyasida xato: {e}")
            return False
    
    def _calculate_unrealized_pnl(self, position: Position) -> float:
        """Unrealized PnL hisoblash"""
        try:
            if position.current_price <= 0:
                return 0.0
            
            if position.side == "BUY":
                pnl = (position.current_price - position.entry_price) * position.quantity
            else:  # SELL
                pnl = (position.entry_price - position.current_price) * position.quantity
            
            return pnl
            
        except Exception as e:
            logger.error(f"Unrealized PnL hisoblashda xato: {e}")
            return 0.0
    
    def _calculate_realized_pnl(self, position: Position) -> float:
        """Realized PnL hisoblash"""
        try:
            if not position.exit_price:
                return 0.0
            
            if position.side == "BUY":
                pnl = (position.exit_price - position.entry_price) * position.quantity
            else:  # SELL
                pnl = (position.entry_price - position.exit_price) * position.quantity
            
            # Komissiya va swap ni ayirish
            pnl -= position.commission + position.swap
            
            return pnl
            
        except Exception as e:
            logger.error(f"Realized PnL hisoblashda xato: {e}")
            return 0.0
    
    def _should_close_position(self, position: Position) -> bool:
        """Pozitsiyani yopish kerakligini tekshirish"""
        try:
            if position.current_price <= 0:
                return False
            
            if position.side == "BUY":
                # Stop loss yoki take profit
                if position.current_price <= position.stop_loss:
                    return True
                if position.current_price >= position.take_profit:
                    return True
            else:  # SELL
                # Stop loss yoki take profit
                if position.current_price >= position.stop_loss:
                    return True
                if position.current_price <= position.take_profit:
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Pozitsiya yopish tekshirishda xato: {e}")
            return False
    
    async def _update_portfolio_stats(self) -> None:
        """Portfolio statistikasini yangilash"""
        try:
            # Unrealized PnL ni hisoblash
            total_unrealized_pnl = sum(
                self._calculate_unrealized_pnl(pos) for pos in self.open_positions.values()
            )
            
            # Equity ni yangilash
            self.portfolio.equity = self.portfolio.balance + total_unrealized_pnl
            self.portfolio.unrealized_pnl = total_unrealized_pnl
            
            # Maksimal balansni yangilash
            if self.portfolio.equity > self.max_balance:
                self.max_balance = self.portfolio.equity
            
            # Drawdown hisoblash
            current_drawdown = ((self.max_balance - self.portfolio.equity) / self.max_balance) * 100
            self.portfolio.drawdown = current_drawdown
            
            if current_drawdown > self.portfolio.max_drawdown:
                self.portfolio.max_drawdown = current_drawdown
            
            # Trade statistikalar
            winning_trades = sum(1 for pos in self.position_history if pos.realized_pnl > 0)
            losing_trades = sum(1 for pos in self.position_history if pos.realized_pnl < 0)
            total_trades = len(self.position_history)
            
            self.portfolio.winning_trades = winning_trades
            self.portfolio.losing_trades = losing_trades
            self.portfolio.total_trades = total_trades
            
            # Win rate
            if total_trades > 0:
                self.portfolio.win_rate = (winning_trades / total_trades) * 100
            
            # Profit factor
            total_wins = sum(pos.realized_pnl for pos in self.position_history if pos.realized_pnl > 0)
            total_losses = abs(sum(pos.realized_pnl for pos in self.position_history if pos.realized_pnl < 0))
            
            if total_losses > 0:
                self.portfolio.profit_factor = total_wins / total_losses
            
            # Oxirgi yangilanish vaqti
            self.portfolio.last_update = datetime.now()
            
        except Exception as e:
            logger.error(f"Portfolio statistika yangilashda xato: {e}")
    
    async def _calculate_period_return(self, days: int) -> float:
        """Berilgan davr uchun return hisoblash"""
        try:
            # PnL tariхidan ma'lumot olish
            period_start = datetime.now() - timedelta(days=days)
            
            period_pnl = []
            for record in self.pnl_history:
                if record.get('date') and record['date'] >= period_start:
                    period_pnl.append(record.get('pnl', 0))
            
            if not period_pnl:
                return 0.0
            
            period_total = sum(period_pnl)
            return (period_total / self.portfolio.balance) * 100
            
        except Exception as e:
            logger.error(f"Davr return hisoblashda xato: {e}")
            return 0.0
    
    async def _calculate_sharpe_ratio(self) -> float:
        """Sharpe ratio hisoblash"""
        try:
            # Kunlik return ma'lumotlari
            daily_returns = []
            for record in self.pnl_history[-30:]:  # Oxirgi 30 kun
                if record.get('pnl'):
                    daily_return = (record['pnl'] / self.portfolio.balance) * 100
                    daily_returns.append(daily_return)
            
            if len(daily_returns) < 5:
                return 0.0
            
            # Pandas orqali hisoblash
            df = pd.DataFrame(daily_returns, columns=['returns'])
            avg_return = df['returns'].mean()
            std_return = df['returns'].std()
            
            if std_return == 0:
                return 0.0
            
            # Risk-free rate = 0 deb hisoblaymiz
            sharpe_ratio = avg_return / std_return
            
            # Yillik Sharpe ratio
            return sharpe_ratio * (252 ** 0.5)  # 252 trading days
            
        except Exception as e:
            logger.error(f"Sharpe ratio hisoblashda xato: {e}")
            return 0.0
    
    async def _calculate_avg_win_loss(self) -> Tuple[float, float]:
        """O'rtacha win va loss hisoblash"""
        try:
            wins = [pos.realized_pnl for pos in self.position_history if pos.realized_pnl > 0]
            losses = [pos.realized_pnl for pos in self.position_history if pos.realized_pnl < 0]
            
            avg_win = sum(wins) / len(wins) if wins else 0.0
            avg_loss = sum(losses) / len(losses) if losses else 0.0
            
            return avg_win, avg_loss
            
        except Exception as e:
            logger.error(f"O'rtacha win/loss hisoblashda xato: {e}")
            return 0.0, 0.0
    
    async def _calculate_largest_win_loss(self) -> Tuple[float, float]:
        """Eng katta win va loss hisoblash"""
        try:
            wins = [pos.realized_pnl for pos in self.position_history if pos.realized_pnl > 0]
            losses = [pos.realized_pnl for pos in self.position_history if pos.realized_pnl < 0]
            
            largest_win = max(wins) if wins else 0.0
            largest_loss = min(losses) if losses else 0.0
            
            return largest_win, largest_loss
            
        except Exception as e:
            logger.error(f"Eng katta win/loss hisoblashda xato: {e}")
            return 0.0, 0.0
    
    async def _calculate_consecutive_trades(self) -> Tuple[int, int]:
        """Ketma-ket win va loss hisoblash"""
        try:
            if not self.position_history:
                return 0, 0
            
            # Oxirgi tradelarni vaqt bo'yicha tartiblash
            sorted_trades = sorted(self.position_history, key=lambda x: x.exit_time or datetime.min)
            
            max_consecutive_wins = 0
            max_consecutive_losses = 0
            current_consecutive_wins = 0
            current_consecutive_losses = 0
            
            for trade in sorted_trades:
                if trade.realized_pnl > 0:
                    current_consecutive_wins += 1
                    current_consecutive_losses = 0
                    max_consecutive_wins = max(max_consecutive_wins, current_consecutive_wins)
                else:
                    current_consecutive_losses += 1
                    current_consecutive_wins = 0
                    max_consecutive_losses = max(max_consecutive_losses, current_consecutive_losses)
            
            return max_consecutive_wins, max_consecutive_losses
            
        except Exception as e:
            logger.error(f"Ketma-ket trade hisoblashda xato: {e}")
            return 0, 0
    
    async def _load_portfolio_data(self) -> None:
        """Database dan portfolio ma'lumotlarini yuklash"""
        try:
            # Database dan portfolio holatini yuklash
            portfolio_data = await self.db_manager.get_portfolio_state()
            if portfolio_data:
                # Portfolio ma'lumotlarini yangilash
                for key, value in portfolio_data.items():
                    if hasattr(self.portfolio, key):
                        setattr(self.portfolio, key, value)
            
            # Ochiq pozitsiyalarni yuklash
            open_positions = await self.db_manager.get_open_positions()
            for pos_data in open_positions:
                position = Position(**pos_data)
                position_id = f"{position.symbol}_{position.side}_{position.entry_time.timestamp()}"
                self.open_positions[position_id] = position
            
            # Pozitsiya tarixini yuklash
            position_history = await self.db_manager.get_position_history(limit=1000)
            self.position_history = [Position(**pos_data) for pos_data in position_history]
            
            # PnL tarixini yuklash
            self.pnl_history = await self.db_manager.get_pnl_history(limit=365)
            
            logger.info("Portfolio ma'lumotlari database dan yuklandi")
            
        except Exception as e:
            logger.error(f"Portfolio ma'lumotlarini yuklashda xato: {e}")
    
    async def _save_portfolio_state(self) -> None:
        """Portfolio holatini database ga saqlash"""
        try:
            await self.db_manager.save_portfolio_state(self.portfolio.to_dict())
            logger.debug("Portfolio holati database ga saqlandi")
            
        except Exception as e:
            logger.error(f"Portfolio holatini saqlashda xato: {e}")
    
    async def _save_position_to_db(self, position: Position) -> None:
        """Pozitsiyani database ga saqlash"""
        try:
            await self.db_manager.save_position(position.to_dict())
            logger.debug(f"Pozitsiya database ga saqlandi: {position.symbol}")
            
        except Exception as e:
            logger.error(f"Pozitsiyani saqlashda xato: {e}")
    
    async def _save_closed_position_to_db(self, position: Position) -> None:
        """Yopilgan pozitsiyani database ga saqlash"""
        try:
            await self.db_manager.update_position_closed(position.to_dict())
            logger.debug(f"Yopilgan pozitsiya database ga saqlandi: {position.symbol}")
            
        except Exception as e:
            logger.error(f"Yopilgan pozitsiyani saqlashda xato: {e}")
    
    async def get_portfolio_summary(self) -> Dict:
        """Portfolio qisqacha ma'lumotlari"""
        try:
            stats = await self.get_portfolio_stats()
            risk_status = await self.check_risk_limits()
            
            return {
                "portfolio": self.portfolio.to_dict(),
                "stats": stats.to_dict(),
                "risk_status": risk_status,
                "open_positions": len(self.open_positions),
                "total_positions": len(self.position_history),
                "last_update": self.portfolio.last_update.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Portfolio qisqacha ma'lumotlarini olishda xato: {e}")
            return {}

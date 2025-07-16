import sys
import os

# 🔑 Bu qatorlar root papkani sys.path ga qo‘shadi
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Pastda sening eski importlaring bo‘lsin
from config.config import ConfigManager

# 🔻 Sening qolgan koding shu yerda davom etadi...





#!/usr/bin/env python3
"""
AI OrderFlow & Signal Bot - Asosiy Fayl
=====================================

AI yordamida DEX Order Flow va Sentiment tahlil qilib, 
real vaqtda signal beruvchi bot. Propshot EA bog'lanadi.

Muallif: AI OrderFlow Bot
Yaratilgan: 2025
Versiya: 1.0.0
"""

import asyncio
import signal
import sys
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path

# Asosiy modullar
from config.config import ConfigManager
from utils.logger import get_logger
from utils.error_handler import ErrorHandler
from utils.fallback_manager import FallbackManager

# API Clients
from api_clients.telegram_client import TelegramClient
from api_clients.oneinch_client import OneInchClient
from api_clients.alchemy_client import AlchemyClient
from api_clients.huggingface_client import HuggingFaceClient
from api_clients.gemini_client import GeminiClient
from api_clients.news_client import NewsClient
from api_clients.reddit_client import RedditClient

# Data Processing
from data_processing.order_flow_analyzer import OrderFlowAnalyzer
from data_processing.sentiment_analyzer import SentimentAnalyzer
from data_processing.market_analyzer import MarketAnalyzer
from data_processing.signal_generator import SignalGenerator

# Risk Management
from risk_management.risk_calculator import RiskCalculator
from risk_management.position_sizer import PositionSizer
from risk_management.trade_monitor import TradeMonitor

# Trading
from trading.strategy_manager import StrategyManager
from trading.execution_engine import ExecutionEngine
from trading.propshot_connector import PropshotConnector
from trading.portfolio_manager import PortfolioManager

# Database
from database.db_manager import DatabaseManager

# Global logger
logger = get_logger(__name__)

@dataclass
class BotStatus:
    """Bot holati ma'lumotlari"""
    is_running: bool = False
    start_time: Optional[datetime] = None
    total_signals: int = 0
    successful_trades: int = 0
    failed_trades: int = 0
    total_profit: float = 0.0
    daily_limit_reached: bool = False
    last_signal_time: Optional[datetime] = None
    current_risk: float = 0.0
    active_positions: int = 0
    last_error: Optional[str] = None
    fallback_active: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Bot holatini dict formatida qaytarish"""
        return {
            'is_running': self.is_running,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'uptime': str(datetime.now() - self.start_time) if self.start_time else None,
            'total_signals': self.total_signals,
            'successful_trades': self.successful_trades,
            'failed_trades': self.failed_trades,
            'win_rate': self.successful_trades / max(self.successful_trades + self.failed_trades, 1) * 100,
            'total_profit': self.total_profit,
            'daily_limit_reached': self.daily_limit_reached,
            'last_signal': self.last_signal_time.isoformat() if self.last_signal_time else None,
            'current_risk': self.current_risk,
            'active_positions': self.active_positions,
            'last_error': self.last_error,
            'fallback_active': self.fallback_active
        }


class AIOrderFlowBot:
    """AI OrderFlow & Signal Bot asosiy class"""
    
    def __init__(self):
        """Bot inicializatsiyasi"""
        logger.info("🚀 AI OrderFlow Bot ishga tushirilmoqda...")
        
        # Status tracking
        self.status = BotStatus()
        self.shutdown_requested = False
        
        # Konfiguratsiya yuklash
        self.config_manager = ConfigManager()
        self.config = self.config_manager.get_config()
        
        # Error handler
        self.error_handler = ErrorHandler()
        
        # Fallback manager
        self.fallback_manager = FallbackManager()
        
        # Database manager
        self.db_manager = DatabaseManager()
        
        # API Clients - fallback bilan
        self.telegram_client: Optional[TelegramClient] = None
        self.oneinch_client: Optional[OneInchClient] = None
        self.alchemy_client: Optional[AlchemyClient] = None
        self.huggingface_client: Optional[HuggingFaceClient] = None
        self.gemini_client: Optional[GeminiClient] = None
        self.news_client: Optional[NewsClient] = None
        self.reddit_client: Optional[RedditClient] = None
        
        # Data processors
        self.order_flow_analyzer: Optional[OrderFlowAnalyzer] = None
        self.sentiment_analyzer: Optional[SentimentAnalyzer] = None
        self.market_analyzer: Optional[MarketAnalyzer] = None
        self.signal_generator: Optional[SignalGenerator] = None
        
        # Risk management
        self.risk_calculator: Optional[RiskCalculator] = None
        self.position_sizer: Optional[PositionSizer] = None
        self.trade_monitor: Optional[TradeMonitor] = None
        
        # Trading components
        self.strategy_manager: Optional[StrategyManager] = None
        self.execution_engine: Optional[ExecutionEngine] = None
        self.propshot_connector: Optional[PropshotConnector] = None
        self.portfolio_manager: Optional[PortfolioManager] = None
        
        # Working hours (UZB vaqti)
        self.work_start_hour = 7   # 07:00
        self.work_end_hour = 19    # 19:30
        self.work_end_minute = 30
        
        logger.info("✅ AI OrderFlow Bot muvaffaqiyatli inicializatsiya qilindi")
    
    async def initialize_components(self) -> bool:
        """Barcha komponentlarni inicializatsiya qilish"""
        try:
            logger.info("⚙️ Komponentlarni inicializatsiya qilish boshlandi...")
            
            # Database inicializatsiya
            await self.db_manager.initialize()
            logger.info("✅ Database inicializatsiya qilindi")
            
            # API Clients inicializatsiya
            await self._initialize_api_clients()
            logger.info("✅ API Clients inicializatsiya qilindi")
            
            # Data processors inicializatsiya
            await self._initialize_data_processors()
            logger.info("✅ Data processors inicializatsiya qilindi")
            
            # Risk management inicializatsiya
            await self._initialize_risk_management()
            logger.info("✅ Risk management inicializatsiya qilindi")
            
            # Trading components inicializatsiya
            await self._initialize_trading_components()
            logger.info("✅ Trading components inicializatsiya qilindi")
            
            # Health check
            health_status = await self._health_check()
            if not health_status:
                logger.error("❌ Health check muvaffaqiyatsiz")
                return False
            
            logger.info("🎉 Barcha komponentlar muvaffaqiyatli inicializatsiya qilindi")
            return True
            
        except Exception as e:
            logger.error(f"❌ Komponentlar inicializatsiya qilishda xato: {e}")
            await self.error_handler.handle_error(e, "initialization")
            return False
    
    async def _initialize_api_clients(self) -> None:
        """API clientlarni inicializatsiya qilish"""
        try:
            # Telegram client (eng muhim)
            self.telegram_client = TelegramClient()
            await self.telegram_client.initialize()
            
            # Order flow clients
            self.oneinch_client = OneInchClient()
            await self.oneinch_client.initialize()
            
            self.alchemy_client = AlchemyClient()
            await self.alchemy_client.initialize()
            
            # AI clients
            self.huggingface_client = HuggingFaceClient()
            await self.huggingface_client.initialize()
            
            self.gemini_client = GeminiClient()
            await self.gemini_client.initialize()
            
            # News clients
            self.news_client = NewsClient()
            await self.news_client.initialize()
            
            self.reddit_client = RedditClient()
            await self.reddit_client.initialize()
            
        except Exception as e:
            logger.error(f"API clientlar inicializatsiya qilishda xato: {e}")
            raise
    
    async def _initialize_data_processors(self) -> None:
        """Data processorlarni inicializatsiya qilish"""
        try:
            # Order flow analyzer
            self.order_flow_analyzer = OrderFlowAnalyzer(
                oneinch_client=self.oneinch_client,
                alchemy_client=self.alchemy_client
            )
            
            # Sentiment analyzer
            self.sentiment_analyzer = SentimentAnalyzer(
                huggingface_client=self.huggingface_client,
                gemini_client=self.gemini_client,
                news_client=self.news_client,
                reddit_client=self.reddit_client
            )
            
            # Market analyzer
            self.market_analyzer = MarketAnalyzer(
                alchemy_client=self.alchemy_client
            )
            
            # Signal generator
            self.signal_generator = SignalGenerator(
                order_flow_analyzer=self.order_flow_analyzer,
                sentiment_analyzer=self.sentiment_analyzer,
                market_analyzer=self.market_analyzer
            )
            
        except Exception as e:
            logger.error(f"Data processors inicializatsiya qilishda xato: {e}")
            raise
    
    async def _initialize_risk_management(self) -> None:
        """Risk management komponentlarni inicializatsiya qilish"""
        try:
            # Risk calculator
            self.risk_calculator = RiskCalculator(
                config=self.config
            )
            
            # Position sizer
            self.position_sizer = PositionSizer(
                risk_calculator=self.risk_calculator,
                config=self.config
            )
            
            # Trade monitor
            self.trade_monitor = TradeMonitor(
                risk_calculator=self.risk_calculator,
                db_manager=self.db_manager
            )
            
        except Exception as e:
            logger.error(f"Risk management inicializatsiya qilishda xato: {e}")
            raise
    
    async def _initialize_trading_components(self) -> None:
        """Trading komponentlarni inicializatsiya qilish"""
        try:
            # Strategy manager
            self.strategy_manager = StrategyManager(
                config=self.config,
                db_manager=self.db_manager
            )
            
            # Execution engine
            self.execution_engine = ExecutionEngine(
                risk_calculator=self.risk_calculator,
                position_sizer=self.position_sizer,
                trade_monitor=self.trade_monitor
            )
            
            # Propshot connector
            self.propshot_connector = PropshotConnector(
                config=self.config
            )
            
            # Portfolio manager
            self.portfolio_manager = PortfolioManager(
                db_manager=self.db_manager,
                risk_calculator=self.risk_calculator
            )
            
        except Exception as e:
            logger.error(f"Trading components inicializatsiya qilishda xato: {e}")
            raise
    
    async def _health_check(self) -> bool:
        """Tizim holatini tekshirish"""
        try:
            logger.info("🔍 Health check boshlandi...")
            health_status = {}
            
            # API clients health check
            if self.telegram_client:
                health_status['telegram'] = await self.telegram_client.health_check()
            if self.oneinch_client:
                health_status['oneinch'] = await self.oneinch_client.health_check()
            if self.alchemy_client:
                health_status['alchemy'] = await self.alchemy_client.health_check()
            
            # Database health check
            health_status['database'] = await self.db_manager.health_check()
            
            # Check critical components
            critical_services = ['telegram', 'database']
            for service in critical_services:
                if service in health_status and not health_status[service]:
                    logger.error(f"❌ Critical service {service} ishlamayapti")
                    return False
            
            # Telegram orqali holatni yuborish
            if self.telegram_client:
                await self.telegram_client.send_message(
                    "🚀 AI OrderFlow Bot ishga tushdi!\n"
                    f"⏰ Vaqt: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                    f"📊 Health Status: {health_status}\n"
                    f"🔄 Ish rejimi: {'Avtomatik' if self.config.get('auto_trading', False) else 'Qo\'lda'}"
                )
            
            logger.info("✅ Health check muvaffaqiyatli tugallandi")
            return True
            
        except Exception as e:
            logger.error(f"Health check da xato: {e}")
            return False
    
    def is_working_hours(self) -> bool:
        """Ish vaqtini tekshirish (UZB vaqti)"""
        try:
            import pytz
            uzb_tz = pytz.timezone('Asia/Tashkent')
            now = datetime.now(uzb_tz)
            
            # Hafta kunlari (Dushanba-Juma)
            if now.weekday() >= 5:  # Shanba, Yakshanba
                return False
            
            # Ish vaqti: 07:00 - 19:30
            work_start = now.replace(hour=self.work_start_hour, minute=0, second=0, microsecond=0)
            work_end = now.replace(hour=self.work_end_hour, minute=self.work_end_minute, second=0, microsecond=0)
            
            return work_start <= now <= work_end
            
        except Exception as e:
            logger.error(f"Ish vaqti tekshirishda xato: {e}")
            return True  # Xato bo'lsa, ishlashda davom etish
    
    async def run_main_loop(self) -> None:
        """Asosiy ishchi tsikl"""
        try:
            logger.info("🔄 Asosiy ishchi tsikl boshlandi")
            self.status.is_running = True
            self.status.start_time = datetime.now()
            
            while not self.shutdown_requested:
                try:
                    # Ish vaqti tekshirish
                    if not self.is_working_hours():
                        logger.debug("⏰ Ish vaqti emas, 60 soniya kutish...")
                        await asyncio.sleep(60)
                        continue
                    
                    # Kunlik limitlar tekshirish
                    daily_risk = await self.trade_monitor.get_daily_risk()
                    if daily_risk > self.config.get('max_daily_risk', 0.05):
                        logger.warning("⚠️ Kunlik risk limiti oshib ketdi")
                        self.status.daily_limit_reached = True
                        await self._send_status_update("🚨 Kunlik risk limiti oshib ketdi!")
                        await asyncio.sleep(300)  # 5 daqiqa kutish
                        continue
                    
                    # Signal generatsiya qilish
                    signal_data = await self._generate_signal()
                    
                    if signal_data and signal_data.get('confidence', 0) > 0.7:
                        logger.info(f"📈 Signal aniqlandi: {signal_data['symbol']} - {signal_data['action']}")
                        
                        # Signal yuborish
                        await self._process_signal(signal_data)
                        
                        # Statistikani yangilash
                        self.status.total_signals += 1
                        self.status.last_signal_time = datetime.now()
                        
                        # Keyingi signal uchun kutish
                        await asyncio.sleep(self.config.get('signal_interval', 300))  # 5 daqiqa
                    
                    else:
                        # Signal yo'q, qisqa kutish
                        await asyncio.sleep(self.config.get('scan_interval', 60))  # 1 daqiqa
                    
                    # Holatni yangilash
                    await self._update_status()
                    
                except Exception as e:
                    logger.error(f"Asosiy tsiklda xato: {e}")
                    await self.error_handler.handle_error(e, "main_loop")
                    self.status.last_error = str(e)
                    await asyncio.sleep(30)  # Xato bo'lsa, 30 soniya kutish
            
            logger.info("🛑 Asosiy ishchi tsikl to'xtatildi")
            
        except Exception as e:
            logger.error(f"Asosiy tsiklda jiddiy xato: {e}")
            await self.error_handler.handle_error(e, "main_loop_critical")
            raise
    
    async def _generate_signal(self) -> Optional[Dict[str, Any]]:
        """Signal generatsiya qilish"""
        try:
            logger.debug("📊 Signal generatsiya qilish boshlandi...")
            
            # Order flow tahlili
            order_flow_data = await self.order_flow_analyzer.analyze()
            if not order_flow_data.success:
                logger.warning("⚠️ Order flow tahlili muvaffaqiyatsiz")
                return None
            
            # Sentiment tahlili
            sentiment_data = await self.sentiment_analyzer.analyze()
            if not sentiment_data.success:
                logger.warning("⚠️ Sentiment tahlili muvaffaqiyatsiz")
                return None
            
            # Market tahlili
            market_data = await self.market_analyzer.analyze()
            if not market_data.success:
                logger.warning("⚠️ Market tahlili muvaffaqiyatsiz")
                return None
            
            # Signal generatsiya
            signal_result = await self.signal_generator.generate_signal(
                order_flow_data.data,
                sentiment_data.data,
                market_data.data
            )
            
            if signal_result.success:
                logger.info(f"✅ Signal muvaffaqiyatli generatsiya qilindi: {signal_result.data}")
                return signal_result.data
            else:
                logger.debug("📊 Signal generatsiya qilinmadi")
                return None
                
        except Exception as e:
            logger.error(f"Signal generatsiya qilishda xato: {e}")
            await self.error_handler.handle_error(e, "signal_generation")
            return None
    
    async def _process_signal(self, signal_data: Dict[str, Any]) -> None:
        """Signalni qayta ishlash va yuborish"""
        try:
            logger.info(f"📤 Signal qayta ishlanmoqda: {signal_data}")
            
            # Risk hisoblash
            risk_result = await self.risk_calculator.calculate_risk(signal_data)
            if not risk_result.success:
                logger.error("❌ Risk hisoblashda xato")
                return
            
            # Position size hisoblash
            position_size = await self.position_sizer.calculate_position_size(
                signal_data, risk_result.data
            )
            
            # Signal ma'lumotlarini to'ldirish
            enhanced_signal = {
                **signal_data,
                'risk_data': risk_result.data,
                'position_size': position_size,
                'timestamp': datetime.now().isoformat(),
                'signal_id': f"SIG_{int(time.time())}"
            }
            
            # Database ga saqlash
            await self.db_manager.save_signal(enhanced_signal)
            
            # Telegram orqali signal yuborish
            await self._send_signal_to_telegram(enhanced_signal)
            
            # Propshot EA ga signal yuborish
            if self.config.get('propshot_enabled', False):
                await self._send_signal_to_propshot(enhanced_signal)
            
            logger.info("✅ Signal muvaffaqiyatli qayta ishlandi va yuborildi")
            
        except Exception as e:
            logger.error(f"Signal qayta ishlashda xato: {e}")
            await self.error_handler.handle_error(e, "signal_processing")
    
    async def _send_signal_to_telegram(self, signal_data: Dict[str, Any]) -> None:
        """Telegram orqali signal yuborish"""
        try:
            if not self.telegram_client:
                logger.error("❌ Telegram client mavjud emas")
                return
            
            # Signal formatini yaratish
            signal_message = self._format_signal_message(signal_data)
            
            # Signal yuborish
            await self.telegram_client.send_signal(signal_message, signal_data)
            
            logger.info("✅ Signal Telegram orqali yuborildi")
            
        except Exception as e:
            logger.error(f"Telegram signal yuborishda xato: {e}")
            await self.error_handler.handle_error(e, "telegram_signal")
    
    async def _send_signal_to_propshot(self, signal_data: Dict[str, Any]) -> None:
        """Propshot EA ga signal yuborish"""
        try:
            if not self.propshot_connector:
                logger.error("❌ Propshot connector mavjud emas")
                return
            
            # Propshot formatiga o'tkazish
            propshot_signal = self.propshot_connector.format_signal(signal_data)
            
            # Signal yuborish
            result = await self.propshot_connector.send_signal(propshot_signal)
            
            if result.success:
                logger.info("✅ Signal Propshot EA ga yuborildi")
            else:
                logger.error(f"❌ Propshot signal yuborishda xato: {result.error}")
            
        except Exception as e:
            logger.error(f"Propshot signal yuborishda xato: {e}")
            await self.error_handler.handle_error(e, "propshot_signal")
    
    def _format_signal_message(self, signal_data: Dict[str, Any]) -> str:
        """Signal xabarini formatlash"""
        try:
            return f"""
🎯 YANGI SIGNAL KELDI!
═══════════════════════
📊 Simbol: {signal_data.get('symbol', 'N/A')}
📈 Harakat: {signal_data.get('action', 'N/A')}
💰 Narx: {signal_data.get('price', 'N/A')}
📊 Lot: {signal_data.get('position_size', 'N/A')}
🛡️ Stop Loss: {signal_data.get('stop_loss', 'N/A')}
🎯 Take Profit: {signal_data.get('take_profit', 'N/A')}
⚡ Ishonch: {signal_data.get('confidence', 0):.1f}%
🔥 Risk: {signal_data.get('risk_percent', 0):.1f}%
═══════════════════════
📝 Sabab: {signal_data.get('reason', 'N/A')}
⏰ Vaqt: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (UZB)
💼 Signal ID: {signal_data.get('signal_id', 'N/A')}

{'🟢 AVTOMATIK REJIM' if self.config.get('auto_trading', False) else '🔶 QO\'LDA REJIM'}
"""
        except Exception as e:
            logger.error(f"Signal message formatlashda xato: {e}")
            return f"❌ Signal format xatosi: {e}"
    
    async def _update_status(self) -> None:
        """Bot holatini yangilash"""
        try:
            # Portfolio holatini yangilash
            if self.portfolio_manager:
                portfolio_data = await self.portfolio_manager.get_portfolio_status()
                if portfolio_data.success:
                    self.status.current_risk = portfolio_data.data.get('current_risk', 0)
                    self.status.active_positions = portfolio_data.data.get('active_positions', 0)
                    self.status.total_profit = portfolio_data.data.get('total_profit', 0)
            
            # Trade statistics yangilash
            if self.trade_monitor:
                trade_stats = await self.trade_monitor.get_daily_stats()
                if trade_stats.success:
                    self.status.successful_trades = trade_stats.data.get('successful_trades', 0)
                    self.status.failed_trades = trade_stats.data.get('failed_trades', 0)
            
        except Exception as e:
            logger.error(f"Status yangilashda xato: {e}")
    
    async def _send_status_update(self, message: str) -> None:
        """Holatni Telegram orqali yuborish"""
        try:
            if self.telegram_client:
                await self.telegram_client.send_message(message)
        except Exception as e:
            logger.error(f"Status update yuborishda xato: {e}")
    
    async def get_status(self) -> Dict[str, Any]:
        """Bot holatini qaytarish"""
        try:
            await self._update_status()
            return self.status.to_dict()
        except Exception as e:
            logger.error(f"Status olishda xato: {e}")
            return {"error": str(e)}
    
    async def shutdown(self) -> None:
        """Botni to'xtatish"""
        try:
            logger.info("🛑 Bot to'xtatilmoqda...")
            self.shutdown_requested = True
            self.status.is_running = False
            
            # Barcha komponentlarni to'xtatish
            if self.telegram_client:
                await self.telegram_client.send_message("🛑 AI OrderFlow Bot to'xtatildi")
                await self.telegram_client.close()
            
            if self.db_manager:
                await self.db_manager.close()
            
            # Boshqa clientlarni to'xtatish
            for client in [self.oneinch_client, self.alchemy_client, 
                          self.huggingface_client, self.gemini_client,
                          self.news_client, self.reddit_client]:
                if client:
                    await client.close()
            
            logger.info("✅ Bot muvaffaqiyatli to'xtatildi")
            
        except Exception as e:
            logger.error(f"Bot to'xtatishda xato: {e}")


async def signal_handler(bot: AIOrderFlowBot):
    """Signal handler"""
    def handle_signal(signum, frame):
        logger.info(f"🔔 Signal qabul qilindi: {signum}")
        asyncio.create_task(bot.shutdown())
    
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)


async def main():
    """Asosiy main function"""
    try:
        # Bot yaratish
        bot = AIOrderFlowBot()
        
        # Signal handler setup
        await signal_handler(bot)
        
        # Komponentlarni inicializatsiya qilish
        if not await bot.initialize_components():
            logger.error("❌ Bot inicializatsiya qilinmadi")
            return 1
        
        # Asosiy tsiklni boshlash
        await bot.run_main_loop()
        
        # Graceful shutdown
        await bot.shutdown()
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("🔔 Keyboard interrupt qabul qilindi")
        if 'bot' in locals():
            await bot.shutdown()
        return 0
        
    except Exception as e:
        logger.error(f"❌ Asosiy xato: {e}")
        if 'bot' in locals():
            await bot.shutdown()
        return 1


if __name__ == "__main__":
    # Asosiy papkani PATH ga qo'shish
    sys.path.insert(0, str(Path(__file__).parent))
    
    # Kerakli papkalarni yaratish
    for directory in ['logs', 'data', 'data/cache', 'data/exports']:
        Path(directory).mkdir(exist_ok=True)
    
    # Botni ishga tushirish
    logger.info("🚀 AI OrderFlow & Signal Bot ishga tushirildi")
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

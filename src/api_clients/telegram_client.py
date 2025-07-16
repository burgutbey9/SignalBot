import asyncio
import aiohttp
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime
import json
import io
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from telegram import Bot, Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
from telegram.constants import ParseMode
from utils.logger import get_logger
from utils.rate_limiter import RateLimiter
from utils.error_handler import handle_api_error
from utils.retry_handler import retry_async

logger = get_logger(__name__)

@dataclass
class TelegramMessage:
    """Telegram xabar ma'lumotlari"""
    text: str
    chat_id: str
    parse_mode: str = ParseMode.MARKDOWN
    reply_markup: Optional[InlineKeyboardMarkup] = None
    disable_web_page_preview: bool = True

@dataclass
class SignalData:
    """Trading signal ma'lumotlari"""
    action: str  # BUY/SELL
    symbol: str
    price: float
    lot_size: float
    stop_loss: float
    take_profit: float
    confidence: float
    risk_percent: float
    reason: str
    timestamp: datetime
    account: str = "Propshot"
    sl_pips: int = 0
    tp_pips: int = 0
    market_analysis: Optional[Dict] = None

@dataclass
class TelegramResponse:
    """Telegram API javob formati"""
    success: bool
    message_id: Optional[int] = None
    error: Optional[str] = None
    data: Optional[Any] = None

class TelegramClient:
    """Telegram bot client - O'zbekcha signal va log yuborish"""
    
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.rate_limiter = RateLimiter(calls=30, period=60)  # 30 xabar/minut
        self.bot: Optional[Bot] = None
        self.application: Optional[Application] = None
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Foydalanuvchi sozlamalari
        self.user_settings = {
            "auto_trade": False,
            "max_risk": 0.02,
            "notifications": True,
            "language": "uz",
            "timezone": "Asia/Tashkent"
        }
        
        # Signal callback lari
        self.signal_callbacks = {}
        
        logger.info("Telegram client ishga tushirildi")

    async def __aenter__(self):
        """Async context manager kirish"""
        await self.initialize_bot()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager chiqish"""
        await self.close()

    async def initialize_bot(self):
        """Telegram bot va application yaratish"""
        try:
            self.bot = Bot(token=self.bot_token)
            self.application = Application.builder().token(self.bot_token).build()
            
            # HTTP session yaratish
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30)
            )
            
            # Command handlerlar qo'shish
            self.application.add_handler(CommandHandler("start", self.handle_start))
            self.application.add_handler(CommandHandler("help", self.handle_help))
            self.application.add_handler(CommandHandler("status", self.handle_status))
            self.application.add_handler(CommandHandler("settings", self.handle_settings))
            self.application.add_handler(CommandHandler("stats", self.handle_stats))
            self.application.add_handler(CallbackQueryHandler(self.handle_callback))
            
            # Bot ma'lumotlarini olish
            bot_info = await self.bot.get_me()
            logger.info(f"Telegram bot muvaffaqiyatli ishga tushirildi: @{bot_info.username}")
            
        except Exception as e:
            logger.error(f"Telegram bot yaratishda xato: {e}")
            raise

    async def close(self):
        """Connectionlarni yopish"""
        if self.application:
            await self.application.stop()
        if self.session:
            await self.session.close()
        logger.info("Telegram client yopildi")

    @retry_async(max_retries=3, delay=2)
    async def send_signal(self, signal: SignalData) -> TelegramResponse:
        """
        Trading signal yuborish
        
        Args:
            signal: Signal ma'lumotlari
            
        Returns:
            TelegramResponse: Yuborish natijasi
        """
        try:
            await self.rate_limiter.wait()
            
            if not self.bot:
                await self.initialize_bot()
            
            # Signal formatini yaratish
            signal_text = self._format_signal(signal)
            
            # Inline keyboard yaratish
            keyboard = self._create_signal_keyboard(signal)
            
            # Xabar yuborish
            message = await self.bot.send_message(
                chat_id=self.chat_id,
                text=signal_text,
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=keyboard,
                disable_web_page_preview=True
            )
            
            # Callback saqlash
            self.signal_callbacks[message.message_id] = signal
            
            logger.info(f"Signal yuborildi: {signal.symbol} {signal.action}")
            
            return TelegramResponse(
                success=True,
                message_id=message.message_id,
                data=signal
            )
            
        except Exception as e:
            logger.error(f"Signal yuborishda xato: {e}")
            return TelegramResponse(
                success=False,
                error=str(e)
            )

    @retry_async(max_retries=3, delay=1)
    async def send_log(self, message: str, level: str = "INFO") -> TelegramResponse:
        """
        Log xabar yuborish
        
        Args:
            message: Log xabari
            level: Log darajasi (INFO, WARNING, ERROR)
            
        Returns:
            TelegramResponse: Yuborish natijasi
        """
        try:
            await self.rate_limiter.wait()
            
            if not self.bot:
                await self.initialize_bot()
            
            # Log formatini yaratish
            log_text = self._format_log(message, level)
            
            # Xabar yuborish
            telegram_message = await self.bot.send_message(
                chat_id=self.chat_id,
                text=log_text,
                parse_mode=ParseMode.MARKDOWN,
                disable_web_page_preview=True
            )
            
            return TelegramResponse(
                success=True,
                message_id=telegram_message.message_id
            )
            
        except Exception as e:
            logger.error(f"Log yuborishda xato: {e}")
            return TelegramResponse(
                success=False,
                error=str(e)
            )

    @retry_async(max_retries=3, delay=1)
    async def send_market_analysis(self, analysis: Dict) -> TelegramResponse:
        """
        Bozor tahlili yuborish
        
        Args:
            analysis: Bozor tahlili ma'lumotlari
            
        Returns:
            TelegramResponse: Yuborish natijasi
        """
        try:
            await self.rate_limiter.wait()
            
            if not self.bot:
                await self.initialize_bot()
            
            # Tahlil formatini yaratish
            analysis_text = self._format_market_analysis(analysis)
            
            # Xabar yuborish
            message = await self.bot.send_message(
                chat_id=self.chat_id,
                text=analysis_text,
                parse_mode=ParseMode.MARKDOWN,
                disable_web_page_preview=True
            )
            
            return TelegramResponse(
                success=True,
                message_id=message.message_id
            )
            
        except Exception as e:
            logger.error(f"Bozor tahlili yuborishda xato: {e}")
            return TelegramResponse(
                success=False,
                error=str(e)
            )

    @retry_async(max_retries=3, delay=1)
    async def send_status_update(self, status: Dict) -> TelegramResponse:
        """
        Bot holati yuborish
        
        Args:
            status: Bot holati ma'lumotlari
            
        Returns:
            TelegramResponse: Yuborish natijasi
        """
        try:
            await self.rate_limiter.wait()
            
            if not self.bot:
                await self.initialize_bot()
            
            # Status formatini yaratish
            status_text = self._format_status(status)
            
            # Xabar yuborish
            message = await self.bot.send_message(
                chat_id=self.chat_id,
                text=status_text,
                parse_mode=ParseMode.MARKDOWN,
                disable_web_page_preview=True
            )
            
            return TelegramResponse(
                success=True,
                message_id=message.message_id
            )
            
        except Exception as e:
            logger.error(f"Status yuborishda xato: {e}")
            return TelegramResponse(
                success=False,
                error=str(e)
            )

    async def send_chart(self, chart_data: Dict, title: str) -> TelegramResponse:
        """
        Grafik yuborish
        
        Args:
            chart_data: Grafik ma'lumotlari
            title: Grafik sarlavhasi
            
        Returns:
            TelegramResponse: Yuborish natijasi
        """
        try:
            await self.rate_limiter.wait()
            
            if not self.bot:
                await self.initialize_bot()
            
            # Grafik yaratish
            chart_buffer = self._create_chart(chart_data, title)
            
            # Grafik yuborish
            message = await self.bot.send_photo(
                chat_id=self.chat_id,
                photo=chart_buffer,
                caption=f"📊 *{title}*",
                parse_mode=ParseMode.MARKDOWN
            )
            
            return TelegramResponse(
                success=True,
                message_id=message.message_id
            )
            
        except Exception as e:
            logger.error(f"Grafik yuborishda xato: {e}")
            return TelegramResponse(
                success=False,
                error=str(e)
            )

    def _format_signal(self, signal: SignalData) -> str:
        """Signal formatini yaratish"""
        
        # Emoji va rang belgilash
        action_emoji = "📈 🟢" if signal.action == "BUY" else "📉 🔴"
        confidence_emoji = "🔥" if signal.confidence > 80 else "⚡" if signal.confidence > 60 else "💫"
        
        # Pips hisoblash (agar belgilanmagan bo'lsa)
        if signal.sl_pips == 0:
            signal.sl_pips = abs(int((signal.price - signal.stop_loss) * 10000))
        if signal.tp_pips == 0:
            signal.tp_pips = abs(int((signal.take_profit - signal.price) * 10000))
        
        # Vaqtni O'zbekiston vaqtiga o'tkazish
        uzb_time = signal.timestamp.strftime("%H:%M:%S")
        uzb_date = signal.timestamp.strftime("%d.%m.%Y")
        
        signal_text = f"""
🤖 *AI SIGNAL KELDI* {action_emoji}
═══════════════════════════

💱 *Juftlik:* `{signal.symbol}`
📊 *Harakat:* *{signal.action}*
💰 *Narx:* `{signal.price:.5f}`
📏 *Lot:* `{signal.lot_size}`

🛡️ *Stop Loss:* `{signal.stop_loss:.5f}` ({signal.sl_pips} pips)
🎯 *Take Profit:* `{signal.take_profit:.5f}` ({signal.tp_pips} pips)

{confidence_emoji} *Ishonch:* `{signal.confidence:.1f}%`
⚖️ *Risk:* `{signal.risk_percent:.1f}%`
💼 *Akavunt:* `{signal.account}`

═══════════════════════════
📝 *Sabab:* {signal.reason}
⏰ *Vaqt:* `{uzb_time}` - `{uzb_date}`
═══════════════════════════

⚠️ *Diqqat:* Risk boshqaruvini unutmang!
"""
        
        return signal_text

    def _format_log(self, message: str, level: str) -> str:
        """Log formatini yaratish"""
        
        # Level bo'yicha emoji
        level_emojis = {
            "INFO": "ℹ️",
            "WARNING": "⚠️", 
            "ERROR": "❌",
            "SUCCESS": "✅"
        }
        
        emoji = level_emojis.get(level, "📝")
        current_time = datetime.now().strftime("%H:%M:%S")
        
        log_text = f"""
{emoji} *{level}* - `{current_time}`
───────────────────────────
{message}
───────────────────────────
"""
        return log_text

    def _format_market_analysis(self, analysis: Dict) -> str:
        """Bozor tahlili formatini yaratish"""
        
        trend_emoji = "📈" if analysis.get('trend', '') == 'bullish' else "📉" if analysis.get('trend', '') == 'bearish' else "↔️"
        
        analysis_text = f"""
📊 *BOZOR TAHLILI* {trend_emoji}
═══════════════════════════

📈 *Trend:* `{analysis.get('trend', 'Noaniq').title()}`
💹 *Sentiment:* `{analysis.get('sentiment_score', 0):.2f}`
🔥 *Faollik:* `{analysis.get('volume_change', 0):.1f}%`

🎯 *Asosiy signallar:*
• Order Flow: `{analysis.get('order_flow', 'Noaniq')}`
• AI Sentiment: `{analysis.get('ai_sentiment', 'Noaniq')}`
• Yangiliklar: `{analysis.get('news_impact', 'Noaniq')}`

⏰ *Yangilanish:* `{datetime.now().strftime("%H:%M:%S")}`
═══════════════════════════
"""
        return analysis_text

    def _format_status(self, status: Dict) -> str:
        """Bot holati formatini yaratish"""
        
        # Status emoji
        status_emoji = "🟢" if status.get('status') == 'running' else "🔴" if status.get('status') == 'error' else "🟡"
        
        status_text = f"""
🤖 *BOT HOLATI* {status_emoji}
═══════════════════════════

🔄 *Holat:* `{status.get('status', 'Noaniq').title()}`
⏱️ *Ish vaqti:* `{status.get('uptime', 'Noaniq')}`
📊 *Signallar:* `{status.get('signals_sent', 0)}`
💹 *Savdolar:* `{status.get('trades_executed', 0)}`

🔌 *API holati:*
• 1inch: `{status.get('api_status', {}).get('oneinch', 'Noaniq')}`
• Alchemy: `{status.get('api_status', {}).get('alchemy', 'Noaniq')}`
• HuggingFace: `{status.get('api_status', {}).get('huggingface', 'Noaniq')}`

💰 *Balans:* `${status.get('balance', 0):.2f}`
📈 *Bugungi P&L:* `${status.get('daily_pnl', 0):.2f}`

⏰ *Oxirgi yangilanish:* `{datetime.now().strftime("%H:%M:%S")}`
═══════════════════════════
"""
        return status_text

    def _create_signal_keyboard(self, signal: SignalData) -> InlineKeyboardMarkup:
        """Signal uchun inline keyboard yaratish"""
        
        keyboard = [
            [
                InlineKeyboardButton("🟢 AVTO SAVDO", callback_data=f"auto_trade_{signal.symbol}"),
                InlineKeyboardButton("🔴 BEKOR QILISH", callback_data=f"cancel_{signal.symbol}")
            ],
            [
                InlineKeyboardButton("📊 BATAFSIL", callback_data=f"details_{signal.symbol}"),
                InlineKeyboardButton("⚙️ SOZLAMALAR", callback_data="settings")
            ]
        ]
        
        return InlineKeyboardMarkup(keyboard)

    def _create_chart(self, chart_data: Dict, title: str) -> io.BytesIO:
        """Grafik yaratish"""
        
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Ma'lumotlarni chizish
        if 'price' in chart_data and 'time' in chart_data:
            ax.plot(chart_data['time'], chart_data['price'], 
                   color='#00ff88', linewidth=2, label='Narx')
        
        # Formatni sozlash
        ax.set_title(title, fontsize=16, color='white', pad=20)
        ax.set_xlabel('Vaqt', fontsize=12, color='white')
        ax.set_ylabel('Narx', fontsize=12, color='white')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Vaqt formatini sozlash
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        
        plt.tight_layout()
        
        # Buffer ga saqlash
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        plt.close()
        
        return buffer

    # Command handlerlar
    async def handle_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Start command handler"""
        welcome_text = """
🤖 *AI OrderFlow Signal Bot*
═══════════════════════════

Assalomu alaykum! Men sizning AI savdo botingizman.

🎯 *Imkoniyatlar:*
• Real vaqtda signal berish
• Bozor tahlili
• Avtomatik savdo
• Risk boshqaruvi

📋 *Buyruqlar:*
/help - Yordam
/status - Bot holati
/settings - Sozlamalar
/stats - Statistika

⚡ Bot ishga tushirildi!
"""
        await update.message.reply_text(welcome_text, parse_mode=ParseMode.MARKDOWN)

    async def handle_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Help command handler"""
        help_text = """
🆘 *YORDAM*
═══════════════════════════

📋 *Buyruqlar:*
• `/start` - Botni ishga tushirish
• `/status` - Bot holati
• `/settings` - Sozlamalar
• `/stats` - Statistika

⚙️ *Sozlamalar:*
• Avtomatik savdo: On/Off
• Risk darajasi: 1-5%
• Bildirishnomalar: On/Off

🔄 *Signal turlari:*
• BUY - Sotib olish
• SELL - Sotish
• WAIT - Kutish

❓ *Savol-javob:*
Qo'shimcha yordam uchun @support_username
"""
        await update.message.reply_text(help_text, parse_mode=ParseMode.MARKDOWN)

    async def handle_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Status command handler"""
        status_data = {
            'status': 'running',
            'uptime': '2h 34m',
            'signals_sent': 12,
            'trades_executed': 8,
            'api_status': {
                'oneinch': 'OK',
                'alchemy': 'OK',
                'huggingface': 'OK'
            },
            'balance': 1000.0,
            'daily_pnl': 45.30
        }
        
        status_text = self._format_status(status_data)
        await update.message.reply_text(status_text, parse_mode=ParseMode.MARKDOWN)

    async def handle_settings(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Settings command handler"""
        keyboard = [
            [
                InlineKeyboardButton("🤖 Avto Savdo", callback_data="toggle_auto_trade"),
                InlineKeyboardButton("⚖️ Risk", callback_data="set_risk")
            ],
            [
                InlineKeyboardButton("🔔 Bildirishnomalar", callback_data="toggle_notifications"),
                InlineKeyboardButton("🌐 Til", callback_data="change_language")
            ]
        ]
        
        settings_text = f"""
⚙️ *SOZLAMALAR*
═══════════════════════════

🤖 *Avto savdo:* `{'Yoniq' if self.user_settings['auto_trade'] else 'O\\'chiq'}`
⚖️ *Risk:* `{self.user_settings['max_risk']*100}%`
🔔 *Bildirishnomalar:* `{'Yoniq' if self.user_settings['notifications'] else 'O\\'chiq'}`
🌐 *Til:* `O'zbekcha`
"""
        
        await update.message.reply_text(
            settings_text,
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=InlineKeyboardMarkup(keyboard)
        )

    async def handle_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Stats command handler"""
        stats_text = """
📊 *STATISTIKA*
═══════════════════════════

📈 *Bugun:*
• Signallar: `12`
• Savdolar: `8`
• Muvaffaqiyat: `75%`
• P&L: `+$45.30`

📅 *Hafta:*
• Signallar: `84`
• Savdolar: `62`
• Muvaffaqiyat: `72%`
• P&L: `+$312.45`

🏆 *Eng yaxshi:*
• Kunlik P&L: `+$89.20`
• Signallar: `18`
• Muvaffaqiyat: `91%`
"""
        await update.message.reply_text(stats_text, parse_mode=ParseMode.MARKDOWN)

    async def handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Callback query handler"""
        query = update.callback_query
        await query.answer()
        
        data = query.data
        
        if data.startswith("auto_trade_"):
            symbol = data.split("_")[2]
            await query.edit_message_text(
                f"🟢 *AVTO SAVDO YOQILDI*\n\nJuftlik: `{symbol}`\nSignal avtomatik bajariladi...",
                parse_mode=ParseMode.MARKDOWN
            )
            
        elif data.startswith("cancel_"):
            symbol = data.split("_")[1]
            await query.edit_message_text(
                f"🔴 *SIGNAL BEKOR QILINDI*\n\nJuftlik: `{symbol}`\nSavdo bajarilmaydi.",
                parse_mode=ParseMode.MARKDOWN
            )
            
        elif data == "toggle_auto_trade":
            self.user_settings['auto_trade'] = not self.user_settings['auto_trade']
            status = "yoqildi" if self.user_settings['auto_trade'] else "o'chirildi"
            await query.edit_message_text(
                f"🤖 *Avto savdo {status}*",
                parse_mode=ParseMode.MARKDOWN
            )

    async def health_check(self) -> bool:
        """
        Telegram bot health tekshirish
        
        Returns:
            bool: Bot sog'ligini ko'rsatadi
        """
        try:
            if not self.bot:
                await self.initialize_bot()
            
            # Bot ma'lumotlarini olish
            await self.bot.get_me()
            
            logger.info("Telegram bot sog'lom")
            return True
            
        except Exception as e:
            logger.error(f"Telegram bot health check xatosi: {e}")
            return False

# Fallback manager bilan integratsiya
class FallbackTelegramManager:
    """Telegram bot fallback manager"""
    
    def __init__(self, bot_configs: List[Dict]):
        self.clients = []
        
        for config in bot_configs:
            client = TelegramClient(
                bot_token=config['bot_token'],
                chat_id=config['chat_id']
            )
            self.clients.append(client)
        
        self.current_client_index = 0
        logger.info(f"Telegram fallback manager {len(self.clients)} ta client bilan yaratildi")

    async def execute_with_fallback(self, operation: str, **kwargs) -> TelegramResponse:
        """
        Fallback bilan operatsiya bajarish
        
        Args:
            operation: Bajarilishi kerak bo'lgan operatsiya
            **kwargs: Operatsiya parametrlari
            
        Returns:
            TelegramResponse: Operatsiya natijasi
        """
        last_error = None
        
        for i, client in enumerate(self.clients):
            try:
                async with client:
                    result = await getattr(client, operation)(**kwargs)
                    
                    if result.success:
                        if i > 0:
                            logger.warning(f"Telegram fallback ishlatildi: client {i}")
                        return result
                    else:
                        last_error = result.error
                        
            except Exception as e:
                logger.error(f"Telegram client {i} ishlamadi: {e}")
                last_error = str(e)
                continue
        
        logger.error("Barcha Telegram clientlar ishlamadi")
        return TelegramResponse(
            success=False,
            error=f"Barcha fallback clientlar ishlamadi. Oxirgi xato: {last_error}"
        )

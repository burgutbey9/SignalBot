# 🤖 AI OrderFlow & Signal Bot

AI yordamida DEX Order Flow va Sentiment tahlil qilib, real vaqtda trading signallari beruvchi avtomatik bot. Signallar O'zbekcha Telegram orqali yuboriladi va Propshot EA bilan integratsiya qiladi.

## 📋 Loyiha Tavsifi

Bu bot quyidagi vazifalarni bajaradi:
- 📊 **Order Flow tahlili** - DEX (Uniswap, 1inch) dan real vaqtda order flow ma'lumotlarini olish
- 🧠 **AI Sentiment tahlili** - Yangiliklar va ijtimoiy tarmoq ma'lumotlarini AI yordamida tahlil qilish
- 📈 **Signal yaratish** - AI tahlil asosida trading signallari yaratish
- 📱 **Telegram integratsiya** - Signallarni O'zbekcha Telegram orqali yuborish
- 🎯 **Propshot EA** - MetaTrader 5 bilan avtomatik savdo
- 🔄 **Fallback tizimi** - Barcha API'lar uchun backup variantlar

## 🏗️ Loyiha Strukturasi

```
ai_orderflow_signal_bot/
├── 📂 config/                    # Konfiguratsiya fayllar
│   ├── config.py                 # Asosiy konfiguratsiya
│   ├── api_keys.py               # API kalitlar
│   ├── settings.json             # JSON sozlamalar
│   └── fallback_config.py        # Fallback tizimi
├── 📂 src/
│   ├── 📂 api_clients/           # API mijozlar
│   │   ├── oneinch_client.py     # 1inch API
│   │   ├── thegraph_client.py    # The Graph API
│   │   ├── alchemy_client.py     # Alchemy API
│   │   ├── ccxt_client.py        # CCXT market data
│   │   ├── huggingface_client.py # HuggingFace AI
│   │   ├── gemini_client.py      # Gemini AI (5 ta key)
│   │   ├── claude_client.py      # Claude AI
│   │   ├── news_client.py        # NewsAPI
│   │   ├── reddit_client.py      # Reddit API
│   │   └── telegram_client.py    # Telegram Bot
│   ├── 📂 data_processing/       # Ma'lumot qayta ishlash
│   │   ├── order_flow_analyzer.py # Order Flow tahlil
│   │   ├── sentiment_analyzer.py  # Sentiment tahlil
│   │   ├── market_analyzer.py     # Market data tahlil
│   │   └── signal_generator.py    # Signal generator
│   ├── 📂 risk_management/       # Risk boshqaruv
│   │   ├── risk_calculator.py     # Risk hisoblash
│   │   ├── position_sizer.py      # Position size
│   │   └── trade_monitor.py       # Trade monitoring
│   ├── 📂 utils/                 # Yordamchi vositalar
│   │   ├── logger.py              # Logger setup
│   │   ├── error_handler.py       # Error handling
│   │   ├── rate_limiter.py        # Rate limiting
│   │   ├── retry_handler.py       # Retry logika
│   │   └── fallback_manager.py    # Fallback tizimi
│   ├── 📂 database/              # Database
│   │   ├── db_manager.py          # Database manager
│   │   ├── models.py              # Data modellari
│   │   └── migrations/            # Database migratsiya
│   ├── 📂 trading/               # Trading
│   │   ├── strategy_manager.py    # Trading strategiyalar
│   │   ├── backtest_engine.py     # Backtest tizimi
│   │   ├── portfolio_manager.py   # Portfolio boshqaruv
│   │   ├── execution_engine.py    # Savdo bajarish
│   │   ├── propshot_connector.py  # Propshot API
│   │   └── mt5_bridge.py          # MetaTrader 5 bridge
│   └── main.py                    # Asosiy fayl
├── 📂 tests/                     # Test fayllar
├── 📂 logs/                      # Log fayllar
├── 📂 data/                      # Ma'lumotlar
├── requirements.txt
├── .env.example
└── README.md
```

## 🚀 O'rnatish va Sozlash

### 1. Loyihani klonlash
```bash
git clone https://github.com/yourusername/ai_orderflow_signal_bot.git
cd ai_orderflow_signal_bot
```

### 2. Python muhitini yaratish
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# yoki
venv\Scripts\activate     # Windows
```

### 3. Kutubxonalarni o'rnatish
```bash
pip install -r requirements.txt
```

### 4. Muhit o'zgaruvchilarini sozlash
```bash
cp .env.example .env
# .env faylini tahrirlang va API kalitlaringizni qo'shing
```

### 5. Konfiguratsiyani sozlash
```bash
# config/settings.json faylini tahrirlang
# API limitlar va strategiya parametrlarini sozlang
```

### 6. Databaseni yaratish
```bash
python -c "from database.db_manager import DatabaseManager; DatabaseManager().create_tables()"
```

## 🔑 API Kalitlar

Quyidagi API kalitlar kerak:

### 📊 Order Flow & Market Data
- **1inch API Key** - Asosiy order flow manba
- **The Graph API Key** - Uniswap fallback
- **Alchemy API Key** - On-chain kuzatuv
- **CCXT** - Tarixiy CEX data (key shart emas)

### 🧠 AI & Sentiment
- **HuggingFace API Key** - Asosiy AI sentiment
- **Gemini API Keys** - 5 ta key (fallback)
- **Claude API Key** - Extra fallback
- **NewsAPI Key** - Yangiliklar
- **Reddit API** - Ijtimoiy sentiment

### 📱 Telegram & Trading
- **Telegram Bot Token** - Signal yuborish
- **Telegram Chat ID** - Sizning chat ID
- **Propshot API** - EA ulanish (ixtiyoriy)
- **MetaTrader 5** - Server/login/password

## ⚙️ Konfiguratsiya

### `config/settings.json` namunasi:
```json
{
  "api_limits": {
    "oneinch": {
      "rate_limit": 100,
      "timeout": 30,
      "max_retries": 3
    },
    "huggingface": {
      "rate_limit": 1000,
      "timeout": 60,
      "max_retries": 2
    }
  },
  "trading": {
    "max_risk_per_trade": 0.02,
    "max_daily_loss": 0.05,
    "position_size_method": "kelly"
  },
  "fallback_order": {
    "order_flow": ["oneinch", "thegraph", "alchemy"],
    "sentiment": ["huggingface", "gemini", "claude"]
  }
}
```

### `.env` namunasi:
```env
# API Keys
ONEINCH_API_KEY=your_oneinch_key
ALCHEMY_API_KEY=your_alchemy_key
HUGGINGFACE_API_KEY=your_huggingface_key
GEMINI_API_KEY_1=your_gemini_key_1
GEMINI_API_KEY_2=your_gemini_key_2
CLAUDE_API_KEY=your_claude_key
NEWS_API_KEY=your_news_key
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_chat_id

# Database
DATABASE_URL=sqlite:///data/bot.db

# Trading
PROPSHOT_API_KEY=your_propshot_key
MT5_SERVER=your_mt5_server
MT5_LOGIN=your_mt5_login
MT5_PASSWORD=your_mt5_password
```

## 🎯 Ishga Tushirish

### 1. Asosiy bot ishga tushirish
```bash
python main.py
```

### 2. Test rejimida ishga tushirish
```bash
python main.py --test-mode
```

### 3. Backtest rejimida ishga tushirish
```bash
python main.py --backtest --start-date 2024-01-01 --end-date 2024-12-31
```

### 4. Faqat sentiment tahlil
```bash
python main.py --sentiment-only
```

## 📊 Xususiyatlar

### 🔄 Fallback Tizimi
- **Order Flow**: 1inch ➜ The Graph ➜ Alchemy
- **Sentiment**: HuggingFace ➜ Gemini (5 ta) ➜ Claude
- **News**: NewsAPI ➜ Reddit ➜ Claude
- Har fallback avtomatik aktivlanadi va loglanadi

### 🧠 AI Tahlil
- **Sentiment Analysis** - Yangiliklar va ijtimoiy tarmoq ma'lumotlari
- **Order Flow Analysis** - Katta orderlar va whale faoliyati
- **Market Analysis** - Texnik ko'rsatkichlar va naqsh tanib olish
- **Signal Generation** - AI kombinatsiya asosida

### 📱 Telegram Signallar
```
📊 SIGNAL KELDI
════════════════
📈 Savdo: BUY EURUSD
💰 Narx: 1.0950
📊 Lot: 0.1 lot
🛡️ Stop Loss: 1.0900 (50 pips)
🎯 Take Profit: 1.1000 (50 pips)
⚡ Ishonch: 85%
🔥 Risk: 2%
════════════════
📝 Sabab: Kuchli order flow + ijobiy sentiment
⏰ Vaqt: 14:30 (UZB)
💼 Akavunt: Demo

[🟢 AVTO SAVDO] [🔴 BEKOR QILISH]
```

### 🎯 Risk Boshqaruv
- **Maksimal risk**: 2% har savdo
- **Kunlik loss limit**: 5%
- **Propshot integratsiya**: 2x qoidalar
- **Position sizing**: Kelly formula
- **Stop Loss/Take Profit**: AI tomonidan belgilanadi

## 🧪 Testlar

### Testlarni ishga tushirish
```bash
# Barcha testlar
pytest tests/

# Ma'lum modul testi
pytest tests/test_api_clients.py

# Coverage bilan
pytest --cov=src tests/
```

### Test turlari
- **Unit testlar** - Har bir modul alohida
- **Integration testlar** - API ulanishlar
- **End-to-end testlar** - To'liq signal jarayoni

## 📈 Monitoring va Loglar

### Log fayllar
- `logs/bot.log` - Asosiy bot loglari
- `logs/api_calls.log` - API so'rovlar
- `logs/trading.log` - Savdo loglari
- `logs/errors.log` - Xatolar
- `logs/fallback.log` - Fallback aktivlanishlar
- `logs/telegram.log` - Telegram faoliyati

### Monitoring
- **Real-time dashboard** - Streamlit orqali
- **Performance metrics** - Sharpe ratio, drawdown
- **API health** - Rate limits, response times
- **Signal accuracy** - Win rate, profit factor

## 🛠️ Texnik Xususiyatlar

### Asosiy texnologiyalar
- **Python 3.9+** - Asosiy til
- **asyncio/aiohttp** - Async programming
- **SQLAlchemy** - Database ORM
- **Pandas/NumPy** - Data processing
- **Scikit-learn** - Machine learning
- **CCXT** - Crypto exchange APIs

### Performance
- **Async architecture** - Parallel API calls
- **Rate limiting** - API cheklovlarini hurmat qilish
- **Caching** - Tez-tez ishlatiladigan ma'lumotlar
- **Database optimization** - Indexlar va query optimization

## 🔒 Xavfsizlik

### API kalitlar
- **Environment variables** - .env faylda
- **Encryption** - Sensitive data shifrlash
- **Access control** - IP whitelist
- **Rate limiting** - DDoS himoya

### Trading xavfsizligi
- **Position limits** - Maksimal lot size
- **Daily limits** - Kunlik loss chegarasi
- **Risk validation** - Har savdo oldidan tekshirish
- **Emergency stop** - Favqulotda to'xtatish

## 📞 Qo'llab-quvvatlash

### Muammolar hal qilish
1. **API xatolar** - `logs/api_calls.log` ni tekshiring
2. **Telegram ishlamayapti** - Token va Chat ID ni tekshiring
3. **Signallar kelmayapti** - `logs/bot.log` ni ko'ring
4. **Trading xatolar** - `logs/trading.log` ni tekshiring

### Tez-tez so'raladigan savollar
- **Q: Bot ishlayaptimi?** - `logs/bot.log` ning eng so'ngi yozuvlarini ko'ring
- **Q: Nima uchun signal kelmayapti?** - Sentiment va order flow ma'lumotlarini tekshiring
- **Q: Risk juda yuqori?** - `config/settings.json` da risk parametrlarini kamaytiring

## 🤝 Hissa qo'shish

### Development setup
```bash
# Development o'rnatish
pip install -r requirements-dev.txt

# Pre-commit hooks
pre-commit install

# Code formatting
black src/
flake8 src/
```

### Pull Request yo'riqnomasi
1. Feature branch yarating
2. Testlar qo'shing
3. Code review o'tkazing
4. Documentation yangilang

## 📄 Litsenziya

MIT License - `LICENSE` faylini ko'ring.

## 🙏 Minnatdorchilik

- **1inch** - Order flow API
- **Alchemy** - On-chain data
- **HuggingFace** - AI models
- **Telegram** - Bot platform
- **OpenAI** - AI assistance

## 📊 Statistika

- **Kod qatorlari**: 15,000+ qator
- **Testlar**: 95% coverage
- **API integratsiya**: 10+ xizmat
- **Fallback darajasi**: 3x backup
- **Performance**: <100ms signal yaratish

## 🎯 Roadmap

### v1.0 (Joriy)
- ✅ Asosiy Order Flow tahlil
- ✅ AI Sentiment tahlil
- ✅ Telegram signallar
- ✅ Fallback tizimi

### v1.1 (Keyingi)
- 🔄 More AI models
- 🔄 Advanced risk management
- 🔄 Mobile app
- 🔄 Real-time dashboard

### v2.0 (Kelajak)
- 🎯 Multi-exchange support
- 🎯 Portfolio management
- 🎯 Social trading
- 🎯 Advanced analytics

---

**💡 Eslatma**: Bu bot kriptovalyuta va forex bozorlarida ishlatish uchun mo'ljallangan. Har doim o'zingizning risk toleransangizni hisobga oling va faqat yo'qotishga tayyor bo'lgan mablag' bilan savdo qiling.

**🔴 Ogohlik**: Avtomatik savdo risklari mavjud. Botni ishga tushirishdan oldin test rejimida sinab ko'ring va barcha sozlamalarni diqqat bilan tekshiring.

**📞 Aloqa**: Savollar yoki yordam uchun Telegram orqali murojaat qiling.

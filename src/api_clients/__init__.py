"""
API Clients moduli - Barcha tashqi API lar bilan ishlash uchun
"""

from .oneinch_client import OneInchClient
from .thegraph_client import TheGraphClient
from .alchemy_client import AlchemyClient
from .ccxt_client import CCXTClient
from .huggingface_client import HuggingFaceClient
from .gemini_client import GeminiClient
from .claude_client import ClaudeClient
from .news_client import NewsClient
from .reddit_client import RedditClient
from .telegram_client import TelegramClient

__all__ = [
    'OneInchClient',
    'TheGraphClient', 
    'AlchemyClient',
    'CCXTClient',
    'HuggingFaceClient',
    'GeminiClient',
    'ClaudeClient',
    'NewsClient',
    'RedditClient',
    'TelegramClient'
]

# API clientlar versiyasi
__version__ = "1.0.0"

# Barcha API clientlar ro'yxati
API_CLIENTS = {
    'orderflow': ['oneinch', 'thegraph', 'alchemy'],
    'sentiment': ['huggingface', 'gemini', 'claude'],
    'news': ['newsapi', 'reddit', 'claude'],
    'trading': ['ccxt', 'propshot', 'mt5'],
    'communication': ['telegram']
}

# Fallback ketma-ketligi
FALLBACK_ORDER = {
    'order_flow': ['OneInchClient', 'TheGraphClient', 'AlchemyClient'],
    'sentiment': ['HuggingFaceClient', 'GeminiClient', 'ClaudeClient'],
    'news': ['NewsClient', 'RedditClient', 'ClaudeClient']
}

def get_client_by_name(client_name: str):
    """Client nomiga ko'ra client classini qaytarish"""
    clients_map = {
        'oneinch': OneInchClient,
        'thegraph': TheGraphClient,
        'alchemy': AlchemyClient,
        'ccxt': CCXTClient,
        'huggingface': HuggingFaceClient,
        'gemini': GeminiClient,
        'claude': ClaudeClient,
        'news': NewsClient,
        'reddit': RedditClient,
        'telegram': TelegramClient
    }
    return clients_map.get(client_name.lower())

def get_fallback_clients(service_type: str):
    """Xizmat turi bo'yicha fallback clientlarini olish"""
    return FALLBACK_ORDER.get(service_type, [])

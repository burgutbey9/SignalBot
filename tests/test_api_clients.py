import pytest
import asyncio
import aiohttp
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any, Optional
import json
import time
from dataclasses import dataclass

# Test uchun imports
from api_clients.oneinch_client import OneInchClient
from api_clients.telegram_client import TelegramClient
from api_clients.huggingface_client import HuggingFaceClient
from api_clients.alchemy_client import AlchemyClient
from api_clients.gemini_client import GeminiClient
from api_clients.thegraph_client import TheGraphClient
from api_clients.news_client import NewsClient
from api_clients.reddit_client import RedditClient
from utils.rate_limiter import RateLimiter
from utils.fallback_manager import FallbackManager


@dataclass
class MockAPIResponse:
    """Mock API response class - API javob sinfi"""
    success: bool
    data: Optional[Dict] = None
    error: Optional[str] = None
    status_code: int = 200
    rate_limit_remaining: int = 100


class TestOneInchClient:
    """1inch API client testlari"""
    
    @pytest.fixture
    def oneinch_client(self):
        """1inch client fixture - test uchun mijoz"""
        return OneInchClient(
            api_key="test_api_key",
            base_url="https://api.1inch.dev"
        )
    
    @pytest.fixture
    def mock_response_data(self):
        """Mock response data - test javob ma'lumotlari"""
        return {
            "tokens": {
                "USDT": {
                    "symbol": "USDT",
                    "name": "Tether",
                    "address": "0xdac17f958d2ee523a2206206994597c13d831ec7",
                    "decimals": 6
                }
            }
        }
    
    @pytest.mark.asyncio
    async def test_get_tokens_success(self, oneinch_client, mock_response_data):
        """Token olish muvaffaqiyatli testi"""
        with patch.object(oneinch_client, 'make_request') as mock_request:
            mock_request.return_value = MockAPIResponse(
                success=True,
                data=mock_response_data
            )
            
            result = await oneinch_client.get_tokens()
            
            assert result.success is True
            assert "tokens" in result.data
            assert mock_request.called
    
    @pytest.mark.asyncio
    async def test_get_tokens_failure(self, oneinch_client):
        """Token olish xato testi"""
        with patch.object(oneinch_client, 'make_request') as mock_request:
            mock_request.return_value = MockAPIResponse(
                success=False,
                error="API xatosi"
            )
            
            result = await oneinch_client.get_tokens()
            
            assert result.success is False
            assert result.error == "API xatosi"
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, oneinch_client):
        """Rate limiting testi"""
        with patch.object(oneinch_client.rate_limiter, 'wait') as mock_wait:
            mock_wait.return_value = None
            
            with patch.object(oneinch_client, 'make_request') as mock_request:
                mock_request.return_value = MockAPIResponse(success=True)
                
                await oneinch_client.get_tokens()
                
                mock_wait.assert_called_once()


class TestTelegramClient:
    """Telegram API client testlari"""
    
    @pytest.fixture
    def telegram_client(self):
        """Telegram client fixture"""
        return TelegramClient(
            bot_token="test_bot_token",
            chat_id="test_chat_id"
        )
    
    @pytest.fixture
    def mock_signal_data(self):
        """Mock signal data - test signal ma'lumotlari"""
        return {
            "action": "BUY",
            "symbol": "EURUSD",
            "price": 1.0850,
            "lot_size": 0.1,
            "stop_loss": 1.0800,
            "take_profit": 1.0900,
            "confidence": 85.5,
            "risk_percent": 2.0,
            "reason": "Strong bullish momentum",
            "time": "2024-01-15 10:30:00",
            "account": "Demo"
        }
    
    @pytest.mark.asyncio
    async def test_send_signal_success(self, telegram_client, mock_signal_data):
        """Signal yuborish muvaffaqiyatli testi"""
        with patch.object(telegram_client, 'send_message') as mock_send:
            mock_send.return_value = MockAPIResponse(success=True)
            
            result = await telegram_client.send_signal(mock_signal_data)
            
            assert result.success is True
            assert mock_send.called
    
    @pytest.mark.asyncio
    async def test_send_signal_failure(self, telegram_client, mock_signal_data):
        """Signal yuborish xato testi"""
        with patch.object(telegram_client, 'send_message') as mock_send:
            mock_send.return_value = MockAPIResponse(
                success=False,
                error="Telegram API xatosi"
            )
            
            result = await telegram_client.send_signal(mock_signal_data)
            
            assert result.success is False
            assert "Telegram API xatosi" in result.error
    
    @pytest.mark.asyncio
    async def test_format_signal_message(self, telegram_client, mock_signal_data):
        """Signal formatini tekshirish testi"""
        formatted_message = telegram_client.format_signal_message(mock_signal_data)
        
        assert "SIGNAL KELDI" in formatted_message
        assert "EURUSD" in formatted_message
        assert "BUY" in formatted_message
        assert "85.5%" in formatted_message
        assert "2.0%" in formatted_message
    
    @pytest.mark.asyncio
    async def test_send_log_message(self, telegram_client):
        """Log xabar yuborish testi"""
        log_message = "API fallback ishga tushirildi"
        
        with patch.object(telegram_client, 'send_message') as mock_send:
            mock_send.return_value = MockAPIResponse(success=True)
            
            result = await telegram_client.send_log_message(log_message)
            
            assert result.success is True
            assert mock_send.called


class TestHuggingFaceClient:
    """HuggingFace API client testlari"""
    
    @pytest.fixture
    def huggingface_client(self):
        """HuggingFace client fixture"""
        return HuggingFaceClient(
            api_key="test_hf_key",
            model_name="cardiffnlp/twitter-roberta-base-sentiment-latest"
        )
    
    @pytest.fixture
    def mock_sentiment_data(self):
        """Mock sentiment data - test sentiment ma'lumotlari"""
        return [
            {
                "label": "POSITIVE",
                "score": 0.8234
            },
            {
                "label": "NEGATIVE",
                "score": 0.1234
            },
            {
                "label": "NEUTRAL",
                "score": 0.0532
            }
        ]
    
    @pytest.mark.asyncio
    async def test_analyze_sentiment_success(self, huggingface_client, mock_sentiment_data):
        """Sentiment tahlil muvaffaqiyatli testi"""
        test_text = "Bitcoin price is going up! Great news for crypto market."
        
        with patch.object(huggingface_client, 'make_request') as mock_request:
            mock_request.return_value = MockAPIResponse(
                success=True,
                data=mock_sentiment_data
            )
            
            result = await huggingface_client.analyze_sentiment(test_text)
            
            assert result.success is True
            assert result.data[0]["label"] == "POSITIVE"
            assert result.data[0]["score"] > 0.8
    
    @pytest.mark.asyncio
    async def test_analyze_sentiment_failure(self, huggingface_client):
        """Sentiment tahlil xato testi"""
        test_text = "Test text"
        
        with patch.object(huggingface_client, 'make_request') as mock_request:
            mock_request.return_value = MockAPIResponse(
                success=False,
                error="HuggingFace API xatosi"
            )
            
            result = await huggingface_client.analyze_sentiment(test_text)
            
            assert result.success is False
            assert "HuggingFace API xatosi" in result.error
    
    @pytest.mark.asyncio
    async def test_process_sentiment_result(self, huggingface_client, mock_sentiment_data):
        """Sentiment natijasini qayta ishlash testi"""
        processed = huggingface_client.process_sentiment_result(mock_sentiment_data)
        
        assert processed["dominant_sentiment"] == "POSITIVE"
        assert processed["confidence"] > 0.8
        assert processed["sentiment_score"] > 0.5


class TestAlchemyClient:
    """Alchemy API client testlari"""
    
    @pytest.fixture
    def alchemy_client(self):
        """Alchemy client fixture"""
        return AlchemyClient(
            api_key="test_alchemy_key",
            network="eth-mainnet"
        )
    
    @pytest.mark.asyncio
    async def test_get_latest_block_success(self, alchemy_client):
        """Eng so'nggi blok olish muvaffaqiyatli testi"""
        mock_block_data = {
            "number": "0x1234567",
            "timestamp": "0x635d5c5f",
            "transactions": ["0xabc123", "0xdef456"]
        }
        
        with patch.object(alchemy_client, 'make_request') as mock_request:
            mock_request.return_value = MockAPIResponse(
                success=True,
                data=mock_block_data
            )
            
            result = await alchemy_client.get_latest_block()
            
            assert result.success is True
            assert "number" in result.data
            assert "timestamp" in result.data
    
    @pytest.mark.asyncio
    async def test_get_transaction_logs(self, alchemy_client):
        """Tranzaksiya loglarini olish testi"""
        tx_hash = "0x123456789abcdef"
        
        mock_logs_data = {
            "logs": [
                {
                    "address": "0x123...",
                    "topics": ["0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"],
                    "data": "0x000000000000000000000000000000000000000000000000000000000000000a"
                }
            ]
        }
        
        with patch.object(alchemy_client, 'make_request') as mock_request:
            mock_request.return_value = MockAPIResponse(
                success=True,
                data=mock_logs_data
            )
            
            result = await alchemy_client.get_transaction_logs(tx_hash)
            
            assert result.success is True
            assert "logs" in result.data
            assert len(result.data["logs"]) > 0


class TestRateLimiting:
    """Rate limiting testlari"""
    
    @pytest.fixture
    def rate_limiter(self):
        """Rate limiter fixture"""
        return RateLimiter(calls=5, period=60)
    
    @pytest.mark.asyncio
    async def test_rate_limiter_within_limit(self, rate_limiter):
        """Rate limit ichida bo'lish testi"""
        start_time = time.time()
        
        for i in range(3):
            await rate_limiter.wait()
        
        end_time = time.time()
        duration = end_time - start_time
        
        # 3 ta so'rov rate limit ichida bo'lishi kerak
        assert duration < 1.0
    
    @pytest.mark.asyncio
    async def test_rate_limiter_exceed_limit(self, rate_limiter):
        """Rate limitni oshirish testi"""
        # Rate limitni to'ldirish
        for i in range(5):
            await rate_limiter.wait()
        
        # Keyingi so'rov kutish vaqtini tekshirish
        start_time = time.time()
        await rate_limiter.wait()
        end_time = time.time()
        
        duration = end_time - start_time
        # Kutish vaqti bo'lishi kerak
        assert duration > 0.1


class TestFallbackSystem:
    """Fallback tizimi testlari"""
    
    @pytest.fixture
    def fallback_manager(self):
        """Fallback manager fixture"""
        return FallbackManager()
    
    @pytest.fixture
    def mock_api_clients(self):
        """Mock API clients - test API mijozlari"""
        primary_client = Mock()
        secondary_client = Mock()
        tertiary_client = Mock()
        
        return [primary_client, secondary_client, tertiary_client]
    
    @pytest.mark.asyncio
    async def test_fallback_primary_success(self, fallback_manager, mock_api_clients):
        """Birinchi API muvaffaqiyatli testi"""
        primary_client = mock_api_clients[0]
        primary_client.make_request.return_value = MockAPIResponse(success=True)
        
        result = await fallback_manager.execute_with_fallback(
            mock_api_clients,
            "make_request",
            endpoint="/test"
        )
        
        assert result.success is True
        # Faqat birinchi client ishlatilishi kerak
        assert primary_client.make_request.called
    
    @pytest.mark.asyncio
    async def test_fallback_primary_fail_secondary_success(self, fallback_manager, mock_api_clients):
        """Birinchi API xato, ikkinchi muvaffaqiyatli testi"""
        primary_client = mock_api_clients[0]
        secondary_client = mock_api_clients[1]
        
        primary_client.make_request.return_value = MockAPIResponse(
            success=False,
            error="Primary API xatosi"
        )
        secondary_client.make_request.return_value = MockAPIResponse(success=True)
        
        result = await fallback_manager.execute_with_fallback(
            mock_api_clients,
            "make_request",
            endpoint="/test"
        )
        
        assert result.success is True
        assert primary_client.make_request.called
        assert secondary_client.make_request.called
    
    @pytest.mark.asyncio
    async def test_fallback_all_fail(self, fallback_manager, mock_api_clients):
        """Barcha API xato testi"""
        for client in mock_api_clients:
            client.make_request.return_value = MockAPIResponse(
                success=False,
                error="API xatosi"
            )
        
        result = await fallback_manager.execute_with_fallback(
            mock_api_clients,
            "make_request",
            endpoint="/test"
        )
        
        assert result.success is False
        assert "Barcha fallback clientlar ishlamadi" in result.error
        
        # Barcha clientlar ishlatilishi kerak
        for client in mock_api_clients:
            assert client.make_request.called


class TestErrorHandling:
    """Error handling testlari"""
    
    @pytest.fixture
    def api_client(self):
        """Generic API client fixture"""
        return OneInchClient(
            api_key="test_key",
            base_url="https://api.test.com"
        )
    
    @pytest.mark.asyncio
    async def test_connection_error_handling(self, api_client):
        """Ulanish xatosi testi"""
        with patch('aiohttp.ClientSession.request') as mock_request:
            mock_request.side_effect = aiohttp.ClientConnectorError(
                connection_key=None,
                os_error=None
            )
            
            result = await api_client.make_request("/test")
            
            assert result.success is False
            assert "connection" in result.error.lower()
    
    @pytest.mark.asyncio
    async def test_timeout_error_handling(self, api_client):
        """Timeout xatosi testi"""
        with patch('aiohttp.ClientSession.request') as mock_request:
            mock_request.side_effect = asyncio.TimeoutError()
            
            result = await api_client.make_request("/test")
            
            assert result.success is False
            assert "timeout" in result.error.lower()
    
    @pytest.mark.asyncio
    async def test_json_decode_error_handling(self, api_client):
        """JSON decode xatosi testi"""
        mock_response = Mock()
        mock_response.status = 200
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        
        with patch('aiohttp.ClientSession.request') as mock_request:
            mock_request.return_value.__aenter__.return_value = mock_response
            
            result = await api_client.make_request("/test")
            
            assert result.success is False
            assert "json" in result.error.lower()


class TestAPIResponseValidation:
    """API response validatsiya testlari"""
    
    def test_valid_response_structure(self):
        """To'g'ri response struktura testi"""
        response = MockAPIResponse(
            success=True,
            data={"key": "value"},
            status_code=200
        )
        
        assert response.success is True
        assert response.data is not None
        assert response.status_code == 200
        assert response.error is None
    
    def test_invalid_response_structure(self):
        """Noto'g'ri response struktura testi"""
        response = MockAPIResponse(
            success=False,
            error="API xatosi",
            status_code=400
        )
        
        assert response.success is False
        assert response.error is not None
        assert response.status_code == 400
        assert response.data is None
    
    def test_response_rate_limit_info(self):
        """Response rate limit ma'lumotlari testi"""
        response = MockAPIResponse(
            success=True,
            data={"test": "data"},
            rate_limit_remaining=50
        )
        
        assert response.rate_limit_remaining == 50
        assert response.rate_limit_remaining > 0


class TestAPIClientIntegration:
    """API client integratsiya testlari"""
    
    @pytest.mark.asyncio
    async def test_client_lifecycle(self):
        """Client lifecycle testi"""
        client = OneInchClient(
            api_key="test_key",
            base_url="https://api.test.com"
        )
        
        # Context manager testi
        async with client as c:
            assert c.session is not None
            
            # Mock request
            with patch.object(c, 'make_request') as mock_request:
                mock_request.return_value = MockAPIResponse(success=True)
                result = await c.make_request("/test")
                assert result.success is True
        
        # Session yopilishi kerak
        assert client.session is None or client.session.closed
    
    @pytest.mark.asyncio
    async def test_multiple_concurrent_requests(self):
        """Bir vaqtda bir nechta so'rov testi"""
        client = OneInchClient(
            api_key="test_key",
            base_url="https://api.test.com"
        )
        
        async with client:
            with patch.object(client, 'make_request') as mock_request:
                mock_request.return_value = MockAPIResponse(success=True)
                
                # Bir vaqtda bir nechta so'rov
                tasks = [
                    client.make_request("/test1"),
                    client.make_request("/test2"),
                    client.make_request("/test3")
                ]
                
                results = await asyncio.gather(*tasks)
                
                assert len(results) == 3
                assert all(result.success for result in results)
                assert mock_request.call_count == 3


# Test configuration
pytest_plugins = ["pytest_asyncio"]

# Test markers
pytestmark = pytest.mark.asyncio

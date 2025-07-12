import pytest
import asyncio
import logging
import time
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import tempfile
import os

# Test qilinadigan modullar
import sys
sys.path.append('../src')

from utils.logger import setup_logger, get_logger
from utils.error_handler import handle_api_error, handle_processing_error, ErrorHandler
from utils.rate_limiter import RateLimiter, RateLimit
from utils.retry_handler import retry_async, RetryHandler
from utils.fallback_manager import FallbackManager, FallbackClient

# Test ma'lumotlari
@dataclass
class TestResponse:
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None

class TestUtils:
    """Test utilities va helper functions"""
    
    @staticmethod
    def create_temp_log_file() -> str:
        """Vaqtinchalik log fayl yaratish"""
        temp_dir = tempfile.mkdtemp()
        log_file = os.path.join(temp_dir, "test.log")
        return log_file
    
    @staticmethod
    def cleanup_temp_file(filepath: str) -> None:
        """Vaqtinchalik faylni o'chirish"""
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
                # Papkani ham o'chirish
                temp_dir = os.path.dirname(filepath)
                if os.path.exists(temp_dir):
                    os.rmdir(temp_dir)
        except Exception:
            pass

class TestLogger:
    """Logger testlari"""
    
    def test_setup_logger_basic(self):
        """Asosiy logger sozlash testi"""
        log_file = TestUtils.create_temp_log_file()
        
        try:
            logger = setup_logger("test_logger", log_file)
            
            # Logger yaratilganini tekshirish
            assert logger is not None
            assert logger.name == "test_logger"
            assert logger.level == logging.INFO
            
            # Handler mavjudligini tekshirish
            assert len(logger.handlers) == 2  # File + Console
            
        finally:
            TestUtils.cleanup_temp_file(log_file)
    
    def test_logger_write_to_file(self):
        """Logger faylga yozish testi"""
        log_file = TestUtils.create_temp_log_file()
        
        try:
            logger = setup_logger("test_file_logger", log_file)
            
            # Test xabar yozish
            test_message = "Bu test xabari"
            logger.info(test_message)
            
            # Fayl mavjudligini tekshirish
            assert os.path.exists(log_file)
            
            # Fayl mazmunini tekshirish
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()
                assert test_message in content
                assert "INFO" in content
                
        finally:
            TestUtils.cleanup_temp_file(log_file)
    
    def test_get_logger_singleton(self):
        """Logger singleton testi"""
        logger1 = get_logger("test_singleton")
        logger2 = get_logger("test_singleton")
        
        # Bir xil logger qaytarilishini tekshirish
        assert logger1 is logger2
        assert logger1.name == logger2.name
    
    def test_logger_levels(self):
        """Logger levellar testi"""
        log_file = TestUtils.create_temp_log_file()
        
        try:
            # DEBUG level logger
            debug_logger = setup_logger("debug_logger", log_file, logging.DEBUG)
            assert debug_logger.level == logging.DEBUG
            
            # ERROR level logger
            error_logger = setup_logger("error_logger", log_file, logging.ERROR)
            assert error_logger.level == logging.ERROR
            
        finally:
            TestUtils.cleanup_temp_file(log_file)


class TestErrorHandler:
    """Error handler testlari"""
    
    def test_handle_api_error_basic(self):
        """Asosiy API error handling testi"""
        error_msg = "API connection failed"
        
        with patch('utils.logger.get_logger') as mock_logger:
            mock_logger.return_value.error = Mock()
            
            result = handle_api_error(error_msg)
            
            # Result tekshirish
            assert result.success == False
            assert result.error == error_msg
            
            # Logger chaqirilganini tekshirish
            mock_logger.return_value.error.assert_called_once()
    
    def test_handle_processing_error_with_context(self):
        """Processing error context bilan testi"""
        error_msg = "Ma'lumot qayta ishlashda xato"
        context = {"module": "sentiment_analyzer", "data_size": 100}
        
        with patch('utils.logger.get_logger') as mock_logger:
            mock_logger.return_value.error = Mock()
            
            result = handle_processing_error(error_msg, context)
            
            # Result tekshirish
            assert result.success == False
            assert result.error == error_msg
            assert "sentiment_analyzer" in str(mock_logger.return_value.error.call_args)
    
    def test_error_handler_class(self):
        """ErrorHandler class testi"""
        handler = ErrorHandler()
        
        # Exception yaratish
        try:
            raise ValueError("Test xatosi")
        except ValueError as e:
            result = handler.handle_exception(e)
            
            # Result tekshirish
            assert result.success == False
            assert "Test xatosi" in result.error
            assert "ValueError" in result.error
    
    def test_error_handler_with_recovery(self):
        """Error handler recovery bilan testi"""
        handler = ErrorHandler()
        
        # Recovery function
        def recovery_func():
            return "Recovery successful"
        
        try:
            raise ConnectionError("Ulanish xatosi")
        except ConnectionError as e:
            result = handler.handle_exception(e, recovery_func)
            
            # Recovery amalga oshganini tekshirish
            assert result.success == True
            assert result.data == "Recovery successful"


class TestRateLimiter:
    """Rate limiter testlari"""
    
    @pytest.mark.asyncio
    async def test_rate_limiter_basic(self):
        """Asosiy rate limiter testi"""
        # 2 so'rov 1 sekundda
        limiter = RateLimiter(calls=2, period=1)
        
        # Birinchi so'rov - tez bo'lishi kerak
        start_time = time.time()
        await limiter.wait()
        first_call_time = time.time() - start_time
        
        # Ikkinchi so'rov - tez bo'lishi kerak
        start_time = time.time()
        await limiter.wait()
        second_call_time = time.time() - start_time
        
        # Uchinchi so'rov - kutishi kerak
        start_time = time.time()
        await limiter.wait()
        third_call_time = time.time() - start_time
        
        # Tekshirish
        assert first_call_time < 0.1  # Tez
        assert second_call_time < 0.1  # Tez
        assert third_call_time > 0.5   # Kutgan
    
    @pytest.mark.asyncio
    async def test_rate_limiter_concurrent(self):
        """Concurrent rate limiter testi"""
        limiter = RateLimiter(calls=3, period=1)
        
        # 5 ta concurrent so'rov
        start_time = time.time()
        tasks = [limiter.wait() for _ in range(5)]
        await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        # Kamida 1 sekund kutgan bo'lishi kerak
        assert total_time >= 1.0
    
    def test_rate_limit_dataclass(self):
        """RateLimit dataclass testi"""
        rate_limit = RateLimit(calls=100, period=60)
        
        # Ma'lumotlar tekshirish
        assert rate_limit.calls == 100
        assert rate_limit.period == 60
        assert rate_limit.remaining == 0
        assert rate_limit.reset_time == 0
    
    @pytest.mark.asyncio
    async def test_rate_limiter_reset(self):
        """Rate limiter reset testi"""
        limiter = RateLimiter(calls=1, period=0.5)
        
        # Birinchi so'rov
        await limiter.wait()
        
        # 0.6 sekund kutish (reset bo'lishi kerak)
        await asyncio.sleep(0.6)
        
        # Ikkinchi so'rov tez bo'lishi kerak
        start_time = time.time()
        await limiter.wait()
        call_time = time.time() - start_time
        
        assert call_time < 0.1


class TestRetryHandler:
    """Retry handler testlari"""
    
    @pytest.mark.asyncio
    async def test_retry_async_success(self):
        """Retry async success testi"""
        call_count = 0
        
        @retry_async(max_retries=3, delay=0.1)
        async def test_function():
            nonlocal call_count
            call_count += 1
            return "Success"
        
        result = await test_function()
        
        # Tekshirish
        assert result == "Success"
        assert call_count == 1
    
    @pytest.mark.asyncio
    async def test_retry_async_failure_then_success(self):
        """Retry async failure keyin success testi"""
        call_count = 0
        
        @retry_async(max_retries=3, delay=0.1)
        async def test_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Xato")
            return "Success"
        
        result = await test_function()
        
        # Tekshirish
        assert result == "Success"
        assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_retry_async_max_retries_exceeded(self):
        """Retry async max retries exceeded testi"""
        call_count = 0
        
        @retry_async(max_retries=2, delay=0.1)
        async def test_function():
            nonlocal call_count
            call_count += 1
            raise ValueError("Doimiy xato")
        
        # Exception kutish
        with pytest.raises(ValueError):
            await test_function()
        
        # 3 marta chaqirilganini tekshirish (1 + 2 retry)
        assert call_count == 3
    
    def test_retry_handler_class(self):
        """RetryHandler class testi"""
        handler = RetryHandler(max_retries=2, delay=0.1)
        
        # Ma'lumotlar tekshirish
        assert handler.max_retries == 2
        assert handler.delay == 0.1
        assert handler.backoff_factor == 2
    
    @pytest.mark.asyncio
    async def test_retry_handler_with_backoff(self):
        """Retry handler backoff bilan testi"""
        handler = RetryHandler(max_retries=3, delay=0.1, backoff_factor=2)
        call_count = 0
        
        async def failing_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Ulanish xatosi")
            return "Success"
        
        start_time = time.time()
        result = await handler.execute(failing_function)
        total_time = time.time() - start_time
        
        # Tekshirish
        assert result == "Success"
        assert call_count == 3
        # Backoff vaqti: 0.1 + 0.2 = 0.3 sekund
        assert total_time >= 0.3


class TestFallbackManager:
    """Fallback manager testlari"""
    
    @pytest.mark.asyncio
    async def test_fallback_manager_primary_success(self):
        """Fallback manager primary success testi"""
        # Mock clientlar
        primary_client = Mock()
        primary_client.get_data = AsyncMock(return_value=TestResponse(success=True, data="Primary data"))
        
        secondary_client = Mock()
        secondary_client.get_data = AsyncMock(return_value=TestResponse(success=True, data="Secondary data"))
        
        # Fallback manager
        manager = FallbackManager()
        manager.add_client("primary", primary_client)
        manager.add_client("secondary", secondary_client)
        
        # Test
        result = await manager.execute("get_data")
        
        # Tekshirish
        assert result.success == True
        assert result.data == "Primary data"
        primary_client.get_data.assert_called_once()
        secondary_client.get_data.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_fallback_manager_primary_fails(self):
        """Fallback manager primary fails testi"""
        # Mock clientlar
        primary_client = Mock()
        primary_client.get_data = AsyncMock(side_effect=Exception("Primary xato"))
        
        secondary_client = Mock()
        secondary_client.get_data = AsyncMock(return_value=TestResponse(success=True, data="Secondary data"))
        
        # Fallback manager
        manager = FallbackManager()
        manager.add_client("primary", primary_client)
        manager.add_client("secondary", secondary_client)
        
        # Test
        result = await manager.execute("get_data")
        
        # Tekshirish
        assert result.success == True
        assert result.data == "Secondary data"
        primary_client.get_data.assert_called_once()
        secondary_client.get_data.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_fallback_manager_all_fail(self):
        """Fallback manager barcha clientlar fail testi"""
        # Mock clientlar
        primary_client = Mock()
        primary_client.get_data = AsyncMock(side_effect=Exception("Primary xato"))
        
        secondary_client = Mock()
        secondary_client.get_data = AsyncMock(side_effect=Exception("Secondary xato"))
        
        # Fallback manager
        manager = FallbackManager()
        manager.add_client("primary", primary_client)
        manager.add_client("secondary", secondary_client)
        
        # Test
        result = await manager.execute("get_data")
        
        # Tekshirish
        assert result.success == False
        assert "Barcha clientlar ishlamadi" in result.error
        primary_client.get_data.assert_called_once()
        secondary_client.get_data.assert_called_once()
    
    def test_fallback_client_dataclass(self):
        """FallbackClient dataclass testi"""
        client = FallbackClient(
            name="test_client",
            instance=Mock(),
            priority=1,
            enabled=True
        )
        
        # Ma'lumotlar tekshirish
        assert client.name == "test_client"
        assert client.priority == 1
        assert client.enabled == True
        assert client.failure_count == 0
    
    @pytest.mark.asyncio
    async def test_fallback_manager_with_disabled_client(self):
        """Fallback manager disabled client bilan testi"""
        # Mock clientlar
        primary_client = Mock()
        primary_client.get_data = AsyncMock(return_value=TestResponse(success=True, data="Primary data"))
        
        disabled_client = Mock()
        disabled_client.get_data = AsyncMock(return_value=TestResponse(success=True, data="Disabled data"))
        
        # Fallback manager
        manager = FallbackManager()
        manager.add_client("primary", primary_client)
        manager.add_client("disabled", disabled_client, enabled=False)
        
        # Primary clientni o'chirish
        primary_client.get_data = AsyncMock(side_effect=Exception("Primary xato"))
        
        # Test
        result = await manager.execute("get_data")
        
        # Tekshirish
        assert result.success == False
        primary_client.get_data.assert_called_once()
        disabled_client.get_data.assert_not_called()


class TestUtilsIntegration:
    """Utils integration testlari"""
    
    @pytest.mark.asyncio
    async def test_integrated_error_handling_with_retry(self):
        """Error handling va retry integration testi"""
        call_count = 0
        
        @retry_async(max_retries=2, delay=0.1)
        async def unstable_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Ulanish xatosi")
            return TestResponse(success=True, data="Success")
        
        # Test
        result = await unstable_function()
        
        # Tekshirish
        assert result.success == True
        assert result.data == "Success"
        assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_integrated_rate_limiting_with_fallback(self):
        """Rate limiting va fallback integration testi"""
        # Rate limiter
        limiter = RateLimiter(calls=1, period=0.5)
        
        # Mock clientlar
        primary_client = Mock()
        primary_client.get_data = AsyncMock(return_value=TestResponse(success=True, data="Primary data"))
        
        secondary_client = Mock()
        secondary_client.get_data = AsyncMock(return_value=TestResponse(success=True, data="Secondary data"))
        
        # Fallback manager
        manager = FallbackManager()
        manager.add_client("primary", primary_client)
        manager.add_client("secondary", secondary_client)
        
        # Rate limited function
        async def rate_limited_execution():
            await limiter.wait()
            return await manager.execute("get_data")
        
        # Test
        start_time = time.time()
        result1 = await rate_limited_execution()
        result2 = await rate_limited_execution()
        total_time = time.time() - start_time
        
        # Tekshirish
        assert result1.success == True
        assert result2.success == True
        assert total_time >= 0.5  # Rate limit ishlaganini tekshirish
    
    def test_comprehensive_logging_integration(self):
        """Comprehensive logging integration testi"""
        log_file = TestUtils.create_temp_log_file()
        
        try:
            # Logger sozlash
            logger = setup_logger("integration_test", log_file)
            
            # Error handler
            handler = ErrorHandler()
            
            # Test scenario
            try:
                raise ValueError("Integration test xatosi")
            except ValueError as e:
                result = handler.handle_exception(e)
                logger.error(f"Error handled: {result.error}")
            
            # Log fayl tekshirish
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()
                assert "Integration test xatosi" in content
                assert "ERROR" in content
                assert "ValueError" in content
                
        finally:
            TestUtils.cleanup_temp_file(log_file)


# Pytest fixtures
@pytest.fixture
def mock_logger():
    """Mock logger fixture"""
    with patch('utils.logger.get_logger') as mock:
        mock_logger_instance = Mock()
        mock.return_value = mock_logger_instance
        yield mock_logger_instance

@pytest.fixture
def temp_log_file():
    """Vaqtinchalik log fayl fixture"""
    log_file = TestUtils.create_temp_log_file()
    yield log_file
    TestUtils.cleanup_temp_file(log_file)

@pytest.fixture
def rate_limiter():
    """Rate limiter fixture"""
    return RateLimiter(calls=10, period=1)

@pytest.fixture
def retry_handler():
    """Retry handler fixture"""
    return RetryHandler(max_retries=3, delay=0.1)

@pytest.fixture
def fallback_manager():
    """Fallback manager fixture"""
    manager = FallbackManager()
    
    # Mock clientlar qo'shish
    primary_client = Mock()
    secondary_client = Mock()
    
    manager.add_client("primary", primary_client)
    manager.add_client("secondary", secondary_client)
    
    return manager, primary_client, secondary_client


# Test ishga tushirish
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

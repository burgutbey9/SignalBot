import pytest
import asyncio
import json
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Test qilinadigan modullar
from data_processing.order_flow_analyzer import OrderFlowAnalyzer, OrderFlowResult
from data_processing.sentiment_analyzer import SentimentAnalyzer, SentimentResult
from data_processing.market_analyzer import MarketAnalyzer, MarketResult
from data_processing.signal_generator import SignalGenerator, SignalResult
from utils.logger import get_logger

logger = get_logger(__name__)

class TestOrderFlowAnalyzer:
    """Order Flow Analyzer testlari"""
    
    @pytest.fixture
    def analyzer(self):
        """Order Flow Analyzer fixture"""
        return OrderFlowAnalyzer()
    
    @pytest.fixture
    def sample_order_flow_data(self):
        """Order flow test ma'lumotlari"""
        return {
            "timestamp": datetime.now().isoformat(),
            "symbol": "EURUSD",
            "buy_orders": [
                {"price": 1.0850, "amount": 1000000, "timestamp": "2024-01-15T10:00:00Z"},
                {"price": 1.0851, "amount": 500000, "timestamp": "2024-01-15T10:01:00Z"},
                {"price": 1.0852, "amount": 750000, "timestamp": "2024-01-15T10:02:00Z"}
            ],
            "sell_orders": [
                {"price": 1.0855, "amount": 800000, "timestamp": "2024-01-15T10:00:30Z"},
                {"price": 1.0856, "amount": 600000, "timestamp": "2024-01-15T10:01:30Z"},
                {"price": 1.0857, "amount": 400000, "timestamp": "2024-01-15T10:02:30Z"}
            ],
            "large_orders": [
                {"price": 1.0853, "amount": 2000000, "side": "buy", "timestamp": "2024-01-15T10:03:00Z"}
            ]
        }
    
    @pytest.mark.asyncio
    async def test_analyze_order_flow_success(self, analyzer, sample_order_flow_data):
        """Order flow tahlil - muvaffaqiyatli holat"""
        # Test bajarish
        result = await analyzer.analyze_order_flow(sample_order_flow_data)
        
        # Natijalarni tekshirish
        assert result.success is True
        assert result.data is not None
        assert "buy_pressure" in result.data
        assert "sell_pressure" in result.data
        assert "net_flow" in result.data
        assert "whale_activity" in result.data
        assert result.confidence > 0.0
        
        # Buy pressure tekshirish
        assert result.data["buy_pressure"] > 0
        # Sell pressure tekshirish  
        assert result.data["sell_pressure"] > 0
        # Net flow hisoblash
        expected_net = result.data["buy_pressure"] - result.data["sell_pressure"]
        assert abs(result.data["net_flow"] - expected_net) < 0.001
        
        logger.info("Order flow tahlil muvaffaqiyatli bajarildi")
    
    @pytest.mark.asyncio
    async def test_analyze_order_flow_empty_data(self, analyzer):
        """Order flow tahlil - bo'sh ma'lumot"""
        empty_data = {
            "symbol": "EURUSD",
            "buy_orders": [],
            "sell_orders": [],
            "large_orders": []
        }
        
        result = await analyzer.analyze_order_flow(empty_data)
        
        assert result.success is False
        assert result.error is not None
        assert "Bo'sh ma'lumot" in result.error or "Empty data" in result.error
        
        logger.info("Bo'sh ma'lumot holati to'g'ri qayta ishlandi")
    
    @pytest.mark.asyncio
    async def test_calculate_whale_activity(self, analyzer, sample_order_flow_data):
        """Whale activity hisoblash testi"""
        result = await analyzer.analyze_order_flow(sample_order_flow_data)
        
        assert result.success is True
        assert "whale_activity" in result.data
        assert result.data["whale_activity"] > 0
        
        # Katta buyurtmalar mavjudligini tekshirish
        whale_score = result.data["whale_activity"]
        assert whale_score >= 0.0 and whale_score <= 1.0
        
        logger.info(f"Whale activity hisoblandi: {whale_score}")


class TestSentimentAnalyzer:
    """Sentiment Analyzer testlari"""
    
    @pytest.fixture
    def analyzer(self):
        """Sentiment Analyzer fixture"""
        return SentimentAnalyzer()
    
    @pytest.fixture
    def sample_news_data(self):
        """News test ma'lumotlari"""
        return {
            "articles": [
                {
                    "title": "EUR/USD rallies on positive economic data",
                    "content": "The euro strengthened against the dollar following better than expected GDP growth",
                    "sentiment_score": 0.7,
                    "timestamp": "2024-01-15T10:00:00Z",
                    "source": "Reuters"
                },
                {
                    "title": "Dollar faces pressure amid inflation concerns",
                    "content": "The US dollar declined as investors worry about rising inflation rates",
                    "sentiment_score": -0.5,
                    "timestamp": "2024-01-15T09:30:00Z",
                    "source": "Bloomberg"
                }
            ],
            "social_sentiment": {
                "reddit_buzz": 0.3,
                "twitter_sentiment": 0.1,
                "overall_social": 0.2
            }
        }
    
    @pytest.mark.asyncio
    async def test_analyze_sentiment_success(self, analyzer, sample_news_data):
        """Sentiment tahlil - muvaffaqiyatli holat"""
        result = await analyzer.analyze_sentiment(sample_news_data)
        
        assert result.success is True
        assert result.data is not None
        assert "news_sentiment" in result.data
        assert "social_sentiment" in result.data
        assert "overall_sentiment" in result.data
        assert result.confidence > 0.0
        
        # Sentiment qiymatlar oralig'ini tekshirish
        assert -1.0 <= result.data["overall_sentiment"] <= 1.0
        assert -1.0 <= result.data["news_sentiment"] <= 1.0
        assert -1.0 <= result.data["social_sentiment"] <= 1.0
        
        logger.info("Sentiment tahlil muvaffaqiyatli bajarildi")
    
    @pytest.mark.asyncio
    async def test_analyze_sentiment_mixed_signals(self, analyzer):
        """Sentiment tahlil - aralash signallar"""
        mixed_data = {
            "articles": [
                {
                    "title": "Bullish EUR/USD outlook",
                    "content": "Strong fundamentals support euro rally",
                    "sentiment_score": 0.8,
                    "timestamp": "2024-01-15T10:00:00Z"
                },
                {
                    "title": "Bearish USD trend continues",
                    "content": "Dollar weakness persists across markets",
                    "sentiment_score": -0.6,
                    "timestamp": "2024-01-15T10:15:00Z"
                }
            ],
            "social_sentiment": {
                "reddit_buzz": -0.2,
                "twitter_sentiment": 0.4,
                "overall_social": 0.1
            }
        }
        
        result = await analyzer.analyze_sentiment(mixed_data)
        
        assert result.success is True
        assert result.data["overall_sentiment"] != 0.0  # Aralash signal
        assert result.confidence < 0.8  # Pastroq ishonch
        
        logger.info("Aralash sentiment signallari to'g'ri qayta ishlandi")
    
    @pytest.mark.asyncio
    async def test_sentiment_confidence_calculation(self, analyzer, sample_news_data):
        """Sentiment confidence hisoblash"""
        result = await analyzer.analyze_sentiment(sample_news_data)
        
        assert result.success is True
        assert 0.0 <= result.confidence <= 1.0
        
        # Confidence faktori tekshirish
        confidence_factors = result.data.get("confidence_factors", {})
        assert "news_count" in confidence_factors
        assert "sentiment_consistency" in confidence_factors
        assert "source_reliability" in confidence_factors
        
        logger.info(f"Sentiment confidence hisoblandi: {result.confidence}")


class TestMarketAnalyzer:
    """Market Analyzer testlari"""
    
    @pytest.fixture
    def analyzer(self):
        """Market Analyzer fixture"""
        return MarketAnalyzer()
    
    @pytest.fixture
    def sample_market_data(self):
        """Market test ma'lumotlari"""
        # 24 soatlik OHLCV ma'lumotlari
        dates = pd.date_range(start='2024-01-15', periods=24, freq='H')
        return {
            "symbol": "EURUSD",
            "timeframe": "1H",
            "data": pd.DataFrame({
                "timestamp": dates,
                "open": np.random.uniform(1.0800, 1.0900, 24),
                "high": np.random.uniform(1.0850, 1.0950, 24),
                "low": np.random.uniform(1.0750, 1.0850, 24),
                "close": np.random.uniform(1.0800, 1.0900, 24),
                "volume": np.random.randint(100000, 1000000, 24)
            })
        }
    
    @pytest.mark.asyncio
    async def test_analyze_market_success(self, analyzer, sample_market_data):
        """Market tahlil - muvaffaqiyatli holat"""
        result = await analyzer.analyze_market(sample_market_data)
        
        assert result.success is True
        assert result.data is not None
        assert "trend_direction" in result.data
        assert "volatility" in result.data
        assert "support_resistance" in result.data
        assert "technical_indicators" in result.data
        assert result.confidence > 0.0
        
        # Trend direction tekshirish
        assert result.data["trend_direction"] in ["bullish", "bearish", "sideways"]
        
        # Volatility tekshirish
        assert result.data["volatility"] > 0.0
        
        # Support/Resistance tekshirish
        sr_levels = result.data["support_resistance"]
        assert "support" in sr_levels
        assert "resistance" in sr_levels
        assert sr_levels["support"] < sr_levels["resistance"]
        
        logger.info("Market tahlil muvaffaqiyatli bajarildi")
    
    @pytest.mark.asyncio
    async def test_calculate_technical_indicators(self, analyzer, sample_market_data):
        """Technical indicators hisoblash"""
        result = await analyzer.analyze_market(sample_market_data)
        
        assert result.success is True
        indicators = result.data["technical_indicators"]
        
        # Asosiy indikatorlar mavjudligini tekshirish
        assert "sma_20" in indicators
        assert "rsi" in indicators
        assert "macd" in indicators
        assert "bollinger_bands" in indicators
        
        # RSI qiymat oralig'ini tekshirish
        assert 0 <= indicators["rsi"] <= 100
        
        # MACD komponenentlari
        macd = indicators["macd"]
        assert "macd_line" in macd
        assert "signal_line" in macd
        assert "histogram" in macd
        
        logger.info("Technical indicators hisoblandi")
    
    @pytest.mark.asyncio
    async def test_market_volatility_calculation(self, analyzer, sample_market_data):
        """Market volatility hisoblash"""
        result = await analyzer.analyze_market(sample_market_data)
        
        assert result.success is True
        volatility = result.data["volatility"]
        
        # Volatility qiymat oralig'ini tekshirish
        assert volatility > 0.0
        assert volatility < 10.0  # Maksimal chegara
        
        # Volatility kategoriyasi
        vol_category = result.data.get("volatility_category", "")
        assert vol_category in ["low", "medium", "high"]
        
        logger.info(f"Market volatility: {volatility} ({vol_category})")


class TestSignalGenerator:
    """Signal Generator testlari"""
    
    @pytest.fixture
    def generator(self):
        """Signal Generator fixture"""
        return SignalGenerator()
    
    @pytest.fixture
    def sample_combined_data(self):
        """Birlashtirilgan test ma'lumotlari"""
        return {
            "order_flow": {
                "buy_pressure": 0.7,
                "sell_pressure": 0.3,
                "net_flow": 0.4,
                "whale_activity": 0.8
            },
            "sentiment": {
                "news_sentiment": 0.6,
                "social_sentiment": 0.4,
                "overall_sentiment": 0.5
            },
            "market": {
                "trend_direction": "bullish",
                "volatility": 0.15,
                "support_resistance": {
                    "support": 1.0820,
                    "resistance": 1.0880
                },
                "technical_indicators": {
                    "rsi": 65,
                    "macd": {"histogram": 0.0015},
                    "sma_20": 1.0850
                }
            },
            "current_price": 1.0860,
            "symbol": "EURUSD"
        }
    
    @pytest.mark.asyncio
    async def test_generate_signal_buy(self, generator, sample_combined_data):
        """Buy signal yaratish testi"""
        result = await generator.generate_signal(sample_combined_data)
        
        assert result.success is True
        assert result.data is not None
        assert "action" in result.data
        assert "confidence" in result.data
        assert "entry_price" in result.data
        assert "stop_loss" in result.data
        assert "take_profit" in result.data
        assert "risk_reward" in result.data
        
        # Buy signal tekshirish
        assert result.data["action"] in ["BUY", "SELL", "HOLD"]
        assert result.confidence > 0.0
        
        # Risk management parametrlari
        if result.data["action"] == "BUY":
            assert result.data["stop_loss"] < result.data["entry_price"]
            assert result.data["take_profit"] > result.data["entry_price"]
        
        logger.info(f"Signal yaratildi: {result.data['action']} - {result.confidence}")
    
    @pytest.mark.asyncio
    async def test_generate_signal_conflicting_data(self, generator):
        """Qarama-qarshi ma'lumotlar bilan signal yaratish"""
        conflicting_data = {
            "order_flow": {
                "buy_pressure": 0.2,  # Zaif buy
                "sell_pressure": 0.8,  # Kuchli sell
                "net_flow": -0.6,
                "whale_activity": 0.3
            },
            "sentiment": {
                "news_sentiment": 0.7,  # Ijobiy yangilik
                "social_sentiment": 0.6,  # Ijobiy social
                "overall_sentiment": 0.65
            },
            "market": {
                "trend_direction": "sideways",
                "volatility": 0.25,  # Yuqori volatility
                "support_resistance": {
                    "support": 1.0820,
                    "resistance": 1.0880
                },
                "technical_indicators": {
                    "rsi": 50,  # Neytral
                    "macd": {"histogram": -0.0005},
                    "sma_20": 1.0850
                }
            },
            "current_price": 1.0850,
            "symbol": "EURUSD"
        }
        
        result = await generator.generate_signal(conflicting_data)
        
        assert result.success is True
        # Qarama-qarshi ma'lumotlarda pastroq confidence
        assert result.confidence < 0.6
        # HOLD signal yoki past confidence
        assert result.data["action"] == "HOLD" or result.confidence < 0.5
        
        logger.info("Qarama-qarshi ma'lumotlar to'g'ri qayta ishlandi")
    
    @pytest.mark.asyncio
    async def test_risk_reward_calculation(self, generator, sample_combined_data):
        """Risk/Reward nisbati hisoblash"""
        result = await generator.generate_signal(sample_combined_data)
        
        assert result.success is True
        
        if result.data["action"] in ["BUY", "SELL"]:
            risk_reward = result.data["risk_reward"]
            assert risk_reward > 0.0
            
            # Risk/Reward hisoblash tekshirish
            entry = result.data["entry_price"]
            sl = result.data["stop_loss"]
            tp = result.data["take_profit"]
            
            if result.data["action"] == "BUY":
                risk = entry - sl
                reward = tp - entry
            else:  # SELL
                risk = sl - entry
                reward = entry - tp
            
            calculated_rr = reward / risk if risk > 0 else 0
            assert abs(risk_reward - calculated_rr) < 0.1
            
            logger.info(f"Risk/Reward nisbati: {risk_reward}")
    
    @pytest.mark.asyncio
    async def test_signal_reasons_explanation(self, generator, sample_combined_data):
        """Signal sabablari tushuntirish"""
        result = await generator.generate_signal(sample_combined_data)
        
        assert result.success is True
        assert "reasons" in result.data
        
        reasons = result.data["reasons"]
        assert isinstance(reasons, list)
        assert len(reasons) > 0
        
        # Sabablarda asosiy faktorlar mavjudligini tekshirish
        reason_text = " ".join(reasons)
        factor_found = any(factor in reason_text.lower() for factor in [
            "order flow", "sentiment", "technical", "trend", "support", "resistance"
        ])
        assert factor_found
        
        logger.info(f"Signal sabablari: {reasons}")


class TestDataProcessingIntegration:
    """Data processing integration testlari"""
    
    @pytest.fixture
    def all_analyzers(self):
        """Barcha tahlilchilar"""
        return {
            "order_flow": OrderFlowAnalyzer(),
            "sentiment": SentimentAnalyzer(),
            "market": MarketAnalyzer(),
            "signal": SignalGenerator()
        }
    
    @pytest.mark.asyncio
    async def test_full_processing_pipeline(self, all_analyzers):
        """To'liq qayta ishlash zanjiri"""
        # Mock ma'lumotlar
        raw_data = {
            "order_flow_data": {
                "symbol": "EURUSD",
                "buy_orders": [{"price": 1.0850, "amount": 1000000}],
                "sell_orders": [{"price": 1.0860, "amount": 800000}],
                "large_orders": []
            },
            "news_data": {
                "articles": [{
                    "title": "EUR strengthens",
                    "content": "Positive economic data",
                    "sentiment_score": 0.7
                }],
                "social_sentiment": {"overall_social": 0.3}
            },
            "market_data": {
                "symbol": "EURUSD",
                "timeframe": "1H",
                "data": pd.DataFrame({
                    "timestamp": pd.date_range('2024-01-15', periods=24, freq='H'),
                    "open": [1.0850] * 24,
                    "high": [1.0870] * 24,
                    "low": [1.0830] * 24,
                    "close": [1.0860] * 24,
                    "volume": [500000] * 24
                })
            }
        }
        
        # Har bir tahlilchi orqali ma'lumotlarni qayta ishlash
        order_flow_result = await all_analyzers["order_flow"].analyze_order_flow(
            raw_data["order_flow_data"]
        )
        assert order_flow_result.success
        
        sentiment_result = await all_analyzers["sentiment"].analyze_sentiment(
            raw_data["news_data"]
        )
        assert sentiment_result.success
        
        market_result = await all_analyzers["market"].analyze_market(
            raw_data["market_data"]
        )
        assert market_result.success
        
        # Signal yaratish
        combined_data = {
            "order_flow": order_flow_result.data,
            "sentiment": sentiment_result.data,
            "market": market_result.data,
            "current_price": 1.0860,
            "symbol": "EURUSD"
        }
        
        signal_result = await all_analyzers["signal"].generate_signal(combined_data)
        assert signal_result.success
        
        logger.info("To'liq qayta ishlash zanjiri muvaffaqiyatli bajarildi")
    
    @pytest.mark.asyncio
    async def test_processing_performance(self, all_analyzers):
        """Qayta ishlash performance testi"""
        start_time = datetime.now()
        
        # Parallel qayta ishlash simulatsiyasi
        tasks = []
        for i in range(10):
            # Mock ma'lumotlar
            mock_data = {
                "symbol": "EURUSD",
                "buy_orders": [{"price": 1.0850, "amount": 1000000}],
                "sell_orders": [{"price": 1.0860, "amount": 800000}],
                "large_orders": []
            }
            task = all_analyzers["order_flow"].analyze_order_flow(mock_data)
            tasks.append(task)
        
        # Parallel bajarish
        results = await asyncio.gather(*tasks)
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # Barcha natijalar muvaffaqiyatli
        assert all(result.success for result in results)
        
        # Performance chegarasi (10 ta tahlil 5 soniyada)
        assert processing_time < 5.0
        
        logger.info(f"10 ta tahlil {processing_time:.2f} soniyada bajarildi")


# Test konfiguratsiya
@pytest.fixture(autouse=True)
def setup_test_environment():
    """Test muhiti sozlash"""
    # Test uchun log darajasini o'zgartirish
    logger.setLevel("DEBUG")
    
    # Test ma'lumotlari tozalash
    yield
    
    # Test tugagandan keyin tozalash
    logger.info("Test muhiti tozalandi")


if __name__ == "__main__":
    # Test ishga tushirish
    pytest.main([__file__, "-v", "--tb=short"])

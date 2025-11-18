"""Tests for BaseService class."""

import pytest
from image_velocimetry_tools.services import BaseService


class TestBaseService:
    """Tests for BaseService base class."""

    @pytest.fixture
    def service(self):
        """Create a BaseService instance."""
        return BaseService()

    def test_initialization(self, service):
        """Test service initialization."""
        assert service is not None
        assert service.logger is not None

    def test_initialization_with_custom_logger_name(self):
        """Test service initialization with custom logger name."""
        service = BaseService(logger_name="custom_logger")
        assert service.logger.name == "custom_logger"

    def test_validate_not_none_with_valid_value(self, service):
        """Test _validate_not_none with valid value."""
        service._validate_not_none("valid_value", "test_param")
        # Should not raise

    def test_validate_not_none_with_none_value(self, service):
        """Test _validate_not_none with None value."""
        with pytest.raises(ValueError, match="test_param cannot be None"):
            service._validate_not_none(None, "test_param")

    def test_validate_positive_with_valid_value(self, service):
        """Test _validate_positive with valid value."""
        service._validate_positive(5.0, "test_param")
        # Should not raise

    def test_validate_positive_with_zero(self, service):
        """Test _validate_positive with zero."""
        with pytest.raises(ValueError, match="test_param must be positive"):
            service._validate_positive(0.0, "test_param")

    def test_validate_positive_with_negative_value(self, service):
        """Test _validate_positive with negative value."""
        with pytest.raises(ValueError, match="test_param must be positive"):
            service._validate_positive(-5.0, "test_param")

    def test_validate_non_negative_with_positive_value(self, service):
        """Test _validate_non_negative with positive value."""
        service._validate_non_negative(5.0, "test_param")
        # Should not raise

    def test_validate_non_negative_with_zero(self, service):
        """Test _validate_non_negative with zero."""
        service._validate_non_negative(0.0, "test_param")
        # Should not raise

    def test_validate_non_negative_with_negative_value(self, service):
        """Test _validate_non_negative with negative value."""
        with pytest.raises(ValueError, match="test_param cannot be negative"):
            service._validate_non_negative(-5.0, "test_param")

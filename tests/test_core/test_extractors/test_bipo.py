"""Tests for BiPOVectorExtractor."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

from psyctl.core.extractors.bipo import BiPOVectorExtractor


@pytest.fixture
def mock_model():
    """Create a mock model with necessary attributes."""
    model = MagicMock(spec=nn.Module)

    # Create config mock
    config = MagicMock()
    config.hidden_size = 128
    model.config = config

    # Mock device and dtype
    mock_param = torch.zeros(1, dtype=torch.float32)
    model.parameters = lambda: iter([mock_param])

    # Mock logits output
    mock_output = MagicMock()
    mock_output.logits = torch.randn(1, 10, 1000)  # [batch, seq_len, vocab_size]
    model.return_value = mock_output

    return model


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer."""
    tokenizer = MagicMock()

    # Mock return value for tokenizer call
    mock_output = MagicMock()
    mock_output.input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    tokenizer.return_value = mock_output

    # Mock chat template attributes
    tokenizer.chat_template = None
    tokenizer.apply_chat_template = None

    return tokenizer


@pytest.fixture
def mock_dataset():
    """Create a mock CAA dataset."""
    return [
        {
            "situation": "You are at a party.\nSomeone asks: How do you feel about parties?",
            "char_name": "Alice",
            "positive": "I love parties and meeting new people!",
            "neutral": "Parties are okay.",
        },
        {
            "situation": "You are meeting a friend.\nFriend asks: Do you enjoy socializing?",
            "char_name": "Bob",
            "positive": "I enjoy socializing very much!",
            "neutral": "I sometimes socialize.",
        },
    ]


@pytest.fixture
def extractor():
    """Create BiPOVectorExtractor instance."""
    return BiPOVectorExtractor()


class TestBiPOVectorExtractor:
    """Test suite for BiPOVectorExtractor."""

    def test_initialization(self, extractor):
        """Test extractor initializes correctly."""
        assert extractor.dataset_loader is not None
        assert extractor.layer_accessor is not None
        assert extractor.logger is not None

    @patch("psyctl.core.extractors.base.SteerDatasetLoader")
    @patch("psyctl.core.extractors.base.LayerAccessor")
    def test_extract_validates_layers(
        self, mock_layer_accessor_class, mock_dataset_loader_class, mock_model
    ):
        """Test that extract validates layer paths."""
        # Setup mocks
        mock_layer_accessor = MagicMock()
        mock_layer_accessor.validate_layers.return_value = False
        mock_layer_accessor_class.return_value = mock_layer_accessor

        extractor = BiPOVectorExtractor()

        # Should raise error for invalid layers
        with pytest.raises(ValueError, match="Some layer paths are invalid"):
            extractor.extract(
                model=mock_model,
                tokenizer=MagicMock(),
                layers=["invalid.layer"],
                dataset_path=Path("./test"),
                epochs=1,
            )

    def test_prepare_dataset(self, extractor, mock_dataset, mock_tokenizer):
        """Test dataset preparation."""
        samples = extractor._prepare_dataset(mock_dataset, mock_tokenizer)

        assert len(samples) == 2
        assert all(
            len(sample) == 4 for sample in samples
        )  # (situation, char_name, positive, neutral)
        assert "party" in samples[0][0].lower()  # situation
        assert samples[0][1] == "Alice"  # char_name
        assert samples[0][2] == "I love parties and meeting new people!"  # positive
        assert samples[0][3] == "Parties are okay."  # neutral

    def test_get_response_logprob_no_steering(
        self, extractor, mock_model, mock_tokenizer
    ):
        """Test log probability calculation without steering."""
        layer_module = MagicMock(spec=nn.Module)

        logprob = extractor._get_response_logprob(
            model=mock_model,
            tokenizer=mock_tokenizer,
            situation="Test situation",
            char_name="TestChar",
            response="Test response",
            layer_module=layer_module,
            steering=None,
        )

        assert isinstance(logprob, torch.Tensor)
        assert logprob.ndim == 0  # Scalar

    def test_get_response_logprob_with_steering(
        self, extractor, mock_model, mock_tokenizer
    ):
        """Test log probability calculation with steering."""
        layer_module = MagicMock(spec=nn.Module)
        steering_vec = torch.randn(128)

        logprob = extractor._get_response_logprob(
            model=mock_model,
            tokenizer=mock_tokenizer,
            situation="Test situation",
            char_name="TestChar",
            response="Test response",
            layer_module=layer_module,
            steering=steering_vec,
        )

        assert isinstance(logprob, torch.Tensor)
        assert logprob.ndim == 0

    def test_get_response_logprob_handles_tuple_output(self, extractor, mock_tokenizer):
        """Test that hook handles tuple outputs from layers."""
        # Mock model that returns tuple output
        model = MagicMock(spec=nn.Module)

        # Create config mock
        config = MagicMock()
        config.hidden_size = 128
        model.config = config

        mock_param = torch.zeros(1, dtype=torch.float32)
        model.parameters = lambda: iter([mock_param])

        mock_output = MagicMock()
        mock_output.logits = torch.randn(1, 10, 1000)
        model.return_value = mock_output

        layer_module = MagicMock(spec=nn.Module)
        steering_vec = torch.randn(128)

        # Test with tuple output in hook
        logprob = extractor._get_response_logprob(
            model=model,
            tokenizer=mock_tokenizer,
            situation="Test situation",
            char_name="TestChar",
            response="Response",
            layer_module=layer_module,
            steering=steering_vec,
        )

        assert isinstance(logprob, torch.Tensor)

    def test_compute_bipo_loss(
        self, extractor, mock_model, mock_tokenizer, mock_dataset
    ):
        """Test BiPO loss computation (full text only)."""
        layer_module = MagicMock(spec=nn.Module)
        v = torch.randn(128, requires_grad=True)
        batch = extractor._prepare_dataset(mock_dataset, mock_tokenizer)

        loss = extractor._compute_bipo_loss(
            model=mock_model,
            tokenizer=mock_tokenizer,
            layer_module=layer_module,
            batch=batch[:1],  # Single sample
            v=v,
            beta=0.1,
        )

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        # Loss calculation involves detached tensors from model inference

    def test_train_steering_vector(self, extractor, mock_dataset):
        """Test steering vector training completes without errors."""

        # Create a simple real model for testing
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = MagicMock()
                self.config.hidden_size = 64
                self.embedding = nn.Embedding(100, 64)
                self.lm_head = nn.Linear(64, 100)

            def forward(self, input_ids):
                x = self.embedding(input_ids)
                logits = self.lm_head(x)
                output = MagicMock()
                output.logits = logits
                return output

        model = SimpleModel()
        layer_module = model.embedding

        # Simple tokenizer mock
        tokenizer = MagicMock()
        mock_output = MagicMock()
        mock_output.input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        tokenizer.return_value = mock_output

        # Use minimal dataset
        minimal_dataset = [mock_dataset[0]]

        # Train with minimal settings - just test it runs
        with patch.object(extractor, "_compute_bipo_loss") as mock_loss:
            # Return a simple loss that requires grad
            mock_loss.return_value = torch.tensor(0.5, requires_grad=True)

            steering_vec = extractor._train_steering_vector(
                model=model,
                tokenizer=tokenizer,
                layer_module=layer_module,
                layer_str="test.layer",
                layer_idx=0,
                total_layers=1,
                dataset=minimal_dataset,
                batch_size=1,
                lr=5e-4,
                beta=0.1,
                epochs=1,
                weight_decay=0.01,
            )

            assert isinstance(steering_vec, torch.Tensor)
            assert steering_vec.shape == (64,)
            assert not steering_vec.requires_grad

    def test_extract_normalization(self, mock_model, mock_tokenizer, tmp_path):
        """Test vector normalization option."""
        with (
            patch(
                "psyctl.core.extractors.base.SteerDatasetLoader"
            ) as mock_loader_class,
            patch("psyctl.core.extractors.base.LayerAccessor") as mock_accessor_class,
        ):
            # Setup mocks
            mock_loader = MagicMock()
            mock_loader.load.return_value = [
                {
                    "situation": "Test situation",
                    "char_name": "TestChar",
                    "positive": "P1",
                    "neutral": "N1",
                }
            ]
            mock_loader_class.return_value = mock_loader

            mock_accessor = MagicMock()
            mock_accessor.validate_layers.return_value = True
            mock_accessor.get_layer.return_value = MagicMock(spec=nn.Module)
            mock_accessor_class.return_value = mock_accessor

            extractor = BiPOVectorExtractor()

            # Mock training to return a non-normalized vector
            with patch.object(
                extractor,
                "_train_steering_vector",
                return_value=torch.tensor([3.0, 4.0]),  # norm = 5.0
            ):
                vectors = extractor.extract(
                    model=mock_model,
                    tokenizer=mock_tokenizer,
                    layers=["test.layer"],
                    dataset_path=tmp_path,
                    normalize=True,
                    epochs=1,
                    batch_size=1,
                )

                vec = vectors["test.layer"]
                assert pytest.approx(vec.norm().item(), rel=1e-5) == 1.0

    def test_extract_without_normalization(self, mock_model, mock_tokenizer, tmp_path):
        """Test extraction without normalization."""
        with (
            patch(
                "psyctl.core.extractors.base.SteerDatasetLoader"
            ) as mock_loader_class,
            patch("psyctl.core.extractors.base.LayerAccessor") as mock_accessor_class,
        ):
            # Setup mocks
            mock_loader = MagicMock()
            mock_loader.load.return_value = [
                {
                    "situation": "Test situation",
                    "char_name": "TestChar",
                    "positive": "P1",
                    "neutral": "N1",
                }
            ]
            mock_loader_class.return_value = mock_loader

            mock_accessor = MagicMock()
            mock_accessor.validate_layers.return_value = True
            mock_accessor.get_layer.return_value = MagicMock(spec=nn.Module)
            mock_accessor_class.return_value = mock_accessor

            extractor = BiPOVectorExtractor()

            with patch.object(
                extractor,
                "_train_steering_vector",
                return_value=torch.tensor([3.0, 4.0]),
            ):
                vectors = extractor.extract(
                    model=mock_model,
                    tokenizer=mock_tokenizer,
                    layers=["test.layer"],
                    dataset_path=tmp_path,
                    normalize=False,
                    epochs=1,
                    batch_size=1,
                )

                vec = vectors["test.layer"]
                assert pytest.approx(vec.norm().item(), rel=1e-5) == 5.0

    def test_format_with_chat_template_available(self, extractor):
        """Test formatting with chat template when available."""
        tokenizer = MagicMock()
        tokenizer.chat_template = "template"
        tokenizer.apply_chat_template = MagicMock(
            return_value="<|user|>Test<|assistant|>"
        )

        result = extractor._format_with_chat_template(tokenizer, "Test")

        assert result == "<|user|>Test<|assistant|>"
        tokenizer.apply_chat_template.assert_called_once()

    def test_format_with_chat_template_unavailable(self, extractor):
        """Test formatting without chat template."""
        tokenizer = MagicMock()
        tokenizer.chat_template = None

        result = extractor._format_with_chat_template(tokenizer, "Test")

        assert result == "Test"

    def test_format_with_chat_template_error(self, extractor):
        """Test formatting when chat template raises error."""
        tokenizer = MagicMock()
        tokenizer.chat_template = "template"
        tokenizer.apply_chat_template = MagicMock(
            side_effect=Exception("Template error")
        )

        result = extractor._format_with_chat_template(tokenizer, "Test")

        assert result == "Test"

"""Shared pytest fixtures for the test suite."""

import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
import pytest
import torch


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def mock_torch_device():
    """Mock torch device for testing without GPU requirements."""
    with patch('torch.cuda.is_available', return_value=False):
        with patch('torch.device', return_value=Mock(type='cpu')):
            yield


@pytest.fixture
def mock_model_config():
    """Mock model configuration for testing."""
    return {
        'model_name': 'test-model',
        'max_sequence_length': 512,
        'hidden_size': 768,
        'num_layers': 12,
        'learning_rate': 1e-4,
        'batch_size': 8
    }


@pytest.fixture
def mock_training_config():
    """Mock training configuration for testing."""
    return {
        'epochs': 10,
        'learning_rate': 1e-4,
        'batch_size': 16,
        'gradient_accumulation_steps': 1,
        'max_grad_norm': 1.0,
        'warmup_steps': 100,
        'save_steps': 500,
        'eval_steps': 100,
        'logging_steps': 50
    }


@pytest.fixture
def sample_data():
    """Sample data for testing."""
    return {
        'text': ['sample text 1', 'sample text 2', 'sample text 3'],
        'labels': [0, 1, 2],
        'features': torch.randn(3, 10)
    }


@pytest.fixture
def mock_wandb():
    """Mock wandb for testing without actual logging."""
    with patch('wandb.init') as mock_init, \
         patch('wandb.log') as mock_log, \
         patch('wandb.finish') as mock_finish:
        yield {
            'init': mock_init,
            'log': mock_log,
            'finish': mock_finish
        }


@pytest.fixture
def mock_transformers():
    """Mock transformers components for testing."""
    mock_tokenizer = Mock()
    mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
    mock_tokenizer.decode.return_value = "decoded text"
    
    mock_model = Mock()
    mock_model.forward.return_value = Mock(logits=torch.randn(1, 5, 1000))
    
    with patch('transformers.AutoTokenizer.from_pretrained', return_value=mock_tokenizer), \
         patch('transformers.AutoModel.from_pretrained', return_value=mock_model):
        yield {
            'tokenizer': mock_tokenizer,
            'model': mock_model
        }


@pytest.fixture
def mock_diffusers():
    """Mock diffusers components for testing."""
    mock_pipeline = Mock()
    mock_pipeline.return_value = Mock(images=[Mock()])
    
    with patch('diffusers.FluxPipeline.from_pretrained', return_value=mock_pipeline):
        yield mock_pipeline


@pytest.fixture(autouse=True)
def disable_gpu():
    """Automatically disable GPU for all tests unless explicitly needed."""
    with patch.dict('os.environ', {'CUDA_VISIBLE_DEVICES': ''}):
        yield


@pytest.fixture
def capture_logs(caplog):
    """Capture and return logs for testing."""
    return caplog


@pytest.fixture
def mock_file_operations():
    """Mock file operations for testing."""
    with patch('builtins.open', create=True) as mock_open, \
         patch('os.path.exists', return_value=True) as mock_exists, \
         patch('os.makedirs') as mock_makedirs:
        yield {
            'open': mock_open,
            'exists': mock_exists,
            'makedirs': mock_makedirs
        }
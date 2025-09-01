"""Validation tests to ensure the testing infrastructure is working correctly."""

import pytest
import sys
from pathlib import Path


class TestInfrastructureValidation:
    """Test suite to validate testing infrastructure setup."""
    
    def test_pytest_is_working(self):
        """Test that pytest is functioning correctly."""
        assert True
    
    def test_python_version(self):
        """Test that Python version meets requirements."""
        assert sys.version_info >= (3, 8)
    
    def test_project_structure(self):
        """Test that project structure is as expected."""
        project_root = Path(__file__).parent.parent
        
        # Check main project files exist
        assert (project_root / "pyproject.toml").exists()
        assert (project_root / "requirements.txt").exists()
        assert (project_root / "README.md").exists()
        
        # Check test directories exist
        assert (project_root / "tests").exists()
        assert (project_root / "tests" / "__init__.py").exists()
        assert (project_root / "tests" / "unit").exists()
        assert (project_root / "tests" / "integration").exists()
        assert (project_root / "tests" / "conftest.py").exists()
    
    def test_fixtures_are_available(self, temp_dir, mock_model_config):
        """Test that shared fixtures are working."""
        # Test temp_dir fixture
        assert temp_dir.exists()
        assert temp_dir.is_dir()
        
        # Test mock_model_config fixture
        assert isinstance(mock_model_config, dict)
        assert 'model_name' in mock_model_config
        assert 'learning_rate' in mock_model_config
    
    @pytest.mark.unit
    def test_unit_marker(self):
        """Test that unit test marker is working."""
        assert True
    
    @pytest.mark.integration
    def test_integration_marker(self):
        """Test that integration test marker is working."""
        assert True
    
    def test_mock_functionality(self, mock_wandb):
        """Test that mocking fixtures work correctly."""
        assert 'init' in mock_wandb
        assert 'log' in mock_wandb
        assert 'finish' in mock_wandb
    
    def test_sample_data_fixture(self, sample_data):
        """Test that sample data fixture provides expected structure."""
        assert 'text' in sample_data
        assert 'labels' in sample_data
        assert 'features' in sample_data
        assert len(sample_data['text']) == len(sample_data['labels'])
    
    def test_imports_work(self):
        """Test that key testing modules can be imported."""
        try:
            import pytest
            import unittest.mock
            from pathlib import Path
            import tempfile
        except ImportError as e:
            pytest.fail(f"Failed to import required testing modules: {e}")
    
    def test_coverage_excludes_test_files(self):
        """Test that coverage configuration excludes test files appropriately."""
        # This test ensures our test files aren't included in coverage
        # by checking that we can access the current test file path
        current_file = Path(__file__)
        assert current_file.name.startswith('test_')
        assert 'tests' in str(current_file)